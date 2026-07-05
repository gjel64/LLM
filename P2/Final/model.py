import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class FFN(nn.Module):
    def __init__(self, in_f, n_hidden, out_f, dropout):
        super().__init__()
        
        self.w1 = nn.Linear(in_f, n_hidden)
        self.w2 =  nn.Linear(n_hidden, out_f)
        self.v = nn.Linear(in_f, n_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ((Swish_1(x @ W1)) * (x @ V)) @ W2 -> SwigLU
        x_w1 = self.w1(x)
        Swish_1 = x_w1 * torch.sigmoid(x_w1)
        x_v = self.v(x)
        out = self.w2(Swish_1 * x_v)
        return self.dropout(out)
    


class GroupedQueryAttention(nn.Module):
    def __init__(self, n_heads, emb_dim, head_size, context_len, dropout, n_groups):
        super().__init__()
        assert n_heads % n_groups == 0
        self.group_size = n_heads // n_groups # the number of heads per group
        self.n_groups = n_groups
        self.n_heads = n_heads
        self.head_size = head_size # for scale in flash_attention
        self.k_proj = nn.Linear(emb_dim, n_groups * head_size, bias=False) # abreviation to GQA L(C, hs) * G = L(C, hs*G)
        self.v_proj = nn.Linear(emb_dim, n_groups * head_size, bias=False)
        self.q_proj = nn.Linear(emb_dim, n_heads * head_size, bias=False) # abreviation to GQA L(C, hs) * H = L(C, hs*H)

        self.proj = nn.Linear(n_heads * head_size, emb_dim)
        self.end_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        self.alpha = 1 # tuneable parameter -> number minimum of rotation
        self.beta = 32 # tuneable parameter -> threshold beyond which pure PI is ok
        self.s = 1.0 # at start no additional scale
        self.scale_temp = (0.1 * math.log(self.s) + 1)**2 # as in YaRN paper to scale indirectly q and k with a temperature : √(1/t) = 0.1*ln(s)+1 -> **2 because i use it as flash attention scale


        theta_i = torch.tensor([10000**(-2*i/head_size) for i in range(head_size//2)])
        self.register_buffer("theta_i", theta_i)
        wave_len = (2 * math.pi) / theta_i
        r = context_len / wave_len
        gamma = torch.clamp((r - self.alpha) / (self.beta - self.alpha), 0, 1) # ntk-aware
        gamma = torch.where(r < self.alpha, torch.zeros_like(gamma), gamma) # PI
        gamma = torch.where(r > self.beta, torch.ones_like(gamma), gamma) # nothing
        self.register_buffer("gamma", gamma)

    def apply_rotation(self, x):
        B, n, T, hs = x.shape

        theta_interpolated = ((1 - self.gamma) * self.theta_i / self.s) + (self.gamma * self.theta_i)
        m = torch.arange(T, device=x.device, dtype=self.theta_i.dtype)
        h = torch.outer(m, theta_interpolated) # (T, head_size/2)
        
        cos = torch.cos(h).view(1, 1, T, hs // 2) # (1, 1, T, head_size/2)
        sin = torch.sin(h).view(1, 1, T, hs // 2) # (1, 1, T, head_size/2)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
    
    def forward(self, x):
        B, T, _ = x.shape

        # better than lot of Ks, Vs, Qs -> big matmul -> parallelism
        K = self.k_proj(x).view(B, T, self.n_groups, self.head_size).transpose(1, 2) # (B, n_groups, T, head_size)
        V = self.v_proj(x).view(B, T, self.n_groups, self.head_size).transpose(1, 2) # (B, n_groups, T, head_size)
        Q = self.q_proj(x).view(B, T, self.n_heads, self.head_size).transpose(1, 2)  # (B, n_heads, T, head_size)

        K = self.apply_rotation(K) # (B, n_groups, T, head_size)
        Q = self.apply_rotation(Q) # (B, n_heads, T, head_size)

        dropout = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            Q, K, V,
            is_causal=True,
            dropout_p=dropout,
            enable_gqa=True, # broadcast G -> H
            scale=self.scale_temp / math.sqrt(self.head_size),
        )  # (B, H, T, hs)

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_size)
        return self.end_dropout(self.proj(out))



class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, context_len, dropout, n_groups):
        super().__init__()

        head_size = emb_dim // n_heads # dk = dv = dmodel / h

        self.multihead = GroupedQueryAttention(n_heads, emb_dim, head_size, context_len, dropout, n_groups)
        self.FFN = FFN(emb_dim, int(emb_dim*(8/3)), emb_dim, dropout) # 8/3 instead of *4 due to swiglu
        self.ln1 = torch.nn.RMSNorm(emb_dim)
        self.ln2 = torch.nn.RMSNorm(emb_dim)
    
    def forward(self, x):
        # pre-norm
        x = x + self.multihead(self.ln1(x)) # (B, T, C)
        x = x + self.FFN(self.ln2(x)) # (B, T, C)
        return x



class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, context_len, n_block, dropout, device, n_groups):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.blocks = nn.Sequential(
            *[DecoderBlock(emb_dim, n_heads, context_len, dropout, n_groups) for _ in range(n_block)]
        )
        self.ln = nn.RMSNorm(emb_dim)
        self.l1 = nn.Linear(emb_dim, vocab_size, bias=False)
        
        self.device = device
        self.initial_context_len = context_len

        self.emb.weight = self.l1.weight # emb and l1 are doing the same thing
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_block))
            

        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params
    
    def forward(self, x, y=None):
        B, T = x.shape # (B, T)
        x = self.emb(x) # (B, T, C)

        x = self.blocks(x) # (B, T, C)
        x = self.ln(x)

        if y is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.l1(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position (Karpathy)
            logits = self.l1(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, context_len):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]
            logits, _ = self(idx_cond)
            logit = logits[:, -1, :]
            probs = F.softmax(logit, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
    
    def change_context_len(self, new_l):
        new_s = new_l / self.initial_context_len
        for block in self.blocks:
            block.multihead.s = new_s
            block.multihead.scale_temp = (0.1 * math.log(new_s) + 1)**2