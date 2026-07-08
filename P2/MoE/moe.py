import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MoE(nn.Module):
 
    def __init__(self, N, K, emb_dim, dropout, m):
        """
        N: nombre d'experts (de base)
        K: nombre d'experts choisis (de base)
        m: granularite (fine-grained experts) -> N_eff = m*N, K_eff = m*K,
           hidden par expert divise par m
        """
        super().__init__()
        self.m = m
        self.N = m * N
        self.K = m * K
        self.emb_dim = emb_dim
        hidden = int(emb_dim * (8 / 3))       # 8/3 au lieu de *4 a cause de SwiGLU
        self.hidden = hidden // m             # hidden par expert
 
        self.gate = nn.Linear(emb_dim, self.N, bias=False)
 
        self.w1 = torch.empty(self.N, emb_dim, self.hidden)
        self.v = torch.empty(self.N, emb_dim, self.hidden)
        self.w2 = torch.empty(self.N, self.hidden, emb_dim)
        self.b1 = torch.zeros(self.N, self.hidden)
        self.bv = torch.zeros(self.N, self.hidden)
        self.b2 = torch.zeros(self.N, emb_dim)
 
        self.dropout = nn.Dropout(dropout)
        self.last_aux_loss = None
        self._reset_expert_params()
 
    def _reset_expert_params(self):
        # same init as all the model
        for p in (self.w1, self.v, self.w2):
            nn.init.normal_(p, mean=0.0, std=0.02)
        for b in (self.b1, self.bv, self.b2):
            nn.init.zeros_(b)
 
    def _grouped_swiglu(self, x_sorted, offs):
        h = torch._grouped_mm(x_sorted, self.w1, offs=offs) # (M, hidden)
        swish = h * torch.sigmoid(h)
        gate = torch._grouped_mm(x_sorted, self.v,  offs=offs) # (M, hidden)
        out = torch._grouped_mm(swish * gate, self.w2, offs=offs) # (M, C)
        return out
 
    def forward(self, x):
        B, T, C = x.shape
        x = x.reshape(B * T, C)
        S = x.shape[0] # = B*T
 
        logits = self.gate(x) # (S, N)
        aff = F.softmax(logits, dim=-1) # softmax on experts
        aff_v, aff_i = aff.topk(self.K, dim=-1)  # (S, K)
        aff_v = aff_v / aff_v.sum(-1, keepdim=True) # norm top-K
 
        with torch.no_grad():
            one_hot = F.one_hot(aff_i, num_classes=self.N).float() # (S, K, N)
            f = one_hot.sum(dim=(0, 1)) / aff_i.numel() # (N)
        P = aff.mean(dim=0) # (N)
        self.last_aux_loss = self.N * (f * P).sum()
 
        expert_idx = aff_i.reshape(-1) # (S*K)
        gates = aff_v.reshape(-1) # (S*K)
        token_idx = torch.arange(S, device=x.device).repeat_interleave(self.K) # (S*K)
 
        sort_order = torch.argsort(expert_idx) # sort per expert
        expert_idx_s = expert_idx[sort_order]
        token_idx_s = token_idx[sort_order]
        gates_s = gates[sort_order]
 
        x_sorted = x[token_idx_s] # (S*K, C)
 
        # nb tokens per expert
        counts = torch.bincount(expert_idx_s, minlength=self.N).tolist()
 
        # grouped GEMM (block-sparse emule)
        out_sorted = self._grouped_swiglu(x_sorted, counts) # (S*K, C)
        out_sorted = out_sorted * gates_s.unsqueeze(-1)
 
        out = torch.zeros_like(x)
        out.index_add_(0, token_idx_s, out_sorted) # somme des K contributions
 
        out = self.dropout(out)
        return out.view(B, T, C)
 


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
    def __init__(self, emb_dim, n_heads, context_len, dropout, n_groups, n_experts, K, m):
        super().__init__()

        head_size = emb_dim // n_heads # dk = dv = dmodel / h

        self.multihead = GroupedQueryAttention(n_heads, emb_dim, head_size, context_len, dropout, n_groups)
        self.moe = MoE(n_experts, K, emb_dim, dropout, m)
        self.ln1 = torch.nn.RMSNorm(emb_dim)
        self.ln2 = torch.nn.RMSNorm(emb_dim)
    
    def forward(self, x):
        # pre-norm
        x = x + self.multihead(self.ln1(x)) # (B, T, C)
        x = x + self.moe(self.ln2(x)) # (B, T, C)
        return x



class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, context_len, n_block, dropout, device, n_groups, n_experts, K, alpha, m):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(emb_dim, n_heads, context_len, dropout, n_groups, n_experts, K, m) for _ in range(n_block)]
        )
        self.ln = nn.RMSNorm(emb_dim)
        self.l1 = nn.Linear(emb_dim, vocab_size, bias=False)
        
        self.device = device
        self.alpha = alpha # coef for MoE loss
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

        aux_losses = []
        for block in self.blocks:
            x = block(x) # (B, T, C)
            aux_losses.append(block.moe.last_aux_loss)
        
        x = self.ln(x)

        aux_loss = torch.stack(aux_losses).mean() # mean over n_block MoE

        if y is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.l1(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            loss = loss + self.alpha * aux_loss # switch transformer
            logits = None # for the auto_chunk
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