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
        # ((Swish_1(x @ W1)) * (x @ V)) @ W2      -> SwigLU
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

        self.ks = nn.ModuleList([nn.Linear(emb_dim, head_size, bias=False) for _ in range(n_groups)])
        self.vs = nn.ModuleList([nn.Linear(emb_dim, head_size, bias=False) for _ in range(n_groups)])
        self.qs = nn.ModuleList([nn.Linear(emb_dim, head_size, bias=False) for _ in range(n_heads)])

        self.proj = nn.Linear(n_heads * head_size, emb_dim)
        self.end_dropout = nn.Dropout(dropout)

        theta_i = torch.tensor([10000**(-2*i/head_size) for i in range(head_size//2)])
        self.register_buffer("theta_i", theta_i)
        self.dropout_p = dropout
        self.n_heads = n_heads
        self.register_buffer("wave_len", (2 * math.pi) / theta_i)
        self.register_buffer("r", context_len / self.wave_len)
        self.alpha = 1 # tuneable parameter -> number minimum of rotation
        self.beta = 32 # tuneable parameter -> threshold beyond which pure PI is ok
        gamma = torch.zeros_like(self.r)
        for i in range(len(gamma)):
            if (self.r[i] < self.alpha):
                gamma[i] = 0 # nothing
            elif (self.r[i] > self.beta):
                gamma[i] = 1 # PI
            else:
                gamma[i] = (self.r[i] - self.alpha) / (self.beta - self.alpha) # ntk-aware
        self.register_buffer("gamma", gamma)
        self.register_buffer("s", torch.tensor(1.0)) # at start no additional scale
        
    
    def apply_rotation(self, matrix):
        # s = L / L'
        B, T, head_size = matrix.shape
        
        theta_interpolated = ((1 - self.gamma) * self.theta_i / self.s) + (self.gamma * self.theta_i)

        m = torch.arange(T, device=matrix.device, dtype=self.theta_i.dtype) # positions

        h = torch.outer(m, theta_interpolated)

        cos = torch.cos(h)  # (T, head_size//2)
        sin = torch.sin(h)  # (T, head_size//2)

        cos = cos.unsqueeze(0) # (1, T, head_size//2)
        sin = sin.unsqueeze(0) # (1, T, head_size//2)

        x1 = matrix[:, :, 0::2]  # (B, T, head_size//2)
        x2 = matrix[:, :, 1::2]  # (B, T, head_size//2)

        new_x1 = x1 * cos - x2 * sin
        new_x2 = x1 * sin + x2 * cos
        out = torch.empty_like(matrix)
        out[:, :, 0::2] = new_x1
        out[:, :, 1::2] = new_x2

        return out
    
    def forward(self, x):
        KS = [self.apply_rotation(self.ks[i](x)) for i in range(self.n_groups)]
        VS = [self.vs[i](x) for i in range(self.n_groups)]

        QS = [self.apply_rotation(self.qs[i](x)) for i in range(len(self.qs))]

        dropout = self.dropout_p if self.training else 0.0
        out = torch.cat( [F.scaled_dot_product_attention(QS[i], KS[(i//self.group_size)], VS[(i//self.group_size)], is_causal=True, dropout_p=dropout) for i in range(self.n_heads)], dim=-1 )
        return self.end_dropout(self.proj(out))



class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, context_len, dropout, n_groups):
        super().__init__()

        head_size = emb_dim // n_heads # dk = dv = dmodel / h

        self.multihead = GroupedQueryAttention(n_heads, emb_dim, head_size, context_len, dropout, n_groups)
        self.FFN = FFN(emb_dim, emb_dim*4, emb_dim, dropout)
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
        self.context_len = context_len

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
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx
    
    def change_context_len(self, new_l):
        new_s = torch.tensor(new_l / self.context_len)
        for block in self.blocks:
            block.multihead.s.fill_(new_s)