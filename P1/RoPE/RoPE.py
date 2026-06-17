import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class Head(nn.Module):
    def __init__(self, emb_dim, head_size, context_len, dropout):
        super().__init__()

        self.k = nn.Linear(emb_dim, head_size, bias=False) # (C, head_size)
        self.q = nn.Linear(emb_dim, head_size, bias=False) # (C, head_size)
        self.v = nn.Linear(emb_dim, head_size, bias=False) # (C, head_size)
        self.dropoutp = dropout
        theta_i = torch.tensor([10000**(-2*i/head_size) for i in range(head_size//2)])
        self.register_buffer("theta_i", theta_i)

    def apply_rotation(self, matrix):
        B, T, head_size = matrix.shape
        
        m = torch.arange(T, device=matrix.device, dtype=self.theta_i.dtype) # positions

        angles = torch.outer(m, self.theta_i) # (T, head_size//2)

        cos = torch.cos(angles)  # (T, head_size//2)
        sin = torch.sin(angles)  # (T, head_size//2)

        cos = cos.unsqueeze(0) # (1, T, head_size//2) for batch dimension
        sin = sin.unsqueeze(0) # (1, T, head_size//2) for batch dimension

        x1 = matrix[:, :, 0::2]  # (B, T, head_size//2)
        x2 = matrix[:, :, 1::2]  # (B, T, head_size//2)

        new_x1 = x1 * cos - x2 * sin
        new_x2 = x1 * sin + x2 * cos

        out = torch.empty_like(matrix)
        out[:, :, 0::2] = new_x1
        out[:, :, 1::2] = new_x2

        return out

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q(x) # (B, T, head_size)
        K = self.k(x) # (B, T, head_size)
        V = self.v(x) # (B, T, head_size)

        Q = self.apply_rotation(Q)
        K = self.apply_rotation(K)

        return F.scaled_dot_product_attention( Q, K, V,
            is_causal=True,
            dropout_p=(self.dropoutp if self.training else 0.0) )
    

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, emb_dim, head_size, context_len, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(emb_dim, head_size, context_len, dropout) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, emb_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))



class FFN(nn.Module):
    def __init__(self, in_f, n_hidden, out_f, dropout):
        super().__init__()
        self.l1 = nn.Linear(in_f, n_hidden)
        self.l2 =  nn.Linear(n_hidden, out_f)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout( self.l2( self.act( self.l1(x) ) ) )



class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, context_len, dropout):
        super().__init__()

        head_size = emb_dim // n_heads # dk = dv = dmodel / h

        self.multihead = MultiHeadAttention(n_heads, emb_dim, head_size, context_len, dropout)
        self.FFN = FFN(emb_dim, emb_dim*4, emb_dim, dropout)
        self.ln1 = torch.nn.LayerNorm(emb_dim)
        self.ln2 = torch.nn.LayerNorm(emb_dim)
    
    def forward(self, x):
        x = x + self.multihead(self.ln1(x)) # (B, T, C)
        x = x + self.FFN(self.ln2(x)) # (B, T, C)
        return x


        

class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, context_len, n_block, dropout, device):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.blocks = nn.Sequential(
            *[DecoderBlock(emb_dim, n_heads, context_len, dropout) for _ in range(n_block)]
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.l1 = nn.Linear(emb_dim, vocab_size, bias=False)
        
        self.device = device
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

        # x = tok_emb + pos_emb -> no need : RoPE

        x = self.blocks(x) # (B, T, C)
        x = self.ln(x)

        if y is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.l1(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
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