import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    matrix m are decomposed into m_D and m_U
    """
    def __init__(self, n_heads, emb_dim, d_latent, context_len, dropout):
        super().__init__()
        head_size = emb_dim // n_heads # dk = dv = dmodel / h
        self.n_heads = n_heads
        self.r = d_latent
        self.register_buffer('mask', torch.tril(torch.ones(context_len, context_len)))
        self.q_d = nn.Linear(emb_dim, self.r, bias=False)
        self.kv_d = nn.Linear(emb_dim, self.r, bias=False)

        # Precomputed matrix multiplications of q_U and k_U, for multiple heads
        self.qk_u = nn.Linear(self.r, n_heads * self.r, bias=False)
        self.v_u = nn.Linear(self.r, emb_dim, bias=False)

        self.o = nn.Linear(emb_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
        """
        = softmax((C_q @ Wu_q @ Wu_k^T @ C_kv^T) / sqrt(d_k) ) @ C_kv @ Wu_c   -> refer to picture
        """
        B, T, C = x.shape

        # Projections of input into latent spaces
        c_q = self.q_d(x) # (B, T, r)
        c_kv = self.kv_d(x) # (B, T, r)

        c_q_qk = self.qk_u(c_q).view(B, T, self.n_heads, self.r).transpose(1, 2) # (B, T, H*r) -> (B, T, H, r) -> (B, H, T, r)
        scores = (c_q_qk @ c_kv.transpose(-2, -1)[:, None, ...]) / math.sqrt(self.r) # (B, H, T, r) @ (B, -1, r, T) = (B, H, T, T)

        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # (B, H, T, T)

        attn_weight = torch.softmax(scores, dim=-1) # (B, H, T, T)
        attn_weight = self.dropout(attn_weight)

        # Restore V from latent space
        V = self.v_u(c_kv).view(B, T, self.n_heads, -1) # (B, T, C) -> (B, T, H, -1)

        output = (attn_weight @ V.transpose(1,2)).transpose(1,2).contiguous() # (B, H, T, -1) -> (B, H, T, -1)

        output = self.o(output.view(B, T, -1))
        return self.dropout(output)
