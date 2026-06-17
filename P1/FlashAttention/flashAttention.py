# ---------------------- Already done by pytorch ----------------------

import torch
import torch.nn as nn
from torch.nn import functional as F



class Head(nn.Module):
    def __init__(self, emb_dim, head_size, context_len, dropout):
        super().__init__()

        self.k = nn.Linear(emb_dim, head_size, bias=False) # (C, head_size)
        self.q = nn.Linear(emb_dim, head_size, bias=False) # (C, head_size)
        self.v = nn.Linear(emb_dim, head_size, bias=False) # (C, head_size)
        self.register_buffer('tril', torch.tril(torch.ones(context_len, context_len)))
        self.dropoutp = dropout

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q(x) # (B, T, head_size)
        K = self.k(x) # (B, T, head_size)
        V = self.v(x) # (B, T, head_size)

        return F.scaled_dot_product_attention( Q, K, V, 
            attn_mask=self.tril, is_causal=True,
            dropout_p=(self.self.dropoutp if self.training else 0.0) )

# ...

    
# ---------------------- Mine ----------------------

import math

# usefull
block_size_q = 1024
block_size_kv = 1024
num_stages = 2
N = 1024 # sequence length
d = 64 # dk = dv = dmodel / h


# try to do mine
def scaled_dot_product_attention(Q, K, V):
    B, N, d = Q.shape
    scale = 1.0 / d**0.5

    out = torch.full((B, N, d), 0)
    l = torch.full((B, N, 1), 0)
    m = torch.full((B, N, 1), float("-inf"))

    nb_qs = N // block_size_q
    nb_ks_and_vs = N // block_size_kv

    for i in range(nb_ks_and_vs):
        k_start, k_end = i * block_size_kv, (i + 1) * block_size_kv
        Kj = K[:, k_start:k_end, :]   # (B, block_size_kv, d)
        Vj = V[:, k_start:k_end, :]   # (B, block_size_kv, d)

        for j in range(nb_qs):
            q_start, q_end = j * block_size_q, (j + 1) * block_size_q
            Qi = Q[:, q_start:q_end, :]  # (B, block_size_q, d)

            S_ij = (Qi @ Kj.transpose(-2, -1)) * scale  # (B, block_size_q, block_size_kv)

            m_ij = S_ij.max(dim=-1, keepdim=True).values  # (B, block_size_q, 1)

            m_old = m[:, q_start:q_end, :]
            m_new = torch.maximum(m_old, m_ij)

            P_ij = torch.exp(S_ij - m_new) 

            l_old = l[:, q_start:q_end, :]
            alpha = torch.exp(m_old - m_new)

            l_new = alpha * l_old + P_ij.sum(dim=-1, keepdim=True)

            O_old = out[:, q_start:q_end, :]
            O_new = alpha * O_old + P_ij @ Vj 

            out[:, q_start:q_end, :] = O_new
            l[:, q_start:q_end, :] = l_new
            m[:, q_start:q_end, :] = m_new

    out = out / l
    return out







