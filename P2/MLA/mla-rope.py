import torch
import torch.nn as nn
from torch.nn import functional as F
import math

class MultiHeadLatentAttention(nn.Module):
    """
    matrix m are decomposed into m_D and m_U
    """
    def __init__(self, n_heads, emb_dim, d_latent, d_rope, context_len, dropout):
        super().__init__()

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

        # ROPE :
        self.alpha = 1 # tuneable parameter -> number minimum of rotation
        self.beta = 32 # tuneable parameter -> threshold beyond which pure PI is ok
        self.s = 1.0 # at start no additional scale
        self.scale_temp = (0.1 * math.log(self.s) + 1)**2 # as in YaRN paper to scale indirectly q and k with a temperature : √(1/t) = 0.1*ln(s)+1 -> **2 because i use it as flash attention scale


        theta_i = torch.tensor([10000**(-2*i/d_rope) for i in range(d_rope//2)])
        self.register_buffer("theta_i", theta_i)
        wave_len = (2 * math.pi) / theta_i
        r = context_len / wave_len
        gamma = torch.clamp((r - self.alpha) / (self.beta - self.alpha), 0, 1) # ntk-aware
        gamma = torch.where(r < self.alpha, torch.zeros_like(gamma), gamma) # PI
        gamma = torch.where(r > self.beta, torch.ones_like(gamma), gamma) # nothing
        self.register_buffer("gamma", gamma)

        self.d_rope = d_rope
        self.q_r = nn.Linear(self.r, n_heads * self.d_rope, bias=False)
        self.k_r = nn.Linear(emb_dim, self.d_rope, bias=False)


    def apply_rotation(self, x):
        B, n, T, hs = x.shape

        theta_interpolated = ((1 - self.gamma) * self.theta_i / self.s) + (self.gamma * self.theta_i)
        m = torch.arange(T, device=x.device, dtype=self.theta_i.dtype)
        h = torch.outer(m, theta_interpolated) # (T, d_rope/2)
        
        cos = torch.cos(h).view(1, 1, T, hs // 2) # (1, 1, T, d_rope/2)
        sin = torch.sin(h).view(1, 1, T, hs // 2) # (1, 1, T, d_rope/2)

        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out
    
    def forward(self, x):
        """
        = softmax((C_q @ Wu_q @ Wu_k^T @ C_kv^T) / sqrt(d_k) ) @ C_kv @ Wu_c   -> refer to picture
        """
        B, T, C = x.shape

        # Projections of input into latent spaces
        c_q = self.q_d(x) # (B, T, r)
        c_kv = self.kv_d(x) # (B, T, r)

        c_q_qk = self.qk_u(c_q).view(B, T, self.n_heads, self.r).transpose(1, 2) # (B, T, H*r) -> (B, T, H, r) -> (B, H, T, r)
        raw_scores = (c_q_qk @ c_kv.transpose(-2, -1)[:, None, ...]) # (B, H, T, r) @ (B, -1, r, T) = (B, H, T, T)

        # ROPE
        q_r = self.q_r(c_q).view(B, T, self.n_heads, self.d_rope).transpose(1, 2) # (B, T, d_rope*H) -> (B, T, H, d_rope) -> (B, H, T, d_rope)
        q_r = self.apply_rotation(q_r) # (B, H, T, d_rope)
        k_r = self.apply_rotation(self.k_r(x).unsqueeze(1)) # (B, 1, T, d_rope) 
        rope_scores = q_r @ k_r.transpose(-2, -1)
        
        # common
        scores = (raw_scores + rope_scores) / math.sqrt(self.r + self.d_rope) * self.scale_temp # YaRN scale 
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf')) # (B, H, T, T)

        attn_weight = torch.softmax(scores, dim=-1) # (B, H, T, T)
        attn_weight = self.dropout(attn_weight)

        # Restore V from latent space
        V = self.v_u(c_kv).view(B, T, self.n_heads, -1) # (B, T, C) -> (B, T, H, -1)

        output = (attn_weight @ V.transpose(1,2)).transpose(1,2).contiguous() # (B, H, T, -1) -> (B, H, T, -1)

        output = self.o(output.view(B, T, -1))
        return self.dropout(output)

