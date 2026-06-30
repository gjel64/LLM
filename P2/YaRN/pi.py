import torch.nn as nn
import torch

class GroupedQueryAttention(nn.Module):
    def __init__(self, n_heads, emb_dim, head_size, context_len, dropout, n_groups):
        super().__init__()
        assert n_heads % n_groups == 0
        self.group_size = n_heads // n_groups # the number of heads per group
        self.n_groups = n_groups

        self.ks = nn.ModuleList([nn.Linear(emb_dim, head_size, bias=False) for _ in range(n_groups)])
        self.vs = nn.ModuleList([nn.Linear(emb_dim, head_size, bias=False) for _ in range(n_groups)])
        self.qs = nn.ModuleList([nn.Linear(emb_dim, head_size, bias=False) for _ in range(n_heads)])

        self.n_heads = n_heads
        self.proj = nn.Linear(n_heads * head_size, emb_dim)
        self.end_dropout = nn.Dropout(dropout)
        self.dropout_p = dropout

        theta_i = torch.tensor([10000**(-2*i/head_size) for i in range(head_size//2)])
        self.register_buffer("theta_i", theta_i)
    
    def apply_rotation(self, matrix, s):
        # s : scale of change -> L / L'
        B, T, head_size = matrix.shape
        
        m = torch.arange(T, device=matrix.device, dtype=self.theta_i.dtype) * s # positions

        angles = torch.outer(m, self.theta_i) # (T, head_size//2)

        cos = torch.cos(angles)  # (T, head_size//2)
        sin = torch.sin(angles)  # (T, head_size//2)

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