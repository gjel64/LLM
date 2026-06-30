import torch
import torch.nn as nn
import math

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

        self.dropout_p = dropout
        self.n_heads = n_heads
        self.head_size = head_size # for scale in flash_attention
        self.alpha = 1 # tuneable parameter -> number minimum of rotation
        self.beta = 32 # tuneable parameter -> threshold beyond which pure PI is ok

        self.register_buffer("s", torch.tensor(1.0)) # at start no additional scale
        self.register_buffer("scale_temp", (0.1 * torch.log(self.s) + 1)**2) # as in YaRN paper to scale indirectly q and k with a temperature : √(1/t) = 0.1*ln(s)+1 -> **2 because i use it as flash attention scale

        theta_i = torch.tensor([10000**(-2*i/head_size) for i in range(head_size//2)])
        self.register_buffer("theta_i", theta_i)
        self.register_buffer("wave_len", (2 * math.pi) / theta_i)
        self.register_buffer("r", context_len / self.wave_len)
        gamma = torch.zeros_like(self.r)
        for i in range(len(gamma)):
            if (self.r[i] < self.alpha):
                gamma[i] = 0 # nothing
            elif (self.r[i] > self.beta):
                gamma[i] = 1 # PI
            else:
                gamma[i] = (self.r[i] - self.alpha) / (self.beta - self.alpha) # ntk-aware
        self.register_buffer("gamma", gamma)

        
    
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
        out = torch.cat( [F.scaled_dot_product_attention(QS[i], KS[(i//self.group_size)], VS[(i//self.group_size)], 
                                                         is_causal=True, dropout_p=dropout, 
                                                         scale = self.scale_temp / math.sqrt(self.head_size)) for i in range(self.n_heads)], dim=-1 )
                                                        # scale = (1 / √head_size) * temp_scale -> temp_scale ref to YaRN
        return self.end_dropout(self.proj(out))