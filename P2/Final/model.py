import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

flex_attention = torch.compile(flex_attention) # to make sure taht this is compiled

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
    


class MoE(nn.Module):
    def __init__(self, N, K, emb_dim, dropout, m):
        """
        N: number of experts
        K: number of experts choosen
        """
        super().__init__()
        self.m = m
        self.N = m * N
        self.K = m * K
        hidden = int(emb_dim * (8/3)) # 8/3 instead of *4 due to swiglu
        self.experts = nn.ModuleList(
            [FFN(emb_dim, hidden // m, emb_dim, dropout) for _ in range(self.N)]
        )
        self.gate = nn.Linear(emb_dim, self.N, bias=False)
        self.last_aux_loss = None
    
    def forward(self, x):
        B, T, C = x.shape
        flat_x = x.view(B*T, C)

        # affinity calcul
        logits = self.gate(flat_x) # (B*T, N)
        aff = F.softmax(logits, dim=-1) # softmax on experts (B*T, N)

        # select good values and norom
        aff_v, aff_i = aff.topk(self.K) # (B*T, K)
        aff_v = aff_v / aff_v.sum(-1, keepdim=True)

        with torch.no_grad():
            # f_i
            one_hot = F.one_hot(aff_i, num_classes=self.N).float() # (B*T, K, N)
            f = one_hot.sum(dim=(0, 1)) / (aff_i.numel()) # (N) sum(B*T*K) normalized
        P = aff.mean(dim=0) # (N)
        self.last_aux_loss = self.N * (f * P).sum()

        flat_out = torch.zeros_like(flat_x) # (B*T, C)
        for e in range(self.N):
            mask = (aff_i == e) # (B*T, K)
            if not mask.any():
                continue
            token_mask = mask.any(dim=-1) # for each token check if need to use this expert
            g = (aff_v * mask).sum(-1) # (B*T)
            flat_out[token_mask] += g[token_mask].unsqueeze(-1) * self.experts[e](flat_x[token_mask]) # (n_e​, 1) × (n_e​, C) = (ne​, C)

        return flat_out.view(B, T, C)



class MultiHeadLatentAttention(nn.Module):
    """
    matrix m are decomposed into m_d and m_u
    """
    def __init__(self, n_heads, emb_dim, d_latent, d_rope, context_len, dropout, device):
        super().__init__()

        self.n_heads = n_heads
        self.r = d_latent
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

        def causal(b, h, q_idx, kv_idx): return q_idx >= kv_idx
        self.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=context_len, KV_LEN=context_len, device=device)


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

        # ROPE
        q_r = self.q_r(c_q).view(B, T, self.n_heads, self.d_rope).transpose(1, 2) # (B, T, d_rope*H) -> (B, T, H, d_rope) -> (B, H, T, d_rope)
        q_r = self.apply_rotation(q_r) # (B, H, T, d_rope)
        k_r = self.apply_rotation(self.k_r(x).unsqueeze(1)) # (B, 1, T, d_rope) 
        
        # common
        Q = torch.cat([c_q_qk, q_r], dim=-1) # (B, H, T, r+d_rope)
        K = torch.cat([c_kv.unsqueeze(1), k_r], dim=-1) # (B, 1, T, r+d_rope)
        V = c_kv.unsqueeze(1) # (B, 1, T, r)

        ctx = flex_attention( # (B, H, T, r)
            Q, K, V,
            block_mask=self.block_mask,
            scale=self.scale_temp / math.sqrt(self.r + self.d_rope),
            enable_gqa=True, # flex attention enable GQA 
        )

        hd = C // self.n_heads
        Wv = self.v_u.weight.view(self.n_heads, hd, self.r) # (H, hd, r)
        out = torch.einsum('bhtr,hdr->bhtd', ctx, Wv) # (B, H, T, hd)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.o(out)

        return self.dropout(out)



class DecoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, context_len, dropout, n_experts, K, m, d_latent, d_rope, device):
        super().__init__()

        head_size = emb_dim // n_heads # dk = dv = dmodel / h
        self.multihead = MultiHeadLatentAttention(n_heads, emb_dim, d_latent, d_rope, context_len, dropout, device)
        self.moe = MoE(n_experts, K, emb_dim, dropout, m)
        self.ln1 = torch.nn.RMSNorm(emb_dim)
        self.ln2 = torch.nn.RMSNorm(emb_dim)
    
    def forward(self, x):
        # pre-norm
        x = x + self.multihead(self.ln1(x)) # (B, T, C)
        x = x + self.moe(self.ln2(x)) # (B, T, C)
        return x



class Transformer(nn.Module):
    def __init__(self, vocab_size, emb_dim, n_heads, context_len, n_block, dropout, device, 
                n_experts, K, alpha, m, d_latent, d_rope):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.blocks = nn.ModuleList(
            [DecoderBlock(emb_dim, n_heads, context_len, dropout, n_experts, K, m, d_latent, d_rope, device) for _ in range(n_block)]
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

            def causal(b, h, q_idx, kv_idx): return q_idx >= kv_idx
            block.multihead.block_mask = create_block_mask(causal, B=None, H=None, Q_LEN=new_l, KV_LEN=new_l, device=self.device)