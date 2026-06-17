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
        self.head_size_sr = head_size ** 0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        Q = self.q(x) # (B, T, head_size)
        K = self.k(x) # (B, T, head_size)
        K = K.transpose(-2, -1) #  K^T (B, head_size, T)
        V = self.v(x) # (B, T, head_size)

        W = (Q @ K) / self.head_size_sr # (B, T, T)
        W = W.masked_fill(self.tril[:T, :T] == 0, float('-inf'))

        W = F.softmax(W, dim=-1) # (B, T, T)
        W = self.dropout(W)

        return W @ V # (B, T, head_size)



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
        self.emb_position = nn.Embedding(context_len, emb_dim)
        self.blocks = nn.Sequential(
            *[DecoderBlock(emb_dim, n_heads, context_len, dropout) for _ in range(n_block)]
        )
        self.ln = nn.LayerNorm(emb_dim)
        self.l1 = nn.Linear(emb_dim, vocab_size)
        self.device = device

        self.emb.weight = self.l1.weight # emb and l1 are doing the same thing

    def forward(self, x):
        B, T = x.shape # (B, T)
        tok_emb = self.emb(x) # (B, T, C)
        pos_emb = self.emb_position(torch.arange(T, device=self.device)) # (B, T, C)

        x = tok_emb + pos_emb # (B, T, C)

        x = self.blocks(x) # (B, T, C)
        x = self.ln(x)
        logits = self.l1(x) # (B, T, vocab_size)

        return logits
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, context_len):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, idx_next], dim=1)
        return idx

    




