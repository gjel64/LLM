import torch
import torch.nn as nn
from torch.nn import functional as F


class FFN(nn.Module):
    def __init__(self, in_f, n_hidden, out_f, dropout):
        super().__init__()
        
        self.w1 = nn.Linear(in_f, n_hidden)
        self.w2 =  nn.Linear(n_hidden, out_f)
        self.v = nn.Linear(in_f, n_hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # ((Swish_1(x @ W1)) * (x @ V)) @ W2
        x_w1 = self.w1(x)
        Swish_1 = x_w1 * torch.sigmoid(x_w1)
        x_v = self.v(x)
        out = self.w2(Swish_1 * x_v)
        return self.dropout(out)