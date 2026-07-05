import torch
from tokenizer import createTokenizer
from dataclasses import dataclass
from model import Transformer


torch.set_float32_matmul_precision('high')
tokenizer = createTokenizer(["<|startoftext|>", "<|endoftext|>"])
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device : {device}")

class Config:
    emb_dim: int = 512 #d_model
    n_heads: int = 4
    n_group: int = 2
    n_iter: int = 20_000
    context_len = 32_768 # block_size
    n_block: int = 4 # n_layers
    dropout: float = 0.0
    vocab_size: int = tokenizer.encode("<|endoftext|>", allowed_special="all")[0] + 1 # get the real size (was pb)
    batch_size: int = 2
    device = device

model = Transformer(Config.vocab_size, Config.emb_dim, 
                    Config.n_heads, Config.context_len, 
                    Config.n_block, Config.dropout, device, Config.n_group)
model = torch.load("model.pt", weights_only=False) # because I save it the wrong way
model = model.to(device)
model.eval()

while (True):
    prompt = input("> ")
    p_en = tokenizer.encode(prompt)
    text = torch.tensor([p_en, p_en], device=device) # because batch size = 2
    res = model.generate(text, 50, Config.context_len)
    r_l = res.cpu().detach().numpy()
    for r in r_l:
        print("--")
        print(tokenizer.decode(r))
    print("\n")
    print("-------------------")
