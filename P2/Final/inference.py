import torch
from tokenizer import createTokenizer
from dataclasses import dataclass
from model import Transformer


torch.set_float32_matmul_precision('high')
tokenizer = createTokenizer(["<|startoftext|>", "<|endoftext|>"])
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device : {device}")

@dataclass
class Config:
    emb_dim: int = 512 #d_model
    n_heads: int = 4
    n_group: int = 2
    n_iter: int = 16000 # found by hand to do 1 epoch
    context_lens = [2_048, 8_192, 32_768] # block_size
    n_iter_per_context_len = [int(n_iter * 0.8), int(n_iter * 0.15), int(n_iter * 0.05) ]
    n_block: int = 4 # n_layers (reduce on MoE)
    dropout: float = 0.0
    lr: float = 1e-3
    vocab_size: int = tokenizer.encode("<|startoftext|>", allowed_special="all")[0] + 1 # get the real size (was pb)
    batch_size: int = 32
    n_iter_eval: int = int(80)
    eval_interval: int = int(400)
    warmup_coef: int = 0.02
    final_lr_frac: float = 0.1
    max_tokens: int = 0 # will be define with the size of the model as said in Chinchilla
    device = device
    n_experts : int = 4
    top_k : int = 1
    m : int = 3
    alpha: float = 0.01

context_len = Config.context_lens[-1]

model = Transformer(Config.vocab_size, Config.emb_dim, 
                    Config.n_heads, context_len, 
                    Config.n_block, Config.dropout, device, Config.n_group, 
                    Config.n_experts, Config.top_k, Config.alpha, Config.m)
model = model.to(device)

state_dict = torch.load("model.pt", map_location=device, weights_only=True)
# kendu compile
state_dict = { (k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k): v 
               for k, v in state_dict.items() }

model.load_state_dict(state_dict)

model.eval()

while (True):
    prompt = input("> ")
    p_en = tokenizer.encode(prompt, allowed_special="all")
    text = torch.tensor([p_en, p_en], device=device) # because batch size = 2
    res = model.generate(text, 50, context_len)
    r_l = res.cpu().detach().numpy()
    for r in r_l:
        print("--")
        print(tokenizer.decode(r))
    print("\n")
    print("-------------------")
