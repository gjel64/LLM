from tokenizer import createTokenizer
from model import Transformer
from dataclasses import dataclass
import torch
from data import download_data, get_batch
from tqdm import tqdm
import math
import json

torch.set_float32_matmul_precision('high')
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device : {device}")
tokenizer = createTokenizer(["<|startoftext|>", "<|endoftext|>"])

@dataclass
class Config:
    emb_dim: int = 512 #d_model
    n_heads: int = 8
    n_group: int = 2
    context_len: int = 1024 # block_size
    n_block: int = 6 # n_layers
    dropout: float = 0.0
    lr: float = 1e-3
    vocab_size: int = tokenizer.encode("<|endoftext|>", allowed_special="all")[0] + 1 # get the real size (was pb)
    batch_size: int = 16
    n_iter: int = 10_000
    n_iter_eval: int = 50
    eval_interval: int = 250
    warmup_steps: int = 200
    warmdown_steps: int = 2000
    final_lr_frac: float = 0.1
    device = device

download_data(tokenizer, max_tokens=200_000_000)



model = Transformer(Config.vocab_size, Config.emb_dim, 
                    Config.n_heads, Config.context_len, 
                    Config.n_block, Config.dropout, device, Config.n_group)
model = model.to(device)

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=Config.lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True
)


@torch.no_grad()
def eval(nb_iter):
    model.eval()
    # eval data
    eval_loss = 0
    for i in range(nb_iter):
        X, Y = get_batch("eval", Config.context_len, device, Config.batch_size)
        logits, loss = model(X, Y)
        eval_loss += loss.item() / nb_iter
    
    train_loss = 0
    for i in range(nb_iter):
        X, Y = get_batch("train", Config.context_len, device, Config.batch_size)
        logits, loss = model(X, Y)
        train_loss += loss.item() / nb_iter
    
    return eval_loss, train_loss

def get_lr_mul(it):
    # 0 -> 1
    if it < Config.warmup_steps:
        return (it + 1) / (Config.warmup_steps)

    # 1
    if it <= Config.n_iter - Config.warmdown_steps:
        return 1.0
    
    # 1 -> 0
    progress = (Config.n_iter - it) / Config.warmdown_steps
    progress = max(0.0, min(1.0, progress)) 
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * (1.0 - progress)))
    return Config.final_lr_frac + (1.0 - Config.final_lr_frac) * cosine_decay


if device == "cuda":
    model = torch.compile(model)

train_loss_stats = []
eval_loss_stats = []
# training loop
for i in tqdm(range(Config.n_iter)):
    model.train()
    for group in optimizer.param_groups:
            group['lr'] = Config.lr * get_lr_mul(i)


    optimizer.zero_grad()
    if (i % Config.eval_interval == 0):
        eval_loss, train_loss = eval(Config.n_iter_eval)
        train_loss_stats.append(train_loss)
        eval_loss_stats.append(eval_loss)
        print(f"iter : {i} | eval_loss : {eval_loss} | train_loss : {train_loss} | lr : {Config.lr * get_lr_mul(i)}")


    X, Y = get_batch("train", Config.context_len, device, Config.batch_size)
    logits, loss = model(X, Y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

with open("stats.txt", "a") as f:
    data = {"train_loss_stats" : train_loss_stats, "eval_loss_stats": eval_loss_stats}
    f.write("\n")
    f.write(json.dumps(data))
