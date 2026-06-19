from bpe import BPE
from model import Transformer
from dataclasses import dataclass
import torch
from tqdm import tqdm
import json
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument("--emb_dim", type=int, default=256)
parser.add_argument("--context_len", type=int, default=512)
parser.add_argument("--n_block", type=int, default=4)
args = parser.parse_args()

@dataclass
class Config:
    emb_dim: int = args.emb_dim #d_model
    n_heads: int = 4
    n_group: int = 2
    context_len: int = args.context_len # block_size
    n_block: int = args.n_block # n_layers
    dropout: float = 0.2
    lr: float = 3e-4
    vocab_size: int = 128 + 256
    batch_size: int = 4
    n_iter: int = 20000
    n_iter_eval: int = 20
    eval_interval: int = 100
    warmup_steps: int = n_iter // 5
    warmdown_steps: int = n_iter // 10
    final_lr_frac: float = 0.1


device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


print("LOG: creating Tokenizer...")
text = open("tinyShakespeare.txt").read()
bpe = BPE(text, Config.vocab_size, special_tokens=False)
bpe.save("bpe.json")
print("LOG: Tokenizing text... ")
encoded_text = bpe.encode(text)
split = int(len(encoded_text) * 0.9)
train_text = encoded_text[:split]
eval_text = encoded_text[split:]
print("text : %.4fM tokens" % (len(encoded_text)/1e6))


print("LOG: creating Model...")
model = Transformer(Config.vocab_size, Config.emb_dim, 
                    Config.n_heads, Config.context_len, 
                    Config.n_block, Config.dropout, device, Config.n_group)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), Config.lr)


def get_batch(context_len, text, batch_size):
    max_start = len(text) - context_len
    ix = torch.randint(max_start, (batch_size,))

    x = torch.stack([torch.tensor(text[i:i+context_len]) for i in ix])
    y = torch.stack([torch.tensor(text[i+1:i+1+context_len]) for i in ix])

    if device == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y

@torch.no_grad()
def eval(nb_iter):
    model.eval()
    # eval data
    eval_loss = 0
    for i in range(nb_iter):
        X, Y = get_batch(Config.context_len, eval_text, Config.batch_size)
        logits, loss = model(X, Y)
        eval_loss += loss.item() / nb_iter
    
    train_loss = 0
    for i in range(nb_iter):
        X, Y = get_batch(Config.context_len, train_text, Config.batch_size)
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






print(f"training for {Config.n_iter*Config.batch_size*Config.context_len / len(train_text)} epochs")


train_loss_stats = []
eval_loss_stats = []
# training loop
for i in tqdm(range(Config.n_iter)):
    model.train()
    optimizer.zero_grad()

    if (i % Config.eval_interval == 0):
        eval_loss, train_loss = eval(Config.n_iter_eval)
        train_loss_stats.append(train_loss)
        eval_loss_stats.append(eval_loss)
        print(f"iter : {i} | eval_loss : {eval_loss} | train_loss : {train_loss} | lr : {Config.lr * get_lr_mul(i)}")

    for group in optimizer.param_groups:
            group['lr'] = Config.lr * get_lr_mul(i)
    X, Y = get_batch(Config.context_len, train_text, Config.batch_size)
    logits, loss = model(X, Y)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

with open("stats.txt", "a") as f:
    data = {"train_loss_stats" : train_loss_stats, "eval_loss_stats": eval_loss_stats}
    f.write("\n")
    f.write(json.dumps(data))



