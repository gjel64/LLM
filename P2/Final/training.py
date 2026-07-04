import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from tokenizer import createTokenizer
from model import Transformer
from dataclasses import dataclass
import torch
from torch.nn.attention import sdpa_kernel, SDPBackend
from data import download_data, get_batch
from tqdm import tqdm
import math
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="unknown")
args = parser.parse_args()

torch.set_float32_matmul_precision('high')
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device : {device}")
tokenizer = createTokenizer(["<|startoftext|>", "<|endoftext|>"])

# auto chunker on compile for the cross_entropy
from torch._inductor import config as inductor_config
inductor_config.auto_chunker.enable = True
# torch.cuda.set_per_process_memory_fraction(0.85)

@dataclass
class Config:
    emb_dim: int = 768 #d_model
    n_heads: int = 8
    n_group: int = 2
    n_iter: int = 10_000
    context_lens = [2_048, 8_192, 32_768] # block_size
    n_iter_per_context_len = [n_iter * 0.8, n_iter * 0.15, n_iter * 0.05]
    n_block: int = 6 # n_layers
    dropout: float = 0.0
    lr: float = 1e-3
    vocab_size: int = tokenizer.encode("<|endoftext|>", allowed_special="all")[0] + 1 # get the real size (was pb)
    batch_size: int = 32
    n_iter_eval: int = 50
    eval_interval: int = 250
    warmup_coef: int = 0.02
    final_lr_frac: float = 0.1
    max_tokens: int = 100_000_000
    device = device

context_len = Config.context_lens[0]

model = Transformer(Config.vocab_size, Config.emb_dim, 
                    Config.n_heads, context_len, 
                    Config.n_block, Config.dropout, device, Config.n_group)
model = model.to(device)

download_data(tokenizer, max_tokens=Config.max_tokens)

# optimizers :

muon_params = [p for n, p in model.named_parameters()
               if p.ndim == 2
               and "emb" not in n
               and "l1" not in n]  

muon_set = {id(p) for p in muon_params}
adamw_params = [p for n, p in model.named_parameters()
                if id(p) not in muon_set]

optimizerAdamW = torch.optim.AdamW(
    adamw_params, 
    lr=Config.lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True
)

optimizerMuon = torch.optim.Muon(
    muon_params,
    lr=Config.lr, 
    weight_decay=0.1,
    adjust_lr_fn = "match_rms_adamw" # as in "Muon is scalable for LLM training" paper
)

optimizers = [optimizerAdamW, optimizerMuon]


@torch.no_grad()
def eval(nb_iter):
    model.eval()
    # eval data
    eval_loss = 0
    for i in range(nb_iter):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            X, Y = get_batch("eval", context_len, device, Config.batch_size)
            logits, loss = model(X, Y)
        eval_loss += loss / nb_iter
    
    train_loss = 0
    for i in range(nb_iter):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            X, Y = get_batch("train", context_len, device, Config.batch_size)
            logits, loss = model(X, Y)
        train_loss += loss / nb_iter
    
    return eval_loss.item(), train_loss.item() # .item() at the end no so snyc GPU and CPU during all the foor loop

def get_lr(relative_it, actual_phase_len):

    # 0 -> 1
    if relative_it / actual_phase_len < Config.warmup_coef:
        return Config.lr * (relative_it + 1) / (Config.warmup_coef * actual_phase_len)

    # cosine lr
    eta_min = 0.1 * Config.lr # decay up to 10%
    return eta_min + 0.5*(Config.lr - eta_min) * (1 + math.cos(relative_it * math.pi / actual_phase_len))



if device == "cuda":
    model = torch.compile(model, dynamic=True)

print("training stats: ")
b_s = Config.batch_size
for i in range(len(Config.context_lens)):
    if (i > 0):
        b_s = b_s // (Config.context_lens[i] / Config.context_lens[i-1])
    print(f"{i} : training for {b_s * Config.context_lens[i] * Config.n_iter_per_context_len[i] / Config.max_tokens} epochs with context_len : {Config.context_lens[i]}")

train_loss_stats = []
eval_loss_stats = []


# training loop
with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
    for i in tqdm(range(Config.n_iter)):
        model.train()
        
        actual_phase = Config.context_lens.index(context_len)
        actual_phase_len = Config.n_iter_per_context_len[actual_phase]

        # update lr
        for optims in optimizers:
            for group in optims.param_groups:
                    group['lr'] = get_lr(i - sum(Config.n_iter_per_context_len[:actual_phase]), actual_phase_len)
        # zero_grad
            optims.zero_grad()

        # evaluation
        if (i % Config.eval_interval == 0):
            # eval
            eval_loss, train_loss = eval(Config.n_iter_eval)
            # stats
            train_loss_stats.append(train_loss)
            eval_loss_stats.append(eval_loss)
            with open("intermediate_stats.txt", "a") as f_stats:
                data = {"name": args.name, "train_loss_stats": train_loss_stats, "eval_loss_stats": eval_loss_stats}
                f_stats.write(json.dumps(data))
                f_stats.write("\n")
            # log
            print(f"iter : {i} | eval_loss : {eval_loss} | train_loss : {train_loss} | lr : {get_lr(i - sum(Config.n_iter_per_context_len[:actual_phase]), actual_phase_len)}")
            # save model
            torch.save(model, "model.pt")

        # update context_len
        if (i >= sum(Config.n_iter_per_context_len[:actual_phase+1]) ) : # divide the training according to n_iter_per_context_len
            before_context_len = context_len
            context_len = Config.context_lens[Config.context_lens.index(context_len) + 1]
            model.change_context_len(context_len)
            Config.batch_size = Config.batch_size // (context_len // before_context_len) # Change the batch size not to explode RAM
            print(f"context_len changed : {context_len}, and batch_size : {Config.batch_size}")

        # training 
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16): # tu upgrade perf -> satisfy flash Attention
            X, Y = get_batch("train", context_len, device, Config.batch_size)
            logits, loss = model(X, Y)

        # update
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        for optims in optimizers:
            optims.step()

with open("stats.txt", "a") as f:
    data = {"name":args.name , "train_loss_stats" : train_loss_stats, "eval_loss_stats": eval_loss_stats}
    f.write("\n")
    f.write(json.dumps(data))
