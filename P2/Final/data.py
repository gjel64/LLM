import json
from datasets import load_dataset
import tiktoken
import os
import numpy as np
import torch
from tqdm import tqdm

def download_data(tokenizer, val_ratio=0.005, max_tokens=10):
    os.makedirs('.cache', exist_ok=True)
    
    if not os.path.isfile('.cache/train.bin') or not os.path.isfile('.cache/val.bin'):
        print("Downloading + tokenizing data...")

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu", 
            "CC-MAIN-2024-51", 
            streaming=True, 
            split="train"
        )
        
        start = tokenizer.encode("<|startoftext|>", allowed_special="all")
        end = tokenizer.encode("<|endoftext|>", allowed_special="all")
        
        # Write en streaming directement sur disque
        train_f = open('.cache/train.bin', 'wb')
        val_f = open('.cache/val.bin', 'wb')
        
        total = 0
        for item in tqdm(dataset, desc="Tokenizing"):
            tokens = start + tokenizer.encode(item['text'], allowed_special="all") + end
            arr = np.array(tokens, dtype=np.uint32)
            
            if total < max_tokens * (1 - val_ratio):
                arr.tofile(train_f)
            else:
                arr.tofile(val_f)
            
            total += len(tokens)
            if total >= max_tokens:
                break

        train_f.close()
        val_f.close()

        print(f"Total tokens : {total:,}")


_memmaps = {}

def get_batch(split, block_size, device, batch_size):
    path = '.cache/train.bin' if split == 'train' else '.cache/val.bin'
    
    # Ouvre une seule fois
    if path not in _memmaps:
        _memmaps[path] = np.memmap(path, dtype=np.uint32, mode='r')
    data = _memmaps[path]
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y