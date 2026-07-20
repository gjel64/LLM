import json
from datasets import load_dataset, interleave_datasets
import tiktoken
import os
import numpy as np
import torch
from tqdm import tqdm

# nb de documents tokenises par appel batch (compromis RAM / debit)
DOCS_PER_BATCH = 2048
NUM_THREADS = os.cpu_count() or 8

SOURCES = [
    {
        "repo": "HuggingFaceFW/fineweb-edu",
        "name": "sample-350BT",
        "text_key": "text",
        "weight": 0.50,
    },
    {
        "repo": "mlfoundations/dclm-baseline-1.0-parquet",
        "name": None,
        "text_key": "text",
        "weight": 0.13,
    },
    {
        "repo": "HuggingFaceTB/smollm-corpus",
        "name": "cosmopedia-v2",
        "text_key": "text",
        "weight": 0.12,
    },
    {
        # StarCoderData: code de qualite, gated -> `huggingface-cli login` requis
        # + accepter les conditions sur la page HF. Organise par langage (data_dir).
        "repo": "bigcode/starcoderdata",
        "data_dir": "python",
        "text_key": "content",
        "weight": 0.08,
    },
    {
        "repo": "bigcode/starcoderdata",
        "data_dir": "javascript",
        "text_key": "content",
        "weight": 0.05,
    },
    {
        "repo": "HuggingFaceTB/finemath",
        "name": "finemath-4plus",
        "text_key": "text",
        "weight": 0.06,
    },
    {
        "repo": "wikimedia/wikipedia",
        "name": "20231101.en",
        "text_key": "text",
        "weight": 0.06,
    },
]


def _prepare(src):
    """Charge un dataset en streaming et l'uniformise a deux colonnes:
    {'text': <contenu>, 'source': <nom>} pour pouvoir interleave proprement."""
    ds = load_dataset(
        src["repo"],
        src.get("name"),
        data_dir=src.get("data_dir"),   # ex: langage pour starcoderdata
        streaming=True,
        split="train",
        trust_remote_code=src.get("trust_remote_code", False),
    )
    key = src["text_key"]
    # label lisible dans le log de mixture (repo + sous-config eventuelle)
    label = src["repo"] + (f"/{src['data_dir']}" if src.get("data_dir") else "")
    # schema commun obligatoire pour interleave_datasets
    ds = ds.map(lambda ex: {"text": ex[key], "source": label})
    ds = ds.select_columns(["text", "source"])
    return ds


def download_data(tokenizer, max_tokens, val_ratio=0.005):
    os.makedirs('.cache', exist_ok=True)

    if not os.path.isfile('.cache/train.bin') or not os.path.isfile('.cache/val.bin'):
        print("Downloading + tokenizing data (quality mixture)...")

        # normalise les poids -> probabilites d'echantillonnage
        weights = [s["weight"] for s in SOURCES]
        total_w = sum(weights)
        probs = [w / total_w for w in weights]

        print("Mixture plan:")
        for s, p in zip(SOURCES, probs):
            print(f"  - {s['repo']:40s} {s.get('name',''):18s} : {p:5.1%}")

        streams = [_prepare(s) for s in SOURCES]
        # all_exhausted -> garde les proportions meme si une source s'epuise
        # (ici on coupe a max_tokens bien avant, mais c'est plus robuste)
        dataset = interleave_datasets(
            streams,
            probabilities=probs,
            seed=42,
            stopping_strategy="all_exhausted",
        )

        start = tokenizer.encode("<|startoftext|>", allowed_special="all")
        end = tokenizer.encode("<|endoftext|>", allowed_special="all")

        # dtype uint16: vocab p50k (~50k) < 65535 -> 2x moins de disque et de RAM
        # memmap qu'en uint32. (assert de securite pour ne pas tronquer un id)
        assert tokenizer.max_token_value < 2**16, "vocab > uint16, repasser en uint32"

        # Write en streaming directement sur disque
        train_f = open('.cache/train.bin', 'wb')
        val_f = open('.cache/val.bin', 'wb')

        total = 0
        per_source = {}
        # Tokenisation par batch: tiktoken.encode_ordinary_batch libere le GIL et
        # parallelise en Rust sur NUM_THREADS -> le vrai goulot single-thread saute.
        buf_txt, buf_src = [], []

        def flush(pbar):
            nonlocal total
            encoded = tokenizer.encode_ordinary_batch(buf_txt, num_threads=NUM_THREADS)
            for enc, src in zip(encoded, buf_src):
                tokens = start + enc + end
                arr = np.array(tokens, dtype=np.uint16)
                if total < max_tokens * (1 - val_ratio):
                    arr.tofile(train_f)
                else:
                    arr.tofile(val_f)
                total += len(tokens)
                per_source[src] = per_source.get(src, 0) + len(tokens)
            pbar.n = total
            pbar.refresh()
            buf_txt.clear()
            buf_src.clear()

        with tqdm(total=max_tokens, desc="Tokenizing", unit="tok", unit_scale=True) as pbar:
            for item in dataset:
                buf_txt.append(item['text'])
                buf_src.append(item['source'])
                if len(buf_txt) >= DOCS_PER_BATCH:
                    flush(pbar)
                    if total >= max_tokens:
                        break
            if total < max_tokens and buf_txt:
                flush(pbar)

        train_f.close()
        val_f.close()

        print(f"Total tokens downloaded : {total:,}")
        print("Realized token mixture:")
        for name, n in sorted(per_source.items(), key=lambda kv: -kv[1]):
            print(f"  - {name:40s} : {n:>14,} ({n/total:5.1%})")


_memmaps = {}

def get_batch(split, context_len, device, batch_size, nb_predict=1):
    path = '.cache/train.bin' if split == 'train' else '.cache/val.bin'

    # Ouvre une seule fois (uint16: doit matcher le dtype d'ecriture ci-dessus)
    if path not in _memmaps:
        _memmaps[path] = np.memmap(path, dtype=np.uint16, mode='r')
    data = _memmaps[path]

    ix = torch.randint(len(data) - context_len - (nb_predict - 1), (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+context_len]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+nb_predict+context_len]).astype(np.int64)) for i in ix])

    x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
    return x, y
