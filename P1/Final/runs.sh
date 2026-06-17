#!/bin/sh

cd ./../..
source .venv/bin/activate
cd P1/Final

python3 training.py --emb_dim 512
python3 training.py --context_len 512
python3 training.py --n_block 8

