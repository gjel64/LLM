import torch
import torch.nn as nn

model = nn.Module # on fait comme si c'était le bon modèle
lr = 1e-3


muon_params = [p for n, p in model.named_parameters()
               if p.ndim == 2
               and "emb" not in n
               and "l1" not in n]  

muon_set = {id(p) for p in muon_params}
adamw_params = [p for n, p in model.named_parameters()
                if id(p) not in muon_set]

optimizerAdamW = torch.optim.AdamW(
    adamw_params, 
    lr=lr,
    betas=(0.9, 0.95),
    weight_decay=0.1,
    fused=True
)

optimizerMuon = torch.optim.Muon(
    muon_params,
    lr=lr, 
    weight_decay=0.1,
    adjust_lr_fn = "match_rms_adamw" # as in Muon is scalable for LLM training paper
)

optimizers = [optimizerAdamW, optimizerMuon]