# Phase 1

On tiny Shakespeare, just to learn the concepts

| Concept | Results | Time | Explanation |
| --- | --- | --- | --- |
| Attention|
| BPE|
| FlashAttention| | | optimize Attention, for the forward case : do the attention per little groups to avoid materializing the full attention matrix in HBM, reducing memory traffic by keeping intermediate computations in SRAM (Bandwidth optimization)|
| RoPE| | | get rid of position emb and instead encodes position by rotating Q and K vectors|
| SwigLU | 2.48 - 1.16 | 40.4m | activation function that uses Swish (sigmoid) and GLU : ((Swish_1(x @ W1)) * (x @ V)) @ W2 |
| RMSNorm | 2.42 - 1.22 | 40.07m | alternative to layer norm : sqrt(eps + 1/n(sum((a_i)^2)->n)) normalizes activations using only their root mean square, without mean subtraction | 
| GQA | 2.44 - 1.25 | 38.40m | somewhere between MHA and MQA : group of heads that have common K and V and personnal Q -> optimisation for time