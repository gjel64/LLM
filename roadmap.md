# Roadmap IA → LLM Agentic CoT
> **Objectif :** maîtriser la pipeline complète d'un LLM agentic avec Chain-of-Thought fort  
> **Prérequis :** tu sais déjà coder GPT-2/3 (cours Karpathy)  
> **Durée estimée :** 12–18 mois  
> **État de l'art :** 2025 (DeepSeek, o1/o3, Llama 3, Qwen 2.5)

---

## Vue d'ensemble

```
Phase 1 → Transformer moderne      (~6 sem)
Phase 2 → Scaling & Pretraining    (~6 sem)
Phase 3 → Post-training / RLHF     (~5 sem)
Phase 4 → Reasoning & CoT          (~6 sem)  ← cœur du sujet
Phase 5 → Agents & multi-agents    (~5 sem)
Phase 6 → Inference & déploiement  (~4 sem)
Phase 7 → Projet final complet     (~8 sem)
```

---

## Phase 1 — Fondations modernes du Transformer

### Papers fondamentaux

| Paper | Auteurs | Année | Pourquoi c'est essentiel |
|---|---|---|---|
| **Attention Is All You Need** ✅ | Vaswani et al. | 2017 | Architecture originale — encore la base de tout | 
| **FlashAttention 1 ✅ & 2** | Dao et al. | 2022/23 | Attention IO-optimisée, base de tout training sérieux |
| **RoPE** ✅ | Su et al. | 2021 | Positional embeddings de tous les LLMs actuels (Llama, Mistral…) |
| **GQA** | Ainslie et al. | 2023 | Grouped Query Attention — inference efficace |
| **SwiGLU / GLU variants** ✅| Shazeer | 2020 | FFN activation standard de presque tous les LLMs open-source |
| **RMSNorm + Pre-Norm** ✅| Zhang & Sennrich | 2019 | Normalisation stabilisante, standard post-2022 |

### Code & vidéos

- **Tri Dao — Flash Attention talks** (Stanford / YouTube) : comprendre l'intuition IO-awareness

### Exercice clé

1. Transformer from scratch ✅
2. BPE from scratch ✅
3. `nn.LayerNorm` → `RMSNorm` ✅
4. Learned positional embeddings → `RoPE` ✅
5. MHA → `GQA`
6. MLP standard → `SwiGLU` ✅

---

## Phase 2 — Scaling, Pretraining & Data
> *Comprendre comment dimensionner un run, construire un dataset et entraîner de façon stable.*

### Scaling Laws

| Paper | Auteurs | Année | Message clé |
|---|---|---|---|
| **Chinchilla / Hoffmann et al.** | Hoffmann et al. | 2022 | Compute-optimal : ratio tokens/params, loi fondamentale |
| **Kaplan Scaling Laws** | Kaplan et al. | 2020 | Fondations historiques — comprendre l'évolution du domaine |
| **Training stability** | Yang et al. | 2022 | μ-parameterization, grad clipping, LR warmup — éviter les loss spikes |
| **Llama 3 tech report** | Meta | 2024 | Pipeline complet open-source : data, tokenizer, architecture |
| **DeepSeek V3** | DeepSeek | 2024 | MoE + multi-token prediction + FP8 training. **SOTA open-source 2025.** |
| **MoE — Sparse routing** | Lepikhin et al. + DeepSeek | 2021/24 | Scale sans coût compute linéaire |
| **Muon** | xx | xx | xx |

### Data pipeline

| Ressource | Type | Description |
|---|---|---|
| **FineWeb / Dolma / RedPajama** | Dataset | Pipelines open-source de curation web — dedup, filtering, quality scoring |
| **minBPE** (Karpathy) | Code | Implémenter BPE complet de zéro, comprendre les edge cases |

### Points de vigilance

- Toujours lire les **data cards** des datasets — la qualité des données > la taille du modèle
- Implémenter le **data mixing** (proportions web / code / math / books) avant de lancer un run
- Monitorer les **loss curves** dès le début — un spike non géré peut ruiner un run de plusieurs jours

---

## Phase 3 — Post-training : RLHF, DPO, SFT
> *Transformer un modèle de base en modèle utile et aligné.*

### Alignment & fine-tuning

| Paper | Auteurs | Année | Importance |
|---|---|---|---|
| **InstructGPT / RLHF** | Ouyang et al. | 2022 | Pipeline historique SFT → Reward Model → PPO. Fondation. |
| **DPO** | Rafailov et al. | 2023 | Plus simple que PPO, **standard actuel** pour l'alignement |
| **GRPO / SimPO** | DeepSeek / Chen et al. | 2024 | Variantes DPO plus stables, utilisées dans DeepSeek-R1 |
| **LoRA / QLoRA** | Hu et al. | 2021/23 | Fine-tuning paramètre-efficient, standard pour adapter un LLM |
| **Constitutional AI / RLAIF** | Anthropic | 2022 | Self-critique, alignment sans labelers humains exhaustifs |

### Outils pratiques

- **TRL library** (HuggingFace) : pipeline complet SFT + DPO en code, le plus utilisé en pratique
- **Axolotl** : wrapper simplifié pour des runs SFT/DPO rapides sur GPU limité

### Ordre de lecture recommandé

```
InstructGPT → DPO → GRPO → implémenter un pipeline SFT+DPO avec TRL
```

---

## Phase 4 — Reasoning & Chain-of-Thought ⭐
> *Le cœur du sujet. La révolution de 2025 : faire émerger le reasoning par RL.*

### CoT & reasoning émergent

| Paper | Auteurs | Année | Message clé |
|---|---|---|---|
| **Chain-of-Thought Prompting** | Wei et al. | 2022 | Découverte du reasoning émergent avec few-shot CoT. Fondation. |
| **Self-Consistency CoT** | Wang et al. | 2023 | Majority vote sur plusieurs raisonnements — amélioration simple et puissante |
| **Tree of Thoughts (ToT)** | Yao et al. | 2023 | Exploration arborescente des raisonnements |
| **DeepSeek-R1** | DeepSeek | 2025 | **RL pur pour faire émerger le reasoning sans SFT supervisé. Révolution.** |
| **OpenAI o1 / o3** | OpenAI | 2024/25 | Process reward models, search at test-time, long thinking traces |
| **Process Reward Models (PRM)** | Lightman et al. | 2023 | Récompenser chaque étape de raisonnement, pas juste le résultat final |
| **MCTS + LLM (rStar-Math)** | Microsoft | 2025 | Monte Carlo Tree Search + PRM pour reasoning mathématique |

### Test-time compute scaling

| Paper | Auteurs | Message clé |
|---|---|---|
| **Scaling LLM Test-time Compute** | Snell et al. | Plus de compute à l'inférence = meilleur reasoning |
| **Best-of-N + PRM pratique** | — | Générer N solutions, scorer avec PRM, sélectionner la meilleure |

### L'arc DeepSeek-R1 à comprendre absolument

```
1. Pretraining base model
2. Cold start : SFT sur quelques milliers d'exemples CoT de qualité
3. GRPO training : reward = exactitude + format (pas de reward model complexe)
4. Rejection sampling : garder les meilleures traces CoT générées
5. SFT final + DPO pour aligner le style
→ Résultat : reasoning de niveau o1 avec beaucoup moins de ressources
```

### Benchmarks de référence

- **MATH-500** — problèmes de mathématiques lycée/prépa
- **AIME** — olympiades mathématiques
- **GSM8K** — arithmétique simple, baseline utile
- **SWE-bench** — coding sur vrais GitHub issues

---

## Phase 5 — LLM Agents & Systèmes multi-agents
> *Donner des outils et de la mémoire au modèle pour agir dans le monde réel.*

### Architectures agents

| Paper | Auteurs | Année | Importance |
|---|---|---|---|
| **ReAct** | Yao et al. | 2023 | Thought → Action → Observation loop. **Base de tous les agents.** |
| **Toolformer** | Schick et al. | 2023 | Self-supervised tool use, API calls dans le contexte |
| **SWE-Agent / SWE-bench** | Yang et al. | 2024 | Coding agent SOTA sur vrais GitHub issues |
| **AutoGen / AgentScope** | Microsoft / Alibaba | 2023/24 | Multi-agent frameworks — agents spécialisés qui se délèguent des tâches |

### Mémoire & contexte long

| Ressource | Description |
|---|---|
| **YaRN / LongRoPE** | Étendre la fenêtre de contexte sans réentraîner from scratch |
| **RAG avancé** | Dense retrieval + reranking + chunk strategy — mémoire externe de l'agent |
| **Ring Attention** | Attention distribuée sur plusieurs GPUs pour contextes > 1M tokens |

### Les 3 types de mémoire d'un agent

```
1. In-context memory    → ce qui est dans la fenêtre de contexte
2. External memory      → vector DB (Faiss, Weaviate, Pinecone)
3. Parametric memory    → ce que le modèle a appris pendant l'entraînement
```

### Stack agent recommandé (2025)

```
LLM backend    : vLLM ou llama.cpp
Orchestration  : code Python custom (éviter les frameworks trop opaques)
Memory         : Faiss ou Chroma pour la vector DB
Tools          : function calling natif du modèle
Streaming CoT  : SSE ou WebSockets pour afficher la pensée en temps réel
```

---

## Phase 6 — Inference, Quantization & Déploiement
> *Faire tourner un grand modèle efficacement en production.*

### Optimisation inference

| Paper / Outil | Auteurs | Année | Impact |
|---|---|---|---|
| **vLLM / PagedAttention** | Kwon et al. | 2023 | Serving haute perf, KV cache management. **Standard production.** |
| **Speculative Decoding** | Leviathan et al. | 2023 | Small draft model + large verifier = 2-3x speedup |
| **GPTQ / AWQ** | Frantar et al. / Lin et al. | 2022/23 | Post-training quantization — INT4 pour faire tourner 70B sur 2 GPUs |
| **KV Cache optimizations** | — | 2024 | Prefix caching, sliding window, quantized KV — critique pour les agents |

### Formats de quantization à connaître

```
GGUF    → llama.cpp, CPU-friendly, le plus polyvalent
GPTQ    → GPU, good quality/speed tradeoff
AWQ     → GPU, meilleure qualité que GPTQ à même taille
FP8     → DeepSeek V3, training + inference (H100 requis)
```

---

## Phase 7 — Projet final : Pipeline LLM Agentic CoT complet ⭐
> *L'objectif : construire de A à Z un LLM qui raisonne et agit.*

### Les 6 étapes du pipeline

#### Étape 1 — Pretraining custom
- Architecture moderne : RoPE + GQA + SwiGLU + RMSNorm
- Data pipeline : curation, dedup, mixing (web / code / math)
- Distributed training : DeepSpeed ZeRO ou FSDP

#### Étape 2 — SFT sur données CoT
- Générer des traces CoT synthétiques avec un modèle fort (GPT-4o, Llama 3.3 70B)
- Rejection sampling : garder uniquement les traces avec bonne réponse finale
- SFT avec TRL ou Axolotl

#### Étape 3 — RL avec GRPO ⭐
- Reward function : exactitude (rule-based) + format du CoT
- GRPO training loop : pas besoin d'un reward model neural complexe
- Monitorer : longueur des traces CoT, taux de bonne réponse, entropy

#### Étape 4 — Agent layer
- ReAct loop en Python pur
- Function calling natif du modèle
- Mémoire externe (RAG) pour les tâches longues
- Streaming CoT pour l'UX

#### Étape 5 — Eval rigoureuse
- Benchmarks standards : MATH-500, AIME, SWE-bench
- Evals custom sur ton domaine cible
- Comparaison before/after chaque étape de training

#### Étape 6 — Serving production
- vLLM avec prefix caching activé
- Batching dynamique
- Monitoring : latence P50/P99, throughput tokens/sec

---

## Ressources de veille continue

### Papers à surveiller (2025 et après)

- **Qwen 2.5 tech report** (Alibaba 2024) — best open-source post-Llama 3
- **Gemini 2.0 Flash Thinking** (Google 2025) — concurrent de o1, multimodal + reasoning
- **Nouveaux papiers DeepSeek** — leur cadence de publication est remarquable

### Où suivre l'actualité

| Ressource | Type | Pourquoi |
|---|---|---|
| **arxiv.org/cs.LG** | Papers | Source primaire |
| **Hugging Face blog** | Blog | Implémentations + commentaires |
| **@karpathy** (X/Twitter) | Veille | Pédagogie + opinions terrain |
| **@ilyasut** | Veille | Chercheur OpenAI, vues profondes |
| **@aidan_goes_places** | Veille | Suivi SOTA très régulier |
| **Latent Space podcast** | Podcast | Interviews des auteurs des papers |
| **Interconnects (Nathan Lambert)** | Newsletter | Alignement + RLHF en profondeur |

---

## Ce qui est délibérément absent

Ces techniques ne sont **plus le standard en 2025** et ne valent pas le temps d'investissement :

- ~~BERT / encoders only~~ → remplacés par les decoders pour presque tout
- ~~RNN / LSTM / GRU~~ → complètement supplantés par les transformers
- ~~Prefix Tuning / Prompt Tuning~~ → LoRA/DPO sont bien supérieurs
- ~~BLEU comme métrique principale~~ → utiliser des LLM-judges ou benchmarks task-specific
- ~~PPO complexe pour l'alignment~~ → DPO / GRPO sont plus simples et aussi efficaces

---

*Roadmap générée en juin 2025 — état de l'art incluant DeepSeek V3/R1, Llama 3, o1/o3, Qwen 2.5*