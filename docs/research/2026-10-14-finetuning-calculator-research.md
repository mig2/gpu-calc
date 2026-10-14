# Deep Research: Fine-Tuning GPU Sizing Heuristics

**Date:** 2026-10-14
**Purpose:** Establish the quantitative foundation for a new **Fine-Tuning** tab alongside the existing Training and Inference modes.
**Question the tab answers:** *How many of GPU X do I need to fine-tune model Y with method Z — and how long will it take and cost?*

---

## 0. The central finding (shapes the whole design)

**Fine-tuning is memory-bound; pretraining is compute-bound.**

The existing Training tab solves `GPUs = total_FLOPs / (deadline × sustained_FLOPs)` — a deadline-driven compute problem. Fine-tuning inverts this. A 70B QLoRA job needs ~52 GB of VRAM and ~3×10^18 FLOPs; a single H100 satisfies the compute in ~10 hours. The binding constraint is almost never the deadline — it is *does the model state fit at all*.

So the Fine-Tuning tab must lead with a **memory feasibility ladder**, not a GPU count:

1. **Does it fit?** Minimum GPU count from memory (per method).
2. **How long on that config?** Time from compute.
3. **What does it cost?** Time × cloud rate.
4. *Secondary:* "To finish by date D, scale to N GPUs" (the Training-tab question).

This is the opposite emphasis from the Training tab and justifies a separate mode rather than a new "training mode" dropdown value on the existing one.

---

## 1. Memory model

### 1.1 Component decomposition

```
M_total = M_weights + M_grads + M_optimizer + M_activations + M_aux + M_overhead
```

| Component | Formula | Notes |
|---|---|---|
| Weights | `N_total × b_w` | `b_w` = 2 (BF16), 1 (FP8), 0.5 (NF4/INT4). Uses **total** params, incl. all MoE experts. |
| Gradients | `N_train × b_g` | `b_g` = 2 (BF16) or 4 (FP32). LoRA: `N_train` = adapter params only. |
| Optimizer | `N_train × b_opt` | See 1.3. |
| Activations | see 1.4 | The only term that scales with batch × sequence length. |
| Auxiliary | reference / reward / critic copies | DPO, PPO, GRPO. See 1.6. |
| Overhead | `~1.5 GB/GPU + 8%` | CUDA context, fragmentation, comm buffers. Existing code already uses a 0.85 usable-memory factor. |

### 1.2 Bytes-per-parameter by method (the headline heuristic)

| Method | Bytes/param (model state, excl. activations) | Source convention |
|---|---|---|
| Full FT, AdamW, mixed precision | **16** = 2 BF16 w + 2 BF16 g + 4 FP32 master + 4 m + 4 v | Megatron / standard mixed-precision accounting |
| Full FT, AdamW, no FP32 master | **12** = 2 + 2 + 8 | Spheron 2026 table convention |
| Full FT, 8-bit Adam | **8** = 2 + 2 + 4 | bitsandbytes block-wise quant: optimizer 8 B → 2 B/param |
| Full FT, Adafactor | **~6–8** | factored second moments, O(n) not O(n²) |
| Full FT, LOMO / AdaLomo | **~2–4** | fused backward update; LLaMA-7B 102.2 GB → 14.6 GB measured |
| LoRA (BF16 base) | **2 × N_total + 16 × N_adapter** | base frozen; 16 B per *trainable* adapter param |
| QLoRA (NF4 base) | **0.5 × N_total + 16 × N_adapter** | 4-bit frozen base |

The two full-FT conventions (12 vs 16) differ by whether the FP32 master copy is counted separately. **Recommendation: expose optimizer bytes as a named preset, default AdamW = 12 B/param + 2 B grads + 2 B weights = 16 total**, and show the breakdown in the formula trace (matching the existing `FormulaTraceDrawer` pattern).

### 1.3 Published VRAM anchors (calibration targets)

| Model | Full FT | LoRA r=64 | QLoRA 4-bit |
|---|---|---|---|
| 7B / 8B | ~88 GB | ~20 GB | ~8 GB |
| 14B | ~174 GB | ~35 GB | ~14 GB |
| 32B | ~394 GB | ~76 GB | ~28 GB |
| 70B / 72B | ~860 GB | ~159 GB | ~52 GB |
| MoE 30B-A3B | ~105 GB | ~69 GB | ~21 GB |

Worked 70B examples from the same source:
- **QLoRA:** 35 (base NF4) + 1.5 (adapters) + 1.5 (grads) + 5.6 (optimizer) + 8 (activations) = **~52 GB**
- **LoRA:** 140 (BF16 base) + 1.5 + 1.5 + 5.6 + 10 = **~159 GB** → 2× H100 minimum (the non-activation floor of ~149 GB already exceeds a single H200's 141 GB)
- **Full FT:** ~860 GB → **11× H100 SXM5** with FSDP2 / ZeRO-3

A second independent source gives 70B full FT as "≈1.12 TB of model state in theory, 400–600 GB aggregate in practice (≈8× H200 or 11× H100 with ZeRO-3)". The 11× H100 figure is corroborated. **These five rows are the regression test suite for the engine.**

### 1.4 LoRA adapter parameter count

Exact, per targeted linear layer: `params = r × (d_in + d_out)`. Full enumeration over a GQA transformer with `L` layers, hidden `d`, KV dim `d_kv`, FFN dim `d_ff`, targeting all 7 projections:

```
per_layer = r × [ (d+d) + (d+d_kv) + (d+d_kv) + (d+d)      # q, k, v, o
                + (d+d_ff) + (d+d_ff) + (d_ff+d) ]          # gate, up, down
N_adapter = L × per_layer
```

**Heuristic fallback** when `L`, `d`, `d_ff` are unknown — calibrated against two published points (8B r=16 → ~21M trainable; 70B r=16 → ~150M trainable, both ≈0.21–0.26% of base):

```
N_adapter ≈ N_total × 0.0026 × (r / 16) × target_factor
target_factor:  all-linear = 1.0   attention-only (q,v) = 0.29   attn+o = 0.4
```

Each trainable adapter param costs **16 bytes** (FP32 weight + FP32 grad + 2 FP32 Adam moments). Dropping r=64 → r=16 cuts adapter memory 4×, but note from the 1.3 table: on Llama 3.1 8B, LoRA → QLoRA moves weights 15.0 → 3.8 GB yet *total* only 27.6 → 23.9 GB, because activations and optimizer state dominate at that scale. **Surfacing this "quantization buys less than you think at small batch" effect is a genuine insight for the UI.**

### 1.5 Activation memory

Per-layer, no recompute, standard attention:
`sbh(34 + 5·a·s/h)` bytes, where `s`=seq len, `b`=batch, `h`=hidden, `a`=heads.
The `5as/h` term is the quadratic attention matrix — **it vanishes under FlashAttention**, which is universal in 2026 fine-tuning stacks. So use:

```
no checkpointing:   M_act ≈ 34 × b × s × d × L
with checkpointing: M_act ≈ 2 × b × s × d × L   +   34 × b × s × d
                            (stashed boundaries)     (one-layer recompute peak)
```

Sanity check, Llama-3 8B (L=32, d=4096), b=1, s=4096, checkpointed: 1.07 GB stash + 0.57 GB peak ≈ **1.6 GB**. At b=8: ≈ **13 GB**. Consistent with practitioner reports and with the "8–20 GB for 70B" anchor.

Checkpointing trade: **40–60% activation memory reduction for 25–40% compute overhead**. Default it **on** — it is the standard fine-tuning configuration.

### 1.6 Preference / RL method multipliers

| Method | Resident model copies | Memory vs SFT | Compute vs SFT |
|---|---|---|---|
| SFT | 1 (trainable) | 1.0× | 1.0× |
| DPO | 2 (policy + frozen ref) | **~2×** | ~3–4× (ref forward + both chosen & rejected completions) |
| ORPO | 1 (reference-free) | ~1.0× | ~2× (two completions) |
| GRPO / RLVR | policy + ref, no critic | **~2–3×** | dominated by G rollouts/prompt |
| PPO | 4 (policy, ref, reward, critic) | ~4× | highest |

GRPO removes the critic via Monte-Carlo group-relative advantage, "roughly halving per-step memory at comparable quality" vs PPO. For GRPO, **rollout generation dominates compute**, so the FLOP model needs a separate generation term: `G × avg_completion_tokens × 2N` per prompt, plus KV cache for `G` concurrent sequences. This is a materially different cost shape and may warrant deferring RL methods to a phase 2.

---

## 2. Compute model

### 2.1 FLOPs per token

| Method | FLOPs/token | Derivation |
|---|---|---|
| Inference / forward only | `2N` | baseline |
| Full fine-tuning | `6N` | 2N forward + 4N backward |
| LoRA / QLoRA | **`≈4N`** | 2N forward + ~2N backward — no weight gradients for the frozen base |
| + gradient checkpointing | `+2N` | one extra forward pass |

The LoRA reduction is exact in form: standard backward costs `4bmn` per linear layer; LoRA's is `2bmn + 4br(m+n)`, and with `r ≪ m,n` the adapter term is negligible — **the LoRA backward pass costs the same as its forward pass.** So full FT 6N → 8N with checkpointing (+33%), LoRA 4N → 6N (+50%).

**MoE:** compute uses `N_active`, memory uses `N_total`. Mixtral-8×7B must hold ~47B params but computes against ~13B per token. This split must be explicit in the engine — it is the single most common sizing error on 2026 models, where every frontier open-weight model is sparse (DeepSeek V4 1T/37B active, Qwen3-235B-A22B, GLM-5.2 744B/40B, Llama 4 Scout 109B/17B, Mistral Small 4 119B/6B).

### 2.2 Empirical throughput anchors (NVIDIA NeMo, DGX-H100, BF16, seq 4096)

| Model | Method | GPUs | tok/s/GPU | Implied MFU @ 4N |
|---|---|---|---|---|
| Llama 3 8B | LoRA | 1 | 12,472.9 | **40.3%** |
| Qwen 2.5 7B | LoRA | 1 | 14,110.1 | **39.9%** (NeMo reports 42.3%) |
| Qwen 2.5 32B | LoRA | 8 | 2,250.3 | **29.1%** |
| Llama 3 70B | LoRA | 8 | 1,091.9 | **30.9%** (NeMo reports 31.6%) |
| Qwen 2.5 32B | LoRA | 16 (2 nodes) | 2,072.9 | 26.8% |
| Llama 3 70B | LoRA | 16 (2 nodes) | 984.9 | 27.9% |

**This is the most valuable finding in the research.** Back-solving MFU from NeMo's published tok/s using the `4N` LoRA FLOP model reproduces NeMo's own reported MFU to within 2.4 percentage points. That independently validates both the `4N` LoRA model and gives default MFUs:

```
MFU defaults for fine-tuning:
  single GPU (model fits)       0.40
  single node, sharded (2-8)    0.30
  multi-node                    0.27
```

Note these are *lower* than the existing Training tab's `defaultMfu` of 0.40–0.45 for pretraining — fine-tuning runs smaller batches and shorter sequences, so it utilizes the hardware less well.

### 2.3 Scaling efficiency

From the 8 → 16 GPU rows above: **~90% efficiency per node doubling** (70B: 1091.9 → 984.9 = 90.2%; 32B: 2250.3 → 2072.9 = 92.1%). Model as `efficiency = 0.90^log2(nodes)`.

LoRA is unusually friendly to data parallelism because only adapter gradients need all-reduce (~0.2% of params). Research confirms the headroom: FCDP reports "up to 100× higher throughput than ZeRO-3 for LoRA by eliminating inter-node communication for frozen parameters." Conversely, generic advice is that assigning *fewer* GPUs per job and running more independent jobs beats scaling one job out.

### 2.4 QLoRA throughput penalty

Reported range is wide: **0.7–0.8× LoRA throughput** (i.e. 25–43% slower) in benchmark studies; some sources say 30%, others claim only 5–10% with recent bitsandbytes, and one measures forward passes ~66% slower. Cause: every 4-bit weight is dequantized to 16-bit on each matmul.

**Recommendation: default QLoRA throughput factor to 0.75 with a visible, editable assumption chip and a `medium-low` confidence badge.** The existing `ConfidenceBadge` / `AssumptionChips` components already support exactly this.

---

## 3. Workload sizing (the user-facing inputs)

The Inference tab's lesson (per `docs/decisions/2026-06-24-inference-ux-presets.md`) applies directly: **users do not think in tokens.** They think "I have ~2,000 support transcripts and want the model to match our tone."

### 3.1 Dataset size heuristics by goal

| Goal | Examples | Notes |
|---|---|---|
| Narrow task, LoRA/QLoRA | 100–500 | simple classification can go lower |
| Well-scoped task (classification, instruction following, code) | 500–2,000 | the common starting point |
| Style / tone / format adoption | ~1,000 | LIMA: 1,000 curated pairs sufficed for strong alignment |
| Broad multi-domain reasoning | 5,000–20,000 | |
| Robust production quality | 10,000+ | "1,000 well-curated examples often outperform 10,000 mediocre ones" |
| Domain knowledge injection (continued pretraining) | 10^8–10^10 tokens | different regime |

**Epochs: 1–3 is standard** (2–3 for 500–2,000 examples). Beyond 5 epochs the model memorizes exact phrasings — fine-tuning sets of 10^2–10^5 examples are vanishingly small against a 2T+ token pretraining corpus.

### 3.2 Token derivation

```
dataset_tokens  = n_examples × avg_tokens_per_example
tokens_processed = dataset_tokens × epochs / packing_efficiency
```

**Sequence packing** concatenates variable-length samples to eliminate padding. Measured speedups vs naive/random batching: **3.8× at 32K seq len, 7.3× at 128K**; packing and sorted batching each roughly halve training time at moderate lengths. Without packing, padding efficiency on a mixed-length corpus is typically 0.4–0.6. **Default packing on (efficiency 0.95); unpacked default 0.55.**

### 3.3 Long context

Activation memory is linear in `s` under FlashAttention but the constant is large: 8B at s=32768, b=1, checkpointed ≈ 12.8 GB of activations alone. Long-context SFT (32K–128K) is where activations overtake optimizer state as the dominant term, and where ChunkFlow-style chunking (up to 4.53× faster than Megatron-LM) and hierarchical balance packing matter. A `sequence length` input with a warning above ~16K is warranted.

---

## 4. Proposed engine shape

Mirrors the existing adapter pattern (`src/engine/adapters/*`, `hardware-estimator.ts`).

```
FineTuneScenario
  # model
  modelId | custom       → N_total, N_active, L, d, d_kv, d_ff, vocab
  # method
  method                 full | lora | qlora | dora | dpo | orpo | grpo
  loraRank, loraTargets
  optimizer              adamw_fp32 | adamw_8bit | adafactor | sgd_momentum | lomo
  weightPrecision        bf16 | fp8 | nf4
  gradientCheckpointing  bool
  # workload
  nExamples, avgTokensPerExample, epochs, sequenceLength, packing
  microBatchSize
  # hardware
  gpuId, availability, mfuOverride, scalingEfficiency, parallelism (ddp|fsdp|zero3)

FineTuneResult
  memoryBreakdownGb      { weights, grads, optimizer, activations, aux, overhead }
  minGpusMemory          ← ceil(M_total / usable_vram)
  fitsOnSingleGpu        bool
  tokensProcessed
  flopsPerToken, totalFlops
  wallClockHours         @ minGpus, and @ user-chosen GPU count
  costUsd                ← reuse src/data/cloud-gpu-pricing.json
  gpusForDeadline        (inverse solve, optional)
  warnings[], trace[], confidence
```

`minGpus` should round up to deployable granularity (1, 2, 4, 8, then multiples of 8) and flag when the memory floor crosses a node boundary.

### Confidence ratings (reusing the existing `ConfidenceLevel` scale)

| Quantity | Confidence | Why |
|---|---|---|
| Weights / grads / optimizer memory | **high** | exact arithmetic |
| LoRA adapter params (catalog model) | **high** | exact enumeration |
| LoRA adapter params (heuristic fallback) | medium | ±30% |
| Activation memory | **medium** | implementation-dependent constants |
| FLOPs/token | **high** | 6N / 4N well established, corroborated by NeMo back-solve |
| Wall-clock (LoRA/full, BF16) | **medium** | MFU validated against NeMo |
| QLoRA wall-clock | **medium-low** | 0.7–0.8× range is wide |
| DPO/ORPO | medium-low | multiplier heuristics |
| GRPO / RL | **low** | rollout-dominated; highly config-sensitive |

### Warnings worth emitting

- Memory floor exceeds single-GPU → "requires FSDP/ZeRO-3 sharding across N GPUs"
- Full FT chosen where QLoRA would fit on 1 GPU → quantify the cost delta
- MoE model: "memory follows 235B total params, compute follows 22B active"
- MoE + QLoRA: **"do not use 4-bit QLoRA for MoE — bitsandbytes lacks native MoE support and quantization interferes with expert routing; use BF16 LoRA"**
- Epochs > 5 → memorization risk
- Examples < 100 → below useful floor
- Sequence length > 16K → activation-dominated; consider chunking
- Dataset tokens > 1/3 of base model's pretraining compute → EU AI Act GPAI provider threshold ("fine-tuning an open model makes you a general-purpose AI provider only if you use more than a third of the original training compute")

---

## 5. Model catalog requirement

A fine-tuning calculator needs per-model architecture data the current `reference-models.ts` does not carry (it has only params/tokens/FLOPs for pretraining comparison). Required fields: `N_total`, `N_active`, `layers`, `hidden`, `kvHeads`/`heads`, `ffnDim`, `vocab`, `contextWindow`, `isMoe`, `license`.

2026 open-weight landscape to cover (every frontier model is sparse MoE):

| Model | Total | Active | Notes |
|---|---|---|---|
| Llama 3 8B / 70B / 405B | dense | — | the calibration standards |
| Qwen 2.5 7B / 32B / 72B | dense | — | NeMo benchmark models |
| Qwen3 235B-A22B | 235B | 22B | |
| Qwen3.8-27B | 27B dense | — | 64 layers, 1M context |
| Llama 4 Scout | 109B | 17B | 10M context |
| DeepSeek-R1 | 671B | 37B | 128K context |
| DeepSeek V4 | ~1T | 37B | |
| GLM-5.2 | 744B | 40B | |
| Gemma 4 26B-A4B | 25.2B | 3.8B | 256K context |
| Gemma 4 31B | 31B dense | — | 60 layers |
| Mistral Small 4 | 119B | 6B | |
| Mixtral 8×7B | 47B | 13B | classic MoE reference |

Exact layer/hidden/FFN dims must be pulled from each model's `config.json` before implementation — the survey sources above give parameter counts reliably but architecture dims only sporadically.

---

## 6. Cost anchors (sanity checks for the cost output)

| Job | Config | Time | Cost |
|---|---|---|---|
| 7B QLoRA | RTX 5090 | 3 h | **$2.76** |
| 70B QLoRA | 1× H100 PCIe | 10 h | **$20.10** |
| 70B full FT | 11× H100 SXM5 | 32 h | **$1,785** |

> **Caveat added 2026-10-14, after engine validation.** These three rows are **not like-for-like** — they differ in model, hardware *and* dataset, so the spread between them is not attributable to the method. Do not cite them as a method comparison.
>
> On a *fixed* workload (70B, 2,000 examples × 800 tokens × 3 epochs, H100 at $3/GPU-hr) the engine gives full FT $12.08 / LoRA $6.90 / QLoRA $6.90 — a **1.75× spread**, not 1000×. Billed by the GPU-hour, `cost ≈ totalFLOPs / (MFU × availability)`: GPU count cancels against wall-clock, and the method only moves FLOPs/token between `8N` and `6N`.
>
> What genuinely varies by **24–27×** on a fixed workload is **memory footprint and minimum GPU count** (1,212 GB / 24 GPUs vs 44 GB / 1 GPU). The tab's value proposition is therefore **feasibility**, not cost arbitrage. See `docs/decisions/2026-10-14-finetuning-tab-decisions.md` Decision 2.

---

## 7. Open design questions (for brainstorming)

1. **Separate tab vs. extending the Training tab?** The Training tab already has a `TrainingMode` enum including `SFT`, `LORA`, `RLHF` — but its formulas are pretraining-oriented (`6ND` with tokens-per-parameter) and its question is deadline-driven. Research says the fine-tuning question is memory-feasibility-driven. Recommendation: separate mode, reuse `GpuSku`, `hardware-estimator`, `FormulaTraceDrawer`, `AssumptionChips`, `ConfidenceBadge`, cloud pricing data.
2. **Primary output card:** "minimum GPUs + time + cost", or a **method comparison matrix** (full / LoRA / QLoRA × a few GPU SKUs) as the hero? The matrix makes the 3-order-of-magnitude spread immediate.
3. **Phase RL out?** DPO/ORPO are cheap multipliers (2× memory). GRPO/PPO need a rollout-generation compute model — genuinely different machinery, `low` confidence. Suggest phase 2.
4. **Model catalog scope:** 12–15 curated models + "custom" (params/layers/hidden), matching the Inference tab's data-file pattern (`src/data/*.json`)?
5. **Preset layer:** fine-tuning equivalent of the inference use-case presets — e.g. "Style/tone adoption (1,000 examples, 3 epochs, LoRA r=16)", "Domain adaptation", "Tool-use / function calling", "Reasoning distillation".
6. **Deadline inverse-solve:** keep it, as a secondary panel rather than the headline?

---

## Sources

**Memory & VRAM sizing**
- [GPU VRAM Requirements to Fine-Tune LLMs in 2026 (Spheron)](https://www.spheron.network/blog/gpu-vram-requirements-fine-tune-llm-2026/) — the 5-row VRAM table, worked breakdowns, cost anchors
- [LoRA Memory Allocation (ApXML)](https://apxml.com/courses/designing-and-sizing-production-llm-infrastructure/chapter-5-fine-tuning-memory-mechanics/lora-memory-allocation) — `2·r·d` adapter math, 16 B/trainable-param
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314) — NF4, paged optimizers
- [How Many GPUs to Fine-Tune a 70B LLM On-Prem? (GPU Mart)](https://rdp.in/gpu-mart/knowledge-base/how-many-gpus-do-you-need-to-fine-tune-a-70b-llm-on-prem/) — 11× H100 / ZeRO-3 corroboration
- [Distributed LLM Training: FSDP, ZeRO-3, Megatron-Core (Spheron)](https://www.spheron.network/blog/distributed-llm-training-fsdp-deepspeed-megatron-multi-node/)
- [LLM Fine-Tuning GPU Requirements: LoRA & QLoRA (GMI Cloud)](https://www.gmicloud.ai/en/blog/fine-tuning-infrastructure-for-llms-lora-qlora-and-full-fine-tuning-gpu-requirements)
- [Fine-Tuning Infrastructure: LoRA, QLoRA, PEFT at Scale (Introl)](https://introl.com/blog/fine-tuning-infrastructure-lora-qlora-peft-scale-guide-2025)

**Activations & recomputation**
- [Reducing Activation Recomputation in Large Transformer Models (MLSys 2023)](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf) — `sbh(34 + 5as/h)`
- [Megatron-Turing NLG 530B](https://arxiv.org/pdf/2201.11990)
- [Estimating memory requirements of transformer networks](https://schartz.github.io/blog/estimating-memory-requirements-of-transformers/)

**Throughput & MFU**
- [NVIDIA NeMo AutoModel Performance Summary](https://docs.nvidia.com/nemo/automodel/latest/performance/performance-summary) — the LoRA tok/s/GPU table; primary MFU calibration
- [A100 vs H100 vs B200 LoRA cost benchmark (VESSL)](https://vessl.ai/en/blog/lora-finetuning-cost-a100-h100-b200)
- [Chronicals: High-Performance LLM Fine-Tuning Framework](https://arxiv.org/html/2601.02609v1)
- [Profiling LoRA/QLoRA Efficiency on Consumer GPUs](https://arxiv.org/pdf/2509.12229)
- [FCDP: Fully Cached Data Parallel](https://arxiv.org/html/2602.06499v1) — LoRA comm elimination
- [mLoRA: Pipeline Parallelism for LoRA Adapters (VLDB)](https://www.vldb.org/pvldb/vol18/p1948-tang.pdf)

**Optimizers**
- [8-bit Optimizers via Block-wise Quantization](https://arxiv.org/pdf/2110.02861) — 8 B → 2 B/param
- [LOMO: Full Parameter Fine-tuning with Limited Resources](https://arxiv.org/pdf/2306.09782) — 102.2 → 14.6 GB
- [AdaLomo](https://arxiv.org/pdf/2310.10195)
- [FlashOptim: Memory Efficient Training](https://arxiv.org/html/2602.23349v1) — 16 B → 7 B/param

**Preference optimization & RL**
- [DPO Fine-Tuning on GPU Cloud (Spheron)](https://www.spheron.network/blog/dpo-fine-tuning-gpu-cloud/) — 2× SFT memory
- [GRPO Fine-Tuning on GPU Cloud (Spheron)](https://www.spheron.network/blog/grpo-fine-tuning-gpu-cloud/) — 2–3× SFT memory
- [Beyond SFT: Sizing DPO and RLVR Post-Training Clusters (GPU Mart)](https://rdp.in/gpu-mart/knowledge-base/beyond-sft-sizing-dpo-rlvr-post-training-infrastructure/)
- [A Technical Survey of RL Techniques for LLMs](https://arxiv.org/pdf/2507.04136)

**Dataset sizing & packing**
- [Post-Training Science for Supervised Fine-Tuning](https://arxiv.org/pdf/2609.01244)
- [Massive SFT Experiments: Data, Layer, and Training Factors](https://arxiv.org/html/2506.14681v1)
- [Supervised Fine-Tuning with SmolLM3 (Hugging Face)](https://huggingface.co/learn/smol-course/en/unit1/3)
- [Hierarchical Balance Packing for Long-Context SFT](https://arxiv.org/pdf/2503.07680) — 3.8× @32K, 7.3× @128K
- [Efficient Long Context Fine-tuning with Chunk Flow](https://arxiv.org/pdf/2503.02356) — 4.53× vs Megatron-LM
- [LongAlign: Long Context Alignment](https://aclanthology.org/2024.findings-emnlp.74.pdf)

**MoE**
- [Fine-Tune MoE LLMs on GPU Cloud (Spheron)](https://www.spheron.network/blog/fine-tune-moe-llms-gpu-cloud-expert-parallelism-load-balancing-moelora-guide/) — the no-QLoRA-for-MoE warning
- [Fine-Tuning MoE: Memory and Routing Realities (GPU Mart)](https://rdp.in/gpu-mart/knowledge-base/fine-tuning-mixture-of-experts-memory-routing-realities/)
- [Expert Parallelism for MoE Models (ApXML)](https://apxml.com/courses/designing-and-sizing-production-llm-infrastructure/chapter-4-distributed-topologies-for-llms/expert-parallelism-moe-models)
- [MixLoRA](https://arxiv.org/abs/2404.15159)

**Model landscape**
- [awesome-open-weight-models](https://github.com/phlx0/awesome-open-weight-models)
- [Best Open-Source LLM Models in 2026 (Hugging Face blog)](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Open-weight models 2026: Kimi K3, Qwen3.8, DeepSeek V4.1](https://slash-digital.io/en/insights/open-weight-models-2026/)
- [LoRA Learns Less and Forgets Less](https://arxiv.org/pdf/2405.09673)
