# Deep Research: LLM Pretraining GPU Sizing

**Date:** 2026-10-14
**Purpose:** Refresh the quantitative foundation of the existing **LLM Training (pretraining)** pane — the `llm` adapter, `hardware-estimator`, `gpu-data`, `reference-models`, `calibration` and `reverse-solve` modules.
**Question the pane answers:** *How many of GPU X do I need to pretrain a model of N parameters on D tokens by date T — and what will it cost?*

---

## 0. The central finding (shapes everything below)

**Pretraining is compute-bound by a factor of ~470×.** Fine-tuning is memory-bound; inference is throughput-and-memory-bound; pretraining is the one regime where the arithmetic is simple and the *only* real uncertainty is achieved utilization.

Concretely, at the repo's own defaults (AdamW 16 B/param, 0.85 usable HBM, 30-day window, H100 @ 40% MFU):

| Model | Memory floor (GPUs) | Compute floor, 30 days (GPUs) | Ratio |
|---|---|---|---|
| 8B / 15.6T tokens | 2 | 892 | **474×** |
| 70B / 15.6T | 17 | 7,806 | **474×** |
| 405B / 15.6T | 96 | 45,162 | **474×** |
| 1T-total MoE / 20T | 236 | 142,962 | **608×** |

`memoryLowerBoundGpus` (`src/engine/hardware-estimator.ts:36-39`) therefore **never binds** for any realistic pretraining scenario. It is correct code answering an irrelevant question. The `memoryLowerBoundGpus > requiredGpus` warning at `src/engine/calculator.ts:73` can only fire for absurd inputs (a multi-trillion-parameter model with a multi-year window).

Three consequences for the pane's design:

1. **All the accuracy lives in one number**: the effective fraction of dense peak the cluster actually delivers, `mfu × availability / overheadFactor`. Everything else is exact arithmetic.
2. **That number is empirically pinned.** Across eleven sourced real runs it lands at **0.14–0.48 of dense BF16 peak**, with a dense-BF16 interquartile band of **0.26–0.40** and a median near **0.34**. The repo's defaults multiply out to **0.400 × 0.90 / 1.10 = 0.327** — within 5.4% of the Llama 3.1 405B anchor. *The current defaults are well-calibrated in aggregate*; the errors are structural (FP8, small models, MoE, long context), not in the central estimate.
3. **The three sliders are one degree of freedom.** `requiredGpus ∝ overheadFactor / (mfu × availability)`. `mfu`, `availability` and `overheadFactor` are perfectly degenerate in the output. Three sliders for one parameter is a UX trap: users will tune all three and double-count. See §10 Q1.

What distinguishes the pretraining question from the other two panes:

| | Pretraining | Fine-tuning | Inference |
|---|---|---|---|
| Binding constraint | **Deadline × FLOPs** | Does model state fit at all | Tokens/s and KV-cache memory |
| Primary unknown | achieved MFU at scale | bytes/param per method | batch × context × cache policy |
| Memory role | never binds (474×slack) | the whole question | co-binding |
| Failure mode of a bad estimate | 2× cluster capex | job OOMs | p99 latency SLO breach |
| What the user actually has | a budget and a date | a dataset | a traffic forecast |

---

## 1. Scaling laws: is 20 tokens/parameter still defensible?

### 1.1 What Chinchilla said, and what the replication says

Hoffmann et al. 2022 found that under a fixed compute budget, `N` and `D` should scale at roughly equal rates, anchored at Chinchilla 70B / 1.4T = **20 TPP**. Epoch AI's replication re-fit the parametric model and found the optimum at **25.6 TPP**, noting DeepMind's own parametric form actually predicts **~70 TPP** at Chinchilla's scale — the three methods in the original paper disagree with each other, which is why the replication exists.

So the honest statement is: **compute-optimal TPP is 20–26, with a parametric-model reading as high as 70.** Keep 20 as the "Chinchilla compute-optimal" preset; it is still the right answer to the question Chinchilla asked.

### 1.2 But nobody trains compute-optimal any more

TPP in actual published runs (computed from sourced `N` and `D`; MoE rows show both total and active):

| Run | Params | Tokens | TPP (total) | TPP (active) |
|---|---|---|---|---|
| GPT-3 175B (2020) | 175B | 300B | **1.7** | — |
| Chinchilla 70B (2022) | 70B | 1.4T | **20.0** | — |
| Falcon 180B (2023) | 180B | 3.5T | **19.4** | — |
| Llama 2 70B (2023) | 70B | 2.0T | 28.6 | — |
| Llama 2 7B (2023) | 7B | 2.0T | 285.7 | — |
| Llama 3.1 405B (2024) | 405B | 15.6T | 38.5 | — |
| Llama 3.1 70B | 70B | 15.6T | 222.9 | — |
| Llama 3.1 8B | 8B | 15.6T | **1,950** | — |
| DeepSeek-V3 (2024) | 671B | 14.8T | 22.1 | **400** |
| Qwen3-235B-A22B (2025) | 235B | 36T | 153.2 | **1,636** |
| OLMo 3 32B (2025) | 32B | 5.5T | 171.9 | — |
| Marin 8B (2025) | 8B | 12.7T | 1,587.5 | — |
| Marin 32B (2025) | 32B | 6.44T | 201.2 | — |
| GLM-5 (2026) | 744B | 28.5T | 38.3 | **712** |
| Liquid LFM2.5-350M (2026) | 0.35B | 28T | **80,000** | — |

Pattern: **small models are massively overtrained; frontier models sit at 20–40 TPP on *total* parameters.** Note DeepSeek-V3 and GLM-5 are nearly Chinchilla-optimal in *total* params (22.1, 38.3) while being 400–712 TPP in *active* params. This is not a coincidence — see §3.3.

### 1.3 Why: inference-aware scaling

Sardana et al. (*Beyond Chinchilla-Optimal*) formalize what everyone does in practice: when you will serve the model to many users, total lifetime compute (train + serve) is minimized by a **smaller, longer-trained** model. The objective changed from "minimize loss at fixed training budget" to "minimize lifetime compute for a target quality," and those optimize differently. Published inference-optimal ratios land at **100–200 TPP**, with the small-model tail going far past that.

Supporting results:
- Over-trained models scale *reliably*: loss remains predictable out to extreme TPP (*Language Models Scale Reliably With Over-Training*, ICLR 2025).
- Loss keeps falling out to ~10,000 TPP, with exponentially diminishing returns.
- **Test-time scaling makes overtraining compute-optimal** (arXiv 2604.01411): if the deployed model will spend reasoning tokens, the optimal pretraining point shifts further toward overtraining.

### 1.4 Data-constrained scaling (the ceiling on D)

Muennighoff et al. (JMLR 2025): with a fixed unique-token corpus, **up to ~4 epochs of repetition is nearly free**; beyond that the marginal value of added compute decays toward zero. For the k-th repetition of `n` tokens, effective unique tokens ≈ `n·δ^(k-1)`.

2026 refinement (arXiv 2605.01640, *Prescriptive Scaling Laws for Data Constrained Training*): **the 4-epoch rule holds only for ~3B models near 1× Chinchilla; the recommended epoch count falls as `N` and TPP grow.** A frontier run at 200 TPP should not assume 4 free epochs.

**Implication for the pane:** TPP is not freely dialable. If `D = TPP × N` exceeds ~4× the user's available unique-token corpus, the estimate is compute-accurate but quality-misleading. Worth a `uniqueTokensAvailable` input and a repetition warning.

### 1.5 Recommendation

| TPP preset | Value | When | Confidence |
|---|---|---|---|
| Chinchilla compute-optimal | **20** | minimize loss per training FLOP | high |
| Epoch-corrected optimal | **26** | the replication's point estimate | high |
| Frontier dense (2026 practice) | **150** | median of Llama 3 70B / OLMo 3 / Marin 32B / Qwen3-total | medium |
| Inference-optimal small model | **1,000** | ≤10B model for high-volume serving | medium |
| MoE (on **active** params) | **400** | DeepSeek-V3 / GLM-5 class | medium |

**Change the default from 20 to 150** (`src/store/scenario-store.ts:119`). The current default understates tokens — and therefore FLOPs, GPUs and cost — by **7.5×** for any realistic 2026 dense run. This is the single largest magnitude error in the pane. Ship it as a labeled preset so the change is visible, not silent.

---

## 2. The 6ND FLOP model: derivation, correction terms, and when it breaks

### 2.1 Derivation

Per token, forward + backward, for a standard decoder-only transformer with `L` layers, hidden `d`, FFN `4d`, sequence `s`:

```
FFN        : L · 48 d²              (2 matmuls × 4d × 2 fwd × 3 fwd+bwd)
QKVO proj  : L · 24 d²
attention  : L ·  6 d (s+1)         (QKᵀ + AV, halved by causal masking)
-----------------------------------------------------------------------
total/token: 72 L d²  +  6 L d (s+1)
```

Since a standard transformer has `N ≈ 12 L d²` non-embedding params, `72 L d² = 6N` exactly. **The 6ND model is not an empirical fit — it is the first two terms of an exact accounting, with the attention term dropped.**

### 2.2 Correction term 1: attention at long context

```
effective_architecture_factor = 6 · ( 1 + s / (12 d) )
```

Exact, and it collapses to one visible multiplier. Magnitudes:

| Model (hidden d) | s=2K | s=8K | s=32K | s=128K |
|---|---|---|---|---|
| 8B (d=4,096) | 6.25 (+4%) | **7.00 (+17%)** | 10.00 (+67%) | **22.00 (+267%)** |
| 70B (d=8,192) | 6.12 (+2%) | 6.50 (+8%) | 8.00 (+33%) | 14.00 (+133%) |
| 405B (d=16,384) | 6.06 (+1%) | **6.25 (+4%)** | 7.00 (+17%) | 10.00 (+67%) |

This reproduces de Vries' published figures exactly (8% of total at 2K, 16–33% overhead at 8–16K, 260% at 128K for a LLaMA-7B shape) and explains why Llama 3's measured MFU fell from 41% at 8K to 38% at 128K — part of that drop is not utilization at all, it is extra model FLOPs that `6ND` does not count.

**Design note:** the right break-even intuition is `s ≈ 12d`. Above that, attention dominates. Modern mitigations (sliding-window attention, GQA, Mamba hybrids, linear attention) reduce the term but no public model eliminates it, and none of them change the fact that **plain 6ND understates a 128K-context 8B pretraining run by 3.7×.**

### 2.3 Correction term 2: embeddings and the output head

The un-embed (logits) matmul costs `6 d V` per token fwd+bwd. As a fraction of `6N`:

| Model | d | V | `dV/N` |
|---|---|---|---|
| 1B | 2,048 | 128,256 | **+21.2%** |
| 8B | 4,096 | 128,256 | **+6.5%** |
| 70B | 8,192 | 128,256 | +1.5% |
| 405B | 16,384 | 128,256 | +0.5% |

Caveat worth stating in the trace: if `N` is the *reported* parameter count it already includes the embedding table, and `6N` therefore charges 6 FLOPs/param to an input lookup that costs ~0. For **tied** embeddings the over-count and the missing logits term roughly cancel. For **untied** embeddings with a 128K+ vocab on a small model, the net is a real understatement of up to ~20%.

OpenAI's `6N` uses *non-embedding* params; DeepMind's Chinchilla accounting includes embeddings and logits. The repo uses whatever the user types, which is almost always the total. Document the convention.

### 2.4 Correction term 3: activation recomputation (do NOT add it)

PaLM 540B reported **46.2% MFU and 57.8% HFU**. The 11.6-point gap is activation rematerialization plus cross-pod gradient reduction. Because MFU is *defined* as model-FLOPs / peak, recompute FLOPs are already excluded from the numerator. **If the user supplies a measured MFU, adding a recompute term to `overheadFactor` double-counts.** `overheadFactor` must cover only non-model-FLOP *wall-clock*: evals, data stalls, restart loss, rollbacks.

### 2.5 Summary: where 6ND is wrong and by how much

| Situation | 6ND error | Fix |
|---|---|---|
| Frontier dense, 8K context | −1% to −4% | none needed |
| Mid-size dense, 8K context | −8% to −17% | `× (1 + s/12d)` |
| Any model, 128K context | −67% to **−267%** | `× (1 + s/12d)` — mandatory |
| Small model (≤2B), large vocab, untied | −20% | `+ dV/N` |
| MoE | **+1,800% if N=total** | use `N_active` |
| Measured-MFU input | +25% if recompute added | don't add it |

**Keep `architectureFactor = 6` as the base**, and add a derived, displayed correction `6 · (1 + s/(12d))` driven by two new optional inputs (`sequenceLength`, `hiddenDim`). Showing the derived factor in `FormulaTraceDrawer` is the honest presentation.

---

## 3. MoE pretraining

### 3.1 The split

**FLOPs follow active parameters; memory follows total parameters.** The training FLOP model is `6 · N_active · D` (+ a negligible router term). This is the single most common sizing error on 2026 models, because *every* frontier open-weight model is sparse.

| Model | Total | Active | Experts | Sparsity | Released |
|---|---|---|---|---|---|
| Mixtral 8×7B | 47B | 13B | 8 (2 active) | 27.7% | 2023 |
| Llama 4 Scout | 109B | 17B | 16 | 15.6% | 2025 |
| Qwen3-235B-A22B | 235B | 22B | 128 (8 active) | 9.4% | 2025 |
| DeepSeek-V3 / R1 | 671B | 37B | 256 + 1 shared | 5.5% | 2024-25 |
| Kimi K2 | 1.0T | 32B | 384 | 3.2% | 2025 |
| GLM-5 | 744B | ~40B | — | 5.4% | Feb 2026 |
| DeepSeek V4 Flash | 284B | 13B | — | 4.6% | Apr 2026 |
| Kimi K3 | 2.8T | 104B | 896 | 3.7% | Jul 2026 |
| GLM-5.3 | 753B | ~40B | 256 + 1 shared | 5.3% | Aug 2026 |
| DeepSeek V4 Pro | 1.65T | 49B | 384 + 1 shared (6 active) | 3.0% | Aug 2026 |

Sparsity has fallen from 28% (2023) to **3–5% (2026)**. Entering a 2026 MoE's *total* parameter count into the current `modelParameters` field overstates FLOPs by **20–34×**.

### 3.2 Granularity and sparsity scaling laws (these now exist)

- *Parameters vs FLOPs: Scaling Laws for Optimal Sparsity* (arXiv 2508.18672): **optimal sparsity is set jointly by active FLOPs and total TPP** — not by either alone.
- *Towards Greater Leverage* (ICLR 2026): fits a unified "efficiency leverage" law over activation ratio, expert granularity `g`, and compute budget across 300+ MoE models up to 28B. **An activation ratio of 3.1% at granularity 12 yields >7× efficiency leverage**; `g=8` is the training-efficiency sweet spot; `g≥2` beats non-fine-grained at equal wall-clock.
- *Compute-Optimal Is Not Cluster-Optimal* (arXiv 2608.10605): the compute-optimal MoE is not the cluster-optimal one once all-to-all and aggregate-HBM constraints are priced in.
- Task asymmetry: **reasoning peaks near 20 TPP; memorization improves at lower TPP.** A single TPP default cannot serve both.

### 3.3 Why frontier MoEs sit at 20–40 total TPP

If `N_total` is set by the memory/serving budget and `N_active` by the FLOP budget, then `D` is chosen against `N_active`. DeepSeek-V3 at 400 active-TPP and 22 total-TPP is *simultaneously* heavily over-trained (active) and Chinchilla-optimal (total). **The pane must ask which parameter count the TPP applies to.** Recommendation: TPP applies to active params, with the total-TPP shown as a derived read-out.

### 3.4 Systems cost of MoE

Wall-clock grows with the expert pool `E` even at constant active FLOPs, through: all-to-all expert-parallel traffic, aggregate HBM for all experts' params/grads/optimizer state, and GPU underutilization from small per-expert GEMMs. Reference points:

- MegaScale-MoE: **1.88× higher MFU than Megatron-LM** on a 352B MoE across 1,440 GPUs — i.e. baseline MoE MFU was roughly half of a well-tuned implementation.
- DeepSeek-V3: **20.2% of the achievable 1,700 TFLOP/s FP8 peak** (17.3% of nominal 1,979), vs ~41% BF16 for dense Llama 3. Even after accounting for FP8 (§5.3), MoE costs utilization.
- 2026 state of the art: >1,200 TFLOP/s/GPU sustained on DeepSeek-V3-685B on GB300, >900 TFLOP/s/GPU on Qwen3-235B (Megatron Core) — i.e. MoE MFU has largely caught up on current software.

**Recommendation:** add `activeParameters` to `LlmConfig`; default `N_active = N_total` (dense); apply a `moeEfficiencyFactor` default **0.85** (medium-low) to MFU when `N_active < N_total`, representing residual all-to-all cost on well-tuned 2026 stacks.

---

## 4. Refreshed GPU SKU table (October 2026)

`GPU_DATA_SPEC_DATE = '2026-06-18'` (`src/engine/gpu-data.ts:65`) is four months stale. Since then B300/GB300, MI355X, TPU v7 Ironwood and Trainium3 all reached GA, and the table carries only 4 NVIDIA SKUs.

### 4.1 Dense-vs-sparse derivation note

NVIDIA publishes tensor-core throughput **with 2:4 structural sparsity** for BF16/FP8/INT8 and marks it with an asterisk; dense is exactly half. Sparsity is not used in pretraining, so **all rows below are dense.** AMD publishes dense FP16/BF16 directly and sparse only for FP8/FP4 — read AMD's numbers carefully. Google and AWS publish dense. The derivations:

- H100/H200 SXM: NVIDIA lists 1,979 TFLOPS BF16* → **989.5 dense**; 3,958 FP8* → **1,979 dense**.
- B200: HGX B200 8-GPU = 72 PFLOPS FP8 dense → **4.5 PF/GPU**; BF16 **2.25 PF/GPU**; FP4 72 PF dense/8 = **9 PF/GPU**.
- GB200 NVL72: 360 PFLOPS FP16/BF16* ÷ 72 ÷ 2 = **2.5 PF dense/GPU**; NVFP4 720 PF dense ÷ 72 = **10 PF/GPU**; HBM 13.4 TB ÷ 72 = **186 GB/GPU**.
- MI355X: AMD gives **2.5 PF dense FP16/BF16**, 5.0 PF dense FP8, 10.1 PF FP4.

### 4.2 The table (directly transcribable into `GpuSku`)

Existing `GpuSku` fields plus three proposed additions (`fp4DenseFlops`, `tdpWatts`, `scaleUpGbps`).

| id | label | bf16Dense | fp8Dense | fp4Dense | memGb | bw TB/s | TDP W | scale-up GB/s | defaultMfu (BF16) | source |
|---|---|---|---|---|---|---|---|---|---|---|
| `h100-sxm` | NVIDIA H100 SXM | 0.9895e15 | 1.979e15 | — | 80 | 3.35 | 700 | 900 | **0.40** | [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) |
| `h200-sxm` | NVIDIA H200 SXM | 0.9895e15 | 1.979e15 | — | 141 | 4.8 | 700 | 900 | **0.42** ↓from 0.45 | [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) |
| `b200-sxm` | NVIDIA B200 SXM / HGX B200 | 2.25e15 | 4.5e15 | 9.0e15 | **180** ↑from 175 | **7.7** (new) | 1000 | 1800 | **0.39** | [Lenovo HGX B200 guide](https://lenovopress.lenovo.com/lp2226.pdf), [DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) |
| `b300-sxm` | **NEW** NVIDIA B300 SXM (Blackwell Ultra) | 2.25e15 | 4.5e15 | 13.5e15 | 288 (268 usable) | 8.0 | 1200–1400 | 900 | **0.40** | [SemiAnalysis B300](https://inferencex.semianalysis.com/chips/b300) |
| `gb200-nvl72-gpu` | GB200 NVL72 per-GPU | 2.5e15 | 5.0e15 | 10.0e15 | 186 | 8.0 | ~1200 | 1800 | **0.42** ↑from 0.40 | [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) |
| `gb300-nvl72-gpu` | **NEW** GB300 NVL72 per-GPU (Ultra) | 2.5e15 | 5.0e15 | 15.0e15 | 288 | 8.0 | ~1400 | 1800 | **0.41** | [Flopper GB300](https://flopper.io/gpu/nvidia-gb300-grace-blackwell-ultra-superchip-252gb) (aggregator) |
| `mi300x` | **NEW** AMD Instinct MI300X | 1.3074e15 | 1.3074e15 | — | 192 | 5.3 | 750 | 896 | **0.30** | [AMD Instinct](https://en.wikipedia.org/wiki/AMD_Instinct), [ASBIS MI300](https://promo.asbis.com/amd_instinct_mi300) |
| `mi325x` | **NEW** AMD Instinct MI325X | 1.3074e15 | 2.6149e15 | — | 256 | 6.0 | 1000 | 896 | **0.31** | [SemiAnalysis MI325X](https://inferencex.semianalysis.com/chips/mi325x) |
| `mi355x` | **NEW** AMD Instinct MI355X | 2.5e15 | 5.0e15 | 10.1e15 | 288 | 8.0 | 1400 | 1075 | **0.33** | [Oracle OCI MI355X](https://blogs.oracle.com/cloud-infrastructure/amd-instinct-mi355x-on-oci-performance-technical-details), [SemiAnalysis](https://inferencex.semianalysis.com/chips/mi355x) |
| `mi450` | **NEW (low conf.)** AMD Instinct MI450 | ~5.0e15 ⚠ | 20e15 | 40e15 | 432 | 19.6–23.3 | ~1800 | 3600 | **0.30** | [WCCFTech MI400](https://wccftech.com/amd-instinct-mi400-accelerator-doubles-compute-40-pflops-432-gb-hbm4-memory-2026-launch/) (aggregator) |
| `tpu-v6e` | **NEW** Google TPU v6e (Trillium) | 0.918e15 | — (no native FP8) | — | 32 | 1.638 | ~? | 1.2 TB/s ICI | **0.45** | [Google TPU v6e docs](https://docs.cloud.google.com/tpu/docs/v6e) |
| `tpu-v7` | **NEW** Google TPU v7 (Ironwood) | 2.307e15 | 4.614e15 | — | 192 | 7.37 | ~? | 1.2 TB/s ICI | **0.40** | [Google TPU7x docs](https://docs.cloud.google.com/tpu/docs/tpu7x) |
| `trainium2` | **NEW** AWS Trainium2 | 0.667e15 | 1.299e15 | — | 96 (GiB) | 2.9 | ~? | — | **0.35** | [AWS Neuron Trn2](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) |
| `trainium3` | **NEW** AWS Trainium3 | 0.671e15 | 2.517e15 (MXFP8) | 2.517e15 (MXFP4) | 144 (GiB) | 4.9 | ~? | — | **0.35** | [AWS Neuron Trn3](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html) |
| `gaudi3` | **NEW** Intel Gaudi 3 | 1.835e15 | 1.835e15 | — | 128 | 3.7 | 900 | 1200 | **0.30** | [Hot Chips 2024 Gaudi 3](https://hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf) |

⚠ **MI450 BF16 is extrapolated, not sourced.** AMD published only FP8 (20 PF) and FP4 (40 PF); BF16 at half of FP8 is the conventional ratio but AMD has not confirmed it. Mark `low` confidence.

**Deliberately excluded:** NVIDIA Vera Rubin VR200 / NVL144. Announced for late 2026 with 288 GB HBM4 and ~22 TB/s, and the rack figures (3.6 EF FP4, 1.2 EF FP8 across 144 GPUs) are published, but the per-GPU dense BF16 is not, and sources conflict on whether the headline "50 PFLOPS FP4" is per-package or per-die. Adding a SKU on that basis would be invention. Add it when the datasheet lands.

### 4.3 Notable corrections to existing rows

- **B200 memory 175 → 180 GB.** HGX B200 is 1,440 GB / 8. The 175 figure (`gpu-data.ts:29`) is unsourced; 180 is confirmed by Lenovo's product guide. Also add `bandwidthTbps: 7.7` (the row currently has none, `gpu-data.ts:24-32`).
- **H200 `defaultMfu: 0.45` → 0.42.** The 0.45 is unsourced. No published pretraining MFU for *any* Hopper part exceeds Llama 3's 43%, and that was on H100. H200's advantage is capacity/bandwidth enabling bigger microbatches — real, but 0.45 > the best measured H100 figure is an unjustified claim. The existing warning at `calculator.ts:68-72` already explains the mechanism correctly; just lower the number.
- **B300 BF16 is *identical* to B200.** Blackwell Ultra's uplift is 1.5× FP4 and 49% more memory; "FP8 and BF16 stay at B200 levels." A user picking B300 for BF16 pretraining gets no compute gain — worth an explicit warning, mirroring the existing H200 one.
- **GB200 `defaultMfu` 0.40 → 0.42.** The 72-GPU NVLink domain at 130 TB/s aggregate lets TP groups span far more GPUs without crossing a slow link; this is the one SKU where topology should buy MFU. Medium confidence.

---

## 5. Achieved MFU in real pretraining runs

### 5.1 Published instantaneous MFU

| Run | Hardware | GPUs | Precision | Context | Reported MFU | Source |
|---|---|---|---|---|---|---|
| GPT-3 175B | V100 | 10,000 | FP16 | 2K | ~21% | Megatron-LM / Epoch |
| PaLM 540B | TPU v4 | 6,144 | BF16 | 2K | **46.2% MFU / 57.8% HFU** | [PaLM paper](https://arxiv.org/pdf/2204.02311) |
| Megatron-LM PTD-P 1T | A100 | 3,072 | BF16 | 2K | **52% of peak** | [Megatron SC21](https://arxiv.org/pdf/2104.04473) |
| Llama 3.1 405B | H100 | 8,192 | BF16 | 8K | **43%** (430 TFLOP/s/GPU) | [Llama 3 Table 4](https://arxiv.org/html/2407.21783v3) |
| Llama 3.1 405B | H100 | 16,384 | BF16 | 8K | **41%** (400 TFLOP/s/GPU) | same |
| Llama 3.1 405B | H100 | 16,384 | BF16 | 128K (CP=16) | **38%** (380 TFLOP/s/GPU) | same |
| Megatron-LM 100B+ class | H100 | — | BF16 | — | **38–52%** end-to-end | public benchmarks |
| DeepSeek-V3 671B/37B | H800 | 2,048 | FP8 | 4K | **20.2%** of achievable FP8 peak | [Medium analysis](https://medium.com/@dlrover/what-is-the-mfu-for-deepseek-v3-training-0d9ea4d42eb4) |
| GB200 NVL72, Llama 405B | GB200 | 512 | FP8 | 8K | 1,960 TFLOP/s/GPU ≈ **39% of FP8 peak** | [MLPerf v5.0 blog](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-up-to-2-6x-higher-performance-in-mlperf-training-v5-0/) |
| DeepSeek-V3-685B | GB300 | — | FP8 | — | >1,200 TFLOP/s/GPU ≈ **24% of FP8 peak** | Megatron Core 2026 |
| Qwen3-235B-A22B | GB300 (reported) | — | FP8 | — | >900 TFLOP/s/GPU ≈ **18% of FP8 peak** | Megatron Core 2026 |

### 5.2 Back-solved end-to-end utilization from GPU-hours

GPU-hours are the better regression target: they are what the calculator outputs, and they include everything. Computed as `6ND / (GPU-hours × 3600 × dense peak)`:

| Run | 6ND | GPU type | GPU-hours | Effective FLOP/s/GPU | % of dense peak |
|---|---|---|---|---|---|
| GPT-3 175B | 3.150e23 | V100 (125 TF) | 3.552e6 | 2.46e13 | **19.7%** |
| Llama 2 7B | 8.400e22 | A100 (312 TF) | 1.843e5 | 1.27e14 | **40.6%** |
| Llama 2 13B | 1.560e23 | A100 | 3.686e5 | 1.18e14 | **37.7%** |
| Llama 2 34B | 4.080e23 | A100 | 1.038e6 | 1.09e14 | **35.0%** |
| Llama 2 70B | 8.400e23 | A100 | 1.720e6 | 1.36e14 | **43.5%** |
| Llama 3.1 8B | 7.488e23 | H100 (989.5 TF) | 1.46e6 | 1.43e14 | **14.4%** |
| Llama 3.1 70B | 6.552e24 | H100 | 7.00e6 | 2.60e14 | **26.3%** |
| Llama 3.1 405B | 3.791e25 | H100 | 30.84e6 | 3.41e14 | **34.5%** |
| DeepSeek-V3 (pretrain) | 3.286e24 | H800 | 2.664e6 | 3.43e14 | **34.6%** of BF16 / **17.3%** of FP8 |
| OLMo 3 32B | 1.056e24 | H100 | 1.05e6 | 2.79e14 | **28.2%** |
| OLMo 3.1 32B | 1.056e24 | H100 | 1.376e6 | 2.13e14 | **21.5%** |
| Falcon 180B | 3.780e24 | A100 40GB | ~7.0e6 | 1.50e14 | **48.1%** ⚠ |

⚠ Falcon 180B's 48.1% exceeds any published instantaneous MFU on A100 and is likely an artifact of a rounded GPU-hour figure. Treat as an outlier, not an anchor.

**Critical caveat: model-card GPU-hours are not consistently defined across labs.** The Llama 2 family back-solves to 35–44% — essentially equal to instantaneous MFU, implying Meta counted only productive time. Llama 3.1 back-solves to 14–35%, implying the same lab counted downtime, evals and restarts. Any regression suite built on GPU-hours must record *which convention* each row uses.

### 5.3 FP8 and FP4: the biggest modeling error in the pane

`getPeakFlops` (`src/engine/hardware-estimator.ts:57-61`) swaps in the FP8 peak but `estimateHardware` (`:25`) uses the *same* MFU. That asserts FP8 delivers a full 2× speedup. It does not.

Measured end-to-end FP8 vs BF16 wall-clock speedups:
- **1.3–1.5×** typical, per Megatron-Bridge's own low-precision guide and corroborated independently.
- **1.26× without MoE parallel folding, 1.30× with**, in MoE Parallel Folding (Megatron Core).
- Default recipes apply FP8 **only to transformer-block linear layers**; softmax, layernorm, residuals, attention masking, optimizer and all-reduce stay BF16/FP32. Amdahl does the rest.

Therefore `MFU_FP8 ≈ MFU_BF16 × (1.3 to 1.5) / 2 = 0.65–0.75 × MFU_BF16`.

**Validation against the one sourced FP8 pretraining run:** DeepSeek-V3 back-solves to 17.3% of nominal FP8 peak (20.2% of the 1,700 TF/s actually available after reserving 20 of 132 SMs for comms) vs ~34.6% of BF16 peak at the same effective throughput — a ratio of 0.50. That is *below* the 0.65–0.75 band, but DeepSeek-V3 is also a 5.5%-sparsity MoE, which carries its own penalty (§3.4). `0.70 × 0.85 (MoE) = 0.595` — close to the observed 0.50 once the MoE factor is applied.

**FP4:** NVFP4 pretraining is no longer speculative. NVIDIA pretrained a 12B hybrid Mamba-Transformer on **10T tokens in NVFP4**, matching an FP8 baseline (MMLU-Pro 62.58 vs 62.62, validation loss within 1% for most of the run, 1.5% during annealing). It requires Random Hadamard transforms, 2-D quantization, stochastic rounding and selective high-precision layers. MLPerf v5.1's 4× Blackwell-Ultra-over-Hopper Llama 405B result implies FP4 training was in use (BF16 peak ratio is only 2.27×).

**Recommendation — add precision-relative MFU scaling:**

```
effectiveMfu = baseMfu(BF16) × precisionFactor × moeFactor
precisionFactor:  BF16 1.00   FP8 0.70   FP4 0.50
moeFactor:        dense 1.00  MoE 0.85
```

Add `'FP4_DENSE'` to `Precision` (`src/engine/types.ts:1`) and `fp4DenseFlops` to `GpuSku`. Without the precision factor, the pane **overstates FP8 throughput by 43%** and would overstate FP4 by 100%.

### 5.4 Scale- and size-dependence of MFU

Llama 3's own 8,192 → 16,384 GPU step (43% → 41%) gives **95.3% efficiency per 2× GPUs** at the 10⁴ scale. But the end-to-end figures show something stronger and stranger:

| Llama 3.1 model | End-to-end % of peak | Ratio to 405B |
|---|---|---|
| 8B | 14.4% | 0.42 |
| 70B | 26.3% | 0.76 |
| 405B | 34.5% | 1.00 |

The Llama 2 family shows the *opposite* (7B 40.6%, 70B 43.5% — nearly flat). The most plausible reconciliation: Llama 3's smaller models ran on the same ~16K-GPU cluster with the same 16M-token global batch, so per-GPU work per step was tiny (405B at 16,384 GPUs = 976 tokens/GPU/step) and the run became comm- and latency-bound. Llama 2's 7B ran on a proportionally smaller slice.

**This is a hypothesis, not a sourced fact** — Meta does not publish per-model cluster sizes for the 8B and 70B runs. Rather than a hard multiplier, emit a warning: *"this configuration implies < 2,000 tokens/GPU/step; MFU degrades sharply when the per-GPU microbatch is small — consider fewer GPUs over a longer window."* Confidence: `low` on the mechanism, `high` that the 8B figure is real.

---

## 6. Cluster realities: goodput, failures, stragglers

This section calibrates `availability` (`scenario-store.ts:124`).

### 6.1 Llama 3.1 405B — the best-documented frontier run

Over a 54-day pretraining snapshot on up to 16,384 H100s:

| Quantity | Value |
|---|---|
| Total interruptions | **466** (47 planned, 419 unexpected) |
| Mean time between unexpected interruptions | **~3 hours** |
| Effective training time | **> 90%** |
| Hardware-caused share of unexpected interruptions | **~78%** |
| GPU failures | 58.7% of all unexpected issues (148 faulty GPUs) |
| HBM3 memory failures | 72 events |
| Parallelism | TP 8 × CP 1 × PP 16 × DP 128 |
| Global batch | 16M tokens, seq 8,192 |

### 6.2 Failure rate scaling

| Cluster size | MTBF / MTTF | Source |
|---|---|---|
| Single H100 | ~50,000 h (~6 years) | Epoch AI |
| 1,024 GPUs | **~7.9 h** | Lablup 504-GPU report / derived |
| 16,384 GPUs | **~1.8–3 h** | Llama 3 (measured 3 h) |
| 100,000 GPUs | **every ~30 min** | Epoch AI |
| 1,000,000 GPUs | **every ~3 min** | Epoch AI |

### 6.3 Overhead decomposition

| Source | Cost | Source |
|---|---|---|
| Checkpoint write (Llama 405B, storage-based) | 2.5 s, optimal interval ~4 min | Epoch AI |
| Total time lost to checkpoint + recovery at 16K scale | **~2.1%** (≈1 pt of MFU) | Epoch AI |
| GPU-memory-based checkpointing | 0.32 s (8× faster); feasible to ~560M GPUs | Epoch AI |
| Restart / state reload | mean ~33 min per failure | Lablup |
| Lost compute per failure | mean ~0.98 h | Lablup |
| Planned maintenance | ~1 event/day | Llama 3 (47 in 54 days) |
| Fail-slow stragglers, 512–1,024 GPU jobs | **59% of jobs affected**, mean JCT delay **34.6%** | 2026 production-cluster study |
| Stragglers, production LLM clusters overall | 42.5% of jobs affected, **10.4% of GPU-hours wasted** | same |
| Idle spare capacity needed, 1M GPUs | ~480 nodes (0.3%) | Epoch AI |
| Industry goodput average | **~90%** | TNW / industry survey |

Epoch's headline conclusion is worth carrying into the UI: **hardware failures will not limit scaling** — storage-based checkpointing stays viable to ~70M GPUs, memory-based replication far beyond. The operative constraint at 10⁴–10⁵ GPUs is stragglers, not crashes.

### 6.4 Recommended `availability` — make it scale-dependent

| Cluster size | `availability` | Rationale | Confidence |
|---|---|---|---|
| ≤ 256 GPUs | **0.95** | MTBF ≫ run length; checkpoint cost trivial | medium |
| 256 – 2,048 | **0.93** | MTBF ~8 h; straggler exposure begins | medium |
| 2,048 – 32,768 | **0.90** | Llama 3's measured >90% at 16K | **high** |
| > 32,768 | **0.85** | failure every ≲30 min; 10.4% straggler loss compounds | medium-low |

The current flat 0.90 is right for the 10³–10⁴ band and is the single best-sourced default in the pane. It is optimistic above ~32K GPUs — exactly where the existing `requiredGpus > 10000` hyperscale warning (`calculator.ts:57-61`) already fires, so the warning and the number should be wired together.

### 6.5 Recommended `overheadFactor`

Back-solve from Llama 3.1 405B: end-to-end 34.5% ÷ instantaneous 41% = **0.842 = availability / overheadFactor**. With `availability = 0.90`, `overheadFactor = 1.069`.

So the empirically calibrated value is **1.07**, not 1.10. The repo's 1.10 is 3% conservative — immaterial, and conservatism in a planning tool is the right direction. **Keep 1.10**, but document that it must cover only non-model-FLOP wall-clock (evals, data stalls, rollbacks) and *not* activation recomputation (§2.4), which measured MFU already excludes.

---

## 7. Parallelism overhead

### 7.1 Pipeline bubble

```
pipeline_efficiency       = MB / (MB + PP − 1)              (naive 1F1B)
interleaved (v stages/GPU) = v·MB / (v·MB + PP − 1)
```

Llama 3.1 405B at PP=16 with 32 microbatches per DP rank would suffer a 31.9% naive bubble; the paper describes a modified interleaved schedule precisely to avoid it. Published bubble ranges: **15–30% typical, >60% pathological**.

### 7.2 Measured scaling efficiency

| Scale | Efficiency vs ideal | Evidence |
|---|---|---|
| ≤ 8 GPUs (1 node, NVLink) | 1.00 | TP stays on NVLink |
| 8 – 512 | ~0.97 | Megatron PTD-P |
| 512 – 4,096 | ~0.93 | Megatron 52% of peak at 3,072 A100s |
| 4,096 – 16,384 | **0.953 per 2×** | Llama 3: 43% @8,192 → 41% @16,384 |
| > 16,384 | ~0.80–0.88 | extrapolation; mark `low` |

Model as `efficiency = 0.95^log2(GPUs/512)` for GPUs > 512, which reproduces the Llama 3 step exactly and is conservative above it.

### 7.3 Per-dimension notes

- **TP** must stay inside a NVLink domain; crossing to InfiniBand all-reduce is the dominant cliff. This is why GB200 NVL72's 72-GPU / 130 TB/s domain deserves a higher `defaultMfu` (§4.3).
- **SP** (sequence parallelism) adds no communication when combined with TP — free memory savings.
- **CP** (context parallelism) is what makes 128K pretraining possible at all; Llama 3 paid 41% → 38% MFU for CP=16, *plus* the `1 + s/12d` FLOP increase (§2.2).
- **EP** (expert parallelism) introduces all-to-all; see §3.4.
- **Nonuniform TP** (CMU/PDL 2026) keeps throughput loss proportional to failed-GPU fraction instead of 10–100× that, which is the direction the 2026 stacks are moving.

---

## 8. Memory feasibility for pretraining

### 8.1 Bytes per parameter, and what the aggregate floor actually means

| Configuration | Bytes/param (aggregate) | Breakdown |
|---|---|---|
| AdamW mixed precision (standard) | **16** | 2 BF16 w + 2 BF16 g + 4 FP32 master + 4 m + 4 v |
| AdamW, no separate FP32 master | 12 | 2 + 2 + 8 |
| 8-bit Adam | 8 | 2 + 2 + 4 |
| Adafactor | ~6–8 | factored second moments |
| + EMA / model-soup copy | +2 to +4 | some labs keep one |

The repo's `memoryBytesPerParameter: 16` (`scenario-store.ts:127`) is the correct standard value.

**The sharding subtlety:** ZeRO-3 / FSDP give per-GPU `16Ψ/N_d`, so the *aggregate* requirement is exactly `16N` and `hardware-estimator.ts:36-39` is a valid lower bound. Megatron's distributed optimizer is different: params and grads are sharded over TP×PP but **replicated over DP**, so the aggregate is roughly `DP·(2N + 2N) + 12N` — it *grows* with data parallelism. So `16N` is a true floor only under full sharding. Document it as a floor, not an estimate.

### 8.2 Real minimum-GPU floor

```
G_memory = ceil( (b_state·N_total + M_activations) / (usable_fraction · HBM_per_gpu) )
M_activations ≈ 2 · tokens_per_step · d · L        (checkpointed boundary stash)
G_min = max( G_memory, TP·PP )                      and TP·PP must divide node size
```

Worked, Llama 3.1 405B (d=16,384, L=126, 16M tokens/step):
- state: 16 × 405e9 = **6.48 TB** → 96 H100s
- activation stash: 2 × 16e6 × 16,384 × 126 = **66 TB** → 971 H100s
- total 72.5 TB of the cluster's 1.31 PB = **5.5% of HBM**

So at frontier batch sizes **activations are 10× the optimizer state in aggregate** — yet both are ~2 orders of magnitude below the compute-driven GPU count. The 0.85 usable fraction (`hardware-estimator.ts:36`) is reasonable for state (CUDA context ~0.5–1 GB, comm buffers, fragmentation); it does not reserve for activations, but since memory never binds, this is a documentation issue rather than a correctness one.

Minor: `gpu.memoryGb * 1e9` treats "80 GB" as 80×10⁹ bytes. H100's 80 GB is 80 GiB ≈ 85.9×10⁹, so the formula is ~7% conservative. Harmless; worth a comment.

---

## 9. Discrepancies with current implementation

The highest-value section. Ordered by magnitude of impact.

| Quantity | Repo's current value | Researched current value | Impact | Recommended action |
|---|---|---|---|---|
| **Llama 2 7B `estimatedFlops`** | `8.4e19` — `reference-models.ts:32` | `6×7e9×2e12 = 8.4e22` | **1,000× too small.** Breaks `findNearestReferences` ordering and every comparison UI | **Fix to `8.4e22`** |
| **Llama 2 13B `estimatedFlops`** | `1.56e20` — `reference-models.ts:39` | `6×13e9×2e12 = 1.56e23` | **1,000× too small**, same blast radius | **Fix to `1.56e23`** |
| **MoE unsupported** | single `modelParameters` — `types.ts:74-82` | FLOPs ∝ `N_active`, memory ∝ `N_total`; 2026 sparsity is 3–5% | Entering a 2026 MoE's total params overstates FLOPs **20–34×** | Add `activeParameters` to `LlmConfig`; `6·N_active·D` |
| **FP8 uses the BF16 MFU** | `hardware-estimator.ts:25` + `:57-61` | FP8 wall-clock gain is 1.3–1.5×, not 2×; MFU_FP8 ≈ 0.70 × MFU_BF16 | **Overstates FP8 throughput 43%**; predicts 1.41M GPU-h for DeepSeek-V3 vs 2.66M actual (−47%) | Add `precisionFactor` {BF16 1.0, FP8 0.70, FP4 0.50} |
| **`tokensPerParameter: 20`** | `scenario-store.ts:119` | No 2025–26 production run uses 20; dense median ≈150–250 | **Understates D, FLOPs, GPUs, cost by 7.5×** | Default **150**; presets 20 / 26 / 150 / 400 / 1000 |
| **`architectureFactor: 6` flat** | `scenario-store.ts:35,65`; `llm-adapter.ts:20` | `6·(1 + s/(12d))` exactly | **−17% at 8B/8K, −267% at 8B/128K** | Add derived long-context correction from `sequenceLength`, `hiddenDim` |
| **Marin 8B tokens** | `1.6e12` — `reference-models.ts:73` | **12.7T** (deeper-starling) | 7.9× low; `estimatedFlops` wrong to match | Fix to `12.7e12` / `6.096e23` |
| **Marin 32B tokens** | `3.2e12` — `reference-models.ts:81` | **6.437T** | 2× low | Fix to `6.437e12` / `1.236e24` |
| **`6` hardcoded in 3 modules** | `calibration.ts:33`, `reverse-solve.ts:13,60`, `IsoFlopExplorer.tsx:36` | should read `scenario.architectureFactor` | Calibration and reverse-solve silently disagree with the adapter whenever the user edits the factor | Thread `architectureFactor` through all four call sites |
| **No FP4 precision** | `Precision` = BF16/FP8 — `types.ts:1` | NVFP4 pretraining validated at 12B / 10T tokens; MLPerf 405B records use it | Can't model Blackwell's actual advantage (B300 BF16 = B200 BF16) | Add `'FP4_DENSE'` + `fp4DenseFlops` |
| **`GPU_DATA_SPEC_DATE`** | `'2026-06-18'` — `gpu-data.ts:65` | 4 months stale | B300, GB300, MI355X, TPU v7, Trn3 all GA and absent | Bump to `'2026-10-14'`; add 11 SKUs (§4.2) |
| **Only 4 NVIDIA SKUs** | `gpu-data.ts:3-43` | AMD MI300X/325X/355X, TPU v6e/v7, Trn2/3, Gaudi 3 all shipping | No non-NVIDIA path at all | Add rows from §4.2 |
| **H200 `defaultMfu: 0.45`** | `gpu-data.ts:21` | Unsourced; best measured Hopper pretraining MFU is 43% (on H100) | Overstates H200 throughput ~7% | **0.42**, confidence medium-low |
| **B200 `memoryGb: 175`** | `gpu-data.ts:29` | **180 GB** (HGX B200 1,440/8) | Memory floor 2.8% conservative (harmless direction) | 180 |
| **B200 missing bandwidth** | no `bandwidthTbps` — `gpu-data.ts:24-32` | **7.7 TB/s** | Display gap | Add 7.7 |
| **GB200 `defaultMfu: 0.40`** | `gpu-data.ts:40` | 72-GPU / 130 TB/s NVLink domain keeps TP on-fabric | ~5% understated | 0.42, confidence medium |
| **`availability: 0.90` flat** | `scenario-store.ts:124` | Llama 3 measured >90% @16K; failure every 30 min @100K | Optimistic above ~32K GPUs — precisely where the hyperscale warning fires | Scale-dependent 0.95/0.93/0.90/0.85 |
| **`overheadFactor: 1.10`** | `scenario-store.ts:125` | Back-solved **1.07** from Llama 3.1 405B | 3% conservative | Keep 1.10; document that it excludes recompute (§2.4) |
| **`ReferenceModel` has no GPU-hours** | `reference-models.ts:1-7` | GPU-hours are the directly comparable regression target | Can't validate the engine against real runs | Add `gpuHours`, `gpuType`, `precision`, `hoursConvention` |
| **No DeepSeek/Qwen/OLMo/GLM references** | `reference-models.ts:13-84` | §11 table | Reference list is 2024-vintage and all-dense | Add the 2025–26 rows, with `activeParameters` |
| **Llama 3 405B tokens `15e12`** | `reference-models.ts:66` | **15.6T** → 3.79e25 (Meta's own 3.8e25) | 4% low | Fix to `15.6e12` / `3.79e25` |
| **Cloud pricing stale** | `lastUpdated: "2026-06-01"` — `src/data/cloud-gpu-pricing.json:2` | Oct 2026 medians: H100 SXM $4.20, H200 $4.62, B200 $8.19, B300 $8.99, GB200 $16.00, MI355X $5.99 | Cost outputs off by up to 2× | Refresh; add B300/GB200/MI355X/TPU rows |
| **`mfu > 0.60` warning** | `calculator.ts:63` | PaLM's 46.2% MFU / 57.8% HFU is the public ceiling | Threshold is fine; message says "30-50%" which matches the data | Keep as-is |
| **`memoryLowerBoundGpus` prominence** | `calculator.ts:73-77` | Memory is 474× less binding than compute in pretraining | Warning is dead code in practice | Demote in UI; keep the check |
| **`usableMemory` 0.85, `×1e9`** | `hardware-estimator.ts:36` | 0.85 reasonable; `1e9` vs GiB makes it ~7% more conservative | Negligible | Add a comment |

### 9.1 Engine accuracy as it stands today

Predicted GPU-hours at current defaults (effective = 0.400 × 0.90 / 1.10 = **0.327** of peak), vs sourced actuals:

| Run | Predicted | Actual | Error |
|---|---|---|---|
| Llama 3.1 405B | 32.5M | 30.84M | **+5.4%** ✅ |
| DeepSeek-V3 (if forced BF16 denominator) | 2.82M | 2.66M | **+5.8%** ✅ |
| Llama 2 34B | 1.11M | 1.04M | +6.9% ✅ |
| OLMo 3 32B | 0.91M | 1.05M | −13.7% |
| Llama 2 13B | 0.42M | 0.37M | +15.1% |
| Llama 3.1 70B | 5.62M | 7.00M | −19.7% |
| Llama 2 7B | 0.23M | 0.18M | +24.0% |
| Llama 2 70B | 2.29M | 1.72M | +32.8% |
| GPT-3 175B | 2.14M | 3.55M | −39.8% |
| **DeepSeek-V3 (FP8, as the engine actually computes it)** | 1.41M | 2.66M | **−47.1%** ❌ |
| Falcon 180B | 10.3M | 7.0M | +46.9% ⚠ |
| **Llama 3.1 8B** | 0.64M | 1.46M | **−56.0%** ❌ |

Read: **the engine is accurate to ±7% on large dense BF16 runs — exactly the case it was designed for — and degrades to −47% on FP8 and −56% on small models.** Those two cells are the work.

---

## 10. Back-solved recommended defaults

| Parameter | Current | Recommended | Justification | Confidence |
|---|---|---|---|---|
| `architectureFactor` | 6 | **6**, plus derived `× (1 + s/(12d))` | Exact accounting (§2.1-2.2); reproduces de Vries' published 128K figures | **high** |
| `tokensPerParameter` | 20 | **150** (presets 20/26/150/400/1000) | Median of Llama 3 70B (223), OLMo 3 (172), Marin 32B (201), Qwen3 total (153) | medium |
| `mfuByGpuId` H100 BF16 | 0.40 | **0.40** | Llama 3 41–43%, Megatron 38–52%, PaLM 46.2% | **high** |
| `mfuByGpuId` H200 BF16 | 0.45 | **0.42** | No measured Hopper figure exceeds 43%; 0.45 unsourced | medium-low |
| `mfuByGpuId` B200 BF16 | 0.40 | **0.39** | MLPerf v5.0: 2.22× Hopper where peak ratio is 2.27× → 0.978 × H100 MFU | medium |
| `mfuByGpuId` B300 BF16 | — | **0.40** | BF16 silicon identical to B200; advantage is FP4 + 288 GB | medium |
| `mfuByGpuId` GB200/GB300 BF16 | 0.40 | **0.42 / 0.41** | 72-GPU, 130 TB/s NVLink domain keeps TP on-fabric | medium |
| `mfuByGpuId` MI300X/325X/355X | — | **0.30 / 0.31 / 0.33** | ROCm maturity gap; no published MoE/dense pretraining MFU at scale | **low** |
| `mfuByGpuId` TPU v6e / v7 | — | **0.45 / 0.40** | PaLM hit 46.2% on v4; Ironwood's ratio is inference-tilted | medium-low / low |
| `mfuByGpuId` Trn2 / Trn3 / Gaudi 3 | — | **0.35 / 0.35 / 0.30** | No published pretraining MFU found for any of them | **low** |
| `precisionFactor` (new) | implicit 1.0 | **BF16 1.00, FP8 0.70, FP4 0.50** | 1.3–1.5× measured FP8 speedup ÷ 2× peak; FP4 extrapolated one further step | medium (FP8) / low (FP4) |
| `moeEfficiencyFactor` (new) | — | **0.85** | All-to-all + small per-expert GEMMs; MegaScale-MoE's 1.88× shows the spread between naive and tuned | medium-low |
| `availability` | 0.90 flat | **0.95 / 0.93 / 0.90 / 0.85** by cluster size | Llama 3 measured >90% at 16K with 466 interruptions in 54 days; Epoch's 30-min MTBF at 100K | **high** at 10³–10⁴, medium-low above |
| `overheadFactor` | 1.10 | **1.10** (back-solved 1.07) | 34.5% end-to-end ÷ 41% instantaneous = 0.842 = avail/overhead | medium |
| `memoryBytesPerParameter` | 16 | **16** | Standard AdamW mixed-precision; ZeRO-3 aggregate is exactly 16N | **high** |
| usable-HBM fraction | 0.85 | **0.85** | Reasonable for state; memory never binds anyway | medium |
| `scalingEfficiency` (new) | — | **`0.95^log2(GPUs/512)`** above 512 | Reproduces Llama 3's 8,192→16,384 step (95.3%) exactly | medium |

### 10.1 The degeneracy problem

`requiredGpus ∝ overheadFactor / (mfu × availability)`. The three are one parameter. Options:

1. **Keep three sliders, add a derived read-out** "effective utilization = 32.7% of dense peak" with the three anchors annotated. Cheapest, and makes the degeneracy visible rather than hidden.
2. Collapse to one "effective utilization" slider with the three as an expandable breakdown.
3. Keep three but make `availability` read-only/derived from the computed GPU count (§6.4) so the user only tunes two.

**Recommendation: (1) plus (3)** — derive `availability` from cluster size, leaving the user with `mfu` (a hardware/software property they may have measured) and `overheadFactor` (an operational property). `AssumptionChips` already has the right shape for displaying the derived effective number.

---

## 11. Regression-target table (test fixtures)

Usable directly as test fixtures. `6ND` computed from the sourced `N` and `D`; `MFU_eff` is the back-solved end-to-end fraction of dense peak that the engine must reproduce.

| # | Run | N_total | N_active | D | 6N_act·D | GPU | GPU-hours | MFU_eff | Hours convention | Confidence |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | GPT-3 175B | 175e9 | — | 300e9 | 3.150e23 | V100 | 3.552e6 | 0.197 | derived (10k × 14.8 d) | medium |
| 2 | Chinchilla 70B | 70e9 | — | 1.4e12 | 5.880e23 | TPU v4 | — | — | FLOPs only (Epoch: 5.8e23) | high (FLOPs) |
| 3 | Llama 2 7B | 7e9 | — | 2.0e12 | 8.400e22 | A100 80GB | 1.843e5 | 0.406 | productive-only | high |
| 4 | Llama 2 13B | 13e9 | — | 2.0e12 | 1.560e23 | A100 | 3.686e5 | 0.377 | productive-only | high |
| 5 | Llama 2 34B | 34e9 | — | 2.0e12 | 4.080e23 | A100 | 1.038e6 | 0.350 | productive-only | high |
| 6 | Llama 2 70B | 70e9 | — | 2.0e12 | 8.400e23 | A100 | 1.720e6 | 0.435 | productive-only | high |
| 7 | Llama 3.1 8B | 8e9 | — | 15.6e12 | 7.488e23 | H100 | 1.46e6 | **0.144** | all-inclusive | high |
| 8 | Llama 3.1 70B | 70e9 | — | 15.6e12 | 6.552e24 | H100 | 7.00e6 | 0.263 | all-inclusive | high |
| 9 | **Llama 3.1 405B** | 405e9 | — | 15.6e12 | 3.791e25 | H100 | 30.84e6 | **0.345** | all-inclusive | **high** — primary anchor |
| 10 | DeepSeek-V3 (pretrain) | 671e9 | 37e9 | 14.8e12 | 3.286e24 | H800, FP8 | 2.664e6 | 0.173 of FP8 peak | pretrain-only | high |
| 11 | DeepSeek-V3 (all stages) | 671e9 | 37e9 | 14.8e12 | — | H800, FP8 | 2.788e6 | — | incl. ctx-ext + post | high |
| 12 | OLMo 3 32B | 32e9 | — | 5.5e12 | 1.056e24 | H100 (1,024) | 1.05e6 | 0.282 | all-inclusive | high |
| 13 | OLMo 3.1 32B | 32e9 | — | 5.5e12 | 1.056e24 | H100 (1,024) | 1.376e6 | 0.215 | all-inclusive | medium |
| 14 | Falcon 180B | 180e9 | — | 3.5e12 | 3.780e24 | A100 40GB (4,096) | ~7.0e6 | 0.481 ⚠ | rounded figure | **low** |
| 15 | PaLM 540B | 540e9 | — | 780e9 | 2.527e24 | TPU v4 (6,144) | — | 0.462 MFU reported | reported MFU | high (MFU) |
| 16 | Marin 8B | 8e9 | — | 12.7e12 | 6.096e23 | TPU v4/v5e | — | — | FLOPs only | high (N, D) |
| 17 | Marin 32B | 32e9 | — | 6.437e12 | 1.236e24 | TPU v5p/v4 | — | — | FLOPs only | high (N, D) |
| 18 | Qwen3-235B-A22B | 235e9 | 22e9 | 36e12 | 4.752e24 | undisclosed | — | — | FLOPs only | medium |
| 19 | GLM-5 | 744e9 | ~40e9 | 28.5e12 | 6.840e24 | undisclosed | — | — | FLOPs only | medium-low |
| 20 | Kimi K3 | 2.8e12 | 104e9 | undisclosed | — | undisclosed | — | — | params only | medium-low |

**Suggested test assertions:**
- Row 9 at defaults (TPP override = 15.6e12, MFU 0.40, avail 0.90, overhead 1.10) → GPU-hours within **±10%** of 30.84M. *This is the single most important regression test in the engine.*
- Row 10 with `precision = FP8_DENSE`, `activeParameters = 37e9`, `precisionFactor = 0.70`, `moeFactor = 0.85` → within **±25%** of 2.664M. (Without those factors it is −47%.)
- Rows 3–6 (Llama 2 family) back-solve to a tight 0.35–0.44 band: assert the engine's effective utilization lands inside it for A100-class inputs.
- Row 7 is the deliberate failure case: assert the "low tokens/GPU/step" warning fires.
- Rows 1, 2, 9, 15: FLOP arithmetic only — assert `6ND` to 3 significant figures.

---

## 12. Cost and energy

### 12.1 October 2026 market rates ($/GPU-hour)

| SKU | Spot / floor | Median on-demand | Hyperscaler on-demand | Index (Silicon Data, Oct 10) |
|---|---|---|---|---|
| H100 PCIe | — | $2.89 | — | — |
| **H100 SXM** | $1.29 (Azure spot) | **$3.60–4.20** | ~$12.29 (AWS p5 ÷ 8) | — |
| **H200 SXM** | $2.20 | **$4.62** | — | — |
| **B200** | $3.40–3.95 | **$8.19** | — | **$5.93** |
| **B300** | $3.80 | **$8.99** (+5% / 90 d) | — | **$6.71** |
| **GB200** | $10.50 | **$16.00** | up to $27.04 | — |
| **MI355X** | $2.95 | **$5.99** | — | — |
| **TPU v7 Ironwood** | $6.00 (flex-start) | **$12.00/chip-h** (Iowa), $13.20 (London) | — | ~$1.60 at negotiated scale (SemiAnalysis est.) |

The repo's `cloud-gpu-pricing.json` carries AWS p5.48xlarge at $98.32/hr ÷ 8 = $12.29/GPU-h against a $3.60–4.20 market median — a 3× spread. Both are real; the file should carry a `tier` dimension (hyperscaler on-demand / neocloud on-demand / reserved / spot) rather than one number per instance, because for a multi-million-GPU-hour pretraining run only the reserved/committed tier is realistic.

### 12.2 Energy anchors

| Quantity | Value | Source |
|---|---|---|
| H100 measured average power while pretraining a 32B | **649 W** (93% of 700 W TDP) | OLMo 3 |
| → energy per H100-hour, GPU only | **0.649 kWh** | derived |
| OLMo 3 32B total GPU energy | **681 MWh** over 1.05M GPU-h | OLMo 3 |
| Llama 3.1 405B GPU-only energy (derived at 0.649) | **20.0 GWh** | derived |
| Llama 3.1 405B location-based emissions | **8,930 tCO₂e** | Llama 3.1 model card |
| → implied grid intensity | **0.446 kgCO₂e/kWh** | derived — plausible for a US grid, so the two figures are mutually consistent |
| Llama 3.1 8B / 70B emissions | 420 / 2,040 tCO₂e | model card |
| Llama 3.1 all three models | 39.3M GPU-h, 11,390 tCO₂e location-based, **0 market-based** | model card |

**Recommended energy model:** `kWh = GPU-hours × 0.65 × node_factor(1.4) × PUE(1.15)` ≈ **1.05 kWh per GPU-hour** at the datacenter meter. Emissions: `× 0.45 kgCO₂e/kWh` default, user-overridable (Meta's market-based figure is 0 because of renewable matching — worth showing both, as the model card does).

### 12.3 Headline cost sanity check

Llama 3.1 405B at 30.84M H100-hours:
- **$77M** at $2.50/h (large reserved commitment)
- **$130M** at $4.20/h (neocloud on-demand median)
- **$379M** at $12.29/h (AWS p5 on-demand)

Widely reported external estimates for the 405B run cluster around $60–100M, which brackets the reserved figure. **The pricing tier matters more than any modeling parameter in this document** — a 5× cost spread dwarfs even the FP8 error. Make the tier a first-class, visible input.

---

## 13. Confidence ratings

Reusing the repo's `ConfidenceLevel` scale (`src/engine/types.ts:54`).

| Quantity | Confidence | Why |
|---|---|---|
| `6ND` arithmetic for dense BF16 at ≤8K context | **high** | exact accounting; validated to ±5.4% against Llama 3.1 405B GPU-hours |
| Long-context correction `1 + s/(12d)` | **high** | exact; reproduces de Vries' published percentages |
| Vocab/logits correction `dV/N` | medium | exact in form, but depends on tied-vs-untied, which users won't know |
| H100 BF16 MFU 0.40 | **high** | four independent sources in 0.38–0.52 |
| B200 / GB200 BF16 MFU | medium | one MLPerf-derived data point each |
| H200 BF16 MFU | medium-low | no published pretraining MFU exists for H200 |
| AMD / TPU / Trainium / Gaudi MFU | **low** | no published pretraining MFU found for any of them; stated values are reasoned, not measured |
| FP8 precision factor 0.70 | medium | 1.3–1.5× speedup is multiply-sourced; the one FP8 run (DeepSeek-V3) is MoE-confounded |
| FP4 precision factor 0.50 | **low** | one 12B/10T validation run; no public MFU figure |
| MoE `6·N_active·D` | **high** | universally agreed form |
| MoE efficiency factor 0.85 | medium-low | MegaScale-MoE's 1.88× shows how wide the naive-vs-tuned spread is |
| `availability` 0.90 at 10³–10⁴ GPUs | **high** | Llama 3's measured >90% with full interruption statistics |
| `availability` above 32K GPUs | medium-low | extrapolated from Epoch's MTBF projections, no measured run |
| `overheadFactor` 1.07–1.10 | medium | one clean back-solve; depends on a GPU-hour convention Meta didn't document |
| Scaling efficiency `0.95^log2(GPUs/512)` | medium | anchored on one measured 2× step |
| TPP = 150 default | medium | a median over 6 heterogeneous runs, not a law |
| Memory floor `16N / (0.85·HBM)` | **high** as a floor | exact under ZeRO-3; not an estimate under Megatron distributed optimizer |
| GPU-hours as regression targets | medium | the convention inconsistency between Llama 2 and Llama 3.1 is unexplained |
| Cloud $/GPU-h | medium | real-time market; 3–5× spread across tiers |
| Energy 0.65 kWh/H100-h | **high** | directly measured and cross-validated against Llama 3's CO₂e |

---

## 14. Warnings worth emitting

Beyond the existing four in `calculator.ts:52-77`:

- **MoE detected (`N_active < N_total`)** → "compute follows 37B active params; aggregate memory follows 671B total. Sparsity 5.5%."
- **`N_total` entered where `N_active` was meant** → heuristic: if `N > 100e9` and the model name matches a known MoE, offer the active count.
- **Long context** → "at s=131,072 and d=4,096 the attention term adds **267%** to 6ND. Effective architecture factor 22.0, not 6."
- **FP8 selected** → "FP8 peak is 2× BF16 but measured end-to-end speedup is 1.3–1.5×. This estimate applies a 0.70 MFU factor."
- **FP4 selected** → "NVFP4 pretraining is validated only to 12B params / 10T tokens and requires Hadamard transforms, stochastic rounding and selective high-precision layers. Treat as experimental." (Replaces the current blanket FP8-is-experimental warning at `llm-adapter.ts:58-60`, which is now out of date — FP8 pretraining is production-standard in 2026.)
- **B300 / GB300 selected with BF16** → "B300's BF16 peak is identical to B200. Its gains are 1.5× NVFP4 and 288 GB HBM. Select FP4 or a memory-bound workload to see a benefit."
- **TPP < 20** → "below compute-optimal; the model is under-trained for its size."
- **TPP > 1,000** → "extreme over-training. Justified only by very high inference volume (Sardana et al.); returns are strongly diminishing."
- **`D > 4 × uniqueTokensAvailable`** → "exceeds the ~4-epoch repetition budget; beyond that the marginal value of compute decays toward zero (Muennighoff et al.), and the 4-epoch allowance itself shrinks at higher N and TPP."
- **`requiredGpus > 32,768`** → "availability below 0.85 is likely: failure every ~30 min at 100K GPUs; 10.4% of GPU-hours lost to stragglers in production clusters."
- **tokens/GPU/step < 2,000** → "small per-GPU microbatch. Llama 3.1's 8B run back-solves to only 14.4% of peak, likely for this reason. Consider fewer GPUs over a longer window."
- **Non-NVIDIA SKU selected** → "MFU default is reasoned, not measured. No public pretraining MFU figure exists for this accelerator — calibrate against your own run."
- **Cost shown** → "the pricing tier (hyperscaler on-demand vs reserved) spans 5× and dominates every modeling parameter here."

---

## 15. Open design questions

1. **Collapse the three degenerate sliders?** `mfu × availability / overheadFactor` is one number. Recommendation in §10.1: derive `availability` from cluster size, keep `mfu` and `overheadFactor` user-facing, and display the effective fraction prominently. This is the highest-leverage UX change in the pane.
2. **Where does the long-context correction live?** As a multiplier on `architectureFactor` (keeps the trace honest, makes the number look "wrong" to users who know 6ND), or as a separate line item in `FormulaTraceDrawer`? Recommendation: separate trace line, `architectureFactor` stays 6.
3. **How to ask for MoE parameters without confusing dense users?** A single `modelParameters` field with an "is MoE" toggle that reveals `activeParameters`, defaulting `N_active = N_total`? Or a model picker that fills both from a catalog (the fine-tuning research already calls for that catalog — §5 of `2026-10-14-finetuning-calculator-research.md`)? Shared catalog is the obvious reuse.
4. **Does `memoryLowerBoundGpus` earn its UI space?** It is 474× from binding. Options: delete the display and keep the check; or repurpose the panel to show the *parallelism* floor (`TP×PP` must divide node size, `G_min ≥ TP×PP`), which is the constraint that actually bites.
5. **Should TPP default change at all?** Changing 20 → 150 moves every output 7.5×. Shipping it as an explicit, labeled preset selection ("Chinchilla compute-optimal" vs "Frontier dense 2026") rather than a silent default change is probably the right call — but the *initial* selection still has to be one of them.
6. **GPU-hours as the primary output?** The pane currently outputs `requiredGpus` for a deadline. GPU-hours is what every published run reports, what cloud bills are denominated in, and what the regression table in §11 contains. Adding it as a co-equal headline output would make the whole pane directly checkable against reality.
7. **Precision as a per-SKU capability rather than a global enum?** TPU v6e has no native FP8; MI325X has no FP4; Trainium3's low-precision format is MXFP8/MXFP4 not NVFP4. The current global `Precision` enum will silently produce numbers for impossible combinations.
8. **Carry the Vera Rubin row?** It ships late 2026 and users will ask. Per §4.2 the per-GPU dense BF16 isn't published and sources conflict on the FP4 figure. Recommendation: a disabled row with a "datasheet pending" note rather than invented numbers.

---

## Sources

**Scaling laws**
- [Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla)](https://arxiv.org/abs/2203.15556)
- [Epoch AI, *Chinchilla Scaling: A Replication Attempt*](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt) — the 25.6 TPP correction and the 70 TPP parametric reading
- [Besiroglu et al., *Chinchilla Scaling: A replication attempt* (arXiv HTML)](https://arxiv.org/html/2404.10102v1)
- [Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in LM Scaling Laws*](https://arxiv.org/html/2401.00448v3)
- [*Language Models Scale Reliably With Over-Training* (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/a91869936a63d814971b6423990ecf6e-Paper-Conference.pdf)
- [*Test-Time Scaling Makes Overtraining Compute-Optimal* (arXiv 2604.01411)](https://arxiv.org/pdf/2604.01411)
- [*Tokens-per-Parameter Coverage Is Critical for Robust LLM Scaling Law Extrapolation* (arXiv 2605.08541)](https://arxiv.org/pdf/2605.08541)
- [Kumar et al., *Scaling Laws for Precision* (ICLR 2025)](https://pehlevan.seas.harvard.edu/sites/g/files/omnuum6471/files/2025-03/Kumar_etal_ICLR_2025.pdf)
- [Muennighoff et al., *Scaling Data-Constrained Language Models* (JMLR 2025)](https://www.jmlr.org/papers/v26/24-1000.html) — the 4-epoch rule
- [*Prescriptive Scaling Laws for Data Constrained Training* (arXiv 2605.01640)](https://arxiv.org/pdf/2605.01640) — the 4-epoch rule holds only near 1× Chinchilla at ~3B
- [*Bridging Compute- and Data-Optimal Pretraining* (arXiv 2607.25271)](https://arxiv.org/html/2607.25271v1)
- [zeroentropy, *Scaling laws: compute, parameters, tokens, and LLM loss*](https://zeroentropy.dev/concepts/scaling-laws/) (aggregator — used for the lifetime-compute framing and the LFM2.5 80,000:1 datapoint)

**FLOP accounting**
- [Harm de Vries, *In the long (context) run*](https://www.harmdevries.com/post/context-length/) — the `6d(L+1)` attention term and the 8% / 16-33% / 260% context-overhead figures
- [Adam Casson, *Transformer FLOPs*](https://www.adamcasson.com/posts/transformer-flops) — Kaplan vs Chinchilla accounting, embeddings/logits terms, the <3% GPT-3 check
- [Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models* (MLSys 2023)](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)

**Real runs and MFU**
- [*The Llama 3 Herd of Models* (arXiv HTML)](https://arxiv.org/html/2407.21783v3) — Table 4 parallelism/MFU, 3.8e25 FLOPs, 15.6T tokens, 466 interruptions, >90% effective time
- [Llama 3.1 model card](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md) — 1.46M / 7.0M / 30.84M GPU-hours, 11,390 tCO₂e
- [Llama 3 model card](https://github.com/meta-llama/llama3/blob/main/MODEL_CARD.md)
- [Touvron et al., *Llama 2*](https://arxiv.org/pdf/2307.09288) — the 7B/13B/34B/70B GPU-hour table
- [Chowdhery et al., *PaLM*](https://arxiv.org/pdf/2204.02311) — 46.2% MFU / 57.8% HFU
- [Narayanan et al., *Efficient Large-Scale LM Training on GPU Clusters Using Megatron-LM* (SC21)](https://arxiv.org/pdf/2104.04473) — PTD-P, 52% of peak at 3,072 GPUs
- [Smith et al., *Megatron-Turing NLG 530B*](https://arxiv.org/pdf/2201.11990)
- [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437) — 2.664M/2.788M H800-hours, 14.8T tokens, 180K GPU-h per trillion tokens
- [*What is the MFU for DeepSeek-V3 training?*](https://medium.com/@dlrover/what-is-the-mfu-for-deepseek-v3-training-0d9ea4d42eb4) (aggregator — the 20.2% / 1,700 TF-available derivation)
- [*Olmo 3* (arXiv 2512.13961)](https://arxiv.org/pdf/2512.13961)
- [*Minipost: Olmo 3 training cost*](https://muxup.com/2025q4/minipost-olmo3-training-cost) — 1.05M H100-h, 649 W measured, 681 MWh
- [*The Falcon Series of Open Language Models*](https://arxiv.org/pdf/2311.16867) — 3.5T tokens, 43,500 PF-days
- [*Qwen3 Technical Report*](https://cdn.jsdelivr.net/gh/yanfeng98/paper-is-all-you-need/papers/00069-Qwen3_Technical_Report.pdf) — 36T tokens
- [Marin 8B Retro report](https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/) — 12.7T tokens
- [Marin 32B Retro report](https://github.com/marin-community/marin/blob/main/docs/reports/marin-32b-retro.md) — 6.437T tokens
- [Google OSS blog, *Training Marin 32B*](https://opensource.googleblog.com/2025/12/training-marin-32b-what-an-open-lab-can-build-with-tpus-jax-and-a-little-persistence.html)
- [Epoch AI, *Estimating training compute of deep learning models*](https://epoch.ai/blog/estimating-training-compute) — methodology, the 0.3 utilization rule of thumb
- [Epoch AI, *Frontier LLM training runs can't get much longer*](https://epoch.ai/data-insights/longest-training-run)

**Precision**
- [NVIDIA, *Pretraining Large Language Models with NVFP4* (arXiv 2509.25149)](https://arxiv.org/pdf/2509.25149) — 12B / 10T tokens, MMLU-Pro 62.58 vs 62.62
- [Megatron-Bridge low-precision discussion #3460](https://github.com/NVIDIA-NeMo/Megatron-Bridge/discussions/3460) — 1.2–1.5× FP8 speedup
- [NVIDIA, *Megatron Bridge Performance Tuning Guide*](https://docs.nvidia.com/nemo/megatron-bridge/latest/performance-guide.html)
- [*MoE Parallel Folding* (arXiv 2504.14960)](https://arxiv.org/pdf/2504.14960) — 1.26×/1.30× FP8 uplift
- [Peng et al., *FP8-LM*](https://arxiv.org/pdf/2310.18313)
- [*A Comprehensive FP8 Training Recipe for Reasoning-Enhanced Language Models* (arXiv 2509.22536)](https://arxiv.org/pdf/2509.22536)
- [*MFU, BF16, FP8 and AI Numeric Formats: The Complete Guide*](https://manishklach.github.io/writings/mfu-bf16-fp8-complete-guide.html) (aggregator — the denominator-mismatch point, 38–52% H100 BF16 band)
- [*Dissecting Outlier Dynamics in LLM NVFP4 Pretraining* (arXiv 2602.02047)](https://arxiv.org/pdf/2602.02047)

**MoE**
- [*Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for MoE LMs* (arXiv 2508.18672)](https://arxiv.org/pdf/2508.18672)
- [*Towards Greater Leverage: Scaling Laws for Efficient MoE LMs* (ICLR 2026)](https://proceedings.iclr.cc/paper_files/paper/2026/file/32b640528f5b67975562210f00c131ed-Paper-Conference.pdf) — efficiency leverage, granularity 12 / 3.1% activation
- [*Hyperparameter Scaling Laws Across MoE Sparsity* (arXiv 2609.08690)](https://arxiv.org/pdf/2609.08690)
- [*Joint MoE Scaling Laws* (arXiv 2502.05172)](https://arxiv.org/pdf/2502.05172)
- [*Compute-Optimal Is Not Cluster-Optimal: Systems-Aware Scaling for Sparse MoE* (arXiv 2608.10605)](https://arxiv.org/html/2608.10605v1)
- [*MegaScale-MoE* (arXiv 2505.11432)](https://arxiv.org/pdf/2505.11432) — 1.88× MFU over Megatron-LM on 352B / 1,440 GPUs
- [NVIDIA Megatron Core MoE docs](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html)
- [*The Llama 4 Herd* (arXiv 2601.11659)](https://arxiv.org/pdf/2601.11659)

**Cluster reliability and goodput**
- [Epoch AI, *Hardware failures won't limit AI scaling*](https://epoch.ai/blog/hardware-failures-wont-limit-ai-scaling) — 50,000 h per-GPU MTBF, 30-min/3-min projections, 2.1% checkpoint loss
- [Lablup, *From Detection to Recovery: Operational Analysis on LLM Pre-training with 504 GPUs* (arXiv 2605.09370)](https://arxiv.org/pdf/2605.09370) — MTTF by scale, restart timings
- [Tom's Hardware, *Faulty H100 GPUs and HBM3 caused half of failures during Llama 3 training*](https://www.tomshardware.com/tech-industry/artificial-intelligence/faulty-nvidia-h100-gpus-and-hbm3-memory-caused-half-of-the-failures-during-llama-3-training-one-failure-every-three-hours-for-metas-16384-gpu-training-cluster) (aggregator — corroborates the paper's interruption table)
- [CoreWeave, *Why Distributed Training Fails at Scale*](https://www.coreweave.com/blog/why-distributed-training-fails-at-scale)
- [Crusoe, *AutoClusters: minimizing GPU cluster failures*](https://www.crusoe.ai/resources/blog/autoclusters-minimizing-hardware-failures-in-large-gpu-clusters)
- [Nebius, *Fault-tolerant training*](https://nebius.com/blog/posts/how-we-build-reliable-clusters)
- [*FFTrainer: Fast Failover in LLM Training* (arXiv 2512.03644)](https://arxiv.org/pdf/2512.03644)
- [*TrainMover: An Interruption-Resilient Runtime for ML Training* (arXiv 2412.12636)](https://arxiv.org/pdf/2412.12636)
- [*Nonuniform Tensor Parallelism* (CMU PDL)](https://www.pdl.cmu.edu/PDL-FTP/BigData/2504.06095v1.pdf)
- [*PipeFill: Using GPUs During Bubbles in Pipeline-Parallel LLM Training* (arXiv 2410.07192)](https://arxiv.org/pdf/2410.07192)
- [*Efficient Training of LLMs on Distributed Infrastructures: A Survey* (arXiv 2407.20018)](https://arxiv.org/pdf/2407.20018)

**Hardware specifications**
- [NVIDIA H100](https://www.nvidia.com/en-us/data-center/h100/) · [NVIDIA H200](https://www.nvidia.com/en-us/data-center/h200/) · [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/) · [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- [Lenovo ThinkSystem HGX B200 180GB product guide](https://lenovopress.lenovo.com/lp2226.pdf) — 180 GB, 1000 W, 7.7 TB/s
- [Lenovo ThinkSystem H200 141GB product guide](https://lenovopress.lenovo.com/lp1944-nvidia-h200-141gb-gpu)
- [SemiAnalysis InferenceX: B300](https://inferencex.semianalysis.com/chips/b300) — 2,250 BF16 / 4,500 FP8 / 13,500 FP4 dense, "FP8 and BF16 stay at B200 levels"
- [SemiAnalysis InferenceX: MI355X](https://inferencex.semianalysis.com/chips/mi355x) · [MI325X](https://inferencex.semianalysis.com/chips/mi325x) · [H200](https://inferencex.semianalysis.com/chips/h200)
- [Oracle, *AMD Instinct MI355X on OCI*](https://blogs.oracle.com/cloud-infrastructure/amd-instinct-mi355x-on-oci-performance-technical-details) — 2.5 PF dense BF16, 288 GB, 8 TB/s, 1400 W
- [Wikipedia, AMD Instinct](https://en.wikipedia.org/wiki/AMD_Instinct) — MI300X 1,307.4 TFLOPS dense BF16
- [ASBIS AMD Instinct MI300 series](https://promo.asbis.com/amd_instinct_mi300)
- [WCCFTech, AMD Instinct MI400 / 432 GB HBM4](https://wccftech.com/amd-instinct-mi400-accelerator-doubles-compute-40-pflops-432-gb-hbm4-memory-2026-launch/) (aggregator — BF16 not published)
- [Google Cloud TPU v6e docs](https://docs.cloud.google.com/tpu/docs/v6e) — 918 TFLOPS BF16, 32 GB, 1,638 GB/s
- [Google Cloud TPU7x (Ironwood) docs](https://docs.cloud.google.com/tpu/docs/tpu7x) — 4,614 FP8 / 2,307 BF16 TFLOPS, 192 GB, 7.37 TB/s
- [AWS Neuron, Trainium2 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium2.html) — 1,299 FP8 / 667 BF16 TFLOPS, 96 GiB, 2.9 TB/s
- [AWS Neuron, Trainium3 architecture](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/trainium3.html) — 2,517 MXFP8 / 671 BF16 TFLOPS, 144 GiB, 4.9 TB/s
- [Intel Gaudi 3, Hot Chips 2024](https://hc2024.hotchips.org/assets/program/conference/day1/60_HC2024.Intel.RomanKaplan.Gaudi3-0826.pdf) — 1,835 BF16/FP8 matrix TFLOPS, 128 GB HBM2e
- [NVIDIA, *Blackwell Delivers up to 2.6× Higher Performance in MLPerf Training v5.0*](https://developer.nvidia.com/blog/nvidia-blackwell-delivers-up-to-2-6x-higher-performance-in-mlperf-training-v5-0/) — 1,960 TFLOP/s/GPU on Llama 405B; 121.09 vs 269.12 min at 512 GPUs
- [MLCommons MLPerf Training](https://mlcommons.org/benchmarks/training/)
- [DCD, *Nvidia announces Vera Rubin Superchip for late 2026*](https://www.datacenterdynamics.com/en/news/nvidia-announces-vera-rubin-superchip-for-late-2026/) — 3.6 EF FP4 / 1.2 EF FP8 per NVL144, 288 GB HBM4

**Pricing**
- [getdeploying, Cloud GPU rental prices (84 providers)](https://getdeploying.com/gpus) · [H100](https://getdeploying.com/gpus/nvidia-h100) · [H200](https://getdeploying.com/gpus/nvidia-h200) · [B300](https://getdeploying.com/gpus/nvidia-b300) · [MI355X](https://getdeploying.com/gpus/amd-mi355x)
- [savrn AI index, GPU prices per hour](https://savrn.com/ai-index/pricing/gpus) — Silicon Data indices, Oct 10 2026
- [getdeploying GPU price trends](https://getdeploying.com/gpu-price-trends)
- [intuitionlabs, Data Center GPU Pricing 2026](https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026) (aggregator)
- [shattered.io, *Google puts $12/hr price on Ironwood TPU*](https://shattered.io/google-ironwood-tpu-pricing-vs-nvidia-2026/) (aggregator — GCP list price, SemiAnalysis negotiated-rate estimate)

**2026 open-weight model landscape**
- [Best Open-Source LLM Models in 2026 (Hugging Face blog)](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Best Open-Source Coding Model 2026: Kimi K3 vs GLM-5.2 vs DeepSeek V4 vs Qwen3](https://www.morphllm.com/best-open-source-coding-model-2026) (aggregator — total/active parameter counts and release dates)
- [Kimi K3 vs DeepSeek V4 Pro vs GLM-5.2 (DeepInfra)](https://deepinfra.com/blog/kimi-k3-vs-deepseek-v4-pro-vs-glm-5-2) (aggregator)
- [Open Source LLM Comparison Table (2026)](https://computingforgeeks.com/open-source-llm-comparison/) (aggregator)
- [Wikipedia, List of large language models](https://en.wikipedia.org/wiki/List_of_large_language_models)

**Energy and emissions**
- [*The Environmental Impacts of Language Model Training Keep Rising* (arXiv 2510.09022)](https://arxiv.org/pdf/2510.09022)
- [*Energy Use of AI Inference, Efficiency Pathways, and Test-Time Scaling* (arXiv 2509.20241)](https://arxiv.org/pdf/2509.20241)
