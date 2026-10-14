# GPU Calculator -- User Guide

## 1. Overview

This calculator has three modes:

- **Training Calculator** -- estimates accelerator requirements for training across four model families (LLM, Time-Series, Tabular Foundation, Classical Tabular / GBDT)
- **Inference Calculator** -- estimates monthly costs for serving an LLM via API (buy) vs self-hosted GPUs (build), and identifies the breakeven point
- **Fine-Tuning Calculator** -- estimates whether a given open-weight model and fine-tuning method fit on a given GPU, how many GPUs the memory floor demands, and what the run costs

**When to use the Training Calculator:** You are planning a pretraining run (or GBDT sweep) and need to estimate GPU count, training time, or cluster size. You know (or can estimate) the model size, training window, and data budget.

**When to use the Inference Calculator:** You have a trained model and want to decide between paying per-token for a managed API or running your own GPU cluster. You know (or can estimate) your request volume.

**When to use the Fine-Tuning Calculator:** You are adapting an existing open-weight model and need to know whether it fits -- which method, which GPU, how many of them. This pane is memory-feasibility-first, where the Training Calculator is deadline-first.

---

## 2. Training Calculator

The Training Calculator estimates GPU cluster requirements for model training. It covers four model families -- from dense transformer pretraining (LLM) to gradient-boosted decision trees (GBDT). For each family it computes total work (FLOPs or work units), divides by sustained hardware throughput, and returns the number of accelerators needed within a given wall-clock window.

**Who it's for:** ML engineers, infrastructure teams, and technical leaders planning pretraining runs, comparing GPU SKUs, or validating vendor estimates.

**Key concept:** GPU count = total compute / (training window x sustained throughput per GPU).

### What it is for

- Planning GPU cluster size for pretraining runs
- Comparing accelerator SKUs (15 built-in, from H100 to MI450, TPU v7, Trainium3 and Gaudi 3) under the same assumptions
- Sensitivity analysis across MFU and training window
- Back-of-envelope validation of vendor or team estimates
- Estimating training time for GBDT hyperparameter sweeps

### What it is NOT for

- Procurement commitments without empirical benchmarking
- MoE, multimodal, or non-transformer architectures (for LLM/TS/tabular modes)
- Cost estimation or pricing (compute cost can be layered on top separately)
- Predicting model quality, loss, or downstream performance
- Detailed distributed-training parallelism planning (TP/PP/CP)

---

### 2.1 Quick Start

1. Select a **model family** tab (LLM, Time-Series, Tabular Foundation, or Classical Tabular).
2. Enter the required inputs for that mode (e.g., model size + training window for LLM).
3. The calculator immediately shows the result using default assumptions.
4. Select additional GPUs to compare, or open Advanced Assumptions to tune.

---

### 2.2 Model Families (LLM, Time-Series, Tabular, Classical)

#### LLM (Dense Transformer)

The original mode. Estimates GPU count for pretraining dense decoder-only transformers using the standard `C = 6ND` FLOP model. Best for: GPT-style, LLaMA-style, and similar dense autoregressive language model pretraining.

**When to use:** You are training a dense language model and know (or can estimate) the parameter count and token budget.

#### Time-Series Foundation Models

Estimates training compute for patch-based transformer models that learn across many time-series datasets. These models tokenize temporal data into patches and process them with transformer architectures.

**When to use:** You are training a foundation model like TimesFM, Chronos, or Moirai on large collections of time-series data.

#### Tabular Foundation Models

Estimates training compute for transformer models pretrained on collections of tabular datasets (e.g., TabPFN, CARTE). Tokenization strategy (row vs cell vs axial) dramatically affects compute.

**When to use:** You are pretraining a transformer on many tabular tasks and want to understand compute scaling under different tokenization approaches.

#### Classical Tabular / GBDT

Estimates training time for gradient-boosted decision tree algorithms (LightGBM, XGBoost, CatBoost, Random Forest). This is NOT a FLOP model -- it uses empirical work units and throughput benchmarks.

**When to use:** You are running a large GBDT hyperparameter sweep and want to estimate wall-clock time.

---

### 2.3 Input Reference

#### LLM Mode

| Input | Description | Default | Valid Range |
|-------|-------------|---------|-------------|
| Model parameters | Number of trainable parameters | 70B | Positive; accepts M/B/T units |
| Training window | Wall-clock time budget | 30 days | Positive; accepts h/d/w units |
| Tokens per parameter (TPP) | Training tokens = TPP x N | 20 | Positive integer; presets 20, 50, 100, 200 |
| Training mode | Type of training run | Full pretraining | Full pretraining, Continued pretraining, SFT, LoRA, RLHF, Distillation |
| Precision | Numerical format | BF16 Dense | BF16 Dense, FP8 Dense (experimental), FP4 Dense (Blackwell / MI355X only) |
| GPU SKU | Accelerator type(s) | H100 SXM | Multi-select across 15 built-in SKUs (NVIDIA, AMD, Google, AWS, Intel) plus custom. See Section 2.9. |
| MFU | Model FLOP utilization per GPU | Per-SKU default, 30-45% | 0-100%; per-GPU override. The effective MFU also carries precision and MoE factors -- see Section 2.4. |
| Availability | Useful wall-clock fraction | 90% | 0-100% |
| Overhead factor | Multiplier for checkpointing, eval, restarts | 1.10x | >= 1.0 |
| Memory bytes/parameter | State memory estimate | 16 | Positive; for memory feasibility check |

#### Unit Shortcuts

- Model size: `7B`, `70B`, `1.5T`, `405M`
- Training window: `30d` (days), `720h` (hours), `4w` (weeks)

#### Quick Estimate

Enter model size and training window. The calculator uses a frontier-dense TPP of 150 by default and shows a single H100 GPU count. This is the minimum-input mode for napkin math.

#### Advanced Estimate

Expand the Advanced Assumptions panel to control:

- **MFU** -- per-GPU utilization (30% conservative, 40% planning, 50% strong)
- **Availability** -- fraction of wall-clock time spent training (accounts for failures, restarts, maintenance)
- **Overhead** -- extra compute from checkpointing, evaluation, data stalls
- **Precision** -- BF16 (standard), FP8 or FP4 (experimental). Lower precisions swap in a higher peak but are penalised by an MFU factor rather than delivering the full peak ratio -- see Section 2.4.
- **Tokens per parameter** -- override the default of 150 (frontier dense). Presets: 20 (Chinchilla), 26 (replicated), 150 (frontier dense), 400 (MoE active), 1,000 (small, served). See Section 2.5 for the evidence behind each.

All results update live as you adjust sliders.

#### GPU Comparison

Select multiple accelerator SKUs (any of the 15 built-in SKUs, or a custom one) in the GPU selector. The comparison table shows side-by-side:

- Required GPU count per SKU
- H100-equivalent count (normalized comparison)
- Sustained FLOP/s per GPU
- Memory lower-bound GPUs

A bar chart visualizes the GPU counts for quick comparison.

#### Memory Feasibility

Every result includes a memory feasibility check. The calculator computes a lower-bound GPU count based on model state memory:

```
memory_lower_bound_GPUs = ceil(N * bytes_per_parameter / (GPU_memory * 0.85))
```

If the memory lower bound exceeds the compute requirement, the result is flagged as **memory-bound** rather than compute-bound. This indicates the model needs more GPUs than compute alone would suggest, due to memory pressure from optimizer states, gradients, and parameters.

---

### 2.4 The Math (Formulas)

#### Token Target

```
D = TPP * N
```

Where D is training tokens, TPP is tokens-per-parameter (default 150), and N is model parameters. At the default, a 70B model trains on 10.5T tokens; at Chinchilla-20 it would be 1.4T.

#### Training FLOPs

```
C = 6 * N * D
```

This is the standard first-order estimate for dense decoder-only transformer training. The factor of 6 accounts for the forward pass (2ND) and backward pass (4ND).

With overhead:

```
C_total = 6 * N * D * overhead_factor
```

#### Sustained Throughput

```
effective_MFU     = base_MFU * precision_factor * moe_factor
sustained_per_GPU = peak_FLOP/s * effective_MFU * availability
```

- `peak_FLOP/s` is the dense (not sparse) tensor-core throughput for the selected precision
- `base_MFU` is the per-SKU default (30-45%) or your per-GPU override
- `availability` is the fraction of wall-clock time spent doing useful training

#### Precision-Relative MFU Factors

Swapping in a lower-precision peak **does not** deliver the full peak ratio, and the calculator no longer pretends it does. Measured end-to-end FP8 wall-clock speedup is **1.3-1.5x, not 2x**, so the achieved MFU at FP8 is roughly `(1.3..1.5) / 2 = 0.65-0.75` of the BF16 MFU:

| Precision | MFU factor |
|-----------|-----------|
| BF16 Dense | 1.00 |
| FP8 Dense | 0.70 |
| FP4 Dense | 0.50 |

The reason is Amdahl's law. Default recipes quantize only the transformer-block linear layers; softmax, layernorm, residuals, attention masking, the optimizer and all-reduce stay at BF16 or FP32, so the 2x (or 4x) peak applies to only part of the step.

> **This changes every FP8 number the tool previously produced.** Earlier versions assumed FP8 delivered a full 2x speedup over BF16. An FP8 estimate now yields roughly `2 x 0.70 = 1.4x` the BF16 throughput instead of `2x`, so required GPU counts at FP8 are about 43% higher than the old model gave. Re-run any FP8 scenario you saved or shared from an earlier version.

#### MoE Factor

| Architecture | MFU factor |
|--------------|-----------|
| Dense | 1.00 |
| Sparse (MoE) | 0.85 |

Sparse architectures pay a systems penalty -- expert routing, all-to-all communication, load imbalance -- on top of the precision factor. Validated against DeepSeek-V3: `0.70 (FP8) x 0.85 (MoE) = 0.595`, close to the observed 0.50 ratio of FP8-peak to BF16-peak utilization.

The Training pane has no MoE input, so it always evaluates the dense factor of 1.00. The total-versus-active parameter split is modelled in the Fine-Tuning pane (Section 4.8).

#### Missing Precision Peaks

Not every accelerator publishes a dense peak at every precision. When the selected precision has no published peak for a SKU, the calculator falls back `FP4 -> FP8 -> BF16` and warns that the number shown is therefore **not** a speedup for that precision -- only the MFU penalty applies. TPU v6e has no FP8 or FP4 peak; MI300X, MI325X, TPU v7, Trainium2 and Gaudi 3 have no FP4 peak.

#### GPU Count

```
GPUs = ceil(C_total / (training_window_seconds * sustained_per_GPU))
```

#### The Quadratic Dependence

Because `D = TPP * N`, the total compute is:

```
C = 6 * N * TPP * N = 6 * TPP * N^2
```

Doubling the model size quadruples the compute. This is the single most important scaling intuition.

#### H100-Equivalent Conversion

```
H100_equivalents = (required_GPUs * sustained_per_GPU) / sustained_per_H100
```

This normalizes any GPU type to an H100-equivalent count for apples-to-apples comparison.

---

### 2.5 LLM Mode (6ND, Chinchilla, TPP)

LLM mode uses the standard `C = 6ND` FLOP model for dense decoder-only transformers. The tokens-per-parameter (TPP) setting controls the data-to-model ratio and has a major impact on total compute.

#### Tokens-Per-Parameter Presets

TPP is tokens of training data per model parameter. Because FLOPs are `6 x TPP x N^2`, it scales the answer linearly -- doubling TPP doubles tokens, FLOPs, GPU count and cost.

**The default is 150, not Chinchilla's 20.** This changed in October 2026. Chinchilla answers "how do I minimise loss for a fixed *training* budget", and 20 is still the right answer to that question. But once a model will be *served*, the thing worth minimising is lifetime compute -- train plus serve -- and that optimises differently, favouring a smaller model trained longer. No 2025-26 production run trains compute-optimal. Defaulting to 20 understated every out-of-the-box estimate by roughly 7.5x.

| Preset | TPP | When to use it |
|--------|-----|----------------|
| Chinchilla | 20 | Minimising loss per training FLOP, with inference cost not a concern. Hoffmann et al. 2022 [R1]. |
| Replicated | 26 | Epoch AI's re-fit puts the optimum at 25.6. DeepMind's own parametric form reads as high as ~70 -- the three methods in the original paper disagree with each other, which is why the replication exists. |
| Frontier dense | **150** (default) | 2026 practice. Median of Llama 3.1 70B (223), OLMo 3 32B (172), Marin 32B (201), Qwen3-235B on total params (153). |
| MoE active | 400 | Sparse models, applied to **active** parameters. DeepSeek-V3 is 400 and GLM-5 is 712 on active, while both sit at 22-38 on *total* params. |
| Small, served | 1,000 | Inference-optimal for a small model at high volume. Llama 3.1 8B is 1,950; Marin 8B 1,588. |

**TPP in real runs**, which is the evidence behind the presets:

| Run | Params | Tokens | TPP (total) | TPP (active) |
|-----|--------|--------|-------------|--------------|
| GPT-3 175B (2020) | 175B | 300B | 1.7 | -- |
| Chinchilla 70B (2022) | 70B | 1.4T | 20.0 | -- |
| Llama 3.1 405B | 405B | 15.6T | 38.5 | -- |
| Llama 3.1 70B | 70B | 15.6T | 222.9 | -- |
| Llama 3.1 8B | 8B | 15.6T | 1,950 | -- |
| DeepSeek-V3 | 671B | 14.8T | 22.1 | 400 |
| Qwen3-235B-A22B | 235B | 36T | 153.2 | 1,636 |
| OLMo 3 32B | 32B | 5.5T | 171.9 | -- |

The pattern: **small models are massively over-trained, frontier models sit at 20-40 on total parameters.** Note that DeepSeek-V3 and GLM-5 are near Chinchilla-optimal on *total* params while being 400-712 on *active* -- so "is this over-trained?" depends entirely on which denominator you use.

**A ceiling you cannot dial past.** `D = TPP x N` has to come from a real corpus. Muennighoff et al. found about 4 epochs of repetition is nearly free, with returns decaying sharply after; a 2026 refinement found even that holds only for ~3B models near 1x Chinchilla, with the free-epoch count falling as N and TPP grow. The calculator warns when the implied corpus exceeds roughly 30T tokens -- past the largest published datasets -- because at that point the compute estimate stays correct while the quality implication does not.

---

### 2.6 Time-Series Mode (Windows, Patches, Tokens)

#### What It Estimates

Time-series foundation models (e.g., TimesFM, Chronos, Moirai) are transformers pretrained on large collections of time-series data. Unlike LLMs where tokens are subwords, here tokens are **patches** -- fixed-length windows of consecutive timesteps, optionally expanded across variables (channels).

The calculator estimates training FLOPs using a transformer-style approximation: `factor * N * effective_tokens`. This is not an empirically calibrated time-series scaling law -- it adapts the LLM approach to time-series geometry.

#### Input Reference

| Input | Description | Default | Notes |
|-------|-------------|---------|-------|
| Model parameters (N) | Trainable parameters | -- | Same as LLM |
| Number of series | Total time-series in training corpus | -- | e.g. 10M series |
| Avg timesteps per series | Mean length of each series | -- | e.g. 1,000 |
| Variables per series | Channels / features per series | -- | e.g. 4 |
| Lookback window | Timesteps in the input context | -- | e.g. 512 |
| Forecast horizon | Timesteps to predict | -- | e.g. 96 |
| Stride | Step size between consecutive windows | -- | Smaller = more windows |
| Patch size | Timesteps per patch | -- | e.g. 16 |
| Tokenization mode | How patches map to tokens | channel_compressed | See below |
| Epochs | Number of passes over the data | 1 | |
| Architecture factor | FLOP multiplier per token | 6 | Analogous to LLM's 6ND factor |

#### Tokenization Modes

- **Channel-compressed**: Each patch is one token regardless of variables. `tokens_per_window = patches_per_window`. Most compact.
- **Channel-expanded**: Each variable gets its own token per patch. `tokens_per_window = variables * patches_per_window`. Can be 10-100x more tokens.
- **Custom**: User specifies tokens per window directly.

#### Formula

```
usable_timesteps = avg_timesteps - lookback - horizon
windows_per_series = floor(usable_timesteps / stride) + 1
patches_per_window = ceil(lookback / patch_size)

tokens_per_window:
  channel_compressed: patches_per_window
  channel_expanded:   variables * patches_per_window
  custom:             user-specified

effective_tokens = series * windows_per_series * tokens_per_window * epochs
base_FLOPs = architecture_factor * N * effective_tokens
total_FLOPs = base_FLOPs * overhead_factor
```

#### Worked Example

**Inputs:** 1B params, 10M series, 1,000 timesteps/series, 4 variables, lookback=512, horizon=96, stride=64, patch_size=16, channel_compressed, 1 epoch, factor=6.

```
usable_timesteps = 1000 - 512 - 96 = 392
windows_per_series = floor(392 / 64) + 1 = 7
patches_per_window = ceil(512 / 16) = 32
tokens_per_window = 32 (channel_compressed)
effective_tokens = 10,000,000 * 7 * 32 * 1 = 2.24e9 (2.24B tokens)
base_FLOPs = 6 * 1e9 * 2.24e9 = 1.344e19
total_FLOPs = 1.344e19 * 1.10 = 1.478e19 FLOPs
```

With **channel-expanded** tokenization (4 variables):

```
tokens_per_window = 4 * 32 = 128
effective_tokens = 10,000,000 * 7 * 128 * 1 = 8.96e9 (8.96B tokens)
base_FLOPs = 6 * 1e9 * 8.96e9 = 5.376e19
```

Channel-expanded tokenization produces 4x more tokens (and 4x more FLOPs) than channel-compressed for 4 variables. With 100 variables, the factor is 100x.

From here, GPU count follows the same formula as LLM mode: `GPUs = ceil(total_FLOPs / (window_seconds * sustained_per_GPU))`.

---

### 2.7 Tabular Foundation Mode (Row/Cell/Axial)

#### What It Estimates

Tabular foundation models (e.g., TabPFN, CARTE) are transformers pretrained on collections of tabular datasets. Each "task" is one tabular dataset. The key design choice is **tokenization**: how rows and columns map to transformer tokens.

#### Tokenization Modes

- **Row-tokenized**: Each row is one token. `tokens_per_task = rows`. Compact but limited expressiveness per token.
- **Cell-tokenized**: Each cell (row x column) is one token. `tokens_per_task = rows * columns`. Very long sequences.
- **Axial**: Additive rather than multiplicative. `tokens_per_task = rows + columns`. Used by axial-attention architectures.
- **Custom**: User specifies tokens per task directly.

#### Formula

```
tokens_per_task:
  row:    rows_per_task
  cell:   rows_per_task * columns_per_task
  axial:  rows_per_task + columns_per_task
  custom: user-specified

effective_tokens = tasks * tokens_per_task * epochs
base_FLOPs = architecture_factor * N * effective_tokens
total_FLOPs = base_FLOPs * overhead_factor * test_time_compute_multiplier
```

The `test_time_compute_multiplier` (default 1) accounts for architectures like TabPFN that perform significant computation at inference time. Set >1 if you want total compute to include test-time budget.

#### Dense Attention Warnings

Because tabular tokenization can create very long sequences, the calculator warns about dense attention feasibility:

| Sequence Length | Warning Level |
|----------------|---------------|
| > 16,384 | Attention warning -- may strain dense attention |
| > 65,536 | High warning -- very large for dense attention, training may be extremely slow |
| > 100,000 | Severe warning -- dense attention infeasible, use sparse/factorized attention |

#### Worked Examples

**Row-tokenized:** 1B params, 1M tasks, 1,024 rows/task, 50 columns, 1 epoch, factor=6.

```
tokens_per_task = 1,024 (row mode)
effective_tokens = 1,000,000 * 1,024 * 1 = 1.024e9 (1.024B tokens)
base_FLOPs = 6 * 1e9 * 1.024e9 = 6.144e18
```

**Cell-tokenized** (same inputs):

```
tokens_per_task = 1,024 * 50 = 51,200 (cell mode)
effective_tokens = 1,000,000 * 51,200 * 1 = 5.12e10 (51.2B tokens)
base_FLOPs = 6 * 1e9 * 5.12e10 = 3.072e20
```

Cell tokenization produces **50x more tokens** (and 50x more FLOPs) than row tokenization for the same data. With 100 columns, it would be 100x. This is the single most impactful decision in tabular foundation model compute planning.

---

### 2.8 Classical Tabular Mode (GBDT, Work Units)

#### How It Works

This mode is fundamentally different from the other three. GBDT algorithms (LightGBM, XGBoost, CatBoost, Random Forest) are not transformer models and cannot be meaningfully estimated with a FLOP model. Instead, the calculator uses **work units** and **empirical throughput benchmarks**.

A work unit represents one row-column-round operation. Total work is:

```
base_work  = rows * columns * boosting_rounds
depth_adj  = base_work * (max_depth / 6)
work_units = depth_adj * cv_folds * hp_trials
```

Training time is estimated by dividing total work by a throughput coefficient (work units per second).

#### Input Reference

| Input | Description | Default | Notes |
|-------|-------------|---------|-------|
| Algorithm | GBDT implementation | LightGBM | LightGBM, XGBoost, CatBoost, Random Forest, Custom |
| Rows | Training dataset rows | -- | GPU coefficients hold at >= 1M rows |
| Columns | Features | -- | Assumed dense numeric; warns above 500 |
| Boosting rounds | Trees / iterations | -- | |
| Max depth | Tree depth | 6 | **Scales the estimate** linearly: `work x depth/6`. Histogram GBDT cost is linear in depth -- at each level the union of nodes covers all rows. |
| Max bins | Histogram bin count | 256 | **Scales GPU throughput only**, as `(256/bins)^0.263`, clamped to 0.7-2.0x. CPU throughput is nearly flat in bin count. |
| CV folds | Cross-validation folds | 1 | Multiplier on total work |
| HP trials | Hyperparameter search trials | 1 | Multiplier on total work |
| CPU / GPU | Implementation target | GPU | Selects which coefficient column is used -- see below |
| Throughput coefficient | Work units per second | Auto | Set to 0 for benchmark defaults |

`max_depth` and `max_bins` were previously collected and ignored. They now both affect the estimate.

#### Default Throughput Coefficients

Both columns, in work units per second. One work unit is `1 row x 1 column x 1 boosting round` -- exactly the quantity this adapter counts.

| Algorithm | GPU | CPU | Measured GPU speedup at >= 1M rows |
|-----------|-----|-----|------------------------------------|
| XGBoost | 8.0e9 | 5.0e8 | 13.4x (range 4.5-23.4x) |
| LightGBM | 1.5e9 | 1.5e9 | **0.69x -- GPU measured slower** on the OpenCL backend; the equal coefficients reflect the newer CUDA backend at rough parity |
| CatBoost | 5.0e9 | 7.0e8 | 9.7x (range 3.9-13.2x) |
| Random Forest | 5.0e9 | 1.0e9 | 3-26x, rising with dataset size |
| Custom | 2.0e9 | 5.0e8 | No measurement exists -- mid-range placeholder, flagged low confidence |

**Provenance.** These are not invented. Each pair is a geometric mean over the >= 1M-row datasets of a published benchmark, recomputed as `rows * columns * rounds / seconds` so it maps onto this adapter's work unit one-for-one.

- **XGBoost / LightGBM / CatBoost** -- GBM-Benchmarks (the XGBoost maintainer's suite), AWS p3.16xlarge, 1x V100-16GB GPU against a 64-vCPU Xeon E5-2686 v4. Datasets: YearPredictionMSD, CoverType, Bosch, Synthetic, Higgs, Airline.
- **LightGBM CPU** is cross-checked against LightGBM's own single-machine experiments (2x E5-2690 v4, 28 cores): 1.13e9 on Higgs, 2.21e9 on MS LTR, 2.17e9 on Yahoo LTR, dense geometric mean 1.76e9.
- **Random Forest** -- cuML accelerator benchmarks, RTX PRO 6000 Blackwell against a 32-core Threadripper PRO 7975WX, 100 estimators. Measured GPU range is 3e9-3.2e10; the conservative low end is used because the CPU baseline there is scikit-learn, a much weaker implementation than LightGBM.

**Valid regime.** Every anchor was measured at **max_depth 6, 500 boosting rounds, >= 1M rows, dense numeric features, and default bin counts (~128-256)**. `max_depth` and `max_bins` scale off those calibration points.

**Known limits.**

- The GPU anchors are V100-era (2018). H100 and Blackwell histogram building is bandwidth-bound and roughly 2-3x faster, so these defaults are conservative on modern hardware. That adjustment is extrapolation and is deliberately not applied.
- The `rows x columns` work model collapses on sparse or one-hot data, by up to ~100x. LightGBM's Exclusive Feature Bundling collapses one-hot columns: its measured rate on Expo (11M x 700 one-hot) is 35x and on Allstate (13.2M x 4,228) is 100x the dense rate. A warning fires above 500 columns; the numbers are not corrected.
- Below ~1M rows, GPU efficiency is scaled down by log-interpolation towards "2.5x slower than the same library's CPU path at 100K rows". That interpolation is derived, not measured.

> **The old "GPU is typically 5-10x faster" claim was wrong as a general statement.** It holds for XGBoost and CatBoost on >= 1M rows of dense numeric features. It does not hold for LightGBM, whose OpenCL backend measured **0.69x -- slower than its own CPU path, on 5 of 6 reference datasets** -- and it does not hold below roughly 500K rows for anything, where transfer and kernel-launch overhead dominate and a GPU is a net loss.

#### GPU Recommendation Verdict

The pane's real question is not "how many GPUs" but **"do I need a GPU at all"**, so the result carries an explicit verdict. Both sides of the comparison always use the default coefficients for the path you did *not* select, so it stays apples-to-apples.

| Verdict | Condition | Meaning |
|---------|-----------|---------|
| **GPU recommended** | GPU >= 2x faster **and** CPU wall-clock > 15 minutes | 1 GPU is enough. Multi-GPU GBDT is a niche; the recommended count is always 1. |
| **CPU sufficient** | GPU < 2x faster, or CPU already finishes in under 15 minutes | The complexity is not worth it at this scale. |
| **GPU likely slower** | Predicted GPU speedup < 0.95x | A GPU is predicted to be slower here outright. |

Where memory binds -- which is rare, since a binned matrix is roughly `rows x columns` bytes, so a 1M x 100 table is ~100 MB -- the answer is external memory on one device, not more GPUs. XGBoost 3.0 external memory trains a 1 TB dataset on a single GH200.

#### How to Calibrate

For better accuracy, calibrate with a known run:

1. Run a small training job and record: rows, columns, rounds, wall-clock seconds.
2. Compute: `throughput = (rows * columns * rounds) / seconds`
3. Enter this value as the throughput coefficient.
4. Confidence improves to **medium**. Without calibration it is **medium-low**, or **low** for a custom algorithm or below the 1M-row regime.

#### Worked Example

**Inputs:** LightGBM, 10M rows, 200 columns, 1,000 rounds, max_depth 6, 256 bins, 5-fold CV, 20 HP trials, GPU.

```
base_work  = 10,000,000 * 200 * 1,000 = 2.0e12 work units
depth_adj  = 2.0e12 * (6 / 6) = 2.0e12
total_work = 2.0e12 * 5 * 20 = 2.0e14 work units
throughput = 1.5e9 /sec (LightGBM GPU default x 1.000 small-data x 1.00 bins)
estimated_time = 2.0e14 / 1.5e9 = 133,333 seconds = 37.0 hours = 1.54 days
```

**Verdict: CPU sufficient.** The CPU path gives exactly the same 37.0 hours, because LightGBM's GPU and CPU coefficients are equal. A GPU buys nothing here.

Switch the algorithm to XGBoost and the picture changes completely: 2.0e14 / 8.0e9 = 25,000 s = **6.9 hours on one GPU** against 2.0e14 / 5.0e8 = 400,000 s = **111 hours on CPU**, a 16x speedup and a **GPU recommended** verdict. The algorithm choice matters more than the hardware choice.

---

### 2.9 GPU SKU Reference

Fifteen built-in accelerator SKUs. All compute figures are **dense**, in TFLOP/s. Spec data as of **2026-10-14**.

| Accelerator | Dense BF16 | Dense FP8 | Dense FP4 | Memory | Bandwidth | Default MFU |
|-------------|-----------|-----------|-----------|--------|-----------|-------------|
| NVIDIA H100 SXM | 989.5 | 1,979 | -- | 80 GB | 3.35 TB/s | 40% |
| NVIDIA H200 SXM | 989.5 | 1,979 | -- | 141 GB | 4.8 TB/s | 42% |
| NVIDIA B200 SXM / HGX B200 | 2,250 | 4,500 | 9,000 | 180 GB | 7.7 TB/s | 39% |
| NVIDIA B300 SXM (Blackwell Ultra) | 2,250 | 4,500 | 13,500 | 288 GB | 8.0 TB/s | 40% |
| NVIDIA GB200 NVL72 (per GPU) | 2,500 | 5,000 | 10,000 | 186 GB | 8.0 TB/s | 42% |
| NVIDIA GB300 NVL72 (per GPU, Ultra) | 2,500 | 5,000 | 15,000 | 288 GB | 8.0 TB/s | 41% |
| AMD Instinct MI300X | 1,307.4 | 1,307.4 | -- | 192 GB | 5.3 TB/s | 30% |
| AMD Instinct MI325X | 1,307.4 | 2,614.9 | -- | 256 GB | 6.0 TB/s | 31% |
| AMD Instinct MI355X | 2,500 | 5,000 | 10,100 | 288 GB | 8.0 TB/s | 33% |
| AMD Instinct MI450 | 5,000 | 20,000 | 40,000 | 432 GB | 19.6 TB/s | 30% |
| Google TPU v6e (Trillium) | 918 | -- | -- | 32 GB | 1.638 TB/s | 45% |
| Google TPU v7 (Ironwood) | 2,307 | 4,614 | -- | 192 GB | 7.37 TB/s | 40% |
| AWS Trainium2 | 667 | 1,299 | -- | 96 GB | 2.9 TB/s | 35% |
| AWS Trainium3 | 671 | 2,517 | 2,517 | 144 GB | 4.9 TB/s | 35% |
| Intel Gaudi 3 | 1,835 | 1,835 | -- | 128 GB | 3.7 TB/s | 30% |

**Dense vs sparse.** NVIDIA publishes tensor-core throughput with 2:4 structural sparsity (asterisked on its datasheets); dense is exactly half, so every NVIDIA row here halves the headline number. AMD publishes dense FP16/BF16 directly but sparse FP8/FP4 -- the AMD rows are the dense figures. Google and AWS publish dense. Structural sparsity is not used in pretraining, so the calculator uses dense exclusively.

**Two honesty caveats carried in the SKU notes:**

> **B300's BF16 and FP8 are identical to B200's** (2,250 and 4,500 TFLOP/s dense). Blackwell Ultra's uplift is 1.5x FP4 (13,500 dense) and 49% more memory (288 GB) -- nothing else. A user selecting B300 for BF16 or FP8 work gets **no compute gain over B200**, and the calculator warns about it at those precisions.

> **MI450's BF16 figure is extrapolated, not published.** AMD released only FP8 (20 PF) and FP4 (40 PF); 5.0 PF BF16 is the conventional half-of-FP8 ratio, but AMD has never confirmed it for MI450. Treat BF16 results on this SKU as an estimate, not a spec. Bandwidth is reported as 19.6-23.3 TB/s (the lower bound is used). Low confidence overall.

**H100 vs H200.** Identical raw dense BF16 peak -- same Hopper compute dies. The H200 advantage is 76% more HBM capacity (141 vs 80 GB) and 43% more bandwidth (4.8 vs 3.35 TB/s), which allows larger microbatches, longer contexts, less activation checkpointing and less aggressive sharding, all of which can raise achieved MFU. The calculator models this with a higher default MFU (42% vs 40%), not a higher peak. The H200 default was lowered from 45% to 42% because no published pretraining MFU for any Hopper part exceeds Llama 3 405B's 43% on H100, so 45% was an unsourced claim.

**Where FP8 buys memory but not time.** On MI300X and Gaudi 3, FP8 runs at exactly the same rate as BF16 -- the matrix engines gain no throughput at FP8, only a smaller memory footprint. On Trainium3, MXFP4 is quoted at the same 2,517 TFLOP/s as MXFP8, so 4-bit buys nothing in compute there either.

**TPU v6e** has no native FP8 or FP4: BF16 is its only training precision, and FP8/FP4 selections fall back to the BF16 peak (see Section 2.4).

**Deliberately excluded.** NVIDIA Vera Rubin VR200 / NVL144. Rack-level figures are published (3.6 EF FP4, 1.2 EF FP8 across 144 GPUs) but per-GPU dense BF16 is not, and sources conflict on whether the headline "50 PFLOPS FP4" is per-package or per-die. The row will be added when the datasheet lands.

---

### 2.10 Confidence Levels

Every estimate includes a confidence label indicating how reliable the estimate is likely to be.

| Level | Meaning |
|-------|---------|
| **High** | Well-understood domain with empirically validated formulas. LLM mode with standard settings. |
| **Medium** | Reasonable approximation but relies on analogies or simplifications. Time-series mode; classical tabular with calibrated throughput. |
| **Medium-low** | Significant uncertainty. Tabular foundation with row or axial tokenization; classical tabular on benchmark defaults. |
| **Low** | Order-of-magnitude estimate at best. Tabular foundation with cell tokenization; classical tabular with a custom algorithm or below the 1M-row regime. |

#### Confidence by Model Family

| Model Family | Settings | Confidence |
|--------------|----------|------------|
| LLM | Standard (6ND, BF16, full pretraining) | High |
| LLM | Non-standard (FP8 or FP4, SFT/LoRA/RLHF) | Medium |
| Time-Series | All tokenization modes | Medium |
| Tabular Foundation | Row or axial tokenization | Medium-low |
| Tabular Foundation | Cell or custom tokenization | Low |
| Classical Tabular | With calibrated throughput | Medium |
| Classical Tabular | Benchmark defaults, >= 1M rows, known algorithm | Medium-low |
| Classical Tabular | Custom algorithm, or below the 1M-row regime | Low |

For the Fine-Tuning pane's per-method confidence tiers, see Section 4.11.

---

### 2.11 Sensitivity Matrix

The sensitivity matrix shows GPU counts across a grid of MFU values (rows) and training windows (columns). This reveals how sensitive your estimate is to the two largest unknowns. Cells are heat-mapped so larger clusters stand out visually.

---

### 2.12 IsoFLOP Explorer

The IsoFLOP Explorer is an interactive chart that visualizes the tradeoff between model size and tokens-per-parameter for a fixed compute budget. It answers the question: "Given my FLOP budget, what combinations of model size and data volume are feasible?"

**The chart:**
- **X axis** -- Tokens per parameter (TPP), ranging from 5 to 300
- **Y axis** -- Maximum model size (log scale)
- **Curve** -- The IsoFLOP frontier: `N = sqrt(budget / (6 * TPP * overhead))`
- **Green dot** -- Your current scenario's (TPP, N) position
- **Amber dots** -- Known reference models that fall within the chart's range, labeled with model names

Points above the curve require more compute than your budget. Points below it are feasible.

**Two entry modes:**

1. **Current Scenario** -- Uses the total FLOPs from your active scenario. No extra input needed.
2. **GPU Budget** -- Enter a GPU count, GPU SKU, training window (days), and MFU. The explorer derives the FLOP budget as: `budget = gpuCount * peakFLOPs * MFU * availability * windowDays * 86400`

The IsoFLOP Explorer is available for all transformer-based modes (LLM, Time-Series, Tabular Foundation) and hidden for Classical Tabular.

**How to read it:** If your green dot is near the curve, your scenario is roughly on the frontier -- you are making efficient use of your compute. If your dot is well below the curve, you have headroom to train a larger model or use more data. If your dot is above the curve (which should not happen for a self-consistent scenario), your scenario exceeds the budget.

---

### 2.13 Reference Model Comparison

Below the main result, the calculator shows a **Compute Context** table comparing your scenario's total FLOPs to known training runs. This provides intuitive context: "Is my run the size of GPT-3, or Llama 3 70B, or something in between?"

The table includes:

| Model | Parameters | Tokens | ~FLOPs | Source |
|-------|-----------|--------|--------|--------|
| Marin 8B | 8B | 1.6T | 7.7e22 | Marin Community 2025 |
| Llama 2 7B | 7B | 2T | 8.4e22 | Touvron et al. 2023 |
| Llama 2 13B | 13B | 2T | 1.6e23 | Touvron et al. 2023 |
| GPT-3 175B | 175B | 300B | 3.2e23 | Brown et al. 2020 |
| Chinchilla 70B | 70B | 1.4T | 5.9e23 | Hoffmann et al. 2022 |
| Marin 32B | 32B | 3.2T | 6.1e23 | Marin Community 2025 |
| Llama 3 8B | 8B | 15T | 7.2e23 | Meta 2024 |
| Llama 2 70B | 70B | 2T | 8.4e23 | Touvron et al. 2023 |
| Llama 3 70B | 70B | 15T | 6.3e24 | Meta 2024 |
| Llama 3 405B | 405B | 15T | 3.6e25 | Meta 2024 |

> **Corrected 2026-10-14.** Llama 2 7B and 13B previously carried 8.4e19 and 1.6e20 -- both 1,000x too low, and the rows were sorted on those wrong values. The correct 6ND figures are 8.4e22 and 1.6e23, which is also why Llama 2 7B now sorts just above Marin 8B rather than at the bottom of the table.

The row closest to your compute budget is highlighted. A "vs Yours" column shows the ratio of your FLOPs to each reference model's FLOPs, giving a quick sense of relative scale.

This table appears for all transformer-based modes (LLM, Time-Series, Tabular Foundation) but not for Classical Tabular / GBDT, which does not use a FLOP model.

---

### 2.14 Reverse Solve & Calibration

#### Reverse Solve

Two sub-modes:

**Time from GPU budget:** Given a fixed GPU count and model size, how long will training take?
- Enter your available GPU count
- The calculator returns training time in days

**Size from GPU budget:** Given a fixed GPU count and training window, what is the largest model you can train?
- Enter your available GPU count
- The calculator returns the maximum model size (uses the quadratic relationship: `N = sqrt(available_FLOPs / (6 * TPP * overhead))`)

#### Calibration

Back-solve achieved MFU from a known training run. Enter:

- Model size, tokens trained, wall-clock days, GPU count, GPU type
- The calculator returns the achieved MFU percentage

If the result is between 10-70%, it is considered reasonable. You can apply the calibrated MFU to your current scenario for more accurate planning.

---

### 2.15 Warnings

#### General Warnings (All Modes)

| Warning | Trigger | What To Do |
|---------|---------|------------|
| Large-cluster warning | Model >70B and window <14 days | Expect significant operational complexity. Consider longer windows or staged training. |
| Distributed-systems warning | Required GPUs >1,024 | Networking, checkpointing, stragglers, and cluster fragmentation become major factors. Plan for operational overhead. |
| Hyperscale warning | Required GPUs >10,000 | A multi-rack, multi-datacenter-class deployment with extreme operational complexity. |
| Optimistic MFU warning | MFU override >60% | Exceeds typical achieved utilization; most production runs achieve 30-50%. |
| Training mode warning | Mode is not Full Pretraining | The 6ND formula is calibrated for full pretraining. SFT/LoRA/RLHF typically need far less compute -- use the Fine-Tuning Calculator (Section 4) instead. |
| FP8 / FP4 warning | FP8 or FP4 precision selected | End-to-end low-precision training is experimental. Actual throughput may differ significantly from the peak-based estimate. |
| Precision MFU factor note | FP8 or FP4 precision selected | States the MFU factor applied (0.70 or 0.50) and that the estimate does not assume the full peak-ratio speedup -- measured FP8 wall-clock gain is 1.3-1.5x, not 2x. |
| FP4 recipe note | FP4 precision selected | NVFP4 pretraining is validated (NVIDIA pretrained a 12B hybrid Mamba-Transformer on 10T tokens in NVFP4, matching an FP8 baseline) but is not a drop-in flag: it needs Random Hadamard transforms, 2-D quantization, stochastic rounding and selective high-precision layers. |
| Missing precision peak | Selected SKU publishes no dense peak at the selected precision | The estimate falls back to a higher-precision peak, so the throughput shown is not a speedup for that precision -- only the MFU penalty applies. |
| H200 note | H200 selected | Reminder that H200 has the same raw BF16 peak as H100. Its advantage is memory and bandwidth, which may improve MFU. |
| B300 note | B300 selected at a precision other than FP4 | B300's dense BF16 and FP8 peaks are identical to B200's. At this precision it delivers no compute gain. |
| Memory bound exceeds compute | Memory lower-bound GPUs > compute GPUs | The model's state memory requires more GPUs than compute alone. The cluster is memory-constrained. |

#### Time-Series Warnings

| Warning | Trigger | What To Do |
|---------|---------|------------|
| Invalid window geometry | lookback + horizon > avg timesteps | Fix inputs: no training windows can be generated. |
| Small stride | stride <= 10% of lookback | Many overlapping windows inflate data volume. Increase stride or verify this is intentional. |
| Channel-expanded inflation | >20 variables with channel_expanded | Consider channel-compressed or custom tokenization to reduce compute. |
| Patch > lookback | patch_size > lookback_window | Each window produces less than one patch. Reduce patch size. |

#### Tabular Foundation Warnings

| Warning | Trigger | What To Do |
|---------|---------|------------|
| Attention strain | Sequence > 16K tokens/task | Consider attention-efficient architectures. |
| High attention | Sequence > 65K tokens/task | Training may be extremely slow with dense attention. |
| Infeasible attention | Sequence > 100K tokens/task | Dense attention is infeasible. Use sparse, factorized, or row-level attention. |
| Cell + many columns | Cell mode, >50 columns | Produces extreme sequence lengths. Switch to row or axial mode. |
| High test-time multiplier | multiplier > 5x | Test-time compute may dominate serving cost. |

#### Classical Tabular Warnings

| Warning | Trigger | What To Do |
|---------|---------|------------|
| No calibration | Throughput coefficient not provided | Benchmark-derived defaults are accurate to roughly 2-3x at best. Calibrate with a known run. |
| Custom algorithm | Algorithm is Custom | No published benchmark exists; the default is a mid-range placeholder. Calibrate it. |
| Search note | folds * trials > 100 | The search is most of the wall-clock. Normal -- TabArena uses 200 configs x 8 inner folds = 1,600 fits per model per outer fold. Diminishing returns land around 50-100 trials. |
| Search dominates | folds * trials > 500 | Prune the search (Hyperband / early stopping) rather than buying hardware. |
| GPU likely slower | GPU selected and predicted speedup < 0.95x | GPU will likely be SLOWER here. GPU histogram building only pays off above roughly 1M rows of dense numeric features; a 100K-row dataset can train 2-3x slower on GPU. |
| GPU gains little | GPU selected and predicted speedup < 2x | The published 5-10x GPU speedup applies to XGBoost and CatBoost on >= 1M rows of dense numeric features, not to every library or every dataset size. |
| CPU selected | CPU implementation and a GPU would not be slower | Names the measured speedup for *this* algorithm -- XGBoost 13.4x, CatBoost 9.7x, cuML random forest 3-26x, LightGBM only 0.5-1.5x -- and the GPU wall-clock for comparison. |
| **Sparse / wide data** | Columns > 500 | If any columns are one-hot encoded or otherwise sparse, the `rows x columns` work model breaks down by up to ~100x, because Exclusive Feature Bundling collapses sparse columns. This estimate assumes dense numeric features and will be a large over-estimate otherwise. |
| Below the measured regime | Rows < 1,000,000 | GPU efficiency is scaled down by log-interpolation, which is derived rather than directly measured. Treat the GPU number as indicative only. |
| VRAM note | Always | VRAM is almost never the binding constraint for GBDTs: a binned matrix is roughly `rows x columns` bytes. If memory binds, the answer is external memory on one device, not more GPUs. |
| Empirical-model note | Always | Classical tabular estimates use empirical throughput models, not theoretical FLOPs. Results depend heavily on implementation, data characteristics and hardware. |

---

### 2.16 Export & Share

The calculator supports four export formats:

- **JSON** -- full scenario and results for programmatic use
- **CSV** -- tabular results for spreadsheets
- **Markdown** -- formatted summary for docs, Slack, or email
- **Share URL** -- all scenario parameters encoded in the URL hash fragment

The share URL encodes model family, model size, training window, TPP, GPU selections, MFU overrides, precision, availability, overhead, training mode, and memory bytes per parameter. Paste it in a browser to restore the exact scenario. URLs without a `modelFamily` parameter default to LLM mode for backwards compatibility.

---

## 3. Inference Calculator

The Inference Calculator estimates monthly costs for serving a large language model, comparing managed API pricing (buy) against self-hosted GPU clusters (build). It identifies the breakeven point where self-hosting becomes cheaper than per-token API fees.

**Who it's for:** Engineering and product teams deciding between API providers and self-hosted inference, or estimating serving costs at a given request volume.

**Key concept:** API cost scales linearly with volume; self-host cost is fixed by GPU provisioning -- the crossover is the breakeven point.

---

> **Pricing data is stale -- check the dates.** The three data files behind this pane -- `api-pricing.json`, `cloud-gpu-pricing.json` and `self-host-throughput.json` -- all carry `lastUpdated: 2026-06-01`. API token prices and cloud GPU rates move faster than that, usually downward. Treat every dollar figure here as **indicative of relative shape**, not as a quote: the breakeven *mechanism* holds, the absolute numbers do not. The last-updated date is shown in the pane. Verify current prices with your provider before committing, and note that the Fine-Tuning pane's costs read from the same cloud GPU file.

---

### 3.1 Overview -- Build vs Buy

The Inference Calculator helps you decide whether to serve an LLM via a managed API (buy) or self-host on your own GPUs (build). It estimates monthly costs for both approaches and identifies the breakeven point.

#### Cost Period Conventions

A "month" is **365/12 = 30.4167 days**, not a flat 30. This matters because it makes the monthly and annual figures consistent with each other:

```
monthly = daily * 30.4167
annual  = 12 * monthly        (exactly equal to daily * 365)
```

Under the previous flat-30 convention, `12 x monthly` came to 360 days and quietly understated the annual figure by 1.4%. Every monthly cost in this pane -- API spend and self-hosted GPU spend alike -- uses 30.4167 days, and the breakeven calculation divides back by the same constant, so the two sides of the comparison stay on one calendar.

---

### 3.2 Use-Case Presets (with Gear Icon Customization)

Select a use-case preset to auto-fill typical input and output token counts per request. You can always override the values after selecting a preset.

| Use Case | Input Tokens | Output Tokens | Typical Scenario |
|----------|-------------|--------------|-----------------|
| Chatbot / Support | 500 | 300 | Customer service, short Q&A turns |
| RAG / Q&A | 4,000 | 500 | Document retrieval + answer generation |
| Summarization | 10,000 | 500 | Legal, financial, or research documents |
| Coding Assistant | 8,000 | 2,000 | Code context + generation |
| Agentic Workflow | 16,000 | 4,000 | Multi-step tool use, autonomous agents |
| Long-context Analysis | 50,000 | 1,000 | Full documents, minimal output |

These presets are derived from Microsoft Azure inference traces (2023-2024), OpenRouter's 100T-token usage study, and enterprise production patterns.

---

### 3.3 Scale Methods (By Users, By Budget, Direct)

Three ways to specify your inference volume:

- **By Users** -- Enter user count and requests per user per day. The calculator multiplies them to get total requests/day. Common example: 1,000 users x 20 requests = 20,000 req/day.
- **By Token Budget** -- Enter a monthly token budget in millions. The calculator back-solves requests/day from the token budget and per-request token sizes (input + output tokens per request).
- **Direct** -- Enter raw requests/day for power users who already know their exact throughput requirements.

---

### 3.4 Buy (API) Panel

Pick a provider and model. The panel shows:

- **Monthly cost** -- total spend at current volume
- **Cost per request** -- unit economics for a single request
- **Annual projection** -- 12-month cost extrapolation
- **Input/output cost split** -- breakdown showing how much of the cost comes from input tokens vs output tokens

---

### 3.5 Build (Self-host) Panel

Pick a model to serve, GPU configuration, and cloud provider. The panel shows:

- **Monthly GPU cost** -- cloud compute spend for the GPU instances
- **Throughput feasibility** -- whether the GPU configuration can handle the target request rate
- **Utilization** -- what fraction of GPU capacity the workload consumes
- **Estimated TTFT** -- approximate time to first token for the configuration

---

### 3.6 Breakeven Analysis

A chart showing API cost (linear, scaling with volume) vs self-host cost (fixed, determined by GPU provisioning). The crossover point is marked on the chart.

The breakeven analysis tells you:

- Whether API or self-hosting is cheaper **at your current volume**
- The **breakeven request rate** -- the volume at which the two approaches cost the same
- How much headroom or savings you get from the cheaper option

Below the breakeven point, API is cheaper (you pay only for what you use). Above it, self-hosting wins (fixed GPU costs are amortized over more requests).

---

### 3.7 Export

The inference calculator shares the same export capabilities as the training calculator (JSON, CSV, Markdown, Share URL). See Section 2.16 for details.

---

## 4. Fine-Tuning Calculator

The Fine-Tuning Calculator estimates accelerator requirements for adapting an existing open-weight model. It answers a different question from the Training Calculator: not "how many GPUs to hit my deadline" but **"does this fit on the GPUs I can actually get, and what does that cost me?"**

**Who it's for:** ML engineers choosing between full fine-tuning, LoRA and QLoRA; infrastructure teams sizing a node for a fine-tuning service; technical leaders checking whether a 70B adaptation needs one GPU or three nodes.

**Key concept:** minimum GPUs = ceil(model state / usable HBM per GPU). Memory sets the GPU count; compute then sets the wall-clock on that configuration.

**Why memory-first?** In pretraining, the memory lower bound is roughly 470x slacker than the compute bound and never binds -- which is why the Training Calculator treats it as a side check. In fine-tuning it *is* the question. On one fixed workload, a 70B model is 1,224 GB and 24 H100s under full fine-tuning, and 47 GB on a single H100 under QLoRA. Same arithmetic, inverted dominance.

### What it is for

- Deciding whether a given model and method fit on the hardware you can obtain
- Comparing full fine-tuning, LoRA, QLoRA, DoRA, DPO, ORPO, GRPO and PPO on one workload
- Finding the minimum deployable GPU count, and seeing when it crosses a node boundary
- Choosing a LoRA rank and target set against an exact adapter-parameter count
- Sizing activation memory for long-context SFT
- Sanity-checking a vendor or team estimate for a fine-tuning job

### What it is NOT for

- Predicting whether the fine-tune will *work* -- no quality, loss, or eval prediction
- Multimodal / vision fine-tuning (image-patch activation accounting is not modelled)
- Embedding or reranker fine-tuning (contrastive losses with large in-batch negatives have a different memory shape)
- Agentic RL with tool calls or environment steps, where environment latency rather than decode binds
- End-to-end RL post-training wall-clock -- the GRPO and PPO rows cover the training side only. See Section 4.9.
- Spot-interruption or failure modelling
- Multi-adapter serving economics (an inference question, not a fine-tuning one)
- Detailed parallelism planning beyond the DDP / ZeRO-2 / FSDP-ZeRO-3 choice

---

### 4.1 Quick Start

1. Pick a **base model** (21 catalogued open-weight models, or Custom).
2. Pick a **goal preset** -- it fills examples, tokens per example, epochs, sequence length, method and LoRA rank.
3. Read the **method x GPU matrix**. Each cell gives GPU count, fit state, model state, wall-clock and cost.
4. Click a cell to open its memory breakdown, warnings and calculation trace.
5. Adjust the raw workload and configuration fields underneath; every preset field stays editable, and editing one clears the preset.

Defaults on first load: Llama 3.1 8B, LoRA r=16 over all linear projections, 2,000 examples x 800 tokens x 3 epochs, sequence length 2,048, micro-batch 1, packing on, AdamW (fp32 master), BF16 base weights, gradient checkpointing on, FSDP/ZeRO-3, availability 95%, overhead 1.10x, on-demand pricing, with H100 SXM / H200 SXM / B200 SXM as the matrix columns.

---

### 4.2 Reading the Method x GPU Matrix

The matrix is the pane's primary output: methods down the rows, selected GPU SKUs across the columns. Each cell shows four things -- GPU count, fit state, total model state, and then wall-clock and cost on the bottom line.

#### Fit states

| State | Meaning |
|-------|---------|
| **FITS** | All model state holds on one GPU. No sharding needed. |
| **SHARDED** | State is split across the listed GPU count using the selected strategy (FSDP/ZeRO-3 or ZeRO-2). |
| **OOM** | Infeasible at any practical GPU count -- either DDP was selected (which replicates rather than shards, so one GPU must hold everything), or the per-GPU fixed terms alone exceed usable memory. |

The GPU count shown is the **memory floor**, rounded up to a deployable count: 1, 2, 4, 8, then whole nodes of 8.

#### Why every cell shows GPU count, wall-clock *and* cost

Because no one of them tells the truth on its own.

**Wall-clock alone inverts the ranking.** Full fine-tuning finishes sooner than QLoRA on the same workload -- not because it is more efficient, but because its memory floor forces it onto 24x more GPUs. Read the time column without the GPU column and you will conclude that the most expensive method is the fastest, which is true and useless.

**Cost alone flattens the comparison**, because on a fixed workload the total bill is nearly method-invariant. Billed by the GPU-hour, `cost ~= total_FLOPs / (MFU x availability)`: GPU count cancels against wall-clock, and the method only moves FLOPs per token between `8N` and `6N`. The engine's own output on a fixed 70B workload spreads just **1.75x** in cost.

What genuinely varies is **memory footprint and minimum GPU count -- 24-27x**. That is the feasibility question this pane exists to answer, and the reason a single-config card could not replace the matrix.

#### Worked example: Llama 3.1 70B, one fixed workload

2,000 examples x 800 tokens x 3 epochs, sequence length 2,048, micro-batch 1, packing on, gradient checkpointing on, AdamW (fp32 master), FSDP/ZeRO-3, availability 95%, overhead 1.10x, on H100 SXM at $3.00/GPU-hour:

| Method | Model state | Min GPUs | Fit | Wall-clock | GPU-hours | Cost |
|--------|------------|----------|-----|-----------|-----------|------|
| Full fine-tune | 1,224 GB | 24 | SHARDED | 0.17 h | 4.06 | $12.17 |
| LoRA r=16 | 161 GB | 4 | SHARDED | 0.58 h | 2.32 | $6.95 |
| QLoRA r=16 | 47 GB | 1 | FITS | 2.32 h | 2.32 | $6.95 |

Memory spreads **26x**, minimum GPU count **24x**, wall-clock **14x the other way** -- and cost **1.75x**. The method choice is about what hardware you can obtain and how long you are willing to wait, not about the bill.

This finding is surfaced in the pane as an assumption chip ("Cost is near method-invariant") and in the note above the matrix, so it does not have to be inferred.

#### Cost columns and missing prices

Cost is `per-GPU hourly rate x GPU count x wall-clock hours`. The rate comes from `cloud-gpu-pricing.json`, taken as the cheapest listed instance price divided by that instance's GPU count, at the selected pricing tier (on-demand, reserved, or spot).

Only H100 SXM currently has a matching row in that file. Every other column -- H200, B200, B300, the AMD, Google, AWS and Intel parts -- shows an em dash for cost rather than borrowing a rate from a different accelerator. The pane names which columns are unpriced underneath the matrix.

---

### 4.3 Methods Reference

Eight methods ship as matrix rows. `N` is **active** parameters for compute and **total** parameters for memory (see Section 4.8).

| Method | FLOPs/token | Trainable params | Model-state bytes | Throughput factor | Confidence |
|--------|------------|------------------|-------------------|-------------------|------------|
| Full fine-tune | `6N` | all | `N_total x 2` weights + `16 B` per param (AdamW fp32) | 1.00 | High |
| LoRA | `4N` | adapters only | `N_total x 2` + `16 B` per adapter param | 1.00 | High |
| QLoRA | `4N` | adapters only | `N_total x 0.5` (NF4) + `16 B` per adapter param | 0.75 | Medium-low |
| DoRA | `4N` | adapters x 1.01 | as LoRA, plus magnitude vectors | 0.90 | Medium |
| DPO | `6N` | all | as full FT, plus a frozen reference copy when resident | 1.00 | Medium-low |
| ORPO | `6N` | all | as full FT (reference-free) | 1.00 | Medium-low |
| GRPO | `6N` | all | as full FT, plus a reference copy when `beta > 0` | 1.00 | Low |
| PPO | `6N` | all | as full FT, plus reward + reference copies and a **trainable** critic | 1.00 | Low |

Gradient checkpointing, which defaults on, adds `+2N` to every row: full fine-tuning becomes `8N` per token and LoRA `6N`.

#### Why LoRA is 4N, not 6N

The frozen base needs no weight gradients. A standard backward pass costs `4bmn` per linear layer; LoRA's costs `2bmn + 4br(m+n)`, and with `r` far smaller than `m` and `n` the adapter term vanishes -- so **LoRA's backward pass costs the same as its forward pass**. `2N` forward + `2N` backward = `4N`.

This is not a guess. Back-solving achieved MFU from NVIDIA NeMo's published LoRA throughputs under the `4N` model reproduces NeMo's own reported MFU to within 2.4 percentage points across four configurations, which validates the FLOP model and the MFU defaults at the same time.

#### Objective multipliers

DPO and ORPO process two completions per preference pair (chosen and rejected), so they multiply the token count rather than the FLOPs per token:

| Method | Reference handling | Multiplier on tokens processed |
|--------|-------------------|-------------------------------|
| DPO | Reference logprobs precomputed offline (default) | 2.00x |
| DPO | Reference model resident | 2.67x |
| ORPO | Reference-free | 2.00x |
| All others | -- | 1.00x |

Precomputing the DPO reference logprobs in one offline pass both removes the reference model from memory and drops the extra reference forward from every step. It is standard practice and defaults on.

#### Auxiliary model copies

| Method | Frozen copies | Trainable copies | Multiplier vs full-FT model state |
|--------|--------------|------------------|-----------------------------------|
| SFT / full FT / LoRA family | 0 | 0 | 1.00x |
| DPO, reference precomputed | 0 | 0 | 1.00x |
| DPO, reference resident | 1 | 0 | 1.125x |
| ORPO | 0 | 0 | 1.00x |
| GRPO, `beta = 0` (the TRL default) | 0 | 0 | 1.00x |
| GRPO, `beta > 0` | 1 | 0 | 1.125x |
| PPO | 2 (reward, reference) | 1 (critic) | 2.25x |

A frozen copy costs only its weight bytes -- it is inference-only and quantizable or offloadable. The PPO critic is trainable and carries its own gradients, master copy and optimizer moments.

#### QLoRA throughput

QLoRA dequantizes every 4-bit weight to 16-bit on each matmul. Benchmark studies put the penalty at 0.7-0.8x LoRA throughput; some report far less with recent bitsandbytes. The engine applies **0.75** and says so in a warning. This is the single widest assumption in the supervised path and the reason QLoRA carries medium-low confidence.

Note that the QLoRA row always forces NF4 base weights regardless of the Base weight precision you selected -- that is what QLoRA *is*. The other rows keep your selection.

---

### 4.4 Goal Presets

Users have "2,000 support transcripts", not a token count. Presets fill the raw workload and method fields and nothing else; every field stays visible and editable, and editing one clears the preset.

| Preset | Examples | Avg tokens | Epochs | Seq len | Method | Basis |
|--------|---------|-----------|--------|---------|--------|-------|
| Style / tone adoption | 1,000 | 600 | 3 | 1,024 | LoRA r=16 | LIMA: 1,000 curated pairs sufficed for strong alignment |
| Narrow task (classification) | 500 | 400 | 3 | 512 | QLoRA r=16 | 100-500 examples is the published floor |
| Instruction following | 10,000 | 800 | 2 | 2,048 | LoRA r=32 | Broad tasks need 5-20K examples |
| Tool use / function calling | 2,000 | 1,200 | 3 | 4,096 | LoRA r=32 | ~2K traces at r=32 |
| Reasoning distillation | 20,000 | 4,000 | 2 | 8,192 | LoRA r=64 | Long outputs drive sequence length, not example count |
| Domain adaptation | 200,000 | 2,000 | 1 | 4,096 | Full fine-tune | Continued-pretraining regime: 10^8-10^10 tokens, 1 epoch |

**Epochs: 1-3 is standard.** Beyond 5 the model starts memorizing exact phrasings, and the calculator warns. A fine-tuning set of 10^2-10^5 examples is vanishingly small against a multi-trillion-token pretraining corpus, so more passes buy memorization rather than generalization.

---

### 4.5 Input Reference

The pane is laid out as five numbered sections, matching the order of the calculation.

#### 1. Which model?

| Input | Description | Default | Notes |
|-------|-------------|---------|-------|
| Base model | Catalogued open-weight model, grouped by family | Llama 3.1 8B | 21 models plus Custom; MoE entries show total/active in the option label |
| Total parameters (B) | Custom only | 8 | Memory follows this |
| Active parameters (B) | Custom only | 8 | Compute follows this |
| Layers, Hidden size, FFN dim, KV heads, Head dim | Custom only, each optional | blank | Blank means "unknown": the engine estimates it and lowers confidence rather than computing adapter parameters off a plausible-looking wrong number |

#### 2. What are you teaching it?

The six goal presets of Section 4.4.

#### 3. Workload

| Input | Description | Default | Presets |
|-------|-------------|---------|---------|
| Training examples | Number of training samples | 2,000 | 500, 1K, 2K, 10K, 20K, 200K |
| Avg tokens / example | Mean length of one example | 800 | -- |
| Epochs | Passes over the dataset | 3 | 1, 2, 3, 5 |
| Sequence length | Max sequence length | 2,048 | 512, 1K, 2K, 4K, 8K, 16K |
| Micro-batch size (per GPU) | Sequences per forward pass per GPU | 1 | -- |
| Sequence packing | Concatenate samples to eliminate padding | On | On = 0.95 efficiency, off = 0.55 |

Sequence length and micro-batch size are the **only** inputs that move activation memory, which is the only batch-dependent term in the whole memory model.

#### 4. Method & configuration

| Input | Description | Default | Options |
|-------|-------------|---------|---------|
| Method | Selects the matrix row | LoRA | Full fine-tune, LoRA, QLoRA, DoRA, DPO, ORPO, GRPO, PPO |
| Optimizer | Sets master-copy and moment bytes | AdamW (fp32 master) | AdamW fp32/bf16 master, 8-bit Adam, Adafactor, SGD + momentum, LOMO |
| Base weight precision | Bytes per resident weight | BF16 (2 B/param) | BF16 2 B, FP8 1 B, NF4 4-bit 0.5 B |
| LoRA rank | Adapter rank; parameters scale linearly with it | 16 | 8, 16, 32, 64, 128 |
| LoRA targets | Which projections get adapters | All linear (q,k,v,o,gate,up,down) | All linear; Attention + output (q,k,v,o); Attention only (q,v) |
| Sharding strategy | How state is distributed when it exceeds one GPU | FSDP / ZeRO-3 | DDP (replicated, cannot reduce the floor), ZeRO-2, FSDP/ZeRO-3 |
| Gradient checkpointing | Recompute activations | On | Trades `+2N` FLOPs/token for a large activation-memory saving |
| Precompute reference logprobs | DPO / ORPO only | On | Offline reference pass; removes the reference from memory |
| Availability | Useful wall-clock fraction | 0.95 | 0.1-1.0 |
| Overhead factor | Multiplier for non-overlapped overhead | 1.10 | >= 1.0 |
| Pricing tier | Which rate column to bill at | On-demand | On-demand, Reserved, Spot |

LoRA rank and targets are inert for the non-PEFT methods, which train the full parameter set; the pane says so rather than silently ignoring them.

#### Optimizer bytes

| Optimizer | Master copy | Moments | Full-FT model state |
|-----------|------------|---------|---------------------|
| AdamW (fp32 master) | 4 B/param | 8 B/param | **16 B/param** (2 weights + 2 gradients + 4 + 8) |
| AdamW (bf16 master) | 2 | 4 | 10 B/param |
| 8-bit Adam | 2 | 2 | 8 B/param |
| Adafactor | 2 | 2 | 8 B/param |
| SGD + momentum | 2 | 4 | 10 B/param |
| LOMO (fused, no state) | 0 | 0 | 4 B/param |

The two conventions in the literature -- 12 versus 16 bytes per parameter for full fine-tuning -- differ only on whether the FP32 master copy is counted separately. The calculation trace shows the breakdown so the number is auditable rather than asserted.

LoRA-family adapters are FP32 in every mainstream implementation regardless of base precision, so each trainable adapter parameter costs 4 B weight + 4 B gradient + the optimizer moments -- **16 B/param under AdamW**.

#### 5. GPU columns

All 15 built-in SKUs from Section 2.9 are selectable as matrix columns. The defaults -- H100 SXM (80 GB), H200 SXM (141 GB) and B200 SXM (180 GB) -- are three distinct memory tiers, chosen so the fit column actually varies. Custom GPUs belong to the Training pane and do not appear here.

---

### 4.6 Model Catalog

The catalog carries per-model architecture, not just a parameter count, because adapter parameters and activation memory both depend on the layer and hidden dimensions.

Architecture dimensions are transcribed field by field from each model's own `config.json`, never from survey articles -- surveys report parameter counts reliably and architecture dimensions only sporadically. `totalParameters` is the safetensors parameter total. For MoE entries, `ffnDim` is the per-expert `moe_intermediate_size` and `activeParameters` is the published per-token activated count.

| Family | Entries |
|--------|---------|
| Llama | Llama 3 8B, Llama 3.1 8B / 70B / 405B, Llama 4 Scout 109B-A17B |
| Qwen | Qwen 2.5 7B / 32B / 72B, Qwen3 235B-A22B, Qwen3.8 27B |
| Mistral | Mistral 7B v0.3, Mistral Small 3 24B, Mixtral 8x7B, Mistral Small 4 119B-A6B |
| Gemma | Gemma 2 9B / 27B, Gemma 4 31B, Gemma 4 26B-A4B |
| Phi | Phi-4 14B |
| DeepSeek | DeepSeek-V3 671B-A37B |
| GLM | GLM-5.2 744B-A40B |
| Custom | user-supplied parameters and optional dimensions |

Catalog last updated **2026-10-14**.

**Verified vs unverified dimensions.** Most entries are marked verified: their dimensions were confirmed against a published config and describe a uniform GQA + SwiGLU stack, so the exact adapter formula applies and full fine-tune / LoRA estimates can earn high confidence. Four entries -- DeepSeek-V3, Mistral Small 4, GLM-5.2 and Qwen3.8 27B -- are faithfully recorded but are *not* uniform GQA + SwiGLU stacks (latent attention, hybrid linear attention). The engine uses its fallback dimensions there, warns that they are unconfirmed, and caps confidence at medium.

---

### 4.7 The Math (Formulas)

#### Memory decomposition

```
M_weights   = N_total x b_w              b_w: BF16 2, FP8 1, NF4 0.5 bytes
M_grads     = N_train x 2                (4 for LoRA adapters, which are FP32)
M_optimizer = N_train x (master + moments)
M_act       = see below
M_aux       = frozen_copies x N_total x b_w
            + trainable_copies x N_total x (b_w + 2 + master + moments)

subtotal    = M_weights + M_grads + M_optimizer + M_act + M_aux
M_overhead  = 1.5 GB + 0.08 x subtotal   (CUDA context, fragmentation, comm buffers)
M_total     = subtotal + M_overhead
```

`N_total` is every resident parameter including all MoE experts; `N_train` is what the method actually trains -- the full parameter set, or adapter parameters only for LoRA, QLoRA and DoRA.

The model reproduces the published VRAM anchors it was built against: 7B 88/20/8 GB, 14B 174/35/14, 32B 394/76/28, 70B 860/159/52 (full fine-tune / LoRA r=64 / QLoRA), MoE-30B-A3B 105/69/21.

#### Activation memory

FlashAttention is assumed -- universal in 2026 fine-tuning stacks -- so the quadratic `5as/h` attention-matrix term of Megatron's `sbh(34 + 5as/h)` drops out:

```
bs = micro_batch_size x sequence_length

with checkpointing (default):  M_act = 2 x bs x d x L  +  34 x bs x d
without checkpointing:         M_act = 34 x bs x d x L
```

The checkpointed form is the stashed layer boundaries plus the peak working set of the single layer being recomputed. Gradient checkpointing buys roughly a 40-60% activation-memory reduction for `+2N` FLOPs per token.

#### Capacity and the GPU floor

```
usable_per_GPU = memory_GB x 1e9 x 0.85 - 1.5 GB

M_total <= usable_per_GPU                      -> 1 GPU, FITS
DDP and it does not fit                        -> OOM (DDP replicates, it cannot shard)

otherwise, with ZeRO-2 or FSDP/ZeRO-3:
  shardable = M_weights + M_grads + M_optimizer + M_aux
  per_GPU_fixed = M_act + 1.5 GB               (activations do not shard)
  smallest n in {1, 2, 4, 8} with shardable/n + per_GPU_fixed <= usable_per_GPU
  beyond one node: n = ceil(ceil(shardable / (usable - per_GPU_fixed)) / 8) x 8
```

Activations are per-GPU and are deliberately excluded from the shardable pool -- sharding model state does not shrink them.

#### LoRA adapter parameters

For a linear layer `d_in -> d_out`, LoRA adds `A (r x d_in)` and `B (d_out x r)`, so `r(d_in + d_out)` parameters. Over a GQA transformer:

```
per_layer = r x [ (d + d)      q
                + (d + d_kv)   k
                + (d + d_kv)   v
                + (d + d)      o
                + (d + d_ffn)  gate
                + (d + d_ffn)  up
                + (d_ffn + d)  down ]

N_adapter = L x per_layer

all linear:         all seven projections
attention + output: q, k, v, o
attention only:     q, v
DoRA:               N_adapter x 1.01   (magnitude vectors)
```

Validation: Llama 3 8B (L=32, d=4096, d_kv=1024, d_ffn=14336) at r=16 all-linear yields **41,943,040** adapter parameters -- PEFT's own reported figure for that configuration.

#### Architecture fallback

When the catalog has no dimensions -- a Custom model with the optional fields left blank -- the engine estimates them from the parameter count, using the empirical `d ~= 128L` aspect ratio observed across the Llama 3 family. With `N = 12Ld^2` and `d = 128L`, `N = 196,608 L^3`:

```
L     = round( (N_active / 196,608)^(1/3) )
d     = 128 x L
d_ffn = 3.5 x d
d_kv  = 0.2 x d
```

**Accuracy caveat.** This is a shape heuristic, not a lookup. It gives L=34 for 8B (actual 32), L=71 for 70B (actual **80**), and L=127 for 405B (actual 126). The 70B case is 11% low. The engine warns whenever the fallback is used and states that adapter-parameter and activation estimates then carry **+/-40%** uncertainty. Supply the real layer count and hidden size if you have them.

#### Compute and wall-clock

```
flops_per_token  = (4 if LoRA/QLoRA/DoRA else 6) x N_active
                 + (2 x N_active if gradient checkpointing)

dataset_tokens   = examples x avg_tokens_per_example
tokens_processed = dataset_tokens x epochs x objective_multiplier / packing_efficiency
total_FLOPs      = tokens_processed x flops_per_token

effective_MFU    = base_MFU x method_factor x scaling_efficiency
sustained        = bf16_dense_peak x effective_MFU x availability
wall_clock_hours = total_FLOPs x overhead_factor / (GPUs x sustained) / 3600
cost             = hourly_rate_per_GPU x GPUs x wall_clock_hours
```

| Term | Values |
|------|--------|
| `base_MFU` by scale | 0.40 single GPU, 0.30 sharded within one node (2-8), 0.27 multi-node |
| `method_factor` | 1.00, except QLoRA 0.75 and DoRA 0.90 |
| `scaling_efficiency` | 1.0 up to 8 GPUs, then `0.90^log2(GPUs/8)` |
| `objective_multiplier` | DPO 2.00 (precomputed reference) or 2.67 (resident), ORPO 2.00, all others 1.00 |
| `packing_efficiency` | 0.95 packed, 0.55 unpacked |

**Fine-tuning MFU is lower than pretraining's** -- the Training pane's NVIDIA defaults sit at 0.39-0.42 (Section 2.9) -- because fine-tuning runs smaller batches and shorter sequences and so utilizes the hardware less well. The three values were back-solved from NeMo's published throughputs; the scaling efficiency was measured from NeMo's 8-to-16 GPU rows (70B LoRA 90.2%, 32B 92.1%).

**Throughput always uses the BF16 dense peak**, even when the base weight precision is FP8 or NF4. Quantization in this pane buys *memory*, not speed -- and for QLoRA it costs speed, via the 0.75 factor.

---

### 4.8 MoE Models

For a sparse mixture-of-experts model the two parameter counts do different jobs, and conflating them is the most common sizing error on 2026 open-weight models:

- **Memory follows `N_total`.** Every expert and the router stay resident. There is no "only the active experts are loaded" -- routing is per token, so all of them must be in memory.
- **Compute follows `N_active`.** FLOPs per token are driven by the parameters actually activated.

The gap is 3.6x on Mixtral 8x7B, 10.6x on Qwen3 235B-A22B, 18.1x on DeepSeek-V3 and 19.9x on Mistral Small 4 -- so treating total as active overstates FLOPs by up to ~20x across this catalog, and more on the sparsest frontier models. The pane states the ratio for the selected model and the engine emits a warning naming both counts.

> **Do not 4-bit QLoRA an MoE.** bitsandbytes lacks native MoE support, and quantization interferes with expert routing. Use BF16 LoRA instead. The engine emits this as a hard warning whenever an MoE model is combined with NF4 weights.

MoE models in the catalog: Mixtral 8x7B (46.7B/12.9B), Qwen3 235B-A22B, Llama 4 Scout 109B-A17B, Gemma 4 26B-A4B, DeepSeek-V3 671B-A37B, Mistral Small 4 119B-A6B, GLM-5.2 744B-A40B.

Note that the Training Calculator's effective-MFU formula also carries a MoE factor of 0.85 (Section 2.4), but the Training pane has no MoE input and always evaluates the dense factor of 1.00. The total/active split is modelled here, in the Fine-Tuning pane.

---

### 4.9 RL Post-Training (GRPO / PPO)

This is the least certain part of the calculator, and the documentation is going to be blunt about why.

#### What the matrix rows actually contain

The GRPO and PPO rows compute the **training side only** -- the gradient steps over the tokens your workload describes, at `6N` per token, plus the auxiliary model copies of Section 4.3. They do not include rollout generation. Every GRPO and PPO cell carries this warning:

> RL post-training cost is dominated by rollout generation (60-80% of step time), not gradient steps. This figure covers the training side only -- use the RL rollout estimator for the generation cost and the inference/training GPU split. Treat as a lower bound.

The rollout estimator referred to there is implemented in the engine and covered by tests, but **is not yet surfaced as a panel in the user interface**. The rest of this section documents how it behaves, because the constants and warnings below are the ones the engine will apply when it is wired up -- and because the shape of the problem matters even if you only read the matrix.

#### RL post-training is an inference problem wearing a training problem's clothes

Each pane has one binding constraint. RL has two in series, and the inference one is bigger:

| Pane | Binding constraint | The question |
|------|-------------------|--------------|
| Training | Compute x deadline | How many GPUs to finish on time? |
| Fine-tuning | Memory capacity | Does the model state fit at all? |
| RL post-training | **Decode bandwidth first, memory capacity second** | **What fraction of my GPUs should be generating rather than training?** |

Rollout generation is **60-80% of step time**. Five independent measurements agree on that band for long chain-of-thought workloads; NeMo-RL's shorter-completion benchmarks put it at 52-64%. Both are right -- the share rises with completion length and with a faster training backend.

So the output is a **GPU split**, not a GPU count. The engine reports a recommended inference share of `r / (1 + r)`, where `r = T_generation / T_training`, and names the published fleets that bracket it: Olmo 3 32B used 20 inference to 8 training nodes (71%), ROLL Flash 24/16 (60%), OpenRLHF 70B 16 vLLM to 16 training (50%).

Sanity bands for `r`, keyed on max completion length: 1.4 at <= 1,024 tokens, 2.5 at <= 4,096, 4.0 at <= 16,384, 6.0 above that. The engine computes `r` from the decode roofline rather than looking it up, and warns when the computed value falls outside 0.8-8.

#### The KV cache usually exceeds the entire model state

This inverts the supervised case. In fine-tuning, model state is the binding memory term and activations are a modest add-on. In RL, the KV pool for `prompts_per_step x group_size` concurrent sequences is routinely larger than weights, gradients and optimizer state combined. The engine warns when the KV pool at full concurrency exceeds `N_total x 16` bytes:

> For most RL configurations KV cache, not model state, is the binding memory term -- the opposite of supervised fine-tuning. Consider FP8 KV (halves it) or fewer prompts per step.

When the pool cannot hold every requested sequence at once, rollouts run in whole **waves**, and the engine says how many and how much VRAM full concurrency would want. Concurrency is also capped at the decode roofline knee (~280 on H100 BF16); past that, extra concurrency buys no throughput.

Generation time is modelled as **decode-bandwidth-bound**, streaming weights plus live KV per decode step -- not as `2N x tokens`, which would be optimistic by 10-30x. Prefill is modelled as compute-bound, with prefix caching on by default: all `G` completions in a GRPO group share the prompt prefix, so prefill runs once per prompt rather than once per completion, a `(G-1)/G` saving.

For MLA models (DeepSeek-V3, GLM-5.2, Mistral Small 4) the KV cache is approximated at 1/8 of the GQA equivalent and explicitly flagged as unreliable.

#### The estimate is a floor

Two realization factors sit between the roofline and reality: `eta_decode = 0.60` for a standalone serving engine, and a further `eta_rollout = 0.45` specific to *synchronous RL* rollout. The second one is a construction, not a published figure. It was back-solved from two anchors that differ by exactly 2x -- NeMo-RL 70B on 64 H100s (0.56) and a 7B vLLM run (0.28) -- and no source states it at all. Its causes are real enough: the effective batch decays as sequences finish so the tail sequence gates the step; colocated engines get a small memory budget; CUDA graphs are re-captured after each weight refit.

Compounding that, against **the one well-documented 70B RL anchor we could source -- Tulu 3's 2,880 GPU-hours** -- the model comes out 3.8x low. The engine rounds this in its own uncalibrated warning:

> Against the one published 70B RL run we could source (Tulu 3, 2,880 GPU-hours) this model runs ~4x optimistic -- treat the wall-clock as a FLOOR, not a prediction.

Part of the gap is explicable (PPO versus GRPO, critic and reward model, a 2024-era stack, unknown step count) and part is not. Until a second 70B anchor exists, treat the GRPO wall-clock as a lower bound.

#### Calibration is the highest-value input you can supply

One number -- **observed seconds per step from your own stack, at your configuration** -- overrides the model entirely. It collapses the three dominant uncertainties (`eta_rollout`, policy MFU and the long-tail factor) into a single measured constant, and moves the estimate from low to medium confidence. The engine then reports your measurement, its own prediction, and the ratio between them, so you can see how far off the model was and by which direction.

If you are going to supply one thing, supply this.

#### Tail gating

A strictly synchronous GRPO step pays for its **slowest** sequence: every completion in a wave must finish before the trainer starts. So the default gating is `max` -- every sequence is assumed to run to the completion cap. That is correct for strict synchronous GRPO and it is the conservative end.

But if completions genuinely cluster below the cap *and* your stack does in-flight replacement under continuous batching, the truth is nearer the mean. The same workload spans roughly **10x** between these two readings: NeMo-RL's 70B row has a mean completion of 398 tokens against a 4,096 cap.

| Gating | Completion length paid for |
|--------|---------------------------|
| `max` (default) | the full cap |
| `mean` | cap x mean completion ratio (default 0.35) |
| `blend` | cap x (0.5 + 0.5 x mean ratio) |

The engine warns when gating is `max` and the cap exceeds 4,096 tokens, naming the up-to-10x band rather than hiding it. The choice is exposed rather than guessed at, because guessing it would be the single largest unstated error in the output.

#### Weight sync and placement

Each step refits the trainer's weights into the inference engine. The engine models this at an effective **10 GB/s**, back-solved from NeMo-RL's measured 141 GB in 14 seconds at 70B on 64 GPUs. That figure covers the whole refit -- re-sharding and CUDA-graph recapture -- not just the NCCL collective, which alone is 20-500 ms. At 70B it is ~14 s per step: 70 minutes of pure weight shuffling over a 300-step run, and the strongest numerical argument for LoRA-based RL, which syncs only adapters and pays roughly zero.

Placement matters for memory: a **colocated** engine holds a second weight copy and gets a small memory budget -- TRL defaults `vllm_gpu_memory_utilization` to 0.3 against 0.90 standalone -- so sleep/wake offload is what makes colocation viable at all. **Disaggregated** placement uses the standard 0.85 usable fraction.

#### PPO versus GRPO

PPO samples **one** completion per prompt rather than `G`, so its rollout is far smaller; but its training step is roughly 2.33x GRPO's (a critic forward and backward, plus a reward-model forward), so its `r` is roughly GRPO's halved. PPO also holds four model copies -- policy, critic, reference, reward -- two of them trainable. GRPO removes the critic via group-relative advantage, trading a compute-bound component for a bandwidth-bound one, which is exactly why the modern default is GRPO-family and why this pane has to be bandwidth-first.

---

### 4.10 Warnings

#### Fine-tuning warnings (shown on the selected cell)

| Warning | Trigger | What To Do |
|---------|---------|------------|
| Does not fit, DDP | Fit state OOM with DDP sharding | DDP replicates model state, so one GPU must hold all of it. Switch to FSDP/ZeRO-3, or use a lower-memory method. |
| Does not fit at any count | Fit state OOM under sharding | Use QLoRA, a smaller rank, or shorter sequences. |
| Sharding required | Fit state SHARDED | Names the GPU count, the sharding strategy and the total model state. |
| MoE model | Catalogued model is MoE | Memory follows total parameters, compute follows active; experts and router stay resident. |
| MoE + 4-bit | MoE model with NF4 weights | Do not 4-bit QLoRA an MoE -- bitsandbytes lacks native MoE support and quantization interferes with expert routing. Use BF16 LoRA. |
| Estimated dimensions | Catalog has no layers / hidden / FFN / KV dims | Dimensions were estimated from parameter count (`d ~= 128L`). Adapter and activation estimates carry +/-40%. |
| Unverified dimensions | Dimensions present but not confirmed against a published config | Verify the layer count, hidden size and FFN dimension before committing hardware. |
| Memorization risk | Epochs > 5 | 1-3 epochs is standard; fine-tuning sets are tiny against a pretraining corpus. |
| Below the example floor | Examples < 100 | Published guidance puts the minimum for a narrow task at 100-500. |
| Activation-dominated | Sequence length > 16,384 | Activations now dominate model state. Consider chunked or balance-packed long-context training. |
| Packing disabled | Sequence packing off | Padding waste assumed at 45%. Packing yields 3.8x at 32K and 7.3x at 128K sequence length versus naive batching. |
| QLoRA throughput | Method is QLoRA | The 0.75 dequantization factor is applied; sources span 0.7-0.8x and some report far less penalty with recent bitsandbytes. |
| Full FT footprint | Full fine-tune with AdamW (fp32 master) | States the full-FT state size against roughly what QLoRA would need on the same model. |
| RL lower bound | Method is GRPO or PPO | Rollout generation is 60-80% of step time and is not in this figure. Treat as a lower bound. |
| High LoRA rank | PEFT method with rank > 128 | Adapter parameters scale linearly with rank and the quality return flattens well below this. |

#### RL rollout estimator warnings (engine; not yet surfaced in the UI)

| Warning | Trigger |
|---------|---------|
| Weights exceed the inference pool | Model weights exceed usable memory on the inference GPUs |
| Rollouts run in waves | KV pool holds fewer concurrent sequences than requested; names the wave count and the VRAM full concurrency wants |
| Past the roofline knee | Concurrency exceeds ~280; decode is compute-bound beyond it |
| Ratio outside the band | Computed `r` falls outside 0.8-8 |
| Rollout share | Always emitted; states the generation share and the published fleet splits that bracket it |
| KV exceeds model state | KV pool at full concurrency > `N_total x 16` bytes |
| Colocated memory budget | Colocated placement; names the 0.3 vs 0.90 utilization default |
| Calibrated | An observed seconds-per-step was supplied; reports the ratio to the modelled figure |
| Uncalibrated floor | No observation supplied; names the Tulu 3 anchor and the ~4x optimism |
| Tail gating | Gating is `max` with a cap above 4,096 tokens; names the up-to-10x band |
| Weight refit cost | Non-PEFT method with refit > 1 s/step; totals the minutes lost over the run |
| MLA / estimated KV | KV bytes derived from latent attention or estimated dimensions |

---

### 4.11 Confidence Levels

Confidence is assigned per method, and then capped by how well the model's architecture is known.

| Method | Confidence | Why |
|--------|-----------|-----|
| Full fine-tune | **High** | Exact memory arithmetic; `6N` is well established |
| LoRA | **High** | Exact adapter enumeration; `4N` corroborated by the NeMo back-solve |
| DoRA | **Medium** | LoRA plus an approximated 1% magnitude-vector term and a 0.90 throughput factor |
| QLoRA | **Medium-low** | The dequantization throughput factor spans 0.7-0.8x across sources |
| DPO | **Medium-low** | Multiplier derived from exact pass counts, but memory depends on the reference strategy |
| ORPO | **Medium-low** | As DPO, reference-free |
| GRPO | **Low** | Rollout-dominated, and the figure shown covers the training side only |
| PPO | **Low** | As GRPO, plus critic and reward-model copies |

**The architecture cap.** If the selected model's dimensions were estimated from the parameter count, or are present but unconfirmed against a published config, full fine-tune and LoRA drop from High to **Medium** -- the adapter and activation terms are then approximations. The methods already at Medium or below are unaffected.

What is reliable, and what is not, within any single estimate:

| Quantity | Confidence |
|----------|-----------|
| Weights, gradients and optimizer memory | High -- exact arithmetic |
| LoRA adapter parameters from catalog dimensions | High -- exact enumeration |
| LoRA adapter parameters from the fallback | Medium -- the shape heuristic is +/-40% |
| Activation memory | Medium -- implementation-dependent constants |
| FLOPs per token | High -- `6N` / `4N`, corroborated by the NeMo back-solve |
| Wall-clock for LoRA and full fine-tuning in BF16 | Medium -- MFU validated against NeMo |
| QLoRA wall-clock | Medium-low |
| GRPO / PPO wall-clock and cost | Low -- and a floor, not a prediction |

---

### 4.12 Export

The Fine-Tuning pane has no export or share-URL support yet. The per-cell calculation trace is expandable and can be copied by hand. JSON, CSV, Markdown and share-URL export exist for the Training and Inference panes only -- see Section 2.16.

---

## 5. Custom GPUs

To add a custom GPU SKU:

1. Open the **Custom GPU** section in the left panel
2. Enter a label, dense BF16 peak (TFLOP/s), optional FP8 peak, memory (GB), and default MFU
3. Click "Add GPU"
4. The custom GPU appears in the GPU selector and can be used in all modes

Custom GPUs are stored in the browser session. They participate in comparison, sensitivity, and reverse solve like built-in SKUs.

---

## 6. Worked Examples

### 70B LLM, 30 Days, H100

**Inputs:**
- Model: 70B parameters (N = 7.0 x 10^10)
- Training window: 30 days
- TPP: 20 (Chinchilla)
- GPU: H100 SXM
- MFU: 40%
- Availability: 90%
- Overhead: 1.10x
- Precision: BF16

**Step 1: Token target**
```
D = 20 * 70e9 = 1.4e12 tokens (1.4 trillion)
```

**Step 2: Base FLOPs**
```
C_base = 6 * 70e9 * 1.4e12 = 5.88e23 FLOPs
```

**Step 3: Total FLOPs with overhead**
```
C_total = 5.88e23 * 1.10 = 6.47e23 FLOPs
```

**Step 4: Sustained throughput per H100**
```
sustained = 989.5e12 * 0.40 * 0.90 = 3.56e14 FLOP/s
```

**Step 5: Training window in seconds**
```
seconds = 30 * 86,400 = 2,592,000
```

**Step 6: Required GPUs**
```
GPUs = ceil(6.47e23 / (2,592,000 * 3.56e14)) = 701
```

**Result: 701 H100 SXM GPUs.**

Sensitivity around this estimate:

| MFU | H100 GPUs |
|-----|-----------|
| 30% | ~935 |
| 40% | ~701 |
| 50% | ~561 |

---

## 7. References

- **[R1]** Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), arXiv:2203.15556, 2022. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- **[R2]** Epoch AI, *Chinchilla scaling: A replication attempt*, Apr. 2024. [https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)
- **[R3]** NVIDIA H100 GPU specifications. [https://www.nvidia.com/en-us/data-center/h100/](https://www.nvidia.com/en-us/data-center/h100/)
- **[R4]** NVIDIA H200 GPU specifications. [https://www.nvidia.com/en-us/data-center/h200/](https://www.nvidia.com/en-us/data-center/h200/)
- **[R5]** NVIDIA HGX B200 specifications. [https://www.nvidia.com/en-us/data-center/hgx/](https://www.nvidia.com/en-us/data-center/hgx/)
- **[R6]** NVIDIA GB200 NVL72 specifications. [https://www.nvidia.com/en-us/data-center/gb200-nvl72/](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- **[R7]** Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448, 2024. [https://arxiv.org/abs/2401.00448](https://arxiv.org/abs/2401.00448)
- **[R8]** Marin Project, *Marin: an open-source research framework for training language models*, 2025. [https://github.com/marin-community/marin](https://github.com/marin-community/marin)
- **[R9]** Liang, Percy et al., Stanford CRFM / Marin: *Building open language models*, 2025. [https://crfm.stanford.edu/](https://crfm.stanford.edu/)
- **[R10]** Touvron et al., *Llama 2: Open Foundation and Fine-Tuned Chat Models*, arXiv:2307.09288, 2023. [https://arxiv.org/abs/2307.09288](https://arxiv.org/abs/2307.09288)
- **[R11]** Meta AI, *The Llama 3 Herd of Models*, arXiv:2407.21783, 2024. [https://arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)
- **[R12]** Patel et al., *Splitwise: Efficient generative LLM inference with disaggregated prefill and decode*, Azure LLM Inference Workload Characterization, arXiv:2512.01644, 2024. [https://arxiv.org/pdf/2512.01644](https://arxiv.org/pdf/2512.01644)
- **[R13]** OpenRouter, *State of AI 2025*, 2025. [https://openrouter.ai/state-of-ai](https://openrouter.ai/state-of-ai)

#### Fine-Tuning and RL Post-Training

- **[R14]** NVIDIA, *NeMo AutoModel Performance Summary* -- the LoRA tok/s/GPU table used to back-solve the fine-tuning MFU defaults. [https://docs.nvidia.com/nemo/automodel/latest/performance/performance-summary](https://docs.nvidia.com/nemo/automodel/latest/performance/performance-summary)
- **[R15]** Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, arXiv:2305.14314. [https://arxiv.org/pdf/2305.14314](https://arxiv.org/pdf/2305.14314)
- **[R16]** Korthikanti et al., *Reducing Activation Recomputation in Large Transformer Models*, MLSys 2023 -- the `sbh(34 + 5as/h)` activation model. [https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)
- **[R17]** Dettmers et al., *8-bit Optimizers via Block-wise Quantization*, arXiv:2110.02861. [https://arxiv.org/pdf/2110.02861](https://arxiv.org/pdf/2110.02861)
- **[R18]** Lv et al., *LOMO: Full Parameter Fine-tuning with Limited Resources*, arXiv:2306.09782. [https://arxiv.org/pdf/2306.09782](https://arxiv.org/pdf/2306.09782)
- **[R19]** *Hierarchical Balance Packing for Long-Context SFT*, arXiv:2503.07680 -- the 3.8x at 32K / 7.3x at 128K packing speedups. [https://arxiv.org/pdf/2503.07680](https://arxiv.org/pdf/2503.07680)
- **[R20]** Lambert et al., *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*, arXiv:2411.15124 -- the 70B RL anchor of 2,880 GPU-hours. [https://arxiv.org/pdf/2411.15124](https://arxiv.org/pdf/2411.15124)
- **[R21]** *ROLL Flash*, arXiv:2510.11345 -- rollout share above 70% of step time. [https://arxiv.org/html/2510.11345](https://arxiv.org/html/2510.11345)
- **[R22]** *RollPacker*, arXiv:2509.21009 -- 70% generation / 21-23% training step breakdown, and tail-batching speedups. [https://arxiv.org/pdf/2509.21009](https://arxiv.org/pdf/2509.21009)
- **[R23]** *Olmo 3*, arXiv:2512.13961 -- the 20-inference / 8-training node RLVR fleet split. [https://arxiv.org/html/2512.13961v2](https://arxiv.org/html/2512.13961v2)

#### Classical Tabular / GBDT Coefficients

- **[R24]** Mitchell, *GBM-Benchmarks* -- the XGBoost / LightGBM / CatBoost GPU and CPU anchors. [https://github.com/RAMitchell/GBM-Benchmarks](https://github.com/RAMitchell/GBM-Benchmarks)
- **[R25]** LightGBM, *Experiments* -- the single-machine CPU cross-check. [https://lightgbm.readthedocs.io/en/latest/Experiments.html](https://lightgbm.readthedocs.io/en/latest/Experiments.html)
- **[R26]** LightGBM, *GPU Performance* -- the small-data GPU penalty. [https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)
- **[R27]** CatBoost, *GPU vs CPU training speed benchmarks* -- the `max_bin` slope. [https://github.com/catboost/catboost/tree/master/catboost/benchmarks/gpu_vs_cpu_training_speed](https://github.com/catboost/catboost/tree/master/catboost/benchmarks/gpu_vs_cpu_training_speed)
- **[R28]** NVIDIA, *cuML accelerator benchmarks* -- the random-forest anchors. [https://docs.nvidia.com/cuml/latest/cuml-accel/benchmarks/](https://docs.nvidia.com/cuml/latest/cuml-accel/benchmarks/)
- **[R29]** *TabArena*, arXiv:2506.16791 -- the 200 configs x 8 inner folds search scale. [https://arxiv.org/abs/2506.16791](https://arxiv.org/abs/2506.16791)

#### Accelerator Specifications

- **[R30]** SemiAnalysis, *NVIDIA B300* -- the B300 FP4-and-memory-only uplift. [https://inferencex.semianalysis.com/chips/b300](https://inferencex.semianalysis.com/chips/b300)
- **[R31]** Lenovo, *ThinkSystem HGX B200 product guide* -- the per-GPU B200 derivation. [https://lenovopress.lenovo.com/lp2226.pdf](https://lenovopress.lenovo.com/lp2226.pdf)
- **[R32]** Google Cloud, *TPU v6e* and *TPU7x* documentation. [https://docs.cloud.google.com/tpu/docs/v6e](https://docs.cloud.google.com/tpu/docs/v6e)

Per-SKU sources, including the honesty markers on B300 and MI450, are recorded in the notes field of each SKU and shown in the calculator.
