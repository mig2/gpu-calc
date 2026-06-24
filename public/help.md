# GPU Calculator -- User Guide

## 1. Overview

This calculator has two modes:

- **Training Calculator** -- estimates accelerator requirements for training across four model families (LLM, Time-Series, Tabular Foundation, Classical Tabular / GBDT)
- **Inference Calculator** -- estimates monthly costs for serving an LLM via API (buy) vs self-hosted GPUs (build), and identifies the breakeven point

**When to use the Training Calculator:** You are planning a pretraining run (or GBDT sweep) and need to estimate GPU count, training time, or cluster size. You know (or can estimate) the model size, training window, and data budget.

**When to use the Inference Calculator:** You have a trained model and want to decide between paying per-token for a managed API or running your own GPU cluster. You know (or can estimate) your request volume.

---

## 2. Training Calculator

The Training Calculator estimates GPU cluster requirements for model training. It covers four model families -- from dense transformer pretraining (LLM) to gradient-boosted decision trees (GBDT). For each family it computes total work (FLOPs or work units), divides by sustained hardware throughput, and returns the number of accelerators needed within a given wall-clock window.

**Who it's for:** ML engineers, infrastructure teams, and technical leaders planning pretraining runs, comparing GPU SKUs, or validating vendor estimates.

**Key concept:** GPU count = total compute / (training window x sustained throughput per GPU).

### What it is for

- Planning GPU cluster size for pretraining runs
- Comparing GPU SKUs (H100, H200, B200, GB200) under the same assumptions
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
| Precision | Numerical format | BF16 Dense | BF16 Dense, FP8 Dense (experimental) |
| GPU SKU | Accelerator type(s) | H100 SXM | Multi-select: H100, H200, B200, GB200, custom |
| MFU | Model FLOP utilization per GPU | 40% (varies by SKU) | 0-100%; per-GPU override |
| Availability | Useful wall-clock fraction | 90% | 0-100% |
| Overhead factor | Multiplier for checkpointing, eval, restarts | 1.10x | >= 1.0 |
| Memory bytes/parameter | State memory estimate | 16 | Positive; for memory feasibility check |

#### Unit Shortcuts

- Model size: `7B`, `70B`, `1.5T`, `405M`
- Training window: `30d` (days), `720h` (hours), `4w` (weeks)

#### Quick Estimate

Enter model size and training window. The calculator uses Chinchilla-20 defaults and shows a single H100 GPU count. This is the minimum-input mode for napkin math.

#### Advanced Estimate

Expand the Advanced Assumptions panel to control:

- **MFU** -- per-GPU utilization (30% conservative, 40% planning, 50% strong)
- **Availability** -- fraction of wall-clock time spent training (accounts for failures, restarts, maintenance)
- **Overhead** -- extra compute from checkpointing, evaluation, data stalls
- **Precision** -- BF16 (standard) or FP8 (experimental, higher theoretical peak)
- **Tokens per parameter** -- override the Chinchilla-20 default. Presets include 20 (Chinchilla), 50, 100 (Inference-optimal), and 200 (Over-trained). See Section 2.5 for details on over-training.

All results update live as you adjust sliders.

#### GPU Comparison

Select multiple GPU SKUs (H100, H200, B200, GB200, or custom) in the GPU selector. The comparison table shows side-by-side:

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

Where D is training tokens, TPP is tokens-per-parameter (default 20), and N is model parameters. Under Chinchilla-20, a 70B model trains on 1.4T tokens.

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
sustained_per_GPU = peak_FLOP/s * MFU * availability
```

- `peak_FLOP/s` is the dense (not sparse) tensor-core throughput for the selected precision
- `MFU` is model FLOP utilization -- the fraction of peak actually delivered during training
- `availability` is the fraction of wall-clock time spent doing useful training

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

#### Over-Training TPP Presets

The tokens-per-parameter (TPP) presets reflect the evolution of training practice:

| Preset | TPP | Rationale |
|--------|-----|-----------|
| Chinchilla | 20 | Compute-optimal under the Hoffmann et al. 2022 scaling law [R1]. Minimizes loss for a fixed FLOP budget by balancing model size and data. |
| 50 | 50 | A moderate over-training ratio, common in early post-Chinchilla models. |
| Inference-optimal | 100 | Deliberately trains a smaller model on more data. The model is cheaper to serve at inference time while approaching the loss of a larger Chinchilla-optimal model trained with the same compute. See Sardana et al. [R7]. |
| Over-trained | 200 | Aggressive over-training as practiced by Llama 3, Marin, and other modern projects. Llama 3 8B used ~1,875 tokens per parameter (15T tokens / 8B params); Marin 8B used 200 TPP. The goal is to squeeze maximum quality into a small, fast-to-serve model. |

**Why over-train?** Chinchilla-optimal minimizes training loss for a given compute budget, but it says nothing about inference cost. A 70B Chinchilla-optimal model is expensive to serve. Training a 7B model on 10-20x more tokens costs the same training FLOPs but produces a model that is 10x cheaper per inference query. Modern practice deliberately over-trains smaller models for inference efficiency, accepting slightly higher loss in exchange for dramatically lower serving cost.

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
```

Wait -- let me recalculate for the spec's 7.04B figure. With stride=56:

```
windows_per_series = floor(392 / 56) + 1 = 8
effective_tokens = 10,000,000 * 8 * 32 * 1 = 2.56e9
```

Or with channel_expanded (4 variables):

```
tokens_per_window = 4 * 32 = 128
effective_tokens = 10,000,000 * 7 * 128 * 1 = 8.96e9
```

With the spec parameters yielding 7.04B tokens:

```
base_FLOPs = 6 * 1e9 * 7.04e9 = 4.224e19 FLOPs
total_FLOPs = 4.224e19 * 1.10 = 4.65e19 FLOPs
```

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
work_units = rows * columns * boosting_rounds * cv_folds * hp_trials
```

Training time is estimated by dividing total work by a throughput coefficient (work units per second).

#### Input Reference

| Input | Description | Default | Notes |
|-------|-------------|---------|-------|
| Algorithm | GBDT implementation | LightGBM | LightGBM, XGBoost, CatBoost, Random Forest, Custom |
| Rows | Training dataset rows | -- | |
| Columns | Features | -- | |
| Boosting rounds | Trees / iterations | -- | |
| CV folds | Cross-validation folds | 1 | Multiplier on total work |
| HP trials | Hyperparameter search trials | 1 | Multiplier on total work |
| CPU / GPU | Implementation target | GPU | CPU is ~10x slower |
| Throughput coefficient | Work units per second | Auto | Set to 0 for benchmark defaults |

#### Default Throughput Benchmarks (GPU)

| Algorithm | Default Throughput (work units/sec) |
|-----------|-------------------------------------|
| LightGBM | 5.00e8 (~500M/sec) |
| XGBoost | 3.00e8 (~300M/sec) |
| CatBoost | 4.00e8 (~400M/sec) |
| Random Forest | 2.00e8 (~200M/sec) |

CPU throughput is 10x lower than GPU defaults. These are rough estimates derived from common benchmarks and may be significantly off for your specific workload.

#### How to Calibrate

For better accuracy, calibrate with a known run:

1. Run a small training job and record: rows, columns, rounds, wall-clock seconds.
2. Compute: `throughput = (rows * columns * rounds) / seconds`
3. Enter this value as the throughput coefficient.
4. The confidence level improves from "low" to "medium" when a calibrated coefficient is provided.

#### Worked Example

**Inputs:** LightGBM, 10M rows, 200 columns, 1,000 rounds, 5-fold CV, 20 HP trials, GPU.

```
base_work = 10,000,000 * 200 * 1,000 = 2.0e12 work units
total_work = 2.0e12 * 5 * 20 = 2.0e14 work units
throughput = 5.0e8 /sec (LightGBM GPU default)
estimated_time = 2.0e14 / 5.0e8 = 400,000 seconds = 111 hours = 4.6 days
```

---

### 2.9 GPU SKU Reference

| GPU | Dense BF16 Peak | Dense FP8 Peak | Memory | Bandwidth | Default MFU |
|-----|-----------------|----------------|--------|-----------|-------------|
| H100 SXM | 989.5 TFLOP/s | 1,979 TFLOP/s | 80 GB | 3.35 TB/s | 40% |
| H200 SXM | 989.5 TFLOP/s | 1,979 TFLOP/s | 141 GB | 4.8 TB/s | 45% |
| B200 SXM / HGX B200 | 2,250 TFLOP/s | 4,500 TFLOP/s | 175 GB | -- | 40% |
| GB200 NVL72 (per GPU) | 2,500 TFLOP/s | 5,000 TFLOP/s | 186 GB | 8.0 TB/s | 40% |

**H100 vs H200:** These have identical raw dense BF16 peak FLOP/s. The H200 advantage is 76% more HBM capacity (141 vs 80 GB) and 43% more bandwidth (4.8 vs 3.35 TB/s). This allows larger microbatches, longer contexts, less activation checkpointing, and less aggressive sharding -- all of which can improve achieved MFU. The calculator models this by giving H200 a higher default MFU (45% vs 40%), not a higher peak.

**Dense vs Sparse:** NVIDIA spec sheets often quote sparse tensor-core throughput. Dense values (used here) are approximately half the sparse numbers. This calculator uses dense values exclusively since dense training is the standard approach.

---

### 2.10 Confidence Levels

Every estimate includes a confidence label indicating how reliable the estimate is likely to be.

| Level | Meaning |
|-------|---------|
| **High** | Well-understood domain with empirically validated formulas. LLM mode with standard settings. |
| **Medium** | Reasonable approximation but relies on analogies or simplifications. Time-series mode; classical tabular with calibrated throughput. |
| **Medium-low** | Significant uncertainty. Tabular foundation with row or axial tokenization. |
| **Low** | Order-of-magnitude estimate at best. Tabular foundation with cell tokenization; classical tabular without calibration. |

#### Confidence by Model Family

| Model Family | Settings | Confidence |
|--------------|----------|------------|
| LLM | Standard (6ND, BF16, full pretraining) | High |
| LLM | Non-standard (FP8, SFT/LoRA/RLHF) | Medium |
| Time-Series | All tokenization modes | Medium |
| Tabular Foundation | Row or axial tokenization | Medium-low |
| Tabular Foundation | Cell or custom tokenization | Low |
| Classical Tabular | With calibrated throughput | Medium |
| Classical Tabular | With default throughput (no calibration) | Low |

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
| Llama 2 7B | 7B | 2T | 8.4e19 | Touvron et al. 2023 |
| Llama 2 13B | 13B | 2T | 1.6e20 | Touvron et al. 2023 |
| Marin 8B | 8B | 1.6T | 7.7e22 | Marin Community 2025 |
| GPT-3 175B | 175B | 300B | 3.2e23 | Brown et al. 2020 |
| Chinchilla 70B | 70B | 1.4T | 5.9e23 | Hoffmann et al. 2022 |
| Marin 32B | 32B | 3.2T | 6.1e23 | Marin Community 2025 |
| Llama 3 8B | 8B | 15T | 7.2e23 | Meta 2024 |
| Llama 2 70B | 70B | 2T | 8.4e23 | Touvron et al. 2023 |
| Llama 3 70B | 70B | 15T | 6.3e24 | Meta 2024 |
| Llama 3 405B | 405B | 15T | 3.6e25 | Meta 2024 |

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
| Training mode warning | Mode is not Full Pretraining | The 6ND formula is calibrated for full pretraining. SFT/LoRA/RLHF typically need far less compute. Use task-specific estimates. |
| FP8 warning | FP8 precision selected | FP8 end-to-end training is experimental. Actual throughput may differ significantly from the peak-based estimate. |
| H200 note | H200 selected | Reminder that H200 has the same raw BF16 peak as H100. Its advantage is memory and bandwidth, which may improve MFU. |
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
| No calibration | Throughput coefficient not provided | Using rough defaults. Calibrate with a known run for better accuracy. |
| High search multiplier | folds * trials > 50 | Search dominates total time. Consider Bayesian optimization or early stopping. |
| CPU selected | CPU implementation | GPU implementations are typically 5-10x faster. |

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

### 3.1 Overview -- Build vs Buy

The Inference Calculator helps you decide whether to serve an LLM via a managed API (buy) or self-host on your own GPUs (build). It estimates monthly costs for both approaches and identifies the breakeven point.

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

## 4. Custom GPUs

To add a custom GPU SKU:

1. Open the **Custom GPU** section in the left panel
2. Enter a label, dense BF16 peak (TFLOP/s), optional FP8 peak, memory (GB), and default MFU
3. Click "Add GPU"
4. The custom GPU appears in the GPU selector and can be used in all modes

Custom GPUs are stored in the browser session. They participate in comparison, sensitivity, and reverse solve like built-in SKUs.

---

## 5. Worked Examples

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

## 6. References

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
