# LLM Training GPU Calculator -- User Guide

## 1. Overview

This calculator estimates the number of GPUs needed to train a dense decoder-only transformer language model within a target time window. It implements the standard `C = 6ND` training FLOP model with configurable assumptions for MFU, availability, overhead, and precision.

**What it is for:**
- Planning GPU cluster size for pretraining runs
- Comparing GPU SKUs (H100, H200, B200, GB200) under the same assumptions
- Sensitivity analysis across MFU and training window
- Back-of-envelope validation of vendor or team estimates

**What it is NOT for:**
- Procurement commitments without empirical benchmarking
- MoE, multimodal, or non-transformer architectures
- Cost estimation or pricing (compute cost can be layered on top separately)
- Predicting model quality, loss, or downstream performance
- Detailed distributed-training parallelism planning (TP/PP/CP)

---

## 2. Quick Start

1. Enter a **model size** (e.g. `70B`) and a **training window** (e.g. `30d`).
2. The calculator immediately shows the required GPU count using default assumptions: Chinchilla-20 token target, H100 SXM, 40% MFU, 90% availability, 1.10x overhead, BF16 precision.
3. Select additional GPUs to compare, or open Advanced Assumptions to tune.

That's it for a quick estimate. Read on for the full feature set.

---

## 3. Input Reference

| Input | Description | Default | Valid Range |
|-------|-------------|---------|-------------|
| Model parameters | Number of trainable parameters | 70B | Positive; accepts M/B/T units |
| Training window | Wall-clock time budget | 30 days | Positive; accepts h/d/w units |
| Tokens per parameter (TPP) | Training tokens = TPP x N | 20 | Positive integer; presets 20, 50, 100 |
| Training mode | Type of training run | Full pretraining | Full pretraining, Continued pretraining, SFT, LoRA, RLHF, Distillation |
| Precision | Numerical format | BF16 Dense | BF16 Dense, FP8 Dense (experimental) |
| GPU SKU | Accelerator type(s) | H100 SXM | Multi-select: H100, H200, B200, GB200, custom |
| MFU | Model FLOP utilization per GPU | 40% (varies by SKU) | 0-100%; per-GPU override |
| Availability | Useful wall-clock fraction | 90% | 0-100% |
| Overhead factor | Multiplier for checkpointing, eval, restarts | 1.10x | >= 1.0 |
| Memory bytes/parameter | State memory estimate | 16 | Positive; for memory feasibility check |

### Unit Shortcuts

- Model size: `7B`, `70B`, `1.5T`, `405M`
- Training window: `30d` (days), `720h` (hours), `4w` (weeks)

---

## 4. Functional Modes

### 4.1 Quick Estimate

Enter model size and training window. The calculator uses Chinchilla-20 defaults and shows a single H100 GPU count. This is the minimum-input mode for napkin math.

### 4.2 Advanced Estimate

Expand the Advanced Assumptions panel to control:

- **MFU** -- per-GPU utilization (30% conservative, 40% planning, 50% strong)
- **Availability** -- fraction of wall-clock time spent training (accounts for failures, restarts, maintenance)
- **Overhead** -- extra compute from checkpointing, evaluation, data stalls
- **Precision** -- BF16 (standard) or FP8 (experimental, higher theoretical peak)
- **Tokens per parameter** -- override the Chinchilla-20 default

All results update live as you adjust sliders.

### 4.3 GPU Comparison

Select multiple GPU SKUs (H100, H200, B200, GB200, or custom) in the GPU selector. The comparison table shows side-by-side:

- Required GPU count per SKU
- H100-equivalent count (normalized comparison)
- Sustained FLOP/s per GPU
- Memory lower-bound GPUs

A bar chart visualizes the GPU counts for quick comparison.

### 4.4 Sensitivity Matrix

The sensitivity matrix shows GPU counts across a grid of MFU values (rows) and training windows (columns). This reveals how sensitive your estimate is to the two largest unknowns. Cells are heat-mapped so larger clusters stand out visually.

### 4.5 Reverse Solve

Two sub-modes:

**Time from GPU budget:** Given a fixed GPU count and model size, how long will training take?
- Enter your available GPU count
- The calculator returns training time in days

**Size from GPU budget:** Given a fixed GPU count and training window, what is the largest model you can train?
- Enter your available GPU count
- The calculator returns the maximum model size (uses the quadratic relationship: `N = sqrt(available_FLOPs / (6 * TPP * overhead))`)

### 4.6 Calibration

Back-solve achieved MFU from a known training run. Enter:

- Model size, tokens trained, wall-clock days, GPU count, GPU type
- The calculator returns the achieved MFU percentage

If the result is between 10-70%, it is considered reasonable. You can apply the calibrated MFU to your current scenario for more accurate planning.

### 4.7 Memory Feasibility

Every result includes a memory feasibility check. The calculator computes a lower-bound GPU count based on model state memory:

```
memory_lower_bound_GPUs = ceil(N * bytes_per_parameter / (GPU_memory * 0.85))
```

If the memory lower bound exceeds the compute requirement, the result is flagged as **memory-bound** rather than compute-bound. This indicates the model needs more GPUs than compute alone would suggest, due to memory pressure from optimizer states, gradients, and parameters.

---

## 5. The Math

### Token Target

```
D = TPP * N
```

Where D is training tokens, TPP is tokens-per-parameter (default 20), and N is model parameters. Under Chinchilla-20, a 70B model trains on 1.4T tokens.

### Training FLOPs

```
C = 6 * N * D
```

This is the standard first-order estimate for dense decoder-only transformer training. The factor of 6 accounts for the forward pass (2ND) and backward pass (4ND).

With overhead:

```
C_total = 6 * N * D * overhead_factor
```

### Sustained Throughput

```
sustained_per_GPU = peak_FLOP/s * MFU * availability
```

- `peak_FLOP/s` is the dense (not sparse) tensor-core throughput for the selected precision
- `MFU` is model FLOP utilization -- the fraction of peak actually delivered during training
- `availability` is the fraction of wall-clock time spent doing useful training

### GPU Count

```
GPUs = ceil(C_total / (training_window_seconds * sustained_per_GPU))
```

### The Quadratic Dependence

Because `D = TPP * N`, the total compute is:

```
C = 6 * N * TPP * N = 6 * TPP * N^2
```

Doubling the model size quadruples the compute. This is the single most important scaling intuition.

### H100-Equivalent Conversion

```
H100_equivalents = (required_GPUs * sustained_per_GPU) / sustained_per_H100
```

This normalizes any GPU type to an H100-equivalent count for apples-to-apples comparison.

---

## 6. GPU SKU Reference

| GPU | Dense BF16 Peak | Dense FP8 Peak | Memory | Bandwidth | Default MFU |
|-----|-----------------|----------------|--------|-----------|-------------|
| H100 SXM | 989.5 TFLOP/s | 1,979 TFLOP/s | 80 GB | 3.35 TB/s | 40% |
| H200 SXM | 989.5 TFLOP/s | 1,979 TFLOP/s | 141 GB | 4.8 TB/s | 45% |
| B200 SXM / HGX B200 | 2,250 TFLOP/s | 4,500 TFLOP/s | 175 GB | -- | 40% |
| GB200 NVL72 (per GPU) | 2,500 TFLOP/s | 5,000 TFLOP/s | 186 GB | 8.0 TB/s | 40% |

**H100 vs H200:** These have identical raw dense BF16 peak FLOP/s. The H200 advantage is 76% more HBM capacity (141 vs 80 GB) and 43% more bandwidth (4.8 vs 3.35 TB/s). This allows larger microbatches, longer contexts, less activation checkpointing, and less aggressive sharding -- all of which can improve achieved MFU. The calculator models this by giving H200 a higher default MFU (45% vs 40%), not a higher peak.

**Dense vs Sparse:** NVIDIA spec sheets often quote sparse tensor-core throughput. Dense values (used here) are approximately half the sparse numbers. This calculator uses dense values exclusively since dense training is the standard approach.

---

## 7. Warnings Explained

| Warning | Trigger | What To Do |
|---------|---------|------------|
| Large-cluster warning | Model >70B and window <14 days | Expect significant operational complexity. Consider longer windows or staged training. |
| Distributed-systems warning | Required GPUs >1,024 | Networking, checkpointing, stragglers, and cluster fragmentation become major factors. Plan for operational overhead. |
| Training mode warning | Mode is not Full Pretraining | The 6ND formula is calibrated for full pretraining. SFT/LoRA/RLHF typically need far less compute. Use task-specific estimates. |
| FP8 warning | FP8 precision selected | FP8 end-to-end training is experimental. Actual throughput may differ significantly from the peak-based estimate. |
| H200 note | H200 selected | Reminder that H200 has the same raw BF16 peak as H100. Its advantage is memory and bandwidth, which may improve MFU. |
| Memory bound exceeds compute | Memory lower-bound GPUs > compute GPUs | The model's state memory requires more GPUs than compute alone. The cluster is memory-constrained. |

---

## 8. Export & Share

The calculator supports four export formats:

- **JSON** -- full scenario and results for programmatic use
- **CSV** -- tabular results for spreadsheets
- **Markdown** -- formatted summary for docs, Slack, or email
- **Share URL** -- all scenario parameters encoded in the URL hash fragment

The share URL encodes model size, training window, TPP, GPU selections, MFU overrides, precision, availability, overhead, training mode, and memory bytes per parameter. Paste it in a browser to restore the exact scenario.

---

## 9. Custom GPUs

To add a custom GPU SKU:

1. Open the **Custom GPU** section in the left panel
2. Enter a label, dense BF16 peak (TFLOP/s), optional FP8 peak, memory (GB), and default MFU
3. Click "Add GPU"
4. The custom GPU appears in the GPU selector and can be used in all modes

Custom GPUs are stored in the browser session. They participate in comparison, sensitivity, and reverse solve like built-in SKUs.

---

## 10. Worked Example: 70B Model, 30 Days, H100

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

## 11. References

- **[R1]** Hoffmann et al., *Training Compute-Optimal Large Language Models* (Chinchilla), arXiv:2203.15556, 2022. [https://arxiv.org/abs/2203.15556](https://arxiv.org/abs/2203.15556)
- **[R2]** Epoch AI, *Chinchilla scaling: A replication attempt*, Apr. 2024. [https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt)
- **[R3]** NVIDIA H100 GPU specifications. [https://www.nvidia.com/en-us/data-center/h100/](https://www.nvidia.com/en-us/data-center/h100/)
- **[R4]** NVIDIA H200 GPU specifications. [https://www.nvidia.com/en-us/data-center/h200/](https://www.nvidia.com/en-us/data-center/h200/)
- **[R5]** NVIDIA HGX B200 specifications. [https://www.nvidia.com/en-us/data-center/hgx/](https://www.nvidia.com/en-us/data-center/hgx/)
- **[R6]** NVIDIA GB200 NVL72 specifications. [https://www.nvidia.com/en-us/data-center/gb200-nvl72/](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
- **[R7]** Sardana et al., *Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws*, arXiv:2401.00448, 2024. [https://arxiv.org/abs/2401.00448](https://arxiv.org/abs/2401.00448)
