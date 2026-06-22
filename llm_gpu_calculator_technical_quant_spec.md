**LLM Training GPU Calculator**

*Technical / Quantitative Specification*

Prepared for Matt Greenwood. Generated 2026-06-18.

Scope note: this specification estimates training infrastructure for dense decoder-only transformer language models. It is intended for planning and comparison, not for procurement commitments without empirical benchmarking.

# 1. Executive Summary

The calculator maps a desired model size and retraining window to an estimated number of accelerator equivalents. The primary path is a Chinchilla-style token target, a dense transformer training FLOP estimate, and an MFU-adjusted GPU throughput model. The output should be a range, not a single pseudo-precise value, because the largest unknown is achieved utilization rather than the algebra.

The simple model is intentionally transparent: parameters determine tokens; parameters and tokens determine training FLOPs; the retraining window and sustained GPU throughput determine GPU count. More advanced modes add memory feasibility, precision selection, empirical calibration, and scenario comparison.

# 2. Definitions and Notation

| **Symbol / Term** | **Meaning**                                                             | **Default / Comment**                                               |
|-------------------|-------------------------------------------------------------------------|---------------------------------------------------------------------|
| N                 | Number of model parameters                                              | User input, usually in billions                                     |
| D                 | Training tokens                                                         | Default D = 20 x N under Chinchilla-20                              |
| TPP               | Tokens per parameter                                                    | Default 20; user-overridable                                        |
| C                 | Training compute in FLOPs                                               | Approx. C = 6 x N x D for dense decoder training                    |
| MFU               | Model FLOP utilization                                                  | Default 0.40 for H100/B200; separate per GPU SKU                    |
| Availability      | Useful wall-clock fraction                                              | Default 0.90                                                        |
| Overhead          | Extra compute/time from checkpointing, evaluation, retries, data stalls | Default 1.10                                                        |
| Dense peak        | Non-sparse tensor-core throughput                                       | For NVIDIA specs marked with sparsity, dense is usually half sparse |

# 3. Scaling-Law Basis

The Chinchilla paper found that, under a fixed compute budget, compute-optimal dense language model training scales model size and training tokens approximately equally. The canonical planning anchor is Chinchilla at 70B parameters trained on 1.4T tokens, which is roughly 20 tokens per parameter \[R1\]\[R2\].

D = tokens_per_parameter x N  
Default: tokens_per_parameter = 20

This is a planning default, not a law of nature. Modern recipes may deliberately train smaller models longer, especially when inference economics dominate. The calculator should therefore expose TPP as a user input, with presets such as 20, 50, 100, and custom \[R7\].

# 4. Core FLOP Model

For a dense decoder-only transformer, a commonly used first-order estimate is:

training_FLOPs ~= 6 x N x D

Combining the above with Chinchilla-20 gives:

D = 20N  
training_FLOPs ~= 120 x N^2

This quadratic dependence is the most important intuition. A model that is twice as large requires roughly four times the training compute if the token target also scales linearly with parameter count.

# 5. GPU Count Model

seconds_available = retraining_window_days x 86,400  
  
sustained_GPU_FLOPs = dense_peak_FLOPs_per_GPU  
x MFU  
x availability  
  
total_training_FLOPs = 6 x N x D x overhead_factor  
  
GPUs_needed = ceil(total_training_FLOPs  
/ (seconds_available x sustained_GPU_FLOPs))

The calculator should always show the assumptions used in the calculation. MFU and availability are intentionally separate because MFU is a kernel/model/parallelism property, while availability captures failures, maintenance, queueing, restart loss, and operational friction.

# 6. GPU SKU Table

The following values are planning defaults derived from NVIDIA product specifications. NVIDIA often reports tensor-core throughput with sparsity; for dense training estimates, use half the sparse number. H100 and H200 have similar raw Hopper compute, but H200 has substantially more HBM capacity and bandwidth, which can improve achieved MFU in memory-stressed training runs \[R3\]\[R4\].

| **GPU**                       | **Dense BF16 peak** | **Dense FP8 peak** | **Memory** | **Bandwidth**             | **Default MFU**   |
|-------------------------------|---------------------|--------------------|------------|---------------------------|-------------------|
| H100 SXM                      | ~0.989 PFLOP/s      | ~1.979 PFLOP/s     | 80 GB      | 3.35 TB/s                 | 0.40              |
| H200 SXM                      | ~0.989 PFLOP/s      | ~1.979 PFLOP/s     | 141 GB     | 4.8 TB/s                  | 0.45 configurable |
| B200 SXM / HGX B200           | ~2.25 PFLOP/s       | ~4.5 PFLOP/s       | ~175 GB    | depends on system         | 0.40              |
| GB200 NVL72 per Blackwell GPU | ~2.5 PFLOP/s        | ~5.0 PFLOP/s       | ~186 GB    | ~8 TB/s aggregate-derived | 0.40              |

H100 and H200 should not be collapsed in the full calculator. They can be equal in a pure peak-FLOP model, but should be split in the SKU table because H200 can support larger microbatches, longer contexts, less activation checkpointing, and less aggressive sharding. The right modeling move is to give H200 the same BF16 peak as H100 but allow a higher achieved MFU range.

# 7. Modeling Approaches

## Approach A: FLOPs-only quick estimate

Inputs: parameter size, retraining days, GPU type. Uses fixed TPP=20, fixed MFU, fixed availability, fixed overhead. This is useful for napkin math, but hides the most important uncertainties.

## Approach B: Assumption-explicit planning model

Adds explicit TPP, MFU, precision, availability, and overhead. This should be the default product mode. It provides honest sensitivity and supports H100-equivalent comparisons.

## Approach C: Memory-feasibility model

Adds a lower-bound memory check. A crude state-memory estimate for Adam-like training is 12-16 bytes per parameter before considering activations and sharding strategy. With FSDP/ZeRO/tensor/pipeline parallelism the state is sharded, but activations, sequence length, microbatching, and checkpointing still matter.

state_memory_lower_bound ~= N x bytes_per_parameter  
minimum_memory_GPUs ~= ceil(state_memory_lower_bound / usable_cluster_HBM)  
recommended_GPUs = max(compute_required_GPUs, memory_lower_bound_GPUs)

## Approach D: Empirical calibration mode

Allows users to enter a known benchmark run: model size, tokens, GPUs, wall-clock days, GPU type. The calculator backs out achieved MFU and uses it for future estimates. This is the most reliable mode for a specific organization, codebase, cluster topology, and model architecture.

# 8. Worked Examples

Assumptions for the examples: TPP=20, dense BF16, MFU=0.40 except H200 sensitivity rows, availability=0.90, overhead=1.10. H100 sustained useful throughput is about 0.989 PFLOP/s x 0.40 x 0.90 = 0.356 PFLOP/s.

| **Model** | **Tokens** | **Window** | **H100 @40% MFU** | **H200 @40% MFU** | **B200 @40% MFU** | **GB200 GPU @40% MFU** |
|-----------|------------|------------|-------------------|-------------------|-------------------|------------------------|
| 7B        | 140B       | 30 days    | 8                 | 8                 | 4                 | 3                      |
| 7B        | 140B       | 7 days     | 31                | 31                | 14                | 12                     |
| 13B       | 260B       | 30 days    | 25                | 25                | 11                | 10                     |
| 13B       | 260B       | 7 days     | 104               | 104               | 46                | 41                     |
| 70B       | 1.40T      | 30 days    | 701               | 701               | 309               | 278                    |
| 70B       | 1.40T      | 7 days     | 3003              | 3003              | 1321              | 1189                   |
| 405B      | 8.10T      | 30 days    | 23450             | 23450             | 10313             | 9282                   |
| 405B      | 8.10T      | 7 days     | 100498            | 100498            | 44197             | 39777                  |

# 9. Example Detail: 70B in 30 Days

N = 70e9  
D = 20 x 70e9 = 1.4e12 tokens  
base_FLOPs = 6 x 70e9 x 1.4e12 = 5.88e23  
total_FLOPs = 5.88e23 x 1.10 = 6.47e23  
seconds = 30 x 86,400 = 2,592,000  
H100_sustained = 0.989e15 x 0.40 x 0.90 = 3.56e14 FLOP/s  
H100s = ceil(6.47e23 / (2.592e6 x 3.56e14)) ~= 701

| **Scenario**           | **GPU estimate** |
|------------------------|------------------|
| H100, MFU 30%          | ~935             |
| H100, MFU 40%          | ~701             |
| H100, MFU 50%          | ~561             |
| H200, same MFU as H100 | ~701             |
| H200, MFU 45%          | ~623             |
| H200, MFU 50%          | ~561             |

# 10. Caveats and Boundaries

- The 6ND FLOP estimate is a first-order dense-transformer training approximation. MoE, multimodal encoders, very long context, specialized attention, and optimizer variants need adjustments.

- Peak FLOPs are not delivered FLOPs. The calculator should lead with achieved MFU ranges and not pretend that theoretical peak is operational capacity.

- H100 and H200 have similar raw dense BF16 peak; H200 should be modeled through memory/bandwidth and MFU improvements, not through a larger compute number.

- B200 and GB200 figures are especially dependent on precision recipe, interconnect topology, and software maturity. Treat defaults as starting points.

- Full retraining from scratch is a different operational mode from continued pretraining, SFT, LoRA/QLoRA, RLHF, distillation, or data refresh fine-tuning. The SPA should make this distinction visible.

# References

**\[R1\]** Hoffmann et al., Training Compute-Optimal Large Language Models (Chinchilla), arXiv:2203.15556, 2022. https://arxiv.org/abs/2203.15556

**\[R2\]** Epoch AI, Chinchilla scaling: A replication attempt, Apr. 2024. https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt

**\[R3\]** NVIDIA H100 GPU product specifications. https://www.nvidia.com/en-us/data-center/h100/

**\[R4\]** NVIDIA H200 GPU product specifications. https://www.nvidia.com/en-us/data-center/h200/

**\[R5\]** NVIDIA HGX Platform product specifications for HGX B200. https://www.nvidia.com/en-us/data-center/hgx/

**\[R6\]** NVIDIA GB200 NVL72 product specifications. https://www.nvidia.com/en-us/data-center/gb200-nvl72/

**\[R7\]** Sardana et al., Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws, arXiv:2401.00448, 2024. https://arxiv.org/abs/2401.00448

**\[R8\]** Held, Hall, Liang, Yang, Relative Scaling Laws for LLMs, arXiv:2510.24626, 2025. https://arxiv.org/abs/2510.24626

**\[R9\]** Wen, Hall, Ma, Liang, Fantastic Pretraining Optimizers and Where to Find Them, 2025.

**\[R10\]** Marin: Open Development of Frontier AI, marin.community. https://marin.community/

**\[R11\]** Marin Community, Levanter: Legible, Scalable, Reproducible Foundation Models with Named Tensors and JAX. https://github.com/marin-community/levanter
