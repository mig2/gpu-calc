**LLM Training GPU Calculator**

*Functional Specification for SPA*

Prepared for Matt Greenwood. Generated 2026-06-18.

Scope note: this specification estimates training infrastructure for dense decoder-only transformer language models. It is intended for planning and comparison, not for procurement commitments without empirical benchmarking.

# 1. Product Goal

Build a single-page application that helps a user estimate accelerator requirements for training or retraining dense language models. The tool should support both quick estimates and assumption-explicit analysis, with special emphasis on sensitivity to MFU, training window, token target, and GPU SKU.

# 2. Primary User Questions

- Given a model size and retraining window, how many H100-equivalent GPUs do I need?

- How does the requirement change for H200, B200, or GB200-class hardware?

- How sensitive is the answer to MFU, tokens-per-parameter, overhead, and availability?

- Is the desired model size obviously memory-infeasible without large-scale parallelism?

- What training duration is implied by a fixed available GPU budget?

# 3. Functional Modes

| **Mode**          | **Purpose**                                          | **Required Inputs**                                                 | **Outputs**                                         |
|-------------------|------------------------------------------------------|---------------------------------------------------------------------|-----------------------------------------------------|
| Quick Estimate    | Fast H100-equivalent answer                          | Parameters, retraining days                                         | GPU count by default SKU, tokens, FLOPs             |
| Advanced Estimate | Explicit assumptions and GPU comparison              | Parameters, TPP, GPU SKU(s), precision, MFU, availability, overhead | GPU counts, H100-equivalents, sensitivity table     |
| Reverse Solve     | Find training time or max model size from GPU budget | GPU count plus either model size or time target                     | Training days or approximate maximum parameter size |
| Calibration       | Fit achieved MFU from a known run                    | Known run parameters, tokens, days, GPUs, SKU                       | Back-solved MFU and reusable preset                 |
| Memory Check      | Warn about feasibility and sharding pressure         | Parameters, GPU SKU, bytes/parameter, usable HBM fraction           | Lower-bound memory GPUs and warnings                |

# 4. Input Requirements

| **Input**              | **Type**      | **Default**      | **Validation / Notes**                                                                                |
|------------------------|---------------|------------------|-------------------------------------------------------------------------------------------------------|
| Model parameters       | number + unit | 70B              | Positive. Accept M/B/T units.                                                                         |
| Retraining window      | number + unit | 30 days          | Positive. Accept hours/days/weeks.                                                                    |
| Tokens per parameter   | number        | 20               | Expose presets: 20, 50, 100, custom.                                                                  |
| Training mode          | enum          | Full pretraining | Full pretraining, continued pretraining, SFT, LoRA, RLHF; v1 formulas optimized for full pretraining. |
| GPU SKU                | multi-select  | H100 SXM         | H100, H200, B200, GB200, custom.                                                                      |
| Precision              | enum          | BF16 dense       | BF16, FP8 experimental/planning.                                                                      |
| MFU                    | percentage    | 40%              | Per SKU, editable. Show ranges.                                                                       |
| Availability           | percentage    | 90%              | Useful wall-clock fraction.                                                                           |
| Overhead factor        | number        | 1.10             | Checkpointing, eval, restarts, data stalls.                                                           |
| Memory bytes/parameter | number        | 16               | For lower-bound state memory check.                                                                   |

# 5. Output Requirements

- Top-line required GPU count for the selected SKU and assumptions.

- H100-equivalent count for every selected SKU.

- Training tokens implied by model size and TPP.

- Base and overhead-adjusted FLOPs.

- Sustained useful FLOP/s per GPU and how it was computed.

- Sensitivity table across MFU values, training windows, and GPU SKUs.

- Warnings when results imply large-cluster operational risk or when memory lower bound exceeds compute requirement.

- Plain-English explanation of why H100 and H200 may be equal in raw FLOPs but different in achieved MFU.

# 6. User Stories and Acceptance Criteria

| **User Story**                                                       | **Acceptance Criteria**                                                                       |
|----------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| As a user, I want to enter 70B and 30 days and get an H100 estimate. | The app shows about 701 H100s under the default assumptions and displays tokens=1.4T.         |
| As a user, I want to compare H100 and H200.                          | The app shows equal raw BF16 peak, different memory/bandwidth, and allows H200 MFU to differ. |
| As a user, I want to understand sensitivity.                         | The app shows GPU counts for MFU 30/40/50% and windows 7/30/60 days.                          |
| As a user, I want to add a custom GPU.                               | The app accepts dense BF16 peak, FP8 peak, memory, bandwidth, and default MFU.                |
| As a user, I want to reverse solve from 256 H100s.                   | The app returns implied training time for a selected model size and TPP.                      |
| As a user, I want to export my scenario.                             | The app exports JSON and a concise text/CSV summary.                                          |

# 7. UI Flow

The SPA should have one main calculation workspace rather than a multi-page wizard. The left side contains assumptions; the right side contains results. Advanced panels can be collapsed by default, but the active assumptions must always remain visible.

1.  User enters model size and retraining window.

2.  App immediately computes default Chinchilla-20 token target and top-line GPU count.

3.  User chooses one or more GPU SKUs for comparison.

4.  User opens Advanced Assumptions to adjust MFU, availability, overhead, precision, and TPP.

5.  Results, sensitivity tables, and warnings update live.

6.  User optionally saves, copies, or exports the scenario.

# 8. Calculation Trace Requirement

Every result should have a trace drawer that shows the exact formula and numbers used. This is critical because the product will be used by technical users who will want to audit the math.

N = 70e9  
D = 20 x N = 1.4e12  
FLOPs = 6 x N x D x 1.10 = 6.47e23  
Sustained H100 = 0.989e15 x 0.40 x 0.90 = 3.56e14 FLOP/s  
GPUs = ceil(6.47e23 / (2,592,000 x 3.56e14)) = 701

# 9. Presets

| **Preset Type**      | **Values**                                                    |
|----------------------|---------------------------------------------------------------|
| Model sizes          | 7B, 13B, 34B, 70B, 130B, 405B, custom                         |
| Training windows     | 7 days, 14 days, 30 days, 60 days, custom                     |
| Tokens per parameter | 20, 50, 100, custom                                           |
| MFU scenario         | Conservative 30%, Planning 40%, Strong 50%                    |
| GPU SKUs             | H100 SXM, H200 SXM, B200 SXM/HGX, GB200 NVL72 per GPU, custom |

# 10. Warnings and Guardrails

- If model size is greater than 70B and retraining window is less than 14 days, display a large-cluster warning.

- If required GPUs exceed 1,024, display a distributed-systems warning about networking, checkpointing, stragglers, and cluster fragmentation.

- If training mode is not full pretraining, state that the full-pretraining formula may overestimate compute and that task-specific formulas are needed.

- If FP8 is selected, warn that FP8 end-to-end training recipes may not deliver the simple peak-based estimate.

- If H200 is selected, state explicitly that its raw BF16 peak is not higher than H100 in the default table; its benefit is memory/bandwidth and possible MFU improvement.

# 11. Non-Goals for v1

- Not a replacement for cluster benchmarking.

- Not a procurement quote or cost model, although cost can be added later.

- Not a complete distributed-training planner for tensor/pipeline/context parallelism.

- Not a quality predictor; it estimates compute, not loss or downstream performance.

- Not a MoE calculator in v1.

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
