**LLM Training GPU Calculator**

*SPA Design Specification*

Prepared for Matt Greenwood. Generated 2026-06-18.

Scope note: this specification estimates training infrastructure for dense decoder-only transformer language models. It is intended for planning and comparison, not for procurement commitments without empirical benchmarking.

# 1. Recommended Product Architecture

Implement the calculator as a client-side SPA using TypeScript. A good v1 stack is React + Vite + TypeScript, with a pure calculation engine isolated from UI components. This keeps the calculator embeddable, testable, and easy to port into internal dashboards later.

| **Layer**          | **Recommendation**                    | **Rationale**                                             |
|--------------------|---------------------------------------|-----------------------------------------------------------|
| UI                 | React + TypeScript                    | Fast iteration, component reuse, strong typing            |
| Build              | Vite                                  | Simple static SPA deployment                              |
| State              | Zustand or React context              | Small calculation state; no heavy global framework needed |
| Charts             | Recharts, ECharts, or lightweight SVG | Sensitivity plots and comparison bars                     |
| Calculation engine | Pure TypeScript module                | Deterministic unit tests and reusable API                 |
| Persistence        | URL params + localStorage             | Shareable scenarios without backend                       |
| Export             | JSON, CSV, copy-to-clipboard Markdown | Supports analysis handoff                                 |

# 2. SPA Information Architecture

The first version should be a single route with panels, not a multi-page application. Suggested layout:

| **Region**                 | **Contents**                                                              |
|----------------------------|---------------------------------------------------------------------------|
| Header                     | Title, scenario name, export/share controls                               |
| Left input rail            | Model size, window, training mode, tokens/parameter, GPU selection        |
| Advanced assumptions panel | MFU, availability, overhead, precision, custom GPU editor                 |
| Main results cards         | Top-line GPU count, H100 equivalents, tokens, FLOPs, sustained throughput |
| Comparison section         | GPU SKU table and bar chart                                               |
| Sensitivity section        | MFU/window matrix and optional heatmap                                    |
| Trace drawer               | Formula expansion with exact numbers                                      |
| Warnings area              | Memory, scale, training-mode caveats                                      |

# 3. Domain Model

type Precision = 'BF16_DENSE' \| 'FP8_DENSE';  
  
type GpuSku = {  
id: string;  
label: string;  
bf16DenseFlops: number; // FLOP/s per GPU  
fp8DenseFlops?: number; // FLOP/s per GPU  
memoryGb: number;  
bandwidthTbps?: number;  
defaultMfu: number;  
notes?: string;  
};  
  
type TrainingScenario = {  
modelParameters: number; // absolute count, not billions  
tokensPerParameter: number;  
trainingWindowSeconds: number;  
precision: Precision;  
selectedGpuIds: string\[\];  
mfuByGpuId: Record\<string, number\>;  
availability: number;  
overheadFactor: number;  
trainingMode: 'FULL_PRETRAINING' \| 'CONTINUED_PRETRAINING' \| 'SFT' \| 'LORA' \| 'RLHF' \| 'DISTILLATION';  
memoryBytesPerParameter: number;  
};  
  
type EstimateResult = {  
gpuId: string;  
tokens: number;  
baseFlops: number;  
totalFlops: number;  
sustainedFlopsPerGpu: number;  
requiredGpus: number;  
h100Equivalents: number;  
memoryLowerBoundGpus: number;  
warnings: string\[\];  
trace: string\[\];  
};

# 4. Calculation Engine

export function estimateTrainingRun(  
scenario: TrainingScenario,  
gpu: GpuSku,  
h100Reference: GpuSku  
): EstimateResult {  
const N = scenario.modelParameters;  
const D = scenario.tokensPerParameter \* N;  
const baseFlops = 6 \* N \* D;  
const totalFlops = baseFlops \* scenario.overheadFactor;  
  
const peak = scenario.precision === 'BF16_DENSE'  
? gpu.bf16DenseFlops  
: gpu.fp8DenseFlops ?? gpu.bf16DenseFlops;  
  
const mfu = scenario.mfuByGpuId\[gpu.id\] ?? gpu.defaultMfu;  
const sustained = peak \* mfu \* scenario.availability;  
const requiredGpus = Math.ceil(totalFlops / (scenario.trainingWindowSeconds \* sustained));  
  
const h100Peak = scenario.precision === 'BF16_DENSE'  
? h100Reference.bf16DenseFlops  
: h100Reference.fp8DenseFlops ?? h100Reference.bf16DenseFlops;  
const h100Mfu = scenario.mfuByGpuId\[h100Reference.id\] ?? h100Reference.defaultMfu;  
const h100Sustained = h100Peak \* h100Mfu \* scenario.availability;  
const h100Equivalents = requiredGpus \* sustained / h100Sustained;  
  
const usableMemBytes = gpu.memoryGb \* 1e9 \* 0.85;  
const memoryLowerBoundGpus = Math.ceil(  
(N \* scenario.memoryBytesPerParameter) / usableMemBytes  
);  
  
return {  
gpuId: gpu.id,  
tokens: D,  
baseFlops,  
totalFlops,  
sustainedFlopsPerGpu: sustained,  
requiredGpus,  
h100Equivalents,  
memoryLowerBoundGpus,  
warnings: buildWarnings(scenario, gpu, requiredGpus, memoryLowerBoundGpus),  
trace: buildTrace(scenario, gpu, D, baseFlops, totalFlops, sustained, requiredGpus)  
};  
}

# 5. Default GPU Data

export const GPU_SKUS: GpuSku\[\] = \[  
{  
id: 'h100-sxm',  
label: 'NVIDIA H100 SXM',  
bf16DenseFlops: 0.9895e15,  
fp8DenseFlops: 1.979e15,  
memoryGb: 80,  
bandwidthTbps: 3.35,  
defaultMfu: 0.40,  
notes: 'Dense values derived by halving NVIDIA sparse tensor-core specs.'  
},  
{  
id: 'h200-sxm',  
label: 'NVIDIA H200 SXM',  
bf16DenseFlops: 0.9895e15,  
fp8DenseFlops: 1.979e15,  
memoryGb: 141,  
bandwidthTbps: 4.8,  
defaultMfu: 0.45,  
notes: 'Same raw Hopper compute class as H100; more HBM and bandwidth.'  
},  
{  
id: 'b200-sxm',  
label: 'NVIDIA B200 SXM / HGX B200',  
bf16DenseFlops: 2.25e15,  
fp8DenseFlops: 4.5e15,  
memoryGb: 175,  
defaultMfu: 0.40,  
notes: 'Per-GPU value derived from HGX B200 8-GPU specs.'  
},  
{  
id: 'gb200-nvl72-gpu',  
label: 'GB200 NVL72 Blackwell GPU equivalent',  
bf16DenseFlops: 2.5e15,  
fp8DenseFlops: 5.0e15,  
memoryGb: 186,  
bandwidthTbps: 8.0,  
defaultMfu: 0.40,  
notes: 'Per-GPU value derived from 72-GPU NVL72 dense rack specs.'  
}  
\];

# 6. Component Design

| **Component**       | **Responsibilities**                                                |
|---------------------|---------------------------------------------------------------------|
| ScenarioForm        | Collect model size, training window, TPP, training mode, precision. |
| GpuSelector         | Select and compare GPU SKUs; open custom GPU editor.                |
| AdvancedAssumptions | MFU sliders, availability, overhead, memory bytes/parameter.        |
| ResultCards         | Top-line GPU count, tokens, FLOPs, sustained throughput.            |
| GpuComparisonTable  | Side-by-side GPU results.                                           |
| SensitivityMatrix   | MFU x training-window grid for a selected SKU.                      |
| FormulaTraceDrawer  | Human-readable formula substitution and calculation trace.          |
| WarningsPanel       | Training-mode, scale, memory, and SKU caveats.                      |
| ExportControls      | Copy markdown, export CSV/JSON, encode scenario in URL.             |

# 7. UX Details

- Use live calculation, but debounce text inputs so that invalid intermediate states do not flash warnings.

- Use parameter units directly in the input: M, B, T. Store internally as absolute parameter count.

- Show both exact and rounded outputs: for example, 701 GPUs and approximately 700 GPUs.

- Keep the formula trace close to the top-line number so technical users can audit the output.

- Avoid hiding assumptions. Even in quick mode, show TPP, MFU, availability, and overhead as chips next to the result.

- Make H100 vs H200 explicit: equal raw BF16 peak, different HBM and default MFU.

# 8. State and URL Encoding

The app should be useful without accounts or a backend. Serialize the scenario into URL query parameters or a compressed hash fragment. Also store recent scenarios in localStorage.

Example URL shape:  
/gpu-calc#model=70B&days=30&tpp=20&gpus=h100,h200,b200&mfu_h100=0.40&mfu_h200=0.45&overhead=1.10&avail=0.90

# 9. Testing Plan

| **Test Type**       | **Examples**                                                                                                     |
|---------------------|------------------------------------------------------------------------------------------------------------------|
| Unit tests          | 70B/30d/H100 default returns 701; 7B/30d/H100 returns 8; H100/H200 same MFU returns same compute count.          |
| Property tests      | GPU count decreases when window increases; GPU count increases with TPP; GPU count decreases when MFU increases. |
| Snapshot tests      | Formula trace text for canonical examples.                                                                       |
| UI tests            | Changing GPU selection updates comparison table; invalid inputs show inline validation.                          |
| Accessibility tests | Keyboard operation, labels, contrast, table headings, ARIA for collapsible panels.                               |

# 10. Performance and Deployment

The calculation is trivial computationally. The app should run entirely client-side and can be hosted as static assets. If integrated into an internal environment, the pure calculation engine can be reused in notebooks, API services, or planning dashboards.

- Use no backend for v1 unless scenario sharing requires central persistence.

- Version the GPU SKU table and show the spec date in the UI.

- Keep source links in metadata so users can audit hardware assumptions.

- Use feature flags for experimental FP8 and non-pretraining modes.

# 11. Implementation Milestones

| **Milestone** | **Deliverable**                                                            |
|---------------|----------------------------------------------------------------------------|
| M1            | Pure TypeScript calculation engine with unit tests and canonical examples. |
| M2            | Static SPA with basic input form and top-line results.                     |
| M3            | GPU comparison, H100-equivalent conversion, and sensitivity matrix.        |
| M4            | Trace drawer, warnings, custom GPU editor, export/share.                   |
| M5            | Calibration mode and memory feasibility improvements.                      |

# References

**\[R1\]** Hoffmann et al., Training Compute-Optimal Large Language Models (Chinchilla), arXiv:2203.15556, 2022. https://arxiv.org/abs/2203.15556

**\[R2\]** Epoch AI, Chinchilla scaling: A replication attempt, Apr. 2024. https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt

**\[R3\]** NVIDIA H100 GPU product specifications. https://www.nvidia.com/en-us/data-center/h100/

**\[R4\]** NVIDIA H200 GPU product specifications. https://www.nvidia.com/en-us/data-center/h200/

**\[R5\]** NVIDIA HGX Platform product specifications for HGX B200. https://www.nvidia.com/en-us/data-center/hgx/

**\[R6\]** NVIDIA GB200 NVL72 product specifications. https://www.nvidia.com/en-us/data-center/gb200-nvl72/

**\[R7\]** Sardana et al., Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws, arXiv:2401.00448, 2024. https://arxiv.org/abs/2401.00448
