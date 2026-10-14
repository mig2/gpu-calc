# GPU Calculator

Three calculators for accelerator planning:

| Mode | Question it answers | Binding constraint |
|---|---|---|
| **Training** | How many GPUs to pretrain model X by date Y? | Compute × deadline |
| **Inference** | API or self-host, and what does serving cost? | Bandwidth, then KV capacity |
| **Fine-Tuning** | Can I fine-tune model Y with method Z, and on what? | Memory capacity |

Training covers four model families: **LLMs**, **time-series foundation models**, **tabular foundation models**, and **classical tabular (GBDT)**.

Each pane answers a structurally different question, which is why they are separate rather than variations on one form. In pretraining the memory floor is roughly 470× slacker than the compute bound and never binds; in fine-tuning it *is* the question. See [`docs/research/README.md`](docs/research/README.md) for the cross-pane comparison.

## Features

### Training
- **Model family tabs** -- LLM, time-series, tabular foundation, classical tabular
- **Quick Estimate** -- model size + training window, GPU count instantly
- **Advanced Estimate** -- explicit MFU, availability, overhead, precision, tokens-per-parameter
- **15 accelerator SKUs** -- NVIDIA H100 through GB300, AMD MI300X/MI355X/MI450, Google TPU v6e/v7, AWS Trainium2/3, Intel Gaudi 3; all dense figures
- **Precision-relative MFU** -- BF16 / FP8 / FP4 scaled by measured end-to-end gain, not peak ratio
- **Sensitivity Matrix**, **IsoFLOP Explorer**, **Reverse Solve**, **Calibration**
- **27 reference runs** -- GPT-3 through DeepSeek-V4, 12 of them MoE, 13 with published GPU-hours

### Inference
- **Buy vs build** -- 38 API models across 7 providers against 32 self-host configurations across 10 clouds
- **Replica sizing** -- capacity and cost scale with the load, auto-sized or pinned
- **Breakeven analysis** -- cost per million tokens on both sides, since with both scaling the deciding variable is utilization rather than volume
- **Use-case presets** -- chatbot, RAG, summarization, coding, agentic, long-context

### Fine-Tuning
- **Method × GPU matrix** -- full / LoRA / QLoRA / DoRA / DPO / ORPO / GRPO / PPO against selected accelerators
- **Memory feasibility** -- fits / sharded / OOM per cell, with a component breakdown
- **30-model catalog** -- architecture dimensions from published configs, MoE total-vs-active throughout
- **RL post-training** -- rollout-dominated cost model with a calibration input

### Throughout
- **Confidence labels** -- high / medium / medium-low / low, per quantity
- **Custom GPUs**, **Export & Share** (JSON/CSV/Markdown + URL hash), **Formula Trace** on every result

## Tech Stack

- **React 19** with TypeScript
- **Zustand** for state management
- **Recharts** for GPU comparison charts
- **Vite** for dev server and builds
- **Vitest** + React Testing Library for tests

## Getting Started

### Prerequisites

- Node.js >= 20
- npm >= 10

### Install and Run

```bash
npm install
npm run dev        # dev server at http://localhost:5173
```

### Test and Build

```bash
npm test           # run tests (vitest)
npm run typecheck  # tsc --noEmit
npm run build      # tsc -b && vite build, to dist/
```

`npm run build` is stricter than `npm run typecheck` -- it applies the project-references config, including `noUnusedLocals`. Run it before pushing; a green typecheck does not imply a green build.

## Architecture

```
src/
  engine/          Pure functions -- no React, no side effects
    calculator.ts    Core 6ND FLOP model and GPU count formula (LLM)
    adapters/
      time-series-adapter.ts   Windowed patch tokenization + FLOP estimate
      tabular-adapter.ts       Row/cell/axial tokenization + FLOP estimate
      classical-tabular-adapter.ts  Work-unit throughput model (GBDT)
    reverse-solve.ts Solve for time or model size from GPU budget
    calibration.ts   Back-solve MFU from known run
    finetune-calculator.ts  Fine-tuning memory + compute model
    rl-calculator.ts        GRPO/PPO rollout model (decode roofline)
    inference-calculator.ts API vs self-host economics
    hardware-estimator.ts   Shared: sustained throughput, precision/MoE MFU factors
    gpu-data.ts      15 accelerator SKUs, dense figures with per-row sources
    export.ts        JSON/CSV/Markdown export + URL hash encoding
    types.ts         All domain types (ModelFamily, Scenario union, adapters)
  store/
    scenario-store.ts  Zustand store -- single source of truth
  data/              Curated JSON + types
    finetune-models.json     30-model architecture catalog
    api-pricing.json         38 API models, vendor-sourced
    cloud-gpu-pricing.json   32 cloud GPU instances, vendor-sourced
    self-host-throughput.json  Serving fixtures
  components/        React components -- read from store, call engine
    App.tsx            Layout shell + mode and model family tabs
    FineTuneCalculator.tsx   Method x GPU matrix + feasibility detail
    InferenceCalculator.tsx  Buy vs build comparison
    ScenarioForm.tsx   Model size, window, TPP, mode, precision
    GpuSelector.tsx    Multi-select GPU checkboxes
    AdvancedAssumptions.tsx  MFU, availability, overhead sliders
    ResultCards.tsx     Top-line GPU count cards + confidence badge
    GpuComparisonTable.tsx   Side-by-side comparison
    SensitivityMatrix.tsx    MFU x window heatmap
    ReverseSolve.tsx         Reverse-solve UI
    CalibrationMode.tsx      Calibration UI
    ...
```

The engine layer is pure math with no UI dependencies. Each model family has a dedicated adapter that computes effective tokens and FLOPs (or work units for classical tabular). The store holds scenario state and derived results. Components subscribe to the store and render.

## Help & Documentation

- **[public/help.html](public/help.html)** -- the in-app user guide: every mode, formula, GPU spec, warning and worked example. Linked from the app header.
- **[docs/research/](docs/research/README.md)** -- the evidence layer. Calibration anchors, back-solved defaults, regression targets and per-quantity confidence ratings for each pane, each with a discrepancy table cited to `file:line`.
- **[docs/decisions/](docs/decisions/)** -- design decisions and their rationale, including corrections where validation overturned an earlier premise.

`public/help.md` and `public/help.html` are maintained **by hand in parallel** -- there is no generator. Any documentation change must land in both.

## References

- Hoffmann et al., [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556), 2022
- Sardana et al., [Beyond Chinchilla-Optimal: Accounting for Inference](https://arxiv.org/abs/2401.00448), 2024
- Epoch AI, [Chinchilla scaling: A replication attempt](https://epoch.ai/publications/chinchilla-scaling-a-replication-attempt), 2024
- Muennighoff et al., [Scaling Data-Constrained Language Models](https://arxiv.org/abs/2305.16264), JMLR 2025
- Dettmers et al., [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314), 2023
- Korthikanti et al., [Reducing Activation Recomputation in Large Transformer Models](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf), MLSys 2023
- NVIDIA [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/), [HGX B200](https://www.nvidia.com/en-us/data-center/hgx/), [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) specs

Per-figure sourcing lives in [`docs/research/`](docs/research/README.md); every data row in `src/data/` carries its own `sourceUrl`.
