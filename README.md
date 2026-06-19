# LLM Training GPU Calculator

Estimate accelerator requirements for training dense decoder-only transformer language models.

## Features

- **Quick Estimate** -- enter model size + training window, get GPU count instantly
- **Advanced Estimate** -- explicit control over MFU, availability, overhead, precision, and tokens-per-parameter
- **GPU Comparison** -- multi-select H100/H200/B200/GB200 with side-by-side table and bar chart
- **Sensitivity Matrix** -- MFU x training-window grid showing how GPU count varies
- **Reverse Solve** -- find training time from a GPU budget, or max model size from a GPU budget
- **Calibration** -- back-solve achieved MFU from a known training run
- **Memory Feasibility** -- detect when model state memory exceeds compute requirements
- **Custom GPUs** -- define custom SKUs with your own peak FLOP/s, memory, and MFU
- **Export & Share** -- JSON, CSV, Markdown export; URL hash encoding for shareable links
- **Formula Trace** -- every result includes a step-by-step calculation audit trail

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
npx vitest run     # run tests
npx tsc --noEmit   # type check
npm run build      # production build to dist/
```

## Architecture

```
src/
  engine/          Pure functions -- no React, no side effects
    calculator.ts    Core 6ND FLOP model and GPU count formula
    reverse-solve.ts Solve for time or model size from GPU budget
    calibration.ts   Back-solve MFU from known run
    gpu-data.ts      GPU SKU table (H100/H200/B200/GB200)
    export.ts        JSON/CSV/Markdown export + URL hash encoding
    types.ts         All domain types
  store/
    scenario-store.ts  Zustand store -- single source of truth
  components/        React components -- read from store, call engine
    App.tsx            Layout shell
    ScenarioForm.tsx   Model size, window, TPP, mode, precision
    GpuSelector.tsx    Multi-select GPU checkboxes
    AdvancedAssumptions.tsx  MFU, availability, overhead sliders
    ResultCards.tsx     Top-line GPU count cards
    GpuComparisonTable.tsx   Side-by-side comparison
    SensitivityMatrix.tsx    MFU x window heatmap
    ReverseSolve.tsx         Reverse-solve UI
    CalibrationMode.tsx      Calibration UI
    ...
```

The engine layer is pure math with no UI dependencies. The store holds scenario state and derived results. Components subscribe to the store and render.

## Help & Documentation

Open the in-app help link or see [public/help.html](public/help.html) for a comprehensive user guide covering all modes, formulas, GPU specs, and worked examples.

## References

- Hoffmann et al., [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556), 2022
- Sardana et al., [Beyond Chinchilla-Optimal: Accounting for Inference](https://arxiv.org/abs/2401.00448), 2024
- NVIDIA [H100](https://www.nvidia.com/en-us/data-center/h100/), [H200](https://www.nvidia.com/en-us/data-center/h200/), [HGX B200](https://www.nvidia.com/en-us/data-center/hgx/), [GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/) specs
