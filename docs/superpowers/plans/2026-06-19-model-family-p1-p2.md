# Model Family Extensions (Phase 1+2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the GPU Calculator into a multi-model-family architecture and add the first non-LLM mode (time-series foundation models).

**Architecture:** Introduce a domain adapter pattern where each model family converts its inputs into `{ effectiveTokens, baseFlops, totalFlops, trace, warnings }`, then a shared hardware estimator converts FLOPs → GPU count. The existing `TrainingScenario` type becomes `LlmScenario` within a discriminated union. The store gains a `modelFamily` field and switches input panels via tabs. All 64 existing LLM tests must pass unchanged throughout.

**Tech Stack:** TypeScript, React, Zustand, Vitest, Recharts (all existing)

**Branch:** `feature/model-family-extensions`

---

## File Structure

```
src/engine/
  types.ts                      — MODIFY: add ModelFamily, ConfidenceLevel, BaseHardwareConfig,
                                   LlmScenario, TimeSeriesScenario, Scenario union, AdapterResult,
                                   HardwareEstimateResult. Keep TrainingScenario as type alias.
  hardware-estimator.ts         — CREATE: shared GPU count from FLOPs (extracted from calculator.ts)
  adapters/
    adapter-types.ts            — CREATE: DomainAdapter interface
    llm-adapter.ts              — CREATE: LLM token/FLOP derivation
    time-series-adapter.ts      — CREATE: time-series window/patch/token derivation
  calculator.ts                 — MODIFY: orchestrator calling adapter → hardware estimator.
                                   Keep estimateTrainingRun signature working for backwards compat.
  calculator.test.ts            — NO CHANGES (all 64 tests must pass as-is)
  time-series-adapter.test.ts   — CREATE: golden cases + property + warning tests
  export.ts                     — MODIFY: add schema_version, modelFamily support
  reverse-solve.ts              — MODIFY: use architectureFactor instead of hardcoded 6
  calibration.ts                — MODIFY: use architectureFactor instead of hardcoded 6

src/store/
  scenario-store.ts             — MODIFY: add modelFamily, time-series fields, family switching

src/components/
  ModelFamilyTabs.tsx           — CREATE: tab bar for LLM / Time Series / Tabular / Classical
  LlmForm.tsx                   — CREATE: extracted from ScenarioForm + architectureFactor + tokensOverride
  TimeSeriesForm.tsx            — CREATE: time-series specific inputs
  TimeSeriesBreakdown.tsx       — CREATE: data-unit breakdown card
  ConfidenceBadge.tsx           — CREATE: confidence label component
  ScenarioForm.tsx              — MODIFY: becomes a switcher that renders LlmForm or TimeSeriesForm
  App.tsx                       — MODIFY: add ModelFamilyTabs, update header title
  ResultCards.tsx                — MODIFY: add confidence badge, show data breakdown for TS
  SensitivityMatrix.tsx         — MODIFY: support time-series sensitivity axes
```

---

## Task 1: New Types (Issue #18)

**Files:**
- Modify: `src/engine/types.ts`

- [ ] **Step 1: Add new types while keeping TrainingScenario intact**

Add to the end of `src/engine/types.ts`:

```typescript
// === Model Family Extension Types ===

export type ModelFamily = 'llm' | 'time_series_foundation' | 'tabular_foundation' | 'classical_tabular';

export type ConfidenceLevel = 'high' | 'medium' | 'medium-low' | 'low';

export type TimeSeriesTokenizationMode = 'channel_compressed' | 'channel_expanded' | 'custom';

export type TimeSeriesArchitectureType =
  | 'decoder_transformer'
  | 'encoder_transformer'
  | 'encoder_decoder'
  | 'patch_transformer'
  | 'custom';

/** Hardware config shared across all model families */
export type BaseHardwareConfig = {
  trainingWindowSeconds: number;
  precision: Precision;
  selectedGpuIds: string[];
  mfuByGpuId: Record<string, number>;
  availability: number;
  overheadFactor: number;
};

/** LLM-specific scenario fields */
export type LlmConfig = {
  modelFamily: 'llm';
  modelParameters: number;
  tokensPerParameter: number;
  trainingTokensOverride?: number;
  architectureFactor: number;
  trainingMode: TrainingMode;
  memoryBytesPerParameter: number;
};

/** Time-series foundation model fields */
export type TimeSeriesConfig = {
  modelFamily: 'time_series_foundation';
  modelParameters: number;
  numberOfSeries: number;
  averageTimestepsPerSeries: number;
  variablesPerSeries: number;
  lookbackWindow: number;
  forecastHorizon: number;
  stride: number;
  patchSize: number;
  tokenizationMode: TimeSeriesTokenizationMode;
  customTokensPerWindow?: number;
  epochs: number;
  architectureType: TimeSeriesArchitectureType;
  architectureFactor: number;
  memoryBytesPerParameter: number;
};

export type Scenario =
  | (BaseHardwareConfig & LlmConfig)
  | (BaseHardwareConfig & TimeSeriesConfig);

/** Result from a domain adapter before hardware estimation */
export type AdapterResult = {
  effectiveTokens: number;
  baseFlops: number;
  totalFlops: number;
  trace: string[];
  warnings: string[];
  confidence: ConfidenceLevel;
  dataBreakdown?: Record<string, string | number>;
};
```

- [ ] **Step 2: Run tests to verify nothing broke**

```bash
npx vitest run
```

Expected: 64 tests pass. We only added new types, no existing code changed.

- [ ] **Step 3: Commit**

```bash
git add src/engine/types.ts
git commit -m "feat: add discriminated union types for model families

ModelFamily, ConfidenceLevel, LlmConfig, TimeSeriesConfig, Scenario
union, AdapterResult. Existing TrainingScenario unchanged.

Refs #18"
```

---

## Task 2: Hardware Estimator (Issue #19)

**Files:**
- Create: `src/engine/hardware-estimator.ts`

- [ ] **Step 1: Extract the shared GPU calculation logic**

Create `src/engine/hardware-estimator.ts`:

```typescript
import type { GpuSku, Precision, BaseHardwareConfig } from './types';

export type HardwareEstimateInput = {
  totalFlops: number;
  modelParameters: number;
  memoryBytesPerParameter: number;
  hardware: BaseHardwareConfig;
};

export type HardwareEstimateResult = {
  gpuId: string;
  requiredGpus: number;
  h100Equivalents: number;
  sustainedFlopsPerGpu: number;
  memoryLowerBoundGpus: number;
  hardwareTrace: string[];
};

export function estimateHardware(
  input: HardwareEstimateInput,
  gpu: GpuSku,
  h100Reference: GpuSku,
): HardwareEstimateResult {
  const peak = getPeakFlops(gpu, input.hardware.precision);
  const mfu = input.hardware.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
  const sustained = peak * mfu * input.hardware.availability;
  const requiredGpus = Math.ceil(
    input.totalFlops / (input.hardware.trainingWindowSeconds * sustained),
  );

  const h100Peak = getPeakFlops(h100Reference, input.hardware.precision);
  const h100Mfu = input.hardware.mfuByGpuId[h100Reference.id] ?? h100Reference.defaultMfu;
  const h100Sustained = h100Peak * h100Mfu * input.hardware.availability;
  const h100Equivalents = (requiredGpus * sustained) / h100Sustained;

  const usableMemBytes = gpu.memoryGb * 1e9 * 0.85;
  const memoryLowerBoundGpus = Math.ceil(
    (input.modelParameters * input.memoryBytesPerParameter) / usableMemBytes,
  );

  const fmt = (n: number) => n.toExponential(2);

  return {
    gpuId: gpu.id,
    requiredGpus,
    h100Equivalents,
    sustainedFlopsPerGpu: sustained,
    memoryLowerBoundGpus,
    hardwareTrace: [
      `${gpu.label} sustained = ${fmt(peak)} × ${mfu} × ${input.hardware.availability} = ${fmt(sustained)} FLOP/s`,
      `Training window = ${input.hardware.trainingWindowSeconds.toLocaleString()} seconds`,
      `GPUs = ceil(${fmt(input.totalFlops)} / (${input.hardware.trainingWindowSeconds.toLocaleString()} × ${fmt(sustained)})) = ${requiredGpus}`,
    ],
  };
}

export function getPeakFlops(gpu: GpuSku, precision: Precision): number {
  return precision === 'BF16_DENSE'
    ? gpu.bf16DenseFlops
    : (gpu.fp8DenseFlops ?? gpu.bf16DenseFlops);
}
```

- [ ] **Step 2: Run tests**

```bash
npx vitest run
```

Expected: 64 tests pass. New file has no consumers yet.

- [ ] **Step 3: Commit**

```bash
git add src/engine/hardware-estimator.ts
git commit -m "feat: extract shared hardware estimator from calculator

Pure function: totalFlops + GPU specs → GPU count, H100 equiv, memory bound.

Refs #19"
```

---

## Task 3: LLM Adapter (Issue #19)

**Files:**
- Create: `src/engine/adapters/adapter-types.ts`
- Create: `src/engine/adapters/llm-adapter.ts`

- [ ] **Step 1: Create adapter interface**

Create `src/engine/adapters/adapter-types.ts`:

```typescript
import type { AdapterResult, GpuSku, Scenario } from '../types';

export type DomainAdapter<T extends Scenario = Scenario> = {
  computeFlops: (scenario: T) => AdapterResult;
};
```

- [ ] **Step 2: Create LLM adapter**

Create `src/engine/adapters/llm-adapter.ts`:

```typescript
import type { AdapterResult, BaseHardwareConfig, LlmConfig, ConfidenceLevel, TrainingMode } from '../types';

type LlmScenario = BaseHardwareConfig & LlmConfig;

function formatSci(n: number, digits = 2): string {
  return n.toExponential(digits);
}

function getConfidence(trainingMode: TrainingMode): ConfidenceLevel {
  switch (trainingMode) {
    case 'FULL_PRETRAINING':
      return 'high';
    case 'CONTINUED_PRETRAINING':
      return 'medium';
    default:
      return 'medium-low';
  }
}

export function computeLlmFlops(scenario: LlmScenario): AdapterResult {
  const N = scenario.modelParameters;
  const D = scenario.trainingTokensOverride ?? (scenario.tokensPerParameter * N);
  const baseFlops = scenario.architectureFactor * N * D;
  const totalFlops = baseFlops * scenario.overheadFactor;

  const tokenSource = scenario.trainingTokensOverride
    ? `D = ${formatSci(scenario.trainingTokensOverride)} (override)`
    : `D = ${scenario.tokensPerParameter} × N = ${formatSci(D)}`;

  const warnings = buildLlmWarnings(scenario, N);

  return {
    effectiveTokens: D,
    baseFlops,
    totalFlops,
    confidence: getConfidence(scenario.trainingMode),
    trace: [
      `N = ${formatSci(N)}`,
      tokenSource,
      `Base FLOPs = ${scenario.architectureFactor} × N × D = ${formatSci(baseFlops)}`,
      `Total FLOPs = ${formatSci(baseFlops)} × ${scenario.overheadFactor} = ${formatSci(totalFlops)}`,
    ],
    warnings,
    dataBreakdown: {
      'Model parameters': formatSci(N),
      'Training tokens': formatSci(D),
      'Token source': scenario.trainingTokensOverride ? 'Manual override' : `${scenario.tokensPerParameter} × N`,
      'Architecture factor': scenario.architectureFactor,
    },
  };
}

function buildLlmWarnings(scenario: LlmScenario, N: number): string[] {
  const warnings: string[] = [];

  if (N > 70e9 && scenario.trainingWindowSeconds < 14 * 86_400) {
    warnings.push(
      'Large-cluster warning: training a model >70B in under 14 days requires a very large cluster with significant operational complexity.',
    );
  }

  if (scenario.trainingMode !== 'FULL_PRETRAINING') {
    warnings.push(
      'The full-pretraining formula (6ND) may overestimate compute for this training mode. Task-specific formulas are recommended.',
    );
  }

  if (scenario.precision === 'FP8_DENSE') {
    warnings.push(
      'FP8 training is experimental. End-to-end FP8 recipes may not deliver the simple peak-based estimate shown here.',
    );
  }

  return warnings;
}
```

- [ ] **Step 3: Run tests**

```bash
npx vitest run
```

Expected: 64 tests pass.

- [ ] **Step 4: Commit**

```bash
git add src/engine/adapters/
git commit -m "feat: LLM domain adapter with architectureFactor and tokensOverride

Extracts LLM-specific token/FLOP derivation from calculator.
Supports configurable architecture factor (default 6) and optional
training tokens override.

Refs #19 #24"
```

---

## Task 4: Rewire Calculator as Orchestrator (Issue #19)

**Files:**
- Modify: `src/engine/calculator.ts`

This is the critical step. The existing `estimateTrainingRun` signature must continue to work identically so all 64 tests pass. Internally, we rewire it to use the adapter + hardware estimator.

- [ ] **Step 1: Rewire calculator internals**

Replace `src/engine/calculator.ts` with:

```typescript
import type { GpuSku, TrainingScenario, EstimateResult } from './types';
import { estimateHardware } from './hardware-estimator';
import { computeLlmFlops } from './adapters/llm-adapter';

/**
 * Main entry point — backwards-compatible with existing TrainingScenario.
 * Internally delegates to LLM adapter + hardware estimator.
 */
export function estimateTrainingRun(
  scenario: TrainingScenario,
  gpu: GpuSku,
  h100Reference: GpuSku,
): EstimateResult {
  // Convert legacy TrainingScenario to LlmScenario shape
  const llmScenario = {
    modelFamily: 'llm' as const,
    modelParameters: scenario.modelParameters,
    tokensPerParameter: scenario.tokensPerParameter,
    architectureFactor: 6,
    trainingMode: scenario.trainingMode,
    memoryBytesPerParameter: scenario.memoryBytesPerParameter,
    trainingWindowSeconds: scenario.trainingWindowSeconds,
    precision: scenario.precision,
    selectedGpuIds: scenario.selectedGpuIds,
    mfuByGpuId: scenario.mfuByGpuId,
    availability: scenario.availability,
    overheadFactor: scenario.overheadFactor,
  };

  const adapterResult = computeLlmFlops(llmScenario);

  const hwResult = estimateHardware(
    {
      totalFlops: adapterResult.totalFlops,
      modelParameters: scenario.modelParameters,
      memoryBytesPerParameter: scenario.memoryBytesPerParameter,
      hardware: {
        trainingWindowSeconds: scenario.trainingWindowSeconds,
        precision: scenario.precision,
        selectedGpuIds: scenario.selectedGpuIds,
        mfuByGpuId: scenario.mfuByGpuId,
        availability: scenario.availability,
        overheadFactor: scenario.overheadFactor,
      },
    },
    gpu,
    h100Reference,
  );

  // Merge adapter warnings with hardware-level warnings
  const warnings = [...adapterResult.warnings];
  if (hwResult.requiredGpus > 1024) {
    warnings.push(
      `Distributed-systems warning: ${hwResult.requiredGpus.toLocaleString()} GPUs implies challenges with networking, checkpointing, stragglers, and cluster fragmentation.`,
    );
  }
  if (gpu.id === 'h200-sxm') {
    warnings.push(
      'H200 has the same raw dense BF16 peak as H100. Its advantage is more HBM capacity and bandwidth, which can improve achieved MFU in memory-stressed runs.',
    );
  }
  if (hwResult.memoryLowerBoundGpus > hwResult.requiredGpus) {
    warnings.push(
      `Memory bound exceeds compute bound: at least ${hwResult.memoryLowerBoundGpus} GPUs needed for model state memory alone (vs ${hwResult.requiredGpus} for compute).`,
    );
  }

  // Combine traces
  const trace = [...adapterResult.trace, ...hwResult.hardwareTrace];

  return {
    gpuId: gpu.id,
    tokens: adapterResult.effectiveTokens,
    baseFlops: adapterResult.baseFlops,
    totalFlops: adapterResult.totalFlops,
    sustainedFlopsPerGpu: hwResult.sustainedFlopsPerGpu,
    requiredGpus: hwResult.requiredGpus,
    h100Equivalents: hwResult.h100Equivalents,
    memoryLowerBoundGpus: hwResult.memoryLowerBoundGpus,
    warnings,
    trace,
  };
}
```

- [ ] **Step 2: Run ALL tests**

```bash
npx vitest run
```

Expected: ALL 64 tests pass. This is the critical gate. If any test fails, the rewiring has a bug — fix before proceeding.

- [ ] **Step 3: Run TypeScript check**

```bash
npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
git add src/engine/calculator.ts
git commit -m "refactor: rewire calculator as orchestrator using adapter + hardware estimator

estimateTrainingRun now delegates to computeLlmFlops → estimateHardware.
All 64 existing tests pass unchanged. This enables adding new model
family adapters without modifying the core calculation path.

Refs #19"
```

---

## Task 5: Time-Series Adapter (Issue #25)

**Files:**
- Create: `src/engine/adapters/time-series-adapter.ts`
- Create: `src/engine/time-series-adapter.test.ts`

- [ ] **Step 1: Write golden test case first**

Create `src/engine/time-series-adapter.test.ts`:

```typescript
import { describe, it, expect } from 'vitest'
import { computeTimeSeriesFlops } from './adapters/time-series-adapter'
import type { BaseHardwareConfig, TimeSeriesConfig } from './types'

function makeDefaultTsScenario(
  overrides: Partial<TimeSeriesConfig & BaseHardwareConfig> = {},
): BaseHardwareConfig & TimeSeriesConfig {
  return {
    modelFamily: 'time_series_foundation',
    modelParameters: 1e9,
    numberOfSeries: 10e6,
    averageTimestepsPerSeries: 1000,
    variablesPerSeries: 4,
    lookbackWindow: 256,
    forecastHorizon: 64,
    stride: 64,
    patchSize: 16,
    tokenizationMode: 'channel_expanded',
    epochs: 1,
    architectureType: 'decoder_transformer',
    architectureFactor: 6,
    memoryBytesPerParameter: 16,
    trainingWindowSeconds: 30 * 86_400,
    precision: 'BF16_DENSE',
    selectedGpuIds: ['h100-sxm'],
    mfuByGpuId: { 'h100-sxm': 0.40 },
    availability: 0.90,
    overheadFactor: 1.10,
    ...overrides,
  }
}

describe('computeTimeSeriesFlops', () => {
  it('golden case: channel-expanded → 7.04B tokens, 4.224e19 base FLOPs', () => {
    const scenario = makeDefaultTsScenario()
    const result = computeTimeSeriesFlops(scenario)
    expect(result.effectiveTokens).toBe(7.04e9)
    expect(result.baseFlops).toBeCloseTo(4.224e19, 14)
    expect(result.confidence).toBe('medium')
  })

  it('channel-compressed returns 1/4 tokens vs channel-expanded with 4 variables', () => {
    const expanded = computeTimeSeriesFlops(makeDefaultTsScenario({ tokenizationMode: 'channel_expanded' }))
    const compressed = computeTimeSeriesFlops(makeDefaultTsScenario({ tokenizationMode: 'channel_compressed' }))
    expect(expanded.effectiveTokens).toBe(compressed.effectiveTokens * 4)
  })

  it('invalid geometry: lookback + horizon > timesteps → zero tokens + warning', () => {
    const scenario = makeDefaultTsScenario({
      averageTimestepsPerSeries: 200,
      lookbackWindow: 256,
      forecastHorizon: 64,
    })
    const result = computeTimeSeriesFlops(scenario)
    expect(result.effectiveTokens).toBe(0)
    expect(result.warnings).toContainEqual(expect.stringContaining('Invalid window geometry'))
  })

  it('more series → more tokens', () => {
    const few = computeTimeSeriesFlops(makeDefaultTsScenario({ numberOfSeries: 1e6 }))
    const many = computeTimeSeriesFlops(makeDefaultTsScenario({ numberOfSeries: 10e6 }))
    expect(many.effectiveTokens).toBeGreaterThan(few.effectiveTokens)
  })

  it('smaller stride → more windows → more tokens', () => {
    const wide = computeTimeSeriesFlops(makeDefaultTsScenario({ stride: 128 }))
    const narrow = computeTimeSeriesFlops(makeDefaultTsScenario({ stride: 32 }))
    expect(narrow.effectiveTokens).toBeGreaterThan(wide.effectiveTokens)
  })

  it('smaller patch → more tokens', () => {
    const big = computeTimeSeriesFlops(makeDefaultTsScenario({ patchSize: 32 }))
    const small = computeTimeSeriesFlops(makeDefaultTsScenario({ patchSize: 8 }))
    expect(small.effectiveTokens).toBeGreaterThan(big.effectiveTokens)
  })

  it('stride=1 warns about overlapping windows', () => {
    const scenario = makeDefaultTsScenario({ stride: 1 })
    const result = computeTimeSeriesFlops(scenario)
    expect(result.warnings).toContainEqual(expect.stringContaining('overlapping windows'))
  })

  it('high variables + channel-expanded warns about compute inflation', () => {
    const scenario = makeDefaultTsScenario({ variablesPerSeries: 100, tokenizationMode: 'channel_expanded' })
    const result = computeTimeSeriesFlops(scenario)
    expect(result.warnings).toContainEqual(expect.stringContaining('compute inflation'))
  })

  it('includes data breakdown with windows, patches, tokens', () => {
    const result = computeTimeSeriesFlops(makeDefaultTsScenario())
    expect(result.dataBreakdown).toBeDefined()
    expect(result.dataBreakdown!['Windows per series']).toBe(11)
    expect(result.dataBreakdown!['Tokens per window']).toBe(64)
  })

  it('trace has at least 6 entries', () => {
    const result = computeTimeSeriesFlops(makeDefaultTsScenario())
    expect(result.trace.length).toBeGreaterThanOrEqual(6)
  })

  it('custom tokenization uses customTokensPerWindow', () => {
    const scenario = makeDefaultTsScenario({
      tokenizationMode: 'custom',
      customTokensPerWindow: 128,
    })
    const result = computeTimeSeriesFlops(scenario)
    // 10M series × 11 windows × 128 tokens × 1 epoch
    expect(result.effectiveTokens).toBe(10e6 * 11 * 128)
  })

  it('multiple epochs multiply tokens', () => {
    const one = computeTimeSeriesFlops(makeDefaultTsScenario({ epochs: 1 }))
    const three = computeTimeSeriesFlops(makeDefaultTsScenario({ epochs: 3 }))
    expect(three.effectiveTokens).toBe(one.effectiveTokens * 3)
  })
})
```

- [ ] **Step 2: Run test to verify it fails**

```bash
npx vitest run src/engine/time-series-adapter.test.ts
```

Expected: FAIL — module not found.

- [ ] **Step 3: Implement time-series adapter**

Create `src/engine/adapters/time-series-adapter.ts`:

```typescript
import type { AdapterResult, BaseHardwareConfig, TimeSeriesConfig } from '../types';

type TimeSeriesScenario = BaseHardwareConfig & TimeSeriesConfig;

function formatSci(n: number, digits = 2): string {
  return n.toExponential(digits);
}

export function computeTimeSeriesFlops(scenario: TimeSeriesScenario): AdapterResult {
  const {
    modelParameters: N,
    numberOfSeries,
    averageTimestepsPerSeries,
    variablesPerSeries,
    lookbackWindow,
    forecastHorizon,
    stride,
    patchSize,
    tokenizationMode,
    customTokensPerWindow,
    epochs,
    architectureFactor,
    overheadFactor,
  } = scenario;

  const warnings: string[] = [];
  const trace: string[] = [];

  // Window generation
  const usableTimesteps = averageTimestepsPerSeries - lookbackWindow - forecastHorizon;

  if (usableTimesteps < 0) {
    warnings.push(
      'Invalid window geometry: lookback + horizon exceeds average timesteps per series. No training windows can be generated.',
    );
    return {
      effectiveTokens: 0,
      baseFlops: 0,
      totalFlops: 0,
      trace: [`N = ${formatSci(N)}`, 'Invalid window geometry — no tokens generated.'],
      warnings,
      confidence: 'medium',
      dataBreakdown: {
        'Windows per series': 0,
        'Tokens per window': 0,
        'Effective training tokens': 0,
      },
    };
  }

  const windowsPerSeries = Math.floor(usableTimesteps / stride) + 1;

  // Patch tokenization
  const patchesPerWindow = Math.ceil(lookbackWindow / patchSize);

  let tokensPerWindow: number;
  switch (tokenizationMode) {
    case 'channel_compressed':
      tokensPerWindow = patchesPerWindow;
      break;
    case 'channel_expanded':
      tokensPerWindow = variablesPerSeries * patchesPerWindow;
      break;
    case 'custom':
      tokensPerWindow = customTokensPerWindow ?? patchesPerWindow;
      break;
    default:
      tokensPerWindow = patchesPerWindow;
  }

  const effectiveTokens = numberOfSeries * windowsPerSeries * tokensPerWindow * epochs;
  const baseFlops = architectureFactor * N * effectiveTokens;
  const totalFlops = baseFlops * overheadFactor;

  // Trace
  trace.push(`N = ${formatSci(N)}`);
  trace.push(`Series = ${numberOfSeries.toLocaleString()}`);
  trace.push(`Windows/series = floor((${averageTimestepsPerSeries} - ${lookbackWindow} - ${forecastHorizon}) / ${stride}) + 1 = ${windowsPerSeries}`);
  trace.push(`Patches/window = ceil(${lookbackWindow} / ${patchSize}) = ${patchesPerWindow}`);
  trace.push(`Tokens/window = ${tokenizationMode === 'channel_expanded' ? `${variablesPerSeries} × ${patchesPerWindow} = ` : ''}${tokensPerWindow}`);
  trace.push(`Effective tokens = ${numberOfSeries.toLocaleString()} × ${windowsPerSeries} × ${tokensPerWindow}${epochs > 1 ? ` × ${epochs}` : ''} = ${formatSci(effectiveTokens)}`);
  trace.push(`Base FLOPs = ${architectureFactor} × ${formatSci(N)} × ${formatSci(effectiveTokens)} = ${formatSci(baseFlops)}`);
  trace.push(`Total FLOPs = ${formatSci(baseFlops)} × ${overheadFactor} = ${formatSci(totalFlops)}`);

  // Warnings
  if (stride <= lookbackWindow * 0.1) {
    warnings.push(
      `Small stride (${stride}) relative to lookback (${lookbackWindow}) creates many overlapping windows and may inflate data volume significantly.`,
    );
  }

  if (tokenizationMode === 'channel_expanded' && variablesPerSeries > 20) {
    warnings.push(
      `Channel-expanded tokenization with ${variablesPerSeries} variables creates ${tokensPerWindow} tokens per window. This may cause significant compute inflation.`,
    );
  }

  if (patchSize > lookbackWindow) {
    warnings.push(
      'Patch size is larger than lookback window. Each window produces less than one patch.',
    );
  }

  warnings.push(
    'Time-series compute estimate uses a transformer-style approximation (factor × N × tokens). This is not an empirically calibrated time-series scaling law.',
  );

  return {
    effectiveTokens,
    baseFlops,
    totalFlops,
    trace,
    warnings,
    confidence: 'medium',
    dataBreakdown: {
      'Series': numberOfSeries.toLocaleString(),
      'Timesteps per series': averageTimestepsPerSeries.toLocaleString(),
      'Windows per series': windowsPerSeries,
      'Patches per window': patchesPerWindow,
      'Tokens per window': tokensPerWindow,
      'Effective training tokens': formatSci(effectiveTokens),
    },
  };
}
```

- [ ] **Step 4: Run ALL tests**

```bash
npx vitest run
```

Expected: 64 existing + 12 new = 76 tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/engine/adapters/time-series-adapter.ts src/engine/time-series-adapter.test.ts
git commit -m "feat: time-series domain adapter with golden tests

Window/patch/token derivation for channel-compressed, channel-expanded,
and custom tokenization. Warnings for invalid geometry, overlap, and
compute inflation. 12 tests including golden case from spec.

Fixes #25
Fixes #26"
```

---

## Task 6: Store Refactor for Model Families (Issue #20)

**Files:**
- Modify: `src/store/scenario-store.ts`

- [ ] **Step 1: Add model family support to store**

Read `src/store/scenario-store.ts` first. Add:

1. `modelFamily: ModelFamily` field (default `'llm'`)
2. `timeSeriesConfig` holding TS-specific fields
3. `setModelFamily(family)` action
4. Time-series setters
5. Update `computeResults` to dispatch to the right adapter

The key design: keep the existing `scenario: TrainingScenario` for LLM mode working. Add a parallel `tsConfig` for time-series fields. When `modelFamily === 'time_series_foundation'`, `computeResults` uses the TS adapter + hardware estimator to produce `EstimateResult[]`.

Add these imports at the top:

```typescript
import type { ModelFamily } from '../engine/types';
import { computeTimeSeriesFlops } from '../engine/adapters/time-series-adapter';
import { estimateHardware } from '../engine/hardware-estimator';
```

Add to `ScenarioStore` type:

```typescript
modelFamily: ModelFamily;
tsConfig: {
  modelParameters: number;
  numberOfSeries: number;
  averageTimestepsPerSeries: number;
  variablesPerSeries: number;
  lookbackWindow: number;
  forecastHorizon: number;
  stride: number;
  patchSize: number;
  tokenizationMode: 'channel_compressed' | 'channel_expanded' | 'custom';
  customTokensPerWindow?: number;
  epochs: number;
  architectureType: 'decoder_transformer' | 'encoder_transformer' | 'encoder_decoder' | 'patch_transformer' | 'custom';
  architectureFactor: number;
  memoryBytesPerParameter: number;
};
setModelFamily: (family: ModelFamily) => void;
setTsField: <K extends keyof ScenarioStore['tsConfig']>(key: K, value: ScenarioStore['tsConfig'][K]) => void;
```

Add default TS config:

```typescript
const defaultTsConfig = {
  modelParameters: 1e9,
  numberOfSeries: 10e6,
  averageTimestepsPerSeries: 1000,
  variablesPerSeries: 1,
  lookbackWindow: 256,
  forecastHorizon: 64,
  stride: 64,
  patchSize: 16,
  tokenizationMode: 'channel_compressed' as const,
  epochs: 1,
  architectureType: 'decoder_transformer' as const,
  architectureFactor: 6,
  memoryBytesPerParameter: 16,
};
```

Update `computeResults` to check `modelFamily`:

```typescript
function computeResults(
  scenario: TrainingScenario,
  customGpus: GpuSku[],
  modelFamily: ModelFamily,
  tsConfig: typeof defaultTsConfig,
): EstimateResult[] {
  if (modelFamily === 'time_series_foundation') {
    return computeTimeSeriesResults(scenario, customGpus, tsConfig);
  }
  // existing LLM path
  const h100 = getH100Reference();
  const allGpus = getAllGpus(customGpus);
  return scenario.selectedGpuIds
    .map((id) => allGpus.find((g) => g.id === id) ?? getGpuById(id))
    .filter((gpu): gpu is NonNullable<typeof gpu> => gpu != null)
    .map((gpu) => estimateTrainingRun(scenario, gpu, h100));
}

function computeTimeSeriesResults(
  scenario: TrainingScenario,
  customGpus: GpuSku[],
  tsConfig: typeof defaultTsConfig,
): EstimateResult[] {
  const h100 = getH100Reference();
  const allGpus = getAllGpus(customGpus);
  const tsScenario = {
    modelFamily: 'time_series_foundation' as const,
    ...tsConfig,
    trainingWindowSeconds: scenario.trainingWindowSeconds,
    precision: scenario.precision,
    selectedGpuIds: scenario.selectedGpuIds,
    mfuByGpuId: scenario.mfuByGpuId,
    availability: scenario.availability,
    overheadFactor: scenario.overheadFactor,
  };

  const adapterResult = computeTimeSeriesFlops(tsScenario);

  return scenario.selectedGpuIds
    .map((id) => allGpus.find((g) => g.id === id) ?? getGpuById(id))
    .filter((gpu): gpu is NonNullable<typeof gpu> => gpu != null)
    .map((gpu) => {
      const hwResult = estimateHardware(
        {
          totalFlops: adapterResult.totalFlops,
          modelParameters: tsConfig.modelParameters,
          memoryBytesPerParameter: tsConfig.memoryBytesPerParameter,
          hardware: {
            trainingWindowSeconds: scenario.trainingWindowSeconds,
            precision: scenario.precision,
            selectedGpuIds: scenario.selectedGpuIds,
            mfuByGpuId: scenario.mfuByGpuId,
            availability: scenario.availability,
            overheadFactor: scenario.overheadFactor,
          },
        },
        gpu,
        h100,
      );
      return {
        gpuId: gpu.id,
        tokens: adapterResult.effectiveTokens,
        baseFlops: adapterResult.baseFlops,
        totalFlops: adapterResult.totalFlops,
        sustainedFlopsPerGpu: hwResult.sustainedFlopsPerGpu,
        requiredGpus: hwResult.requiredGpus,
        h100Equivalents: hwResult.h100Equivalents,
        memoryLowerBoundGpus: hwResult.memoryLowerBoundGpus,
        warnings: adapterResult.warnings,
        trace: [...adapterResult.trace, ...hwResult.hardwareTrace],
      };
    });
}
```

Update every setter to pass `modelFamily` and `tsConfig` to `computeResults`. Add `setModelFamily` and `setTsField` actions.

- [ ] **Step 2: Run tests**

```bash
npx vitest run
npx tsc --noEmit
```

Expected: 76 tests pass, TypeScript clean.

- [ ] **Step 3: Commit**

```bash
git add src/store/scenario-store.ts
git commit -m "feat: add model family support to Zustand store

Adds modelFamily field, time-series config, family switching.
computeResults dispatches to TS adapter when family is time_series.
Existing LLM path unchanged.

Refs #20"
```

---

## Task 7: Model Family Tabs + LLM/TS Form Switching (Issues #21, #27)

**Files:**
- Create: `src/components/ModelFamilyTabs.tsx`
- Create: `src/components/LlmForm.tsx`
- Create: `src/components/TimeSeriesForm.tsx`
- Modify: `src/components/ScenarioForm.tsx`
- Modify: `src/components/App.tsx`
- Modify: `src/index.css`

- [ ] **Step 1: Create ModelFamilyTabs**

Create `src/components/ModelFamilyTabs.tsx`:

```tsx
import { useScenarioStore } from '../store/scenario-store'
import type { ModelFamily } from '../engine/types'

const TABS: { family: ModelFamily; label: string; enabled: boolean }[] = [
  { family: 'llm', label: 'LLM', enabled: true },
  { family: 'time_series_foundation', label: 'Time Series', enabled: true },
  { family: 'tabular_foundation', label: 'Tabular', enabled: false },
  { family: 'classical_tabular', label: 'Classical', enabled: false },
]

export function ModelFamilyTabs() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const setModelFamily = useScenarioStore((s) => s.setModelFamily)

  return (
    <div className="family-tabs" role="tablist" aria-label="Model family">
      {TABS.map((tab) => (
        <button
          key={tab.family}
          role="tab"
          aria-selected={modelFamily === tab.family}
          className={`family-tab ${modelFamily === tab.family ? 'active' : ''} ${!tab.enabled ? 'disabled' : ''}`}
          onClick={() => tab.enabled && setModelFamily(tab.family)}
          disabled={!tab.enabled}
        >
          {tab.label}
          {!tab.enabled && <span className="coming-soon">Soon</span>}
        </button>
      ))}
    </div>
  )
}
```

- [ ] **Step 2: Create LlmForm (extract from ScenarioForm)**

Create `src/components/LlmForm.tsx` — copy the existing ScenarioForm content but renamed. This is the LLM-specific input panel. Identical to the current ScenarioForm.

```tsx
import { useState, useEffect } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import type { TrainingMode } from '../engine/types'
import { Tooltip } from './Tooltip'

const MODEL_PRESETS = [
  { label: '7B', value: 7e9 },
  { label: '13B', value: 13e9 },
  { label: '34B', value: 34e9 },
  { label: '70B', value: 70e9 },
  { label: '130B', value: 130e9 },
  { label: '405B', value: 405e9 },
]

const WINDOW_PRESETS = [
  { label: '7 days', value: 7 * 86_400 },
  { label: '14 days', value: 14 * 86_400 },
  { label: '30 days', value: 30 * 86_400 },
  { label: '60 days', value: 60 * 86_400 },
]

const TPP_PRESETS = [
  { label: '20 (Chinchilla)', value: 20 },
  { label: '50', value: 50 },
  { label: '100', value: 100 },
]

const TRAINING_MODES: { label: string; value: TrainingMode }[] = [
  { label: 'Full pretraining', value: 'FULL_PRETRAINING' },
  { label: 'Continued pretraining', value: 'CONTINUED_PRETRAINING' },
  { label: 'SFT', value: 'SFT' },
  { label: 'LoRA', value: 'LORA' },
  { label: 'RLHF', value: 'RLHF' },
  { label: 'Distillation', value: 'DISTILLATION' },
]

function parseModelSize(input: string): number | null {
  const match = input.trim().match(/^(\d+(?:\.\d+)?)\s*([MBTmbt])?$/i)
  if (!match) return null
  const num = parseFloat(match[1])
  if (num <= 0) return null
  const unit = (match[2] || 'B').toUpperCase()
  switch (unit) {
    case 'M': return num * 1e6
    case 'B': return num * 1e9
    case 'T': return num * 1e12
    default: return null
  }
}

function formatModelSize(params: number): string {
  if (params >= 1e12) return `${params / 1e12}T`
  if (params >= 1e9) return `${params / 1e9}B`
  return `${params / 1e6}M`
}

function parseWindow(input: string): number | null {
  const match = input.trim().match(/^(\d+(?:\.\d+)?)\s*(h|hours?|d|days?|w|weeks?)?$/i)
  if (!match) return null
  const num = parseFloat(match[1])
  if (num <= 0) return null
  const unit = (match[2] || 'd')[0].toLowerCase()
  switch (unit) {
    case 'h': return num * 3600
    case 'd': return num * 86_400
    case 'w': return num * 7 * 86_400
    default: return num * 86_400
  }
}

function formatWindow(seconds: number): string {
  const days = seconds / 86_400
  if (days >= 7 && days % 7 === 0) return `${days / 7}w`
  if (Number.isInteger(days)) return `${days}d`
  const hours = seconds / 3600
  return `${hours}h`
}

export function LlmForm() {
  const scenario = useScenarioStore((s) => s.scenario)
  const {
    setModelParameters,
    setTokensPerParameter,
    setTrainingWindowSeconds,
    setTrainingMode,
    setPrecision,
  } = useScenarioStore()

  const [modelInput, setModelInput] = useState(formatModelSize(scenario.modelParameters))
  const [windowInput, setWindowInput] = useState(formatWindow(scenario.trainingWindowSeconds))
  const [tppInput, setTppInput] = useState(String(scenario.tokensPerParameter))
  const [modelError, setModelError] = useState('')
  const [windowError, setWindowError] = useState('')

  useEffect(() => {
    const timer = setTimeout(() => {
      const parsed = parseModelSize(modelInput)
      if (parsed) { setModelError(''); setModelParameters(parsed) }
      else if (modelInput.trim()) { setModelError('Enter a number with M, B, or T (e.g. 70B)') }
    }, 400)
    return () => clearTimeout(timer)
  }, [modelInput, setModelParameters])

  useEffect(() => {
    const timer = setTimeout(() => {
      const parsed = parseWindow(windowInput)
      if (parsed) { setWindowError(''); setTrainingWindowSeconds(parsed) }
      else if (windowInput.trim()) { setWindowError('Enter a number with h, d, or w (e.g. 30d)') }
    }, 400)
    return () => clearTimeout(timer)
  }, [windowInput, setTrainingWindowSeconds])

  return (
    <>
      <fieldset>
        <legend><Tooltip text="Total number of model parameters. The calculator uses this with tokens-per-parameter to determine training compute.">Model Size</Tooltip></legend>
        <div className="input-with-presets">
          <input type="text" value={modelInput} onChange={(e) => setModelInput(e.target.value)} aria-label="Model parameters" placeholder="e.g. 70B" />
          {modelError && <span className="input-error">{modelError}</span>}
          <div className="presets">
            {MODEL_PRESETS.map((p) => (
              <button key={p.label} className={scenario.modelParameters === p.value ? 'active' : ''} onClick={() => { setModelInput(p.label); setModelParameters(p.value) }}>{p.label}</button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Wall-clock time available for training. Shorter windows require more GPUs.">Training Window</Tooltip></legend>
        <div className="input-with-presets">
          <input type="text" value={windowInput} onChange={(e) => setWindowInput(e.target.value)} aria-label="Training window" placeholder="e.g. 30d" />
          {windowError && <span className="input-error">{windowError}</span>}
          <div className="presets">
            {WINDOW_PRESETS.map((p) => (
              <button key={p.label} className={scenario.trainingWindowSeconds === p.value ? 'active' : ''} onClick={() => { setWindowInput(formatWindow(p.value)); setTrainingWindowSeconds(p.value) }}>{p.label}</button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend><Tooltip text="How many tokens to train on per parameter. Chinchilla-optimal is ~20. Modern recipes often use 50-100+ for inference efficiency.">Tokens per Parameter</Tooltip></legend>
        <div className="input-with-presets">
          <input type="number" value={tppInput} min={1} onChange={(e) => { setTppInput(e.target.value); const val = parseFloat(e.target.value); if (val > 0) setTokensPerParameter(val) }} aria-label="Tokens per parameter" />
          <div className="presets">
            {TPP_PRESETS.map((p) => (
              <button key={p.label} className={scenario.tokensPerParameter === p.value ? 'active' : ''} onClick={() => { setTppInput(String(p.value)); setTokensPerParameter(p.value) }}>{p.label}</button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Type of training run. The 6ND formula is calibrated for full pretraining; other modes may need less compute.">Training Mode</Tooltip></legend>
        <select value={scenario.trainingMode} onChange={(e) => setTrainingMode(e.target.value as TrainingMode)} aria-label="Training mode">
          {TRAINING_MODES.map((m) => (<option key={m.value} value={m.value}>{m.label}</option>))}
        </select>
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Numerical precision for training. BF16 is standard. FP8 is experimental and may not deliver theoretical peak.">Precision</Tooltip></legend>
        <select value={scenario.precision} onChange={(e) => setPrecision(e.target.value as 'BF16_DENSE' | 'FP8_DENSE')} aria-label="Precision">
          <option value="BF16_DENSE">BF16 Dense</option>
          <option value="FP8_DENSE">FP8 Dense (experimental)</option>
        </select>
      </fieldset>
    </>
  )
}
```

- [ ] **Step 3: Create TimeSeriesForm**

Create `src/components/TimeSeriesForm.tsx`:

```tsx
import { useScenarioStore } from '../store/scenario-store'
import { Tooltip } from './Tooltip'

export function TimeSeriesForm() {
  const tsConfig = useScenarioStore((s) => s.tsConfig)
  const setTsField = useScenarioStore((s) => s.setTsField)
  const scenario = useScenarioStore((s) => s.scenario)
  const { setTrainingWindowSeconds, setPrecision } = useScenarioStore()

  return (
    <>
      <fieldset>
        <legend><Tooltip text="Total model parameters for the time-series foundation model.">Model Parameters</Tooltip></legend>
        <input type="text" value={tsConfig.modelParameters >= 1e9 ? `${tsConfig.modelParameters / 1e9}B` : `${tsConfig.modelParameters / 1e6}M`}
          onChange={(e) => {
            const match = e.target.value.trim().match(/^(\d+(?:\.\d+)?)\s*([MBT])?$/i)
            if (match) {
              const num = parseFloat(match[1])
              const unit = (match[2] || 'B').toUpperCase()
              const val = unit === 'T' ? num * 1e12 : unit === 'M' ? num * 1e6 : num * 1e9
              if (val > 0) setTsField('modelParameters', val)
            }
          }}
          aria-label="Model parameters" placeholder="e.g. 1B" />
        <div className="presets">
          {[100e6, 500e6, 1e9, 5e9].map((v) => (
            <button key={v} className={tsConfig.modelParameters === v ? 'active' : ''} onClick={() => setTsField('modelParameters', v)}>
              {v >= 1e9 ? `${v / 1e9}B` : `${v / 1e6}M`}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Count of independent time series in the training corpus.">Number of Series</Tooltip></legend>
        <input type="number" value={tsConfig.numberOfSeries} min={1}
          onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('numberOfSeries', v) }}
          aria-label="Number of series" />
        <div className="presets">
          {[100e3, 1e6, 10e6, 100e6].map((v) => (
            <button key={v} className={tsConfig.numberOfSeries === v ? 'active' : ''} onClick={() => setTsField('numberOfSeries', v)}>
              {v >= 1e6 ? `${v / 1e6}M` : `${v / 1e3}K`}
            </button>
          ))}
        </div>
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Average number of timesteps per series after resampling and cleaning.">Avg Timesteps / Series</Tooltip></legend>
        <input type="number" value={tsConfig.averageTimestepsPerSeries} min={1} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('averageTimestepsPerSeries', v) }} aria-label="Timesteps per series" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Number of channels, features, or variates per series.">Variables / Series</Tooltip></legend>
        <input type="number" value={tsConfig.variablesPerSeries} min={1} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('variablesPerSeries', v) }} aria-label="Variables per series" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Input context length in timesteps.">Lookback Window</Tooltip></legend>
        <input type="number" value={tsConfig.lookbackWindow} min={1} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('lookbackWindow', v) }} aria-label="Lookback window" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Output forecast horizon in timesteps.">Forecast Horizon</Tooltip></legend>
        <input type="number" value={tsConfig.forecastHorizon} min={1} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('forecastHorizon', v) }} aria-label="Forecast horizon" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Window step size. Smaller = more windows = more data.">Stride</Tooltip></legend>
        <input type="number" value={tsConfig.stride} min={1} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('stride', v) }} aria-label="Stride" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Timesteps per token/patch. Smaller = more tokens per window.">Patch Size</Tooltip></legend>
        <input type="number" value={tsConfig.patchSize} min={1} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('patchSize', v) }} aria-label="Patch size" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="How multivariate series are tokenized. Channel-expanded multiplies tokens by variable count.">Tokenization Mode</Tooltip></legend>
        <select value={tsConfig.tokenizationMode} onChange={(e) => setTsField('tokenizationMode', e.target.value as typeof tsConfig.tokenizationMode)} aria-label="Tokenization mode">
          <option value="channel_compressed">Channel-compressed</option>
          <option value="channel_expanded">Channel-expanded</option>
          <option value="custom">Custom</option>
        </select>
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Number of passes over the data.">Epochs</Tooltip></legend>
        <input type="number" value={tsConfig.epochs} min={1} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setTsField('epochs', v) }} aria-label="Epochs" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="FLOPs multiplier per token. Default 6 matches dense transformer training.">Architecture Factor</Tooltip></legend>
        <input type="number" value={tsConfig.architectureFactor} min={1} step={1} onChange={(e) => { const v = parseFloat(e.target.value); if (v > 0) setTsField('architectureFactor', v) }} aria-label="Architecture factor" />
      </fieldset>

      <fieldset>
        <legend><Tooltip text="Wall-clock time available for training.">Training Window</Tooltip></legend>
        <input type="text" defaultValue="30d"
          onChange={(e) => {
            const match = e.target.value.trim().match(/^(\d+(?:\.\d+)?)\s*(h|d|w)?$/i)
            if (match) {
              const num = parseFloat(match[1])
              const unit = (match[2] || 'd')[0].toLowerCase()
              const secs = unit === 'h' ? num * 3600 : unit === 'w' ? num * 7 * 86_400 : num * 86_400
              if (secs > 0) setTrainingWindowSeconds(secs)
            }
          }}
          aria-label="Training window" placeholder="e.g. 30d" />
      </fieldset>

      <fieldset>
        <legend>Precision</legend>
        <select value={scenario.precision} onChange={(e) => setPrecision(e.target.value as 'BF16_DENSE' | 'FP8_DENSE')} aria-label="Precision">
          <option value="BF16_DENSE">BF16 Dense</option>
          <option value="FP8_DENSE">FP8 Dense (experimental)</option>
        </select>
      </fieldset>
    </>
  )
}
```

- [ ] **Step 4: Update ScenarioForm to be a switcher**

Replace `src/components/ScenarioForm.tsx`:

```tsx
import { useScenarioStore } from '../store/scenario-store'
import { LlmForm } from './LlmForm'
import { TimeSeriesForm } from './TimeSeriesForm'

export function ScenarioForm() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)

  return (
    <div className="scenario-form">
      <h2>Scenario</h2>
      {modelFamily === 'llm' && <LlmForm />}
      {modelFamily === 'time_series_foundation' && <TimeSeriesForm />}
      {(modelFamily === 'tabular_foundation' || modelFamily === 'classical_tabular') && (
        <p className="coming-soon-msg">This model family is coming soon.</p>
      )}
    </div>
  )
}
```

- [ ] **Step 5: Update App.tsx**

Add `ModelFamilyTabs` at the top of the input rail, update header title:

```tsx
import { ModelFamilyTabs } from './ModelFamilyTabs'
// ... existing imports

export default function App() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  // ... existing useEffect

  const title = modelFamily === 'llm'
    ? 'LLM Training GPU Calculator'
    : modelFamily === 'time_series_foundation'
    ? 'Time-Series GPU Calculator'
    : 'GPU Calculator'

  return (
    <div className="app">
      <header className="app-header">
        <h1>{title}</h1>
        <p className="subtitle">
          Estimate accelerator requirements for training models
        </p>
        {/* ... existing header-actions */}
      </header>
      <main className="app-main">
        <aside className="input-rail">
          <ModelFamilyTabs />
          <ScenarioForm />
          <GpuSelector />
          <AdvancedAssumptions />
          <CustomGpuEditor />
        </aside>
        {/* ... existing results-area */}
      </main>
    </div>
  )
}
```

- [ ] **Step 6: Add CSS for tabs**

Append to `src/index.css`:

```css
/* Model Family Tabs */
.family-tabs {
  display: flex;
  gap: 0;
  margin-bottom: 1.25rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
}

.family-tab {
  flex: 1;
  padding: 0.5rem 0.5rem;
  background: var(--surface);
  border: none;
  border-right: 1px solid var(--border);
  color: var(--text-muted);
  font-size: 0.75rem;
  font-weight: 600;
  cursor: pointer;
  transition: all 0.15s;
  text-align: center;
  position: relative;
}

.family-tab:last-child {
  border-right: none;
}

.family-tab:hover:not(.disabled) {
  background: var(--surface-hover);
  color: var(--text);
}

.family-tab.active {
  background: var(--accent);
  color: white;
}

.family-tab.disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.family-tab .coming-soon {
  display: block;
  font-size: 0.55rem;
  font-weight: 400;
  text-transform: uppercase;
  letter-spacing: 0.04em;
  opacity: 0.7;
}

.coming-soon-msg {
  color: var(--text-muted);
  font-size: 0.85rem;
  padding: 1rem 0;
}
```

- [ ] **Step 7: Run tests and tsc**

```bash
npx vitest run
npx tsc --noEmit
```

Expected: 76 tests pass, TypeScript clean.

- [ ] **Step 8: Commit**

```bash
git add src/components/ModelFamilyTabs.tsx src/components/LlmForm.tsx src/components/TimeSeriesForm.tsx src/components/ScenarioForm.tsx src/components/App.tsx src/index.css
git commit -m "feat: model family tabs with LLM and Time-Series input panels

Tab bar at top of input rail. Switching tabs changes the input form.
LlmForm extracted from ScenarioForm. TimeSeriesForm with all TS inputs.
Tabular and Classical tabs shown as coming soon.

Fixes #21
Fixes #27"
```

---

## Task 8: Confidence Badge (Issue #22)

**Files:**
- Create: `src/components/ConfidenceBadge.tsx`
- Modify: `src/components/ResultCards.tsx`
- Modify: `src/index.css`

- [ ] **Step 1: Create ConfidenceBadge**

Create `src/components/ConfidenceBadge.tsx`:

```tsx
import type { ConfidenceLevel } from '../engine/types'
import { Tooltip } from './Tooltip'

const LABELS: Record<ConfidenceLevel, { text: string; className: string; tip: string }> = {
  high: { text: 'High confidence', className: 'confidence-high', tip: 'Formula is widely used and inputs are well-defined.' },
  medium: { text: 'Medium confidence', className: 'confidence-medium', tip: 'Formula is plausible but architecture/data-unit choices matter.' },
  'medium-low': { text: 'Medium-low confidence', className: 'confidence-medium-low', tip: 'Estimate is useful for comparison but not procurement.' },
  low: { text: 'Low confidence', className: 'confidence-low', tip: 'Estimate depends heavily on custom implementation or empirical calibration.' },
}

export function ConfidenceBadge({ level }: { level: ConfidenceLevel }) {
  const info = LABELS[level]
  return (
    <Tooltip text={info.tip}>
      <span className={`confidence-badge ${info.className}`}>{info.text}</span>
    </Tooltip>
  )
}
```

- [ ] **Step 2: Add confidence to ResultCards**

In `ResultCards.tsx`, import `ConfidenceBadge` and `useScenarioStore`. Determine confidence based on model family. Add the badge after the GPU count heading. For LLM, derive from training mode. For TS, always 'medium'.

```tsx
import { ConfidenceBadge } from './ConfidenceBadge'
import type { ConfidenceLevel } from '../engine/types'

// Inside ResultCards:
const modelFamily = useScenarioStore((s) => s.modelFamily)
const scenario = useScenarioStore((s) => s.scenario)

function getConfidence(): ConfidenceLevel {
  if (modelFamily === 'time_series_foundation') return 'medium'
  if (modelFamily === 'llm') {
    if (scenario.trainingMode === 'FULL_PRETRAINING') return 'high'
    if (scenario.trainingMode === 'CONTINUED_PRETRAINING') return 'medium'
    return 'medium-low'
  }
  return 'low'
}

// Add after result-card-header h3:
<ConfidenceBadge level={getConfidence()} />
```

- [ ] **Step 3: Add CSS**

```css
/* Confidence Badges */
.confidence-badge {
  display: inline-block;
  padding: 0.125rem 0.5rem;
  border-radius: 999px;
  font-size: 0.65rem;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.confidence-high { background: rgba(34, 197, 94, 0.15); color: var(--success); }
.confidence-medium { background: rgba(99, 102, 241, 0.15); color: var(--accent); }
.confidence-medium-low { background: rgba(251, 191, 36, 0.15); color: var(--warning-text); }
.confidence-low { background: rgba(239, 68, 68, 0.15); color: var(--danger); }
```

- [ ] **Step 4: Run tests and tsc**

```bash
npx vitest run
npx tsc --noEmit
```

- [ ] **Step 5: Commit**

```bash
git add src/components/ConfidenceBadge.tsx src/components/ResultCards.tsx src/index.css
git commit -m "feat: confidence badges on result cards

High/medium/medium-low/low labels based on model family and training mode.
Color-coded with tooltips explaining each level.

Fixes #22"
```

---

## Task 9: Time-Series Data Breakdown Card (Issue #28)

**Files:**
- Create: `src/components/TimeSeriesBreakdown.tsx`
- Modify: `src/components/App.tsx`

- [ ] **Step 1: Create TimeSeriesBreakdown**

Create `src/components/TimeSeriesBreakdown.tsx`:

```tsx
import { useScenarioStore } from '../store/scenario-store'
import { computeTimeSeriesFlops } from '../engine/adapters/time-series-adapter'

export function TimeSeriesBreakdown() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const tsConfig = useScenarioStore((s) => s.tsConfig)
  const scenario = useScenarioStore((s) => s.scenario)

  if (modelFamily !== 'time_series_foundation') return null

  const tsScenario = {
    modelFamily: 'time_series_foundation' as const,
    ...tsConfig,
    trainingWindowSeconds: scenario.trainingWindowSeconds,
    precision: scenario.precision,
    selectedGpuIds: scenario.selectedGpuIds,
    mfuByGpuId: scenario.mfuByGpuId,
    availability: scenario.availability,
    overheadFactor: scenario.overheadFactor,
  }

  const result = computeTimeSeriesFlops(tsScenario)

  if (!result.dataBreakdown) return null

  return (
    <div className="ts-breakdown">
      <h3>Data Pipeline</h3>
      <dl className="result-details">
        {Object.entries(result.dataBreakdown).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
```

- [ ] **Step 2: Add to App.tsx before ResultCards**

```tsx
import { TimeSeriesBreakdown } from './TimeSeriesBreakdown'
// In results-area, before ResultCards:
<TimeSeriesBreakdown />
```

- [ ] **Step 3: Add CSS**

```css
.ts-breakdown {
  margin-bottom: 1rem;
}

.ts-breakdown h3 {
  font-size: 1rem;
  margin-bottom: 0.5rem;
}
```

- [ ] **Step 4: Run tests, tsc, commit**

```bash
npx vitest run
npx tsc --noEmit
git add src/components/TimeSeriesBreakdown.tsx src/components/App.tsx src/index.css
git commit -m "feat: time-series data breakdown card

Shows pipeline: series → windows → patches → tokens → FLOPs.
Only visible in time-series mode.

Fixes #28"
```

---

## Task 10: Update Export for Model Families (Issue #32)

**Files:**
- Modify: `src/engine/export.ts`

- [ ] **Step 1: Update export functions**

Read `src/engine/export.ts`. Add `schema_version` and `modelFamily` to JSON export. Update URL hash to include `family` parameter. Keep backwards compatibility.

In `exportToJson`:
```typescript
export function exportToJson(
  scenario: TrainingScenario,
  results: EstimateResult[],
  modelFamily: string = 'llm',
): string {
  return JSON.stringify({ schema_version: '2.0', model_family: modelFamily, scenario, results }, null, 2);
}
```

In `encodeScenarioToHash`, add:
```typescript
params.set('family', modelFamily || 'llm');
```

In `decodeScenarioFromHash`, read `family` param with fallback to `'llm'`.

- [ ] **Step 2: Update ExportControls to pass modelFamily**

Read and update `src/components/ExportControls.tsx` to read `modelFamily` from store and pass to export functions.

- [ ] **Step 3: Run tests, tsc, commit**

```bash
npx vitest run
npx tsc --noEmit
git add src/engine/export.ts src/components/ExportControls.tsx
git commit -m "feat: versioned export with model family support

JSON exports include schema_version 2.0 and model_family.
URL hash includes family parameter. Old URLs default to LLM.

Fixes #32"
```

---

## Task 11: Final Verification and Push

- [ ] **Step 1: Run full test suite**

```bash
npx vitest run
```

Expected: 76+ tests pass (64 original LLM + 12 time-series adapter).

- [ ] **Step 2: TypeScript check**

```bash
npx tsc --noEmit
```

- [ ] **Step 3: Push feature branch**

```bash
git push -u origin feature/model-family-extensions
```
