import { describe, it, expect } from 'vitest'
import { estimateClassicalTabular } from './adapters/classical-tabular-adapter'
import type { BaseHardwareConfig, ClassicalTabularConfig } from './types'

function makeDefaultClassicalScenario(
  overrides: Partial<ClassicalTabularConfig & BaseHardwareConfig> = {},
): BaseHardwareConfig & ClassicalTabularConfig {
  return {
    modelFamily: 'classical_tabular',
    algorithm: 'lightgbm',
    rows: 1e6,
    columns: 100,
    boostingRounds: 1000,
    maxDepth: 8,
    bins: 256,
    cvFolds: 1,
    hyperparameterTrials: 1,
    cpuOrGpu: 'gpu',
    throughputCoefficient: 0,
    trainingWindowSeconds: 30 * 86_400,
    precision: 'BF16_DENSE',
    selectedGpuIds: ['h100-sxm'],
    mfuByGpuId: { 'h100-sxm': 0.40 },
    availability: 0.90,
    overheadFactor: 1.10,
    ...overrides,
  }
}

describe('estimateClassicalTabular', () => {
  it('computes work units including the depth factor', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    // 1M rows x 100 cols x 1000 rounds x (depth 8 / depth-6 calibration) x 1 fold x 1 trial
    expect(result.workUnits).toBeCloseTo(1e11 * (8 / 6), -4)
  })

  it('work units are linear in maxDepth, normalised at the depth-6 calibration point', () => {
    const atCalibration = estimateClassicalTabular(makeDefaultClassicalScenario({ maxDepth: 6 }))
    const doubled = estimateClassicalTabular(makeDefaultClassicalScenario({ maxDepth: 12 }))
    expect(atCalibration.workUnits).toBe(1e11)
    expect(doubled.workUnits).toBeCloseTo(atCalibration.workUnits * 2, -4)
  })

  it('CV folds and HP trials multiply work', () => {
    const base = estimateClassicalTabular(makeDefaultClassicalScenario())
    const multi = estimateClassicalTabular(makeDefaultClassicalScenario({ cvFolds: 5, hyperparameterTrials: 10 }))
    expect(multi.workUnits).toBeCloseTo(base.workUnits * 50, -6)
  })

  it('uses calibrated throughput when provided', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ throughputCoefficient: 1e9 }))
    expect(result.throughputUsed).toBe(1e9)
    expect(result.confidence).toBe('medium')
  })

  it('uses default throughput when not calibrated', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ throughputCoefficient: 0 }))
    expect(result.throughputUsed).toBeGreaterThan(0)
    // Benchmark-anchored defaults in their valid regime (>=1M dense rows) are
    // 'medium-low', not 'low'. 'low' is now reserved for extrapolated regimes.
    expect(result.confidence).toBe('medium-low')
  })

  it('drops to low confidence below the 1M-row calibration regime', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ rows: 1e5 }))
    expect(result.confidence).toBe('low')
  })

  it('drops to low confidence for a custom implementation', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'custom' }))
    expect(result.confidence).toBe('low')
  })

  it('lowering max_bin raises GPU throughput but leaves CPU flat', () => {
    const gpu256 = estimateClassicalTabular(makeDefaultClassicalScenario({ bins: 256 }))
    const gpu32 = estimateClassicalTabular(makeDefaultClassicalScenario({ bins: 32 }))
    const cpu256 = estimateClassicalTabular(makeDefaultClassicalScenario({ cpuOrGpu: 'cpu', bins: 256 }))
    const cpu32 = estimateClassicalTabular(makeDefaultClassicalScenario({ cpuOrGpu: 'cpu', bins: 32 }))
    // CatBoost Epsilon: 9.26x at 128 bins -> 13.33x at 32 bins = 1.44x over two halvings.
    expect(gpu32.throughputUsed / gpu256.throughputUsed).toBeCloseTo(1.73, 1)
    expect(cpu32.throughputUsed).toBe(cpu256.throughputUsed)
  })

  it('CPU is slower than GPU for XGBoost, but at parity for LightGBM', () => {
    // The old global CPU_THROUGHPUT_FACTOR = 0.1 made CPU unconditionally 10x
    // slower. GBM-Benchmarks measured 13.4x for XGBoost but 0.69x for
    // LightGBM-OpenCL, i.e. GPU *slower*, so the one-size factor was wrong in sign.
    const xgbGpu = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'xgboost', cpuOrGpu: 'gpu' }))
    const xgbCpu = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'xgboost', cpuOrGpu: 'cpu' }))
    expect(xgbCpu.estimatedSeconds / xgbGpu.estimatedSeconds).toBeCloseTo(16, 0)

    const lgbmGpu = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'lightgbm', cpuOrGpu: 'gpu' }))
    const lgbmCpu = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'lightgbm', cpuOrGpu: 'cpu' }))
    expect(lgbmCpu.estimatedSeconds).toBeCloseTo(lgbmGpu.estimatedSeconds, 5)
  })

  it('warns when no throughput coefficient', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    expect(result.warnings).toContainEqual(expect.stringContaining('No empirical throughput'))
  })

  it('trace has at least 5 entries', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    expect(result.trace.length).toBeGreaterThanOrEqual(5)
  })

  it('data breakdown includes key fields', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    expect(result.dataBreakdown['Algorithm']).toBe('lightgbm')
    expect(result.dataBreakdown['Implementation']).toBe('GPU')
    expect(String(result.dataBreakdown['Max depth'])).toContain('8')
    expect(String(result.dataBreakdown['Max bins'])).toContain('256')
  })
})

describe('estimateClassicalTabular: CV x HPO warnings', () => {
  // The old threshold was > 50, which fires on every realistic configuration:
  // TabArena itself uses 200 configs x 8 inner folds = 1,600 fits per model per
  // outer fold. 50 is now the informational floor, 500 the "dominates" line.
  it('stays quiet at a light-tuning multiplier', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ cvFolds: 5, hyperparameterTrials: 10 }))
    expect(result.warnings).not.toContainEqual(expect.stringContaining('dominates'))
    expect(result.warnings).not.toContainEqual(expect.stringContaining('most of the wall-clock'))
  })

  it('notes, but does not escalate, a 200x multiplier', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ cvFolds: 10, hyperparameterTrials: 20 }))
    expect(result.warnings).toContainEqual(expect.stringContaining('most of the wall-clock'))
    expect(result.warnings).not.toContainEqual(expect.stringContaining('dominates'))
  })

  it('escalates above 500x', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ cvFolds: 8, hyperparameterTrials: 200 }))
    expect(result.warnings).toContainEqual(expect.stringContaining('dominates'))
  })
})

describe('estimateClassicalTabular: GPU guidance warnings', () => {
  it('warns that GPU will be SLOWER below the 1M-row regime', () => {
    const result = estimateClassicalTabular(
      makeDefaultClassicalScenario({ algorithm: 'xgboost', rows: 1e5, cpuOrGpu: 'gpu' }),
    )
    expect(result.warnings).toContainEqual(expect.stringContaining('likely be SLOWER'))
    expect(result.gpuVerdict).toBe('gpu_likely_slower')
  })

  it('does not claim a 5-10x GPU speedup for LightGBM', () => {
    const cpu = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'lightgbm', cpuOrGpu: 'cpu' }))
    const joined = cpu.warnings.join(' ')
    expect(joined).toContain('0.5-1.5x')
    expect(joined).not.toContain('typically 5-10x faster')
  })

  it('cites the measured XGBoost speedup range when CPU is selected on large data', () => {
    const result = estimateClassicalTabular(
      makeDefaultClassicalScenario({ algorithm: 'xgboost', rows: 11e6, columns: 28, boostingRounds: 500, cpuOrGpu: 'cpu' }),
    )
    expect(result.warnings).toContainEqual(expect.stringContaining('13.4x'))
  })

  it('warns that the rows x columns model breaks on wide/sparse data', () => {
    const narrow = estimateClassicalTabular(makeDefaultClassicalScenario({ columns: 100 }))
    const wide = estimateClassicalTabular(makeDefaultClassicalScenario({ columns: 700 }))
    expect(narrow.warnings).not.toContainEqual(expect.stringContaining('one-hot encoded'))
    expect(wide.warnings).toContainEqual(expect.stringContaining('one-hot encoded'))
    expect(wide.warnings).toContainEqual(expect.stringContaining('~100x'))
  })
})

describe('estimateClassicalTabular: "do I need a GPU at all" verdict', () => {
  it('recommends exactly one GPU when a GPU is the answer', () => {
    // Airline: 115M x 13 x 500 rounds. 25 min on CPU, ~1.5 min on one V100.
    const result = estimateClassicalTabular(
      makeDefaultClassicalScenario({
        algorithm: 'xgboost', rows: 115_069_017, columns: 13, boostingRounds: 500, maxDepth: 6,
      }),
    )
    expect(result.gpuVerdict).toBe('gpu_recommended')
    expect(result.recommendedGpuCount).toBe(1)
    expect(result.gpuVerdictReason).toContain('multi-GPU GBDT is a niche')
  })

  it('says CPU is sufficient when the CPU job is already short', () => {
    const result = estimateClassicalTabular(
      makeDefaultClassicalScenario({
        algorithm: 'catboost', rows: 11e6, columns: 28, boostingRounds: 500, maxDepth: 6,
      }),
    )
    expect(result.gpuVerdict).toBe('cpu_sufficient')
    expect(result.cpuAlternativeSeconds).toBeLessThan(900)
  })

  it('says CPU is sufficient for LightGBM, where GPU buys nothing', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'lightgbm' }))
    expect(result.gpuVerdict).toBe('cpu_sufficient')
    expect(result.gpuVerdictReason).toContain('buys you nothing')
  })

  it('reports both alternatives regardless of which path is selected', () => {
    const onGpu = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'xgboost', cpuOrGpu: 'gpu' }))
    const onCpu = estimateClassicalTabular(makeDefaultClassicalScenario({ algorithm: 'xgboost', cpuOrGpu: 'cpu' }))
    expect(onGpu.cpuAlternativeSeconds).toBeCloseTo(onCpu.cpuAlternativeSeconds, 5)
    expect(onGpu.gpuAlternativeSeconds).toBeCloseTo(onCpu.gpuAlternativeSeconds, 5)
    expect(onGpu.estimatedSeconds).toBeCloseTo(onGpu.gpuAlternativeSeconds, 5)
    expect(onCpu.estimatedSeconds).toBeCloseTo(onCpu.cpuAlternativeSeconds, 5)
  })
})

/**
 * Regression fixtures from docs/research/2026-10-14-alt-model-families-research.md §6.3.
 *
 * TOLERANCE: 3x, asserted two-sided. These coefficients are geometric means of
 * published wall-clock over datasets whose per-dataset rates span 10x within a
 * single library (GBM-Benchmarks XGBoost GPU: 1.52e8 on CoverType to 1.78e10 on
 * Bosch), measured on 2018 V100 hardware, with the CPU baselines on a different
 * machine again. A single scalar cannot do better than a small multiple. 3x is
 * the honest band; the observed worst case across these fixtures is 1.8x, so the
 * tolerance has real headroom and the test still fails on a coefficient that is
 * wrong by an order of magnitude — which is exactly the defect it guards against.
 */
describe('estimateClassicalTabular: §6.3 regression fixtures', () => {
  const TOLERANCE = 3

  type Fixture = {
    name: string
    config: Partial<ClassicalTabularConfig>
    measuredSeconds: number
  }

  const fixtures: Fixture[] = [
    {
      // The headline anchor: 28-64-core CPU server, 500 rounds, Higgs.
      name: 'lgbm-cpu-higgs-official (2x E5-2690v4, 28 cores)',
      config: { algorithm: 'lightgbm', rows: 10.5e6, columns: 28, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'cpu' },
      measuredSeconds: 130.094,
    },
    {
      name: 'lgbm-cpu-msltr',
      config: { algorithm: 'lightgbm', rows: 2_270_296, columns: 137, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'cpu' },
      measuredSeconds: 70.417,
    },
    {
      name: 'xgb-gpu-higgs (V100-16GB)',
      config: { algorithm: 'xgboost', rows: 11e6, columns: 28, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'gpu' },
      measuredSeconds: 34.08,
    },
    {
      name: 'xgb-cpu-higgs (64 vCPU E5-2686v4)',
      config: { algorithm: 'xgboost', rows: 11e6, columns: 28, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'cpu' },
      measuredSeconds: 470.09,
    },
    {
      name: 'xgb-gpu-airline (V100)',
      config: { algorithm: 'xgboost', rows: 115_069_017, columns: 13, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'gpu' },
      measuredSeconds: 144.86,
    },
    {
      name: 'xgb-gpu-synthetic (V100)',
      config: { algorithm: 'xgboost', rows: 10e6, columns: 100, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'gpu' },
      measuredSeconds: 38.49,
    },
    {
      name: 'cat-gpu-higgs (V100)',
      config: { algorithm: 'catboost', rows: 11e6, columns: 28, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'gpu' },
      measuredSeconds: 30.15,
    },
  ]

  for (const { name, config, measuredSeconds } of fixtures) {
    it(`${name}: within ${TOLERANCE}x of ${measuredSeconds}s`, () => {
      const result = estimateClassicalTabular(makeDefaultClassicalScenario(config))
      expect(result.estimatedSeconds).toBeGreaterThan(measuredSeconds / TOLERANCE)
      expect(result.estimatedSeconds).toBeLessThan(measuredSeconds * TOLERANCE)
    })
  }

  it('cuml-rf-large: within 5x of 1.12s, the wider band the source justifies', () => {
    // cuML measured 2.23e10 work units/s on a Blackwell RTX PRO 6000, but §3.3
    // deliberately recommends 5e9 — the conservative low end of the 3e9-3.2e10
    // range — because the cuML CPU baseline is scikit-learn, a much weaker
    // implementation than LightGBM. Holding that conservatism costs ~4.5x here.
    const result = estimateClassicalTabular(
      makeDefaultClassicalScenario({
        algorithm: 'random_forest', rows: 1_953_125, columns: 128, boostingRounds: 100, maxDepth: 6, cpuOrGpu: 'gpu',
      }),
    )
    expect(result.estimatedSeconds).toBeGreaterThan(1.12 / 5)
    expect(result.estimatedSeconds).toBeLessThan(1.12 * 5)
  })

  it('lgbm-gpu-higgs: the OpenCL backend is outside the model, but the verdict is right', () => {
    // Measured 529.54s on a GTX 1080 via OpenCL vs 306.98s on CPU. The shipped
    // coefficient is for the newer CUDA backend, so the seconds are ~5x optimistic.
    // What the calculator must get right is the decision, not that number.
    const result = estimateClassicalTabular(
      makeDefaultClassicalScenario({
        algorithm: 'lightgbm', rows: 11e6, columns: 28, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'gpu',
      }),
    )
    expect(result.gpuVerdict).toBe('cpu_sufficient')
    expect(result.gpuAlternativeSeconds).toBeLessThanOrEqual(result.cpuAlternativeSeconds)
  })

  describe('sparse counter-fixtures: the model must be shown to fail here', () => {
    it('lgbm-cpu-expo (11M x 700 one-hot) over-estimates and warns', () => {
      const result = estimateClassicalTabular(
        makeDefaultClassicalScenario({
          algorithm: 'lightgbm', rows: 11e6, columns: 700, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'cpu',
        }),
      )
      // Measured 62.607s. EFB collapses the one-hot columns, so rows x columns
      // over-counts by ~35x. The estimate is wrong on purpose; the warning is the fix.
      expect(result.estimatedSeconds).toBeGreaterThan(62.607 * 10)
      expect(result.warnings).toContainEqual(expect.stringContaining('one-hot encoded'))
    })

    it('lgbm-cpu-allstate (13.2M x 4,228 one-hot) over-estimates and warns', () => {
      const result = estimateClassicalTabular(
        makeDefaultClassicalScenario({
          algorithm: 'lightgbm', rows: 13_184_290, columns: 4228, boostingRounds: 500, maxDepth: 6, cpuOrGpu: 'cpu',
        }),
      )
      // Measured 148.231s; rows x columns over-counts by ~100x here.
      expect(result.estimatedSeconds).toBeGreaterThan(148.231 * 10)
      expect(result.warnings).toContainEqual(expect.stringContaining('one-hot encoded'))
    })
  })
})

describe('estimateClassicalTabular: the repo default scenario', () => {
  // Before this fix, the default scenario (1M x 100 x 1000 rounds, LightGBM,
  // max_depth 8) reported 2,000s = 33 minutes on CPU and 200s on GPU, because
  // the GPU coefficient was 30x too low and CPU was a flat 0.1x of that.
  // Research §0.3: this is "a ~2-5 minute CPU job".
  it('is a minutes-long CPU job, not an hours-long one', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ cpuOrGpu: 'cpu' }))
    expect(result.estimatedSeconds).toBeGreaterThan(30)
    expect(result.estimatedSeconds).toBeLessThan(600)
    expect(result.estimatedHours).toBeLessThan(1 / 6)
  })

  it('needs no GPU', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    expect(result.gpuVerdict).toBe('cpu_sufficient')
  })

  it('becomes a real job once a realistic CV x HPO multiplier is applied', () => {
    // §3.6: 5 folds x 50 trials = 250x is the real wall-clock answer, and the
    // only place a GPU starts to matter for mid-size data.
    const result = estimateClassicalTabular(
      makeDefaultClassicalScenario({ cvFolds: 5, hyperparameterTrials: 50, cpuOrGpu: 'cpu' }),
    )
    expect(result.estimatedHours).toBeGreaterThan(5)
    expect(result.estimatedHours).toBeLessThan(10)
  })
})
