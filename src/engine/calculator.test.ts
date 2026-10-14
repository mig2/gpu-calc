import { describe, it, expect } from 'vitest'
import { estimateTrainingRun } from './calculator'
import { getH100Reference, getGpuById } from './gpu-data'
import type { TrainingScenario } from './types'

function makeDefaultScenario(overrides: Partial<TrainingScenario> = {}): TrainingScenario {
  return {
    modelParameters: 70e9,
    tokensPerParameter: 20,
    trainingWindowSeconds: 30 * 86_400,
    precision: 'BF16_DENSE',
    selectedGpuIds: ['h100-sxm'],
    mfuByGpuId: { 'h100-sxm': 0.40 },
    availability: 0.90,
    overheadFactor: 1.10,
    trainingMode: 'FULL_PRETRAINING',
    memoryBytesPerParameter: 16,
    ...overrides,
  }
}

describe('estimateTrainingRun', () => {
  const h100 = getH100Reference()

  it('70B/30d/H100 returns 701 GPUs', () => {
    const scenario = makeDefaultScenario()
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.requiredGpus).toBe(701)
  })

  it('computes correct tokens for 70B model', () => {
    const scenario = makeDefaultScenario()
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.tokens).toBe(1.4e12)
  })

  it('computes correct base FLOPs for 70B model', () => {
    const scenario = makeDefaultScenario()
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.baseFlops).toBe(6 * 70e9 * 1.4e12)
  })

  it('computes correct total FLOPs with overhead', () => {
    const scenario = makeDefaultScenario()
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.totalFlops).toBeCloseTo(6 * 70e9 * 1.4e12 * 1.10, -15)
  })
})

describe('worked examples from Tech Spec §8', () => {
  const h100 = getH100Reference()

  // Each row: [modelParams, tokens, windowDays, h100Gpus, h200Gpus, b200Gpus, gb200Gpus]
  const examples: [number, number, number, number, number, number, number][] = [
    [7e9,   140e9,  30, 8,     8,     4,    3],
    [7e9,   140e9,   7, 31,    31,    14,   12],
    [13e9,  260e9,  30, 25,    25,    11,   10],
    [13e9,  260e9,   7, 104,   104,   46,   41],
    [70e9,  1.4e12, 30, 701,   701,   309,  278],
    [70e9,  1.4e12,  7, 3003,  3003,  1321, 1189],
    [405e9, 8.1e12, 30, 23450, 23450, 10313, 9282],
    [405e9, 8.1e12,  7, 100498, 100498, 44197, 39777],
  ]

  const skuExpected: [string, number][] = [
    ['h100-sxm', 0],
    ['h200-sxm', 1],
    ['b200-sxm', 2],
    ['gb200-nvl72-gpu', 3],
  ]

  for (const [modelParams, , windowDays, ...gpuCounts] of examples) {
    const modelLabel = modelParams >= 1e9 ? `${modelParams / 1e9}B` : `${modelParams / 1e6}M`

    for (const [skuId, idx] of skuExpected) {
      const expectedGpus = gpuCounts[idx]
      const gpu = getGpuById(skuId)!
      // All examples use 0.40 MFU for comparison (even H200)
      it(`${modelLabel}/${windowDays}d/${gpu.label} → ${expectedGpus} GPUs`, () => {
        const scenario = makeDefaultScenario({
          modelParameters: modelParams,
          tokensPerParameter: 20,
          trainingWindowSeconds: windowDays * 86_400,
          selectedGpuIds: [skuId],
          mfuByGpuId: { [skuId]: 0.40 },
        })
        const result = estimateTrainingRun(scenario, gpu, h100)
        expect(result.requiredGpus).toBe(expectedGpus)
      })
    }
  }
})

describe('sensitivity examples from Tech Spec §9', () => {
  const h100 = getH100Reference()

  const sensitivityCases: [string, number, number][] = [
    ['H100, MFU 30%', 0.30, 935],
    ['H100, MFU 40%', 0.40, 701],
    ['H100, MFU 50%', 0.50, 561],
  ]

  for (const [label, mfu, expectedGpus] of sensitivityCases) {
    it(`70B/30d ${label} → ${expectedGpus}`, () => {
      const scenario = makeDefaultScenario({
        mfuByGpuId: { 'h100-sxm': mfu },
      })
      const result = estimateTrainingRun(scenario, h100, h100)
      expect(result.requiredGpus).toBe(expectedGpus)
    })
  }

  it('H200 at 45% MFU → 623', () => {
    const h200 = getGpuById('h200-sxm')!
    const scenario = makeDefaultScenario({
      selectedGpuIds: ['h200-sxm'],
      mfuByGpuId: { 'h200-sxm': 0.45 },
    })
    const result = estimateTrainingRun(scenario, h200, h100)
    expect(result.requiredGpus).toBe(623)
  })

  it('H200 at 50% MFU → 561', () => {
    const h200 = getGpuById('h200-sxm')!
    const scenario = makeDefaultScenario({
      selectedGpuIds: ['h200-sxm'],
      mfuByGpuId: { 'h200-sxm': 0.50 },
    })
    const result = estimateTrainingRun(scenario, h200, h100)
    expect(result.requiredGpus).toBe(561)
  })
})

describe('property tests', () => {
  const h100 = getH100Reference()

  it('GPU count decreases when training window increases', () => {
    const short = makeDefaultScenario({ trainingWindowSeconds: 7 * 86_400 })
    const long = makeDefaultScenario({ trainingWindowSeconds: 60 * 86_400 })
    const shortResult = estimateTrainingRun(short, h100, h100)
    const longResult = estimateTrainingRun(long, h100, h100)
    expect(shortResult.requiredGpus).toBeGreaterThan(longResult.requiredGpus)
  })

  it('GPU count increases with TPP', () => {
    const low = makeDefaultScenario({ tokensPerParameter: 20 })
    const high = makeDefaultScenario({ tokensPerParameter: 100 })
    const lowResult = estimateTrainingRun(low, h100, h100)
    const highResult = estimateTrainingRun(high, h100, h100)
    expect(highResult.requiredGpus).toBeGreaterThan(lowResult.requiredGpus)
  })

  it('GPU count decreases when MFU increases', () => {
    const lowMfu = makeDefaultScenario({ mfuByGpuId: { 'h100-sxm': 0.30 } })
    const highMfu = makeDefaultScenario({ mfuByGpuId: { 'h100-sxm': 0.50 } })
    const lowResult = estimateTrainingRun(lowMfu, h100, h100)
    const highResult = estimateTrainingRun(highMfu, h100, h100)
    expect(lowResult.requiredGpus).toBeGreaterThan(highResult.requiredGpus)
  })

  it('H100 and H200 at same MFU return same GPU count', () => {
    const h200 = getGpuById('h200-sxm')!
    const scenarioH100 = makeDefaultScenario({ mfuByGpuId: { 'h100-sxm': 0.40 } })
    const scenarioH200 = makeDefaultScenario({
      selectedGpuIds: ['h200-sxm'],
      mfuByGpuId: { 'h200-sxm': 0.40 },
    })
    const h100Result = estimateTrainingRun(scenarioH100, h100, h100)
    const h200Result = estimateTrainingRun(scenarioH200, h200, h100)
    expect(h100Result.requiredGpus).toBe(h200Result.requiredGpus)
  })
})

describe('warnings', () => {
  const h100 = getH100Reference()

  it('warns for >70B model with <14 day window', () => {
    const scenario = makeDefaultScenario({
      modelParameters: 130e9,
      trainingWindowSeconds: 7 * 86_400,
    })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('Large-cluster warning'),
    )
  })

  it('does NOT warn for 70B model (boundary: not greater than 70B)', () => {
    const scenario = makeDefaultScenario({
      modelParameters: 70e9,
      trainingWindowSeconds: 7 * 86_400,
    })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.warnings).not.toContainEqual(
      expect.stringContaining('Large-cluster warning'),
    )
  })

  it('warns for >1024 GPUs', () => {
    const scenario = makeDefaultScenario({
      modelParameters: 70e9,
      trainingWindowSeconds: 7 * 86_400,
    })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.requiredGpus).toBeGreaterThan(1024)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('Distributed-systems warning'),
    )
  })

  it('warns for non-pretraining mode', () => {
    const scenario = makeDefaultScenario({ trainingMode: 'SFT' })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('may overestimate'),
    )
  })

  it('warns for FP8 precision', () => {
    const scenario = makeDefaultScenario({ precision: 'FP8_DENSE' })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('FP8 training is experimental'),
    )
  })

  it('explains the FP8 MFU factor and why it is applied', () => {
    const scenario = makeDefaultScenario({ precision: 'FP8_DENSE' })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('0.70x MFU factor is applied for FP8'),
    )
    expect(result.warnings).toContainEqual(
      expect.stringContaining('transformer-block linear layers'),
    )
  })

  it('explains the FP4 MFU factor', () => {
    const b200 = getGpuById('b200-sxm')!
    const scenario = makeDefaultScenario({
      precision: 'FP4_DENSE',
      selectedGpuIds: ['b200-sxm'],
      mfuByGpuId: { 'b200-sxm': 0.39 },
    })
    const result = estimateTrainingRun(scenario, b200, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('0.50x MFU factor is applied for FP4'),
    )
  })

  it('warns that NVFP4 pretraining is validated but not a drop-in flag', () => {
    const b200 = getGpuById('b200-sxm')!
    const scenario = makeDefaultScenario({
      precision: 'FP4_DENSE',
      selectedGpuIds: ['b200-sxm'],
      mfuByGpuId: { 'b200-sxm': 0.39 },
    })
    const result = estimateTrainingRun(scenario, b200, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('Random Hadamard transforms'),
    )
    expect(result.warnings).toContainEqual(
      expect.stringContaining('10T tokens'),
    )
  })

  it('does not emit the FP4 recipe warning at BF16', () => {
    const scenario = makeDefaultScenario()
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.warnings).not.toContainEqual(
      expect.stringContaining('Random Hadamard transforms'),
    )
    expect(result.warnings).not.toContainEqual(
      expect.stringContaining('MFU factor is applied'),
    )
  })

  it('warns that B300 has no BF16 compute gain over B200', () => {
    const b300 = getGpuById('b300-sxm')!
    const scenario = makeDefaultScenario({
      selectedGpuIds: ['b300-sxm'],
      mfuByGpuId: { 'b300-sxm': 0.40 },
    })
    const result = estimateTrainingRun(scenario, b300, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('same dense BF16 and FP8 peak as B200'),
    )
  })

  it('warns about B300 at FP8 too, where it is also B200-equivalent', () => {
    const b300 = getGpuById('b300-sxm')!
    const scenario = makeDefaultScenario({
      precision: 'FP8_DENSE',
      selectedGpuIds: ['b300-sxm'],
      mfuByGpuId: { 'b300-sxm': 0.40 },
    })
    const result = estimateTrainingRun(scenario, b300, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('same dense BF16 and FP8 peak as B200'),
    )
  })

  it('does not warn about B300 at FP4, where the uplift is real', () => {
    const b300 = getGpuById('b300-sxm')!
    const scenario = makeDefaultScenario({
      precision: 'FP4_DENSE',
      selectedGpuIds: ['b300-sxm'],
      mfuByGpuId: { 'b300-sxm': 0.40 },
    })
    const result = estimateTrainingRun(scenario, b300, h100)
    expect(result.warnings).not.toContainEqual(
      expect.stringContaining('same dense BF16 and FP8 peak as B200'),
    )
  })

  it('warns when FP4 falls back to a higher-precision peak on Hopper', () => {
    const scenario = makeDefaultScenario({ precision: 'FP4_DENSE' })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('no dense FP4 peak'),
    )
  })

  it('warns for H200 about same BF16 peak', () => {
    const h200 = getGpuById('h200-sxm')!
    const scenario = makeDefaultScenario({
      selectedGpuIds: ['h200-sxm'],
      mfuByGpuId: { 'h200-sxm': 0.45 },
    })
    const result = estimateTrainingRun(scenario, h200, h100)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('same raw dense BF16 peak as H100'),
    )
  })

  it('warns when memory bound exceeds compute bound', () => {
    const scenario = makeDefaultScenario({
      modelParameters: 7e9,
      memoryBytesPerParameter: 128,
    })
    const result = estimateTrainingRun(scenario, h100, h100)
    expect(result.memoryLowerBoundGpus).toBeGreaterThan(result.requiredGpus)
    expect(result.warnings).toContainEqual(
      expect.stringContaining('Memory bound exceeds compute bound'),
    )
  })
})

describe('formula trace', () => {
  const h100 = getH100Reference()

  it('produces trace for 70B canonical example', () => {
    const scenario = makeDefaultScenario()
    const result = estimateTrainingRun(scenario, h100, h100)
    // 8 entries, not 7: the hardware trace now opens with the explicit
    // effective-MFU derivation (base x precision factor x MoE factor).
    expect(result.trace).toHaveLength(8)
    expect(result.trace[0]).toContain('7.00e+10')
    expect(result.trace[1]).toContain('1.40e+12')
    expect(result.trace[4]).toContain('Effective MFU')
    expect(result.trace[7]).toContain('701')
  })
})
