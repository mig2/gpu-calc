import { describe, it, expect } from 'vitest'
import { estimateTrainingRun } from './calculator'
import { getH100Reference, GPU_SKUS } from './gpu-data'
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
