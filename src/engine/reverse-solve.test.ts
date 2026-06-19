import { describe, it, expect } from 'vitest'
import { solveForTrainingTime, solveForMaxModelSize } from './reverse-solve'
import { getH100Reference } from './gpu-data'
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

describe('solveForTrainingTime', () => {
  const h100 = getH100Reference()

  it('701 H100s for 70B should give ~30 days', () => {
    const scenario = makeDefaultScenario()
    const result = solveForTrainingTime(scenario, h100, 701)
    // Should be approximately 30 days (won't be exact due to ceil in forward direction)
    expect(result.trainingDays).toBeGreaterThan(29)
    expect(result.trainingDays).toBeLessThan(31)
  })

  it('256 H100s for 70B should give longer than 30 days', () => {
    const scenario = makeDefaultScenario()
    const result = solveForTrainingTime(scenario, h100, 256)
    expect(result.trainingDays).toBeGreaterThan(30)
  })

  it('produces trace with 6 entries', () => {
    const scenario = makeDefaultScenario()
    const result = solveForTrainingTime(scenario, h100, 701)
    expect(result.trace).toHaveLength(6)
  })
})

describe('solveForMaxModelSize', () => {
  const h100 = getH100Reference()

  it('701 H100s in 30 days should support ~70B', () => {
    const scenario = makeDefaultScenario()
    const result = solveForMaxModelSize(scenario, h100, 701)
    // Should be approximately 70B (within 5%)
    expect(result.maxParameters).toBeGreaterThan(65e9)
    expect(result.maxParameters).toBeLessThan(75e9)
  })

  it('8 H100s in 30 days should support ~7B', () => {
    const scenario = makeDefaultScenario({ modelParameters: 7e9 })
    const result = solveForMaxModelSize(scenario, h100, 8)
    expect(result.maxParameters).toBeGreaterThan(5e9)
    expect(result.maxParameters).toBeLessThan(9e9)
  })

  it('produces trace with 4 entries', () => {
    const scenario = makeDefaultScenario()
    const result = solveForMaxModelSize(scenario, h100, 701)
    expect(result.trace).toHaveLength(4)
  })
})
