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
  it('computes correct work units', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    // 1M rows x 100 cols x 1000 rounds x 1 fold x 1 trial = 1e11
    expect(result.workUnits).toBe(1e11)
  })

  it('CV folds and HP trials multiply work', () => {
    const base = estimateClassicalTabular(makeDefaultClassicalScenario())
    const multi = estimateClassicalTabular(makeDefaultClassicalScenario({ cvFolds: 5, hyperparameterTrials: 10 }))
    expect(multi.workUnits).toBe(base.workUnits * 50)
  })

  it('uses calibrated throughput when provided', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ throughputCoefficient: 1e9 }))
    expect(result.throughputUsed).toBe(1e9)
    expect(result.confidence).toBe('medium')
  })

  it('uses default throughput when not calibrated', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ throughputCoefficient: 0 }))
    expect(result.throughputUsed).toBeGreaterThan(0)
    expect(result.confidence).toBe('low')
  })

  it('CPU throughput is lower than GPU', () => {
    const gpu = estimateClassicalTabular(makeDefaultClassicalScenario({ cpuOrGpu: 'gpu' }))
    const cpu = estimateClassicalTabular(makeDefaultClassicalScenario({ cpuOrGpu: 'cpu' }))
    expect(cpu.estimatedSeconds).toBeGreaterThan(gpu.estimatedSeconds)
  })

  it('warns when no throughput coefficient', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    expect(result.warnings).toContainEqual(expect.stringContaining('No empirical throughput'))
  })

  it('warns about high CV x HP multiplier', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ cvFolds: 10, hyperparameterTrials: 20 }))
    expect(result.warnings).toContainEqual(expect.stringContaining('dominates'))
  })

  it('warns about CPU implementation', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario({ cpuOrGpu: 'cpu' }))
    expect(result.warnings).toContainEqual(expect.stringContaining('CPU implementation'))
  })

  it('trace has at least 5 entries', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    expect(result.trace.length).toBeGreaterThanOrEqual(5)
  })

  it('data breakdown includes key fields', () => {
    const result = estimateClassicalTabular(makeDefaultClassicalScenario())
    expect(result.dataBreakdown['Algorithm']).toBe('lightgbm')
    expect(result.dataBreakdown['Implementation']).toBe('GPU')
  })
})
