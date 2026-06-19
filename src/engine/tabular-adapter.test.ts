import { describe, it, expect } from 'vitest'
import { computeTabularFlops } from './adapters/tabular-adapter'
import type { BaseHardwareConfig, TabularFoundationConfig } from './types'

function makeDefaultTabScenario(
  overrides: Partial<TabularFoundationConfig & BaseHardwareConfig> = {},
): BaseHardwareConfig & TabularFoundationConfig {
  return {
    modelFamily: 'tabular_foundation',
    modelParameters: 100e6,
    numberOfPretrainingTasks: 1e6,
    rowsPerTask: 1024,
    columnsPerTask: 100,
    tokenizationMode: 'row',
    epochs: 1,
    architectureType: 'row_transformer',
    architectureFactor: 6,
    testTimeComputeMultiplier: 1,
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

describe('computeTabularFlops', () => {
  it('golden case: row-tokenized → 1.024B tokens, 6.144e17 base FLOPs', () => {
    const result = computeTabularFlops(makeDefaultTabScenario())
    expect(result.effectiveTokens).toBe(1.024e9)
    expect(result.baseFlops).toBeCloseTo(6.144e17, 12)
    expect(result.confidence).toBe('medium-low')
  })

  it('golden case: cell-tokenized → 102.4B tokens, 6.144e19 base FLOPs', () => {
    const result = computeTabularFlops(makeDefaultTabScenario({ tokenizationMode: 'cell' }))
    expect(result.effectiveTokens).toBe(102.4e9)
    expect(result.baseFlops).toBeCloseTo(6.144e19, 14)
    expect(result.confidence).toBe('low')
  })

  it('cell-tokenized has 100x more tokens than row-tokenized', () => {
    const row = computeTabularFlops(makeDefaultTabScenario({ tokenizationMode: 'row' }))
    const cell = computeTabularFlops(makeDefaultTabScenario({ tokenizationMode: 'cell' }))
    expect(cell.effectiveTokens).toBe(row.effectiveTokens * 100)
  })

  it('axial tokenization is additive (rows + columns)', () => {
    const result = computeTabularFlops(makeDefaultTabScenario({ tokenizationMode: 'axial' }))
    // 1M tasks × (1024 + 100) = 1.124B tokens
    expect(result.effectiveTokens).toBe(1e6 * (1024 + 100))
  })

  it('custom tokenization uses customTokensPerTask', () => {
    const result = computeTabularFlops(makeDefaultTabScenario({
      tokenizationMode: 'custom',
      customTokensPerTask: 2048,
    }))
    expect(result.effectiveTokens).toBe(1e6 * 2048)
  })

  it('multiple epochs multiply tokens', () => {
    const one = computeTabularFlops(makeDefaultTabScenario({ epochs: 1 }))
    const three = computeTabularFlops(makeDefaultTabScenario({ epochs: 3 }))
    expect(three.effectiveTokens).toBe(one.effectiveTokens * 3)
  })

  it('test-time compute multiplier increases total FLOPs', () => {
    const base = computeTabularFlops(makeDefaultTabScenario({ testTimeComputeMultiplier: 1 }))
    const multi = computeTabularFlops(makeDefaultTabScenario({ testTimeComputeMultiplier: 10 }))
    expect(multi.totalFlops).toBe(base.totalFlops * 10)
  })

  it('severe attention warning for cell-tokenized with sequence > 100K', () => {
    const result = computeTabularFlops(makeDefaultTabScenario({
      tokenizationMode: 'cell',
      rowsPerTask: 1024,
      columnsPerTask: 100,
    }))
    // sequence = 1024 * 100 = 102,400 > 100,000
    expect(result.warnings).toContainEqual(expect.stringContaining('Severe attention warning'))
  })

  it('medium attention warning for sequence 16K-65K', () => {
    const result = computeTabularFlops(makeDefaultTabScenario({
      tokenizationMode: 'cell',
      rowsPerTask: 200,
      columnsPerTask: 100,
    }))
    // sequence = 200 * 100 = 20,000
    expect(result.warnings).toContainEqual(expect.stringContaining('Attention warning'))
  })

  it('no attention warning for row-tokenized with 1024 rows', () => {
    const result = computeTabularFlops(makeDefaultTabScenario({ tokenizationMode: 'row' }))
    expect(result.warnings).not.toContainEqual(expect.stringContaining('attention warning'))
  })

  it('warns about high test-time multiplier', () => {
    const result = computeTabularFlops(makeDefaultTabScenario({ testTimeComputeMultiplier: 10 }))
    expect(result.warnings).toContainEqual(expect.stringContaining('test-time compute multiplier'))
  })

  it('includes data breakdown', () => {
    const result = computeTabularFlops(makeDefaultTabScenario())
    expect(result.dataBreakdown).toBeDefined()
    expect(result.dataBreakdown!['Tokens per task']).toBe('1,024')
    expect(result.dataBreakdown!['Tokenization mode']).toBe('row')
  })

  it('more tasks → more tokens', () => {
    const few = computeTabularFlops(makeDefaultTabScenario({ numberOfPretrainingTasks: 100e3 }))
    const many = computeTabularFlops(makeDefaultTabScenario({ numberOfPretrainingTasks: 10e6 }))
    expect(many.effectiveTokens).toBeGreaterThan(few.effectiveTokens)
  })

  it('trace has at least 6 entries', () => {
    const result = computeTabularFlops(makeDefaultTabScenario())
    expect(result.trace.length).toBeGreaterThanOrEqual(6)
  })
})
