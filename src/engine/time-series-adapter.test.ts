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
    expect(result.effectiveTokens).toBe(10e6 * 11 * 128)
  })

  it('multiple epochs multiply tokens', () => {
    const one = computeTimeSeriesFlops(makeDefaultTsScenario({ epochs: 1 }))
    const three = computeTimeSeriesFlops(makeDefaultTsScenario({ epochs: 3 }))
    expect(three.effectiveTokens).toBe(one.effectiveTokens * 3)
  })
})
