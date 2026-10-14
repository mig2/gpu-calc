import { describe, it, expect } from 'vitest'
import { REFERENCE_MODELS, findNearestReferences, formatFlopsShort } from './reference-models'

describe('REFERENCE_MODELS', () => {
  // Guards against order-of-magnitude typos in the hand-entered FLOP figures
  // (e.g. e19 where e22 was meant), which silently corrupt nearest-reference
  // ordering, the ReferenceComparison ratio column and the IsoFlop filter window.
  it.each(REFERENCE_MODELS)('$name estimatedFlops matches 6ND within 1%', (model) => {
    const expected = 6 * model.parameters * model.trainingTokens
    expect(Math.abs(model.estimatedFlops - expected) / expected).toBeLessThan(0.01)
  })

  it('has a unique name and positive figures for every entry', () => {
    const names = new Set(REFERENCE_MODELS.map((m) => m.name))
    expect(names.size).toBe(REFERENCE_MODELS.length)
    for (const model of REFERENCE_MODELS) {
      expect(model.parameters).toBeGreaterThan(0)
      expect(model.trainingTokens).toBeGreaterThan(0)
      expect(model.estimatedFlops).toBeGreaterThan(0)
      expect(model.source).not.toBe('')
    }
  })
})

describe('findNearestReferences', () => {
  it('brackets a budget between the nearest smaller and larger runs', () => {
    // Between Llama 2 7B (8.4e22) and Llama 2 13B (1.56e23)
    const { smaller, larger, closest } = findNearestReferences(1e23)
    expect(smaller?.name).toBe('Llama 2 7B')
    expect(larger?.name).toBe('Llama 2 13B')
    expect(closest.name).toBe('Llama 2 7B')
  })

  it('returns no smaller run below the smallest reference', () => {
    const { smaller, larger } = findNearestReferences(1e18)
    expect(smaller).toBeNull()
    expect(larger?.name).toBe('Marin 8B')
  })

  it('returns no larger run above the largest reference', () => {
    const { smaller, larger } = findNearestReferences(1e26)
    expect(smaller?.name).toBe('Llama 3 405B')
    expect(larger).toBeNull()
  })
})

describe('formatFlopsShort', () => {
  it('formats to the nearest supported exponent', () => {
    expect(formatFlopsShort(8.4e22)).toBe('8.4e22')
    expect(formatFlopsShort(3.645e25)).toBe('36.5e24')
  })
})
