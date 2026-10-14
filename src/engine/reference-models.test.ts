import { describe, it, expect } from 'vitest'
import { REFERENCE_MODELS, findNearestReferences, formatFlopsShort } from './reference-models'

describe('REFERENCE_MODELS', () => {
  // Guards against order-of-magnitude typos in the hand-entered FLOP figures
  // (e.g. e19 where e22 was meant), which silently corrupt nearest-reference
  // ordering, the ReferenceComparison ratio column and the IsoFlop filter window.
  //
  // Compute follows *active* parameters, so for MoE rows the 6ND check must use
  // activeParameters. Using total params on a 2026-era MoE overstates FLOPs by 20-34x,
  // which is exactly the error this assertion exists to catch.
  it.each(REFERENCE_MODELS)('$name estimatedFlops matches 6ND within 1%', (model) => {
    const expected = 6 * (model.activeParameters ?? model.parameters) * model.trainingTokens
    expect(Math.abs(model.estimatedFlops - expected) / expected).toBeLessThan(0.01)
  })

  it('never claims more active parameters than total', () => {
    for (const model of REFERENCE_MODELS) {
      if (model.activeParameters === undefined) continue
      expect(model.activeParameters).toBeGreaterThan(0)
      expect(model.activeParameters).toBeLessThanOrEqual(model.parameters)
    }
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

  // Every row must be traceable to something fetchable — a confidently wrong figure is
  // worse than a missing row, and an un-sourced row is indistinguishable from a guess.
  it('cites a URL for every entry', () => {
    for (const model of REFERENCE_MODELS) {
      expect(model.source, model.name).toMatch(/https?:\/\//)
    }
  })

  it('reports positive GPU-hours wherever hours are claimed', () => {
    for (const model of REFERENCE_MODELS) {
      if (model.gpuHours === undefined) continue
      expect(model.gpuHours, model.name).toBeGreaterThan(0)
      // An hours figure with no stated convention is unusable: Llama 2's productive-only
      // hours and Llama 3.1's all-inclusive hours differ by roughly 2x for the same work.
      expect(model.hoursConvention, model.name).toBeTruthy()
    }
  })

  it('spans 2020-era through 2026-era runs with MoE coverage', () => {
    expect(REFERENCE_MODELS.length).toBeGreaterThanOrEqual(20)
    const moe = REFERENCE_MODELS.filter(
      (m) => m.activeParameters !== undefined && m.activeParameters < m.parameters
    )
    expect(moe.length).toBeGreaterThanOrEqual(8)
    const withHours = REFERENCE_MODELS.filter((m) => m.gpuHours !== undefined)
    expect(withHours.length).toBeGreaterThanOrEqual(10)
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
    expect(larger?.name).toBe('Llama 2 7B')
  })

  it('returns no larger run above the largest reference', () => {
    const { smaller, larger } = findNearestReferences(1e26)
    expect(smaller?.name).toBe('Llama 3.1 405B')
    expect(larger).toBeNull()
  })

  it('brackets a sparse-MoE budget using active-parameter FLOPs', () => {
    // DeepSeek-V3 sits at 3.29e24 on 37B active params despite being 671B resident.
    // If a future edit ever recomputed its FLOPs from total params it would land near
    // 6e25 and this bracket would break.
    const { closest } = findNearestReferences(3.3e24)
    expect(closest.name).toBe('DeepSeek-V3 671B-A37B')
  })
})

describe('formatFlopsShort', () => {
  it('formats to the nearest supported exponent', () => {
    expect(formatFlopsShort(8.4e22)).toBe('8.4e22')
    expect(formatFlopsShort(3.645e25)).toBe('36.5e24')
  })
})
