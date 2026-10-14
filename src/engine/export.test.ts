import { describe, it, expect } from 'vitest'
import { encodeScenarioToHash, decodeScenarioFromHash } from './export'
import type { Precision, TrainingScenario } from './types'

function makeScenario(precision: Precision): TrainingScenario {
  return {
    modelParameters: 70e9,
    tokensPerParameter: 20,
    trainingWindowSeconds: 30 * 86_400,
    precision,
    selectedGpuIds: ['b200-sxm'],
    mfuByGpuId: { 'b200-sxm': 0.39 },
    availability: 0.90,
    overheadFactor: 1.10,
    trainingMode: 'FULL_PRETRAINING',
    memoryBytesPerParameter: 16,
  }
}

describe('precision URL round-trip', () => {
  const precisions: Precision[] = ['BF16_DENSE', 'FP8_DENSE', 'FP4_DENSE']

  for (const precision of precisions) {
    it(`survives a round-trip for ${precision}`, () => {
      const hash = encodeScenarioToHash(makeScenario(precision))
      const decoded = decodeScenarioFromHash(hash)
      expect(decoded?.scenario.precision).toBe(precision)
    })
  }

  it('drops an unrecognised precision rather than trusting it', () => {
    const hash = encodeScenarioToHash(makeScenario('FP4_DENSE')).replace(
      'precision=FP4_DENSE',
      'precision=FP2_NONSENSE',
    )
    const decoded = decodeScenarioFromHash(hash)
    expect(decoded?.scenario.precision).toBeUndefined()
  })
})
