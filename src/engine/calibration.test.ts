import { describe, it, expect } from 'vitest'
import { calibrateMfu } from './calibration'
import { getH100Reference, getGpuById } from './gpu-data'

describe('calibrateMfu', () => {
  const h100 = getH100Reference()

  it('back-solves ~40% MFU for the canonical 70B/30d/701 GPU scenario', () => {
    const result = calibrateMfu({
      modelParameters: 70e9,
      tokensTrainedOn: 1.4e12,
      wallClockDays: 30,
      gpuCount: 701,
      gpu: h100,
      precision: 'BF16_DENSE',
      overheadFactor: 1.10,
      availability: 0.90,
    })
    // 701 GPUs was computed from ceil() with 40% MFU, so back-solved should be close to 40%
    expect(result.achievedMfu).toBeGreaterThan(0.39)
    expect(result.achievedMfu).toBeLessThanOrEqual(0.40)
    expect(result.isReasonable).toBe(true)
  })

  it('flags unreasonable MFU below 10%', () => {
    const result = calibrateMfu({
      modelParameters: 70e9,
      tokensTrainedOn: 1.4e12,
      wallClockDays: 30,
      gpuCount: 10000,
      gpu: h100,
      precision: 'BF16_DENSE',
      overheadFactor: 1.10,
      availability: 0.90,
    })
    expect(result.achievedMfu).toBeLessThan(0.10)
    expect(result.isReasonable).toBe(false)
  })

  it('flags unreasonable MFU above 70%', () => {
    const result = calibrateMfu({
      modelParameters: 70e9,
      tokensTrainedOn: 1.4e12,
      wallClockDays: 30,
      gpuCount: 50,
      gpu: h100,
      precision: 'BF16_DENSE',
      overheadFactor: 1.10,
      availability: 0.90,
    })
    expect(result.achievedMfu).toBeGreaterThan(0.70)
    expect(result.isReasonable).toBe(false)
  })

  it('produces trace with 8 entries', () => {
    const result = calibrateMfu({
      modelParameters: 70e9,
      tokensTrainedOn: 1.4e12,
      wallClockDays: 30,
      gpuCount: 701,
      gpu: h100,
      precision: 'BF16_DENSE',
      overheadFactor: 1.10,
      availability: 0.90,
    })
    expect(result.trace).toHaveLength(8)
  })

  it('works with H200 GPU', () => {
    const h200 = getGpuById('h200-sxm')!
    const result = calibrateMfu({
      modelParameters: 70e9,
      tokensTrainedOn: 1.4e12,
      wallClockDays: 30,
      gpuCount: 623,
      gpu: h200,
      precision: 'BF16_DENSE',
      overheadFactor: 1.10,
      availability: 0.90,
    })
    // 623 GPUs was from H200 at 45% MFU
    expect(result.achievedMfu).toBeGreaterThan(0.44)
    expect(result.achievedMfu).toBeLessThanOrEqual(0.45)
    expect(result.isReasonable).toBe(true)
  })
})
