import { describe, it, expect } from 'vitest'
import {
  estimateHardware,
  getPeakFlops,
  hasNativePeak,
  PRECISION_MFU_FACTORS,
  MOE_MFU_FACTORS,
  type HardwareEstimateInput,
} from './hardware-estimator'
import { GPU_SKUS, getGpuById, getH100Reference } from './gpu-data'
import type { GpuSku, Precision } from './types'

function makeInput(overrides: Partial<HardwareEstimateInput> = {}): HardwareEstimateInput {
  return {
    totalFlops: 6 * 70e9 * 1.4e12 * 1.10,
    modelParameters: 70e9,
    memoryBytesPerParameter: 16,
    hardware: {
      trainingWindowSeconds: 30 * 86_400,
      precision: 'BF16_DENSE',
      selectedGpuIds: ['h100-sxm'],
      mfuByGpuId: { 'h100-sxm': 0.40 },
      availability: 0.90,
      overheadFactor: 1.10,
    },
    ...overrides,
  }
}

/** Rebuild the input with a different precision, keeping the rest fixed. */
function withPrecision(precision: Precision, isMoe = false): HardwareEstimateInput {
  const base = makeInput()
  return { ...base, isMoe, hardware: { ...base.hardware, precision } }
}

describe('PRECISION_MFU_FACTORS', () => {
  it('matches the research §5.3 recommendation', () => {
    expect(PRECISION_MFU_FACTORS.BF16_DENSE).toBe(1.00)
    expect(PRECISION_MFU_FACTORS.FP8_DENSE).toBe(0.70)
    expect(PRECISION_MFU_FACTORS.FP4_DENSE).toBe(0.50)
  })

  it('covers every Precision member', () => {
    const precisions: Precision[] = ['BF16_DENSE', 'FP8_DENSE', 'FP4_DENSE']
    for (const p of precisions) {
      expect(PRECISION_MFU_FACTORS[p]).toBeGreaterThan(0)
      expect(PRECISION_MFU_FACTORS[p]).toBeLessThanOrEqual(1)
    }
  })
})

describe('MOE_MFU_FACTORS', () => {
  it('matches the research §5.3 recommendation', () => {
    expect(MOE_MFU_FACTORS.dense).toBe(1.00)
    expect(MOE_MFU_FACTORS.moe).toBe(0.85)
  })
})

describe('getPeakFlops', () => {
  const b200 = getGpuById('b200-sxm')!
  const h100 = getGpuById('h100-sxm')!
  const mi300x = getGpuById('mi300x')!

  it('returns the BF16 peak for BF16', () => {
    expect(getPeakFlops(b200, 'BF16_DENSE')).toBe(2.25e15)
  })

  it('returns the FP8 peak for FP8', () => {
    expect(getPeakFlops(b200, 'FP8_DENSE')).toBe(4.5e15)
  })

  it('returns the FP4 peak for FP4 when published', () => {
    expect(getPeakFlops(b200, 'FP4_DENSE')).toBe(9.0e15)
  })

  it('falls back FP4 -> FP8 when the GPU has no FP4 (Hopper)', () => {
    expect(h100.fp4DenseFlops).toBeUndefined()
    expect(getPeakFlops(h100, 'FP4_DENSE')).toBe(h100.fp8DenseFlops)
  })

  it('falls back FP4 -> BF16 when the GPU has neither FP4 nor a distinct FP8 rate', () => {
    const tpuV6e = getGpuById('tpu-v6e')!
    expect(tpuV6e.fp4DenseFlops).toBeUndefined()
    expect(tpuV6e.fp8DenseFlops).toBeUndefined()
    expect(getPeakFlops(tpuV6e, 'FP4_DENSE')).toBe(tpuV6e.bf16DenseFlops)
    expect(getPeakFlops(tpuV6e, 'FP8_DENSE')).toBe(tpuV6e.bf16DenseFlops)
  })

  it('returns the BF16 peak for FP8 on MI300X, where FP8 has no rate advantage', () => {
    expect(getPeakFlops(mi300x, 'FP8_DENSE')).toBe(mi300x.bf16DenseFlops)
  })
})

describe('hasNativePeak', () => {
  it('is always true for BF16', () => {
    for (const gpu of GPU_SKUS) {
      expect(hasNativePeak(gpu, 'BF16_DENSE')).toBe(true)
    }
  })

  it('is false for FP4 on Hopper and true on Blackwell', () => {
    expect(hasNativePeak(getGpuById('h100-sxm')!, 'FP4_DENSE')).toBe(false)
    expect(hasNativePeak(getGpuById('b200-sxm')!, 'FP4_DENSE')).toBe(true)
  })

  it('is false for FP8 on TPU v6e, which has no native FP8', () => {
    expect(hasNativePeak(getGpuById('tpu-v6e')!, 'FP8_DENSE')).toBe(false)
  })
})

describe('estimateHardware precision scaling', () => {
  const h100 = getH100Reference()

  it('applies no factor at BF16', () => {
    const result = estimateHardware(withPrecision('BF16_DENSE'), h100, h100)
    // 0.9895e15 x 0.40 x 0.90
    expect(result.sustainedFlopsPerGpu).toBeCloseTo(0.9895e15 * 0.40 * 0.90, -10)
  })

  it('applies the 0.70 factor at FP8 instead of the full 2x peak swap', () => {
    const result = estimateHardware(withPrecision('FP8_DENSE'), h100, h100)
    expect(result.sustainedFlopsPerGpu).toBeCloseTo(1.979e15 * 0.40 * 0.70 * 0.90, -10)
  })

  it('applies the 0.50 factor at FP4', () => {
    const b200 = getGpuById('b200-sxm')!
    const input = withPrecision('FP4_DENSE')
    input.hardware.mfuByGpuId = { 'b200-sxm': 0.40, 'h100-sxm': 0.40 }
    const result = estimateHardware(input, b200, h100)
    expect(result.sustainedFlopsPerGpu).toBeCloseTo(9.0e15 * 0.40 * 0.50 * 0.90, -10)
  })

  it('gives FP8 a 1.4x throughput gain over BF16, not 2x', () => {
    const bf16 = estimateHardware(withPrecision('BF16_DENSE'), h100, h100)
    const fp8 = estimateHardware(withPrecision('FP8_DENSE'), h100, h100)
    const ratio = fp8.sustainedFlopsPerGpu / bf16.sustainedFlopsPerGpu
    expect(ratio).toBeCloseTo(1.4, 5)
    expect(ratio).toBeLessThan(1.5)
  })

  it('needs more GPUs at FP8 than an uncorrected 2x model would imply', () => {
    const bf16 = estimateHardware(withPrecision('BF16_DENSE'), h100, h100)
    const fp8 = estimateHardware(withPrecision('FP8_DENSE'), h100, h100)
    expect(fp8.requiredGpus).toBeGreaterThan(bf16.requiredGpus / 2)
    expect(fp8.requiredGpus).toBeLessThan(bf16.requiredGpus)
  })

  it('shows the factors explicitly in the trace', () => {
    const result = estimateHardware(withPrecision('FP8_DENSE'), h100, h100)
    const mfuLine = result.hardwareTrace.find((l) => l.includes('Effective MFU'))
    expect(mfuLine).toBeDefined()
    expect(mfuLine).toContain('0.7')
    expect(mfuLine).toContain('FP8 factor')
    expect(mfuLine).toContain('dense factor')
  })
})

describe('estimateHardware MoE scaling', () => {
  const h100 = getH100Reference()

  it('defaults to the dense factor when isMoe is omitted', () => {
    const omitted = estimateHardware(makeInput(), h100, h100)
    const explicitDense = estimateHardware({ ...makeInput(), isMoe: false }, h100, h100)
    expect(omitted.sustainedFlopsPerGpu).toBe(explicitDense.sustainedFlopsPerGpu)
  })

  it('applies the 0.85 factor for MoE', () => {
    const dense = estimateHardware(withPrecision('BF16_DENSE', false), h100, h100)
    const moe = estimateHardware(withPrecision('BF16_DENSE', true), h100, h100)
    expect(moe.sustainedFlopsPerGpu / dense.sustainedFlopsPerGpu).toBeCloseTo(0.85, 10)
  })

  it('compounds with the precision factor (FP8 MoE = 0.595 of base MFU)', () => {
    const result = estimateHardware(withPrecision('FP8_DENSE', true), h100, h100)
    expect(result.sustainedFlopsPerGpu).toBeCloseTo(1.979e15 * 0.40 * 0.595 * 0.90, -10)
  })

  it('labels the MoE factor in the trace', () => {
    const result = estimateHardware(withPrecision('BF16_DENSE', true), h100, h100)
    expect(result.hardwareTrace.find((l) => l.includes('Effective MFU'))).toContain('MoE factor')
  })
})

describe('estimateHardware fallback warnings', () => {
  const h100 = getH100Reference()

  it('warns when FP4 is requested on a GPU with no FP4 peak', () => {
    const result = estimateHardware(withPrecision('FP4_DENSE'), h100, h100)
    expect(result.warnings).toContainEqual(expect.stringContaining('no dense FP4 peak'))
  })

  it('does not warn when the GPU publishes the requested peak', () => {
    const b200 = getGpuById('b200-sxm')!
    const input = withPrecision('FP4_DENSE')
    input.hardware.mfuByGpuId = { 'b200-sxm': 0.40, 'h100-sxm': 0.40 }
    const result = estimateHardware(input, b200, h100)
    expect(result.warnings).toHaveLength(0)
  })

  it('warns when FP8 is requested on TPU v6e, which has no native FP8', () => {
    const tpu = getGpuById('tpu-v6e')!
    const input = withPrecision('FP8_DENSE')
    input.hardware.mfuByGpuId = { 'tpu-v6e': 0.45, 'h100-sxm': 0.40 }
    const result = estimateHardware(input, tpu, h100)
    expect(result.warnings).toContainEqual(expect.stringContaining('no dense FP8 peak'))
  })

  it('never warns at BF16', () => {
    for (const gpu of GPU_SKUS) {
      const input = withPrecision('BF16_DENSE')
      input.hardware.mfuByGpuId = { [gpu.id]: gpu.defaultMfu, 'h100-sxm': 0.40 }
      const result = estimateHardware(input, gpu, h100)
      expect(result.warnings).toHaveLength(0)
    }
  })
})

describe('GPU_SKUS table integrity', () => {
  it('has unique ids', () => {
    const ids = GPU_SKUS.map((g) => g.id)
    expect(new Set(ids).size).toBe(ids.length)
  })

  it('gives every row a non-empty notes string', () => {
    for (const gpu of GPU_SKUS) {
      expect(typeof gpu.notes).toBe('string')
      expect((gpu.notes ?? '').length).toBeGreaterThan(0)
    }
  })

  it('cites a source in every notes string', () => {
    for (const gpu of GPU_SKUS) {
      expect(gpu.notes!.toLowerCase()).toContain('source')
    }
  })

  it('keeps the four original ids so saved URLs still resolve', () => {
    for (const id of ['h100-sxm', 'h200-sxm', 'b200-sxm', 'gb200-nvl72-gpu']) {
      expect(getGpuById(id)).toBeDefined()
    }
  })

  it('has plausible positive values on every row', () => {
    for (const gpu of GPU_SKUS) {
      expect(gpu.bf16DenseFlops).toBeGreaterThan(0)
      expect(gpu.memoryGb).toBeGreaterThan(0)
      expect(gpu.defaultMfu).toBeGreaterThan(0)
      expect(gpu.defaultMfu).toBeLessThanOrEqual(0.60)
      if (gpu.fp8DenseFlops !== undefined) {
        expect(gpu.fp8DenseFlops).toBeGreaterThanOrEqual(gpu.bf16DenseFlops)
      }
      if (gpu.fp4DenseFlops !== undefined) {
        expect(gpu.fp4DenseFlops).toBeGreaterThanOrEqual(gpu.fp8DenseFlops ?? gpu.bf16DenseFlops)
      }
    }
  })

  it('applies the §4.3 corrections', () => {
    expect(getGpuById('b200-sxm')!.memoryGb).toBe(180)
    expect(getGpuById('b200-sxm')!.bandwidthTbps).toBe(7.7)
    expect(getGpuById('h200-sxm')!.defaultMfu).toBe(0.42)
    expect(getGpuById('gb200-nvl72-gpu')!.defaultMfu).toBe(0.42)
  })

  it('marks B300 as BF16/FP8-identical to B200', () => {
    const b200 = getGpuById('b200-sxm')!
    const b300 = getGpuById('b300-sxm')!
    expect(b300.bf16DenseFlops).toBe(b200.bf16DenseFlops)
    expect(b300.fp8DenseFlops).toBe(b200.fp8DenseFlops)
    expect(b300.fp4DenseFlops).toBeGreaterThan(b200.fp4DenseFlops!)
    expect(b300.notes).toContain('IDENTICAL to B200')
  })

  it('marks the MI450 BF16 figure as extrapolated', () => {
    expect(getGpuById('mi450')!.notes).toContain('EXTRAPOLATED, NOT PUBLISHED')
  })

  it('excludes Vera Rubin / VR200, whose per-GPU dense BF16 is unpublished', () => {
    const ids = GPU_SKUS.map((g) => g.id).join(' ')
    expect(ids).not.toContain('vr200')
    expect(ids).not.toContain('rubin')
  })

  it('has a GPU count matching the 15 rows of research §4.2', () => {
    // §4.2 lists 15 SKUs: 6 NVIDIA, 4 AMD, 2 Google, 2 AWS, 1 Intel.
    expect(GPU_SKUS).toHaveLength(15)
  })
})

describe('h100Equivalents', () => {
  const h100 = getH100Reference()

  it('equals requiredGpus when the GPU is the H100 reference', () => {
    const result = estimateHardware(makeInput(), h100, h100)
    expect(result.h100Equivalents).toBeCloseTo(result.requiredGpus, 6)
  })

  it('is unaffected by the precision factor, which cancels on both sides', () => {
    const b200 = getGpuById('b200-sxm')!
    const mfuByGpuId: Record<string, number> = { 'b200-sxm': 0.40, 'h100-sxm': 0.40 }
    const bf16 = makeInput()
    bf16.hardware = { ...bf16.hardware, precision: 'BF16_DENSE', mfuByGpuId }
    const fp8 = makeInput()
    fp8.hardware = { ...fp8.hardware, precision: 'FP8_DENSE', mfuByGpuId }

    const bf16Result = estimateHardware(bf16, b200, h100)
    const fp8Result = estimateHardware(fp8, b200, h100)
    // Both GPUs double their peak at FP8, so the ratio per GPU is identical;
    // only the (smaller) GPU count moves the aggregate.
    const perGpuBf16 = bf16Result.h100Equivalents / bf16Result.requiredGpus
    const perGpuFp8 = fp8Result.h100Equivalents / fp8Result.requiredGpus
    expect(perGpuFp8).toBeCloseTo(perGpuBf16, 6)
  })
})

describe('custom GPU rows still work', () => {
  const h100 = getH100Reference()

  it('handles a BF16-only custom GPU at FP4 by falling back and warning', () => {
    const custom: GpuSku = {
      id: 'custom-bf16-only',
      label: 'Custom BF16-only accelerator',
      bf16DenseFlops: 1.0e15,
      memoryGb: 100,
      defaultMfu: 0.35,
    }
    const input = withPrecision('FP4_DENSE')
    input.hardware.mfuByGpuId = { 'custom-bf16-only': 0.35, 'h100-sxm': 0.40 }
    const result = estimateHardware(input, custom, h100)
    expect(result.sustainedFlopsPerGpu).toBeCloseTo(1.0e15 * 0.35 * 0.50 * 0.90, -10)
    expect(result.warnings).toContainEqual(expect.stringContaining('no dense FP4 peak'))
  })
})
