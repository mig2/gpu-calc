import type { GpuSku, Precision, BaseHardwareConfig } from './types';

export type HardwareEstimateInput = {
  totalFlops: number;
  modelParameters: number;
  memoryBytesPerParameter: number;
  hardware: BaseHardwareConfig;
  /** MoE architectures pay a systems penalty on top of the precision factor. Default false. */
  isMoe?: boolean;
};

export type HardwareEstimateResult = {
  gpuId: string;
  requiredGpus: number;
  h100Equivalents: number;
  sustainedFlopsPerGpu: number;
  memoryLowerBoundGpus: number;
  hardwareTrace: string[];
  warnings: string[];
};

/**
 * MFU multipliers applied on top of the BF16 baseline MFU
 * (research §5.3). Swapping in the lower-precision peak without these asserts a
 * full 2x (FP8) or 4x (FP4) speedup, which measurement does not support:
 * end-to-end FP8 wall-clock speedup is 1.3-1.5x, so
 * MFU_FP8 ~= MFU_BF16 x (1.3..1.5) / 2 = 0.65-0.75 x MFU_BF16.
 * Default recipes quantize only transformer-block linears; softmax, layernorm,
 * residuals, attention masking, optimizer and all-reduce stay BF16/FP32, and
 * Amdahl does the rest.
 */
export const PRECISION_MFU_FACTORS: Record<Precision, number> = {
  BF16_DENSE: 1.00,
  FP8_DENSE: 0.70,
  FP4_DENSE: 0.50,
};

/**
 * MFU multiplier for sparse (MoE) vs dense architectures (research §3.4, §5.3).
 * Validated against DeepSeek-V3: 0.70 (FP8) x 0.85 (MoE) = 0.595, close to the
 * observed 0.50 ratio of FP8-peak to BF16-peak utilization.
 */
export const MOE_MFU_FACTORS = {
  dense: 1.00,
  moe: 0.85,
} as const;

export function estimateHardware(
  input: HardwareEstimateInput,
  gpu: GpuSku,
  h100Reference: GpuSku,
): HardwareEstimateResult {
  const precision = input.hardware.precision;
  const warnings: string[] = [];

  const peak = getPeakFlops(gpu, precision);
  if (precision !== 'BF16_DENSE' && !hasNativePeak(gpu, precision)) {
    const label = precision === 'FP4_DENSE' ? 'FP4' : 'FP8';
    warnings.push(
      `${gpu.label} publishes no dense ${label} peak, so the estimate falls back to a higher-precision peak for this GPU. The ${label} throughput shown is therefore not a ${label} speedup — only the precision MFU penalty applies.`,
    );
  }

  const baseMfu = input.hardware.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
  const precisionFactor = PRECISION_MFU_FACTORS[precision];
  const moeFactor = input.isMoe ? MOE_MFU_FACTORS.moe : MOE_MFU_FACTORS.dense;
  const effectiveMfu = baseMfu * precisionFactor * moeFactor;

  const sustained = peak * effectiveMfu * input.hardware.availability;
  const requiredGpus = Math.ceil(
    input.totalFlops / (input.hardware.trainingWindowSeconds * sustained),
  );

  const h100Peak = getPeakFlops(h100Reference, precision);
  const h100BaseMfu = input.hardware.mfuByGpuId[h100Reference.id] ?? h100Reference.defaultMfu;
  const h100EffectiveMfu = h100BaseMfu * precisionFactor * moeFactor;
  const h100Sustained = h100Peak * h100EffectiveMfu * input.hardware.availability;
  const h100Equivalents = (requiredGpus * sustained) / h100Sustained;

  const usableMemBytes = gpu.memoryGb * 1e9 * 0.85;
  const memoryLowerBoundGpus = Math.ceil(
    (input.modelParameters * input.memoryBytesPerParameter) / usableMemBytes,
  );

  const fmt = (n: number) => n.toExponential(2);

  return {
    gpuId: gpu.id,
    requiredGpus,
    h100Equivalents,
    sustainedFlopsPerGpu: sustained,
    memoryLowerBoundGpus,
    warnings,
    hardwareTrace: [
      `Effective MFU = ${baseMfu} (base) × ${precisionFactor} (${precisionLabel(precision)} factor) × ${moeFactor} (${input.isMoe ? 'MoE' : 'dense'} factor) = ${effectiveMfu.toFixed(4)}`,
      `${gpu.label} sustained = ${fmt(peak)} × ${effectiveMfu.toFixed(4)} × ${input.hardware.availability} = ${fmt(sustained)} FLOP/s`,
      `Training window = ${input.hardware.trainingWindowSeconds.toLocaleString()} seconds`,
      `GPUs = ceil(${fmt(input.totalFlops)} / (${input.hardware.trainingWindowSeconds.toLocaleString()} × ${fmt(sustained)})) = ${requiredGpus}`,
    ],
  };
}

function precisionLabel(precision: Precision): string {
  switch (precision) {
    case 'BF16_DENSE':
      return 'BF16';
    case 'FP8_DENSE':
      return 'FP8';
    case 'FP4_DENSE':
      return 'FP4';
  }
}

/** True when the GPU publishes a dense peak for this precision (no fallback needed). */
export function hasNativePeak(gpu: GpuSku, precision: Precision): boolean {
  switch (precision) {
    case 'BF16_DENSE':
      return true;
    case 'FP8_DENSE':
      return gpu.fp8DenseFlops !== undefined;
    case 'FP4_DENSE':
      return gpu.fp4DenseFlops !== undefined;
  }
}

/**
 * Dense peak FLOP/s for a precision, falling back through FP4 -> FP8 -> BF16
 * when the GPU does not publish the requested one.
 */
export function getPeakFlops(gpu: GpuSku, precision: Precision): number {
  switch (precision) {
    case 'BF16_DENSE':
      return gpu.bf16DenseFlops;
    case 'FP8_DENSE':
      return gpu.fp8DenseFlops ?? gpu.bf16DenseFlops;
    case 'FP4_DENSE':
      return gpu.fp4DenseFlops ?? gpu.fp8DenseFlops ?? gpu.bf16DenseFlops;
  }
}
