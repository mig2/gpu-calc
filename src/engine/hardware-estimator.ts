import type { GpuSku, Precision, BaseHardwareConfig } from './types';

export type HardwareEstimateInput = {
  totalFlops: number;
  modelParameters: number;
  memoryBytesPerParameter: number;
  hardware: BaseHardwareConfig;
};

export type HardwareEstimateResult = {
  gpuId: string;
  requiredGpus: number;
  h100Equivalents: number;
  sustainedFlopsPerGpu: number;
  memoryLowerBoundGpus: number;
  hardwareTrace: string[];
};

export function estimateHardware(
  input: HardwareEstimateInput,
  gpu: GpuSku,
  h100Reference: GpuSku,
): HardwareEstimateResult {
  const peak = getPeakFlops(gpu, input.hardware.precision);
  const mfu = input.hardware.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
  const sustained = peak * mfu * input.hardware.availability;
  const requiredGpus = Math.ceil(
    input.totalFlops / (input.hardware.trainingWindowSeconds * sustained),
  );

  const h100Peak = getPeakFlops(h100Reference, input.hardware.precision);
  const h100Mfu = input.hardware.mfuByGpuId[h100Reference.id] ?? h100Reference.defaultMfu;
  const h100Sustained = h100Peak * h100Mfu * input.hardware.availability;
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
    hardwareTrace: [
      `${gpu.label} sustained = ${fmt(peak)} × ${mfu} × ${input.hardware.availability} = ${fmt(sustained)} FLOP/s`,
      `Training window = ${input.hardware.trainingWindowSeconds.toLocaleString()} seconds`,
      `GPUs = ceil(${fmt(input.totalFlops)} / (${input.hardware.trainingWindowSeconds.toLocaleString()} × ${fmt(sustained)})) = ${requiredGpus}`,
    ],
  };
}

export function getPeakFlops(gpu: GpuSku, precision: Precision): number {
  return precision === 'BF16_DENSE'
    ? gpu.bf16DenseFlops
    : (gpu.fp8DenseFlops ?? gpu.bf16DenseFlops);
}
