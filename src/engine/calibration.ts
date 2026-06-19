import type { GpuSku, Precision } from './types';

export type CalibrationInput = {
  modelParameters: number;
  tokensTrainedOn: number;
  wallClockDays: number;
  gpuCount: number;
  gpu: GpuSku;
  precision: Precision;
  overheadFactor: number;
  availability: number;
};

export type CalibrationResult = {
  achievedMfu: number;
  achievedMfuPercent: string;
  isReasonable: boolean;
  trace: string[];
};

/**
 * Back-solve for MFU from known run parameters.
 *
 * From: GPUs = ceil(totalFlops / (seconds * sustained))
 * We know GPUs, totalFlops, seconds. We solve for sustained, then MFU.
 *
 * sustained = totalFlops / (GPUs * seconds)
 * MFU = sustained / (peak * availability)
 */
export function calibrateMfu(input: CalibrationInput): CalibrationResult {
  const N = input.modelParameters;
  const D = input.tokensTrainedOn;
  const baseFlops = 6 * N * D;
  const totalFlops = baseFlops * input.overheadFactor;

  const seconds = input.wallClockDays * 86_400;
  const achievedSustained = totalFlops / (input.gpuCount * seconds);

  const peak =
    input.precision === 'BF16_DENSE'
      ? input.gpu.bf16DenseFlops
      : (input.gpu.fp8DenseFlops ?? input.gpu.bf16DenseFlops);

  // sustained = peak * MFU * availability
  // MFU = sustained / (peak * availability)
  const achievedMfu = achievedSustained / (peak * input.availability);

  const isReasonable = achievedMfu > 0.10 && achievedMfu < 0.70;

  const fmt = (n: number) => n.toExponential(2);

  return {
    achievedMfu,
    achievedMfuPercent: `${(achievedMfu * 100).toFixed(1)}%`,
    isReasonable,
    trace: [
      `N = ${fmt(N)}`,
      `D = ${fmt(D)} tokens`,
      `Base FLOPs = 6 × N × D = ${fmt(baseFlops)}`,
      `Total FLOPs = ${fmt(baseFlops)} × ${input.overheadFactor} = ${fmt(totalFlops)}`,
      `Wall-clock = ${input.wallClockDays} days = ${seconds.toLocaleString()} seconds`,
      `Achieved sustained/GPU = ${fmt(totalFlops)} / (${input.gpuCount} × ${seconds.toLocaleString()}) = ${fmt(achievedSustained)} FLOP/s`,
      `Peak = ${fmt(peak)} FLOP/s`,
      `MFU = ${fmt(achievedSustained)} / (${fmt(peak)} × ${input.availability}) = ${(achievedMfu * 100).toFixed(1)}%`,
    ],
  };
}
