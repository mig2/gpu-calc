import type { TrainingScenario, GpuSku } from './types';

/**
 * Given a fixed GPU count and model size, solve for training time in seconds.
 */
export function solveForTrainingTime(
  scenario: TrainingScenario,
  gpu: GpuSku,
  gpuCount: number,
): { trainingSeconds: number; trainingDays: number; trace: string[] } {
  const N = scenario.modelParameters;
  const D = scenario.tokensPerParameter * N;
  const totalFlops = 6 * N * D * scenario.overheadFactor;

  const peak =
    scenario.precision === 'BF16_DENSE'
      ? gpu.bf16DenseFlops
      : (gpu.fp8DenseFlops ?? gpu.bf16DenseFlops);
  const mfu = scenario.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
  const sustained = peak * mfu * scenario.availability;

  const trainingSeconds = totalFlops / (gpuCount * sustained);
  const trainingDays = trainingSeconds / 86_400;

  const fmt = (n: number) => n.toExponential(2);

  return {
    trainingSeconds,
    trainingDays,
    trace: [
      `N = ${fmt(N)}`,
      `D = ${scenario.tokensPerParameter} × N = ${fmt(D)}`,
      `Total FLOPs = 6 × N × D × ${scenario.overheadFactor} = ${fmt(totalFlops)}`,
      `Sustained/GPU = ${fmt(peak)} × ${mfu} × ${scenario.availability} = ${fmt(sustained)} FLOP/s`,
      `Training seconds = ${fmt(totalFlops)} / (${gpuCount} × ${fmt(sustained)}) = ${Math.round(trainingSeconds).toLocaleString()}`,
      `Training days = ${trainingDays.toFixed(1)}`,
    ],
  };
}

/**
 * Given a fixed GPU count and training window, solve for max model parameters.
 * Uses closed-form solution since the relationship is N^2 (via 6 * N * TPP * N).
 */
export function solveForMaxModelSize(
  scenario: TrainingScenario,
  gpu: GpuSku,
  gpuCount: number,
): { maxParameters: number; maxParametersLabel: string; trace: string[] } {
  const peak =
    scenario.precision === 'BF16_DENSE'
      ? gpu.bf16DenseFlops
      : (gpu.fp8DenseFlops ?? gpu.bf16DenseFlops);
  const mfu = scenario.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
  const sustained = peak * mfu * scenario.availability;

  // totalFlops = 6 * N * TPP * N * overhead = 6 * TPP * overhead * N^2
  // Available FLOPs = gpuCount * sustained * windowSeconds
  const availableFlops = gpuCount * sustained * scenario.trainingWindowSeconds;
  const coefficient = 6 * scenario.tokensPerParameter * scenario.overheadFactor;
  // N^2 = availableFlops / coefficient
  const nSquared = availableFlops / coefficient;
  const maxN = Math.floor(Math.sqrt(nSquared));

  const fmt = (n: number) => n.toExponential(2);

  function formatParams(n: number): string {
    if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
    if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
    if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
    return n.toLocaleString();
  }

  return {
    maxParameters: maxN,
    maxParametersLabel: formatParams(maxN),
    trace: [
      `Available FLOPs = ${gpuCount} GPUs × ${fmt(sustained)} FLOP/s × ${scenario.trainingWindowSeconds.toLocaleString()}s = ${fmt(availableFlops)}`,
      `Coefficient = 6 × TPP(${scenario.tokensPerParameter}) × overhead(${scenario.overheadFactor}) = ${coefficient}`,
      `N² = ${fmt(availableFlops)} / ${coefficient} = ${fmt(nSquared)}`,
      `Max N = floor(√${fmt(nSquared)}) = ${fmt(maxN)} ≈ ${formatParams(maxN)}`,
    ],
  };
}
