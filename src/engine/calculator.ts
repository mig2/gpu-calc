import type { GpuSku, TrainingScenario, EstimateResult } from './types';

export function estimateTrainingRun(
  scenario: TrainingScenario,
  gpu: GpuSku,
  h100Reference: GpuSku,
): EstimateResult {
  const N = scenario.modelParameters;
  const D = scenario.tokensPerParameter * N;
  const baseFlops = 6 * N * D;
  const totalFlops = baseFlops * scenario.overheadFactor;

  const peak =
    scenario.precision === 'BF16_DENSE'
      ? gpu.bf16DenseFlops
      : (gpu.fp8DenseFlops ?? gpu.bf16DenseFlops);

  const mfu = scenario.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
  const sustained = peak * mfu * scenario.availability;
  const requiredGpus = Math.ceil(
    totalFlops / (scenario.trainingWindowSeconds * sustained),
  );

  // H100-equivalent conversion
  const h100Peak =
    scenario.precision === 'BF16_DENSE'
      ? h100Reference.bf16DenseFlops
      : (h100Reference.fp8DenseFlops ?? h100Reference.bf16DenseFlops);
  const h100Mfu =
    scenario.mfuByGpuId[h100Reference.id] ?? h100Reference.defaultMfu;
  const h100Sustained = h100Peak * h100Mfu * scenario.availability;
  const h100Equivalents = (requiredGpus * sustained) / h100Sustained;

  // Memory lower-bound
  const usableMemBytes = gpu.memoryGb * 1e9 * 0.85;
  const memoryLowerBoundGpus = Math.ceil(
    (N * scenario.memoryBytesPerParameter) / usableMemBytes,
  );

  return {
    gpuId: gpu.id,
    tokens: D,
    baseFlops,
    totalFlops,
    sustainedFlopsPerGpu: sustained,
    requiredGpus,
    h100Equivalents,
    memoryLowerBoundGpus,
    warnings: buildWarnings(scenario, gpu, requiredGpus, memoryLowerBoundGpus),
    trace: buildTrace(scenario, gpu, D, baseFlops, totalFlops, sustained, requiredGpus),
  };
}

function buildWarnings(
  scenario: TrainingScenario,
  gpu: GpuSku,
  requiredGpus: number,
  memoryLowerBoundGpus: number,
): string[] {
  const warnings: string[] = [];

  if (
    scenario.modelParameters > 70e9 &&
    scenario.trainingWindowSeconds < 14 * 86_400
  ) {
    warnings.push(
      'Large-cluster warning: training a model >70B in under 14 days requires a very large cluster with significant operational complexity.',
    );
  }

  if (requiredGpus > 1024) {
    warnings.push(
      `Distributed-systems warning: ${requiredGpus.toLocaleString()} GPUs implies challenges with networking, checkpointing, stragglers, and cluster fragmentation.`,
    );
  }

  if (scenario.trainingMode !== 'FULL_PRETRAINING') {
    warnings.push(
      'The full-pretraining formula (6ND) may overestimate compute for this training mode. Task-specific formulas are recommended.',
    );
  }

  if (scenario.precision === 'FP8_DENSE') {
    warnings.push(
      'FP8 training is experimental. End-to-end FP8 recipes may not deliver the simple peak-based estimate shown here.',
    );
  }

  if (gpu.id === 'h200-sxm') {
    warnings.push(
      'H200 has the same raw dense BF16 peak as H100. Its advantage is more HBM capacity and bandwidth, which can improve achieved MFU in memory-stressed runs.',
    );
  }

  if (memoryLowerBoundGpus > requiredGpus) {
    warnings.push(
      `Memory bound exceeds compute bound: at least ${memoryLowerBoundGpus} GPUs needed for model state memory alone (vs ${requiredGpus} for compute).`,
    );
  }

  return warnings;
}

function formatSci(n: number, digits = 2): string {
  return n.toExponential(digits);
}

function buildTrace(
  scenario: TrainingScenario,
  gpu: GpuSku,
  tokens: number,
  baseFlops: number,
  totalFlops: number,
  sustained: number,
  requiredGpus: number,
): string[] {
  const N = scenario.modelParameters;
  const peak =
    scenario.precision === 'BF16_DENSE'
      ? gpu.bf16DenseFlops
      : (gpu.fp8DenseFlops ?? gpu.bf16DenseFlops);
  const mfu = scenario.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;

  return [
    `N = ${formatSci(N)}`,
    `D = ${scenario.tokensPerParameter} × N = ${formatSci(tokens)}`,
    `Base FLOPs = 6 × N × D = ${formatSci(baseFlops)}`,
    `Total FLOPs = ${formatSci(baseFlops)} × ${scenario.overheadFactor} = ${formatSci(totalFlops)}`,
    `${gpu.label} sustained = ${formatSci(peak)} × ${mfu} × ${scenario.availability} = ${formatSci(sustained)} FLOP/s`,
    `Training window = ${scenario.trainingWindowSeconds.toLocaleString()} seconds`,
    `GPUs = ceil(${formatSci(totalFlops)} / (${scenario.trainingWindowSeconds.toLocaleString()} × ${formatSci(sustained)})) = ${requiredGpus}`,
  ];
}
