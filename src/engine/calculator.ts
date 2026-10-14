import type { GpuSku, Precision, TrainingScenario, EstimateResult } from './types';
import { estimateHardware, PRECISION_MFU_FACTORS } from './hardware-estimator';
import { computeLlmFlops } from './adapters/llm-adapter';

/**
 * Warnings that depend only on the chosen GPU and precision, shared by the
 * calculator entry point and the store's multi-GPU path.
 */
export function buildHardwareWarnings(gpu: GpuSku, precision: Precision): string[] {
  const warnings: string[] = [];

  if (gpu.id === 'h200-sxm') {
    warnings.push(
      'H200 has the same raw dense BF16 peak as H100. Its advantage is more HBM capacity and bandwidth, which can improve achieved MFU in memory-stressed runs.',
    );
  }

  if (gpu.id === 'b300-sxm' && precision !== 'FP4_DENSE') {
    warnings.push(
      'B300 (Blackwell Ultra) has the same dense BF16 and FP8 peak as B200. Its uplift is 1.5x FP4 throughput and 49% more memory only, so at this precision it delivers no compute gain over B200.',
    );
  }

  if (precision === 'FP4_DENSE') {
    warnings.push(
      'NVFP4 pretraining is validated — NVIDIA pretrained a 12B hybrid Mamba-Transformer on 10T tokens in NVFP4, matching an FP8 baseline — but it is not a drop-in flag. It requires Random Hadamard transforms, 2-D quantization, stochastic rounding and selective high-precision layers.',
    );
  }

  if (precision === 'FP8_DENSE' || precision === 'FP4_DENSE') {
    const label = precision === 'FP4_DENSE' ? 'FP4' : 'FP8';
    const factor = PRECISION_MFU_FACTORS[precision];
    warnings.push(
      `A ${factor.toFixed(2)}x MFU factor is applied for ${label}, so the estimate does not assume the full peak-ratio speedup. Default recipes quantize only transformer-block linear layers; softmax, layernorm, residuals, optimizer state and all-reduce stay at higher precision, and Amdahl's law caps the end-to-end gain (measured FP8 wall-clock speedup is 1.3-1.5x, not 2x).`,
    );
  }

  return warnings;
}

/**
 * Main entry point — backwards-compatible with existing TrainingScenario.
 * Internally delegates to LLM adapter + hardware estimator.
 */
export function estimateTrainingRun(
  scenario: TrainingScenario,
  gpu: GpuSku,
  h100Reference: GpuSku,
): EstimateResult {
  const llmScenario = {
    modelFamily: 'llm' as const,
    modelParameters: scenario.modelParameters,
    tokensPerParameter: scenario.tokensPerParameter,
    architectureFactor: scenario.architectureFactor ?? 6,
    trainingTokensOverride: scenario.trainingTokensOverride,
    trainingMode: scenario.trainingMode,
    memoryBytesPerParameter: scenario.memoryBytesPerParameter,
    trainingWindowSeconds: scenario.trainingWindowSeconds,
    precision: scenario.precision,
    selectedGpuIds: scenario.selectedGpuIds,
    mfuByGpuId: scenario.mfuByGpuId,
    availability: scenario.availability,
    overheadFactor: scenario.overheadFactor,
  };

  const adapterResult = computeLlmFlops(llmScenario);

  const hwResult = estimateHardware(
    {
      totalFlops: adapterResult.totalFlops,
      modelParameters: scenario.modelParameters,
      memoryBytesPerParameter: scenario.memoryBytesPerParameter,
      hardware: {
        trainingWindowSeconds: scenario.trainingWindowSeconds,
        precision: scenario.precision,
        selectedGpuIds: scenario.selectedGpuIds,
        mfuByGpuId: scenario.mfuByGpuId,
        availability: scenario.availability,
        overheadFactor: scenario.overheadFactor,
      },
    },
    gpu,
    h100Reference,
  );

  // Merge adapter warnings with hardware-level warnings
  const warnings = [...adapterResult.warnings];
  if (hwResult.requiredGpus > 1024) {
    warnings.push(
      `Distributed-systems warning: ${hwResult.requiredGpus.toLocaleString()} GPUs implies challenges with networking, checkpointing, stragglers, and cluster fragmentation.`,
    );
  }
  if (hwResult.requiredGpus > 10000) {
    warnings.push(
      `Hyperscale warning: ${hwResult.requiredGpus.toLocaleString()} GPUs implies a multi-rack, multi-datacenter-class deployment with extreme operational complexity.`,
    );
  }
  const mfu = scenario.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
  if (mfu > 0.60) {
    warnings.push(
      `Optimistic MFU warning: ${(mfu * 100).toFixed(0)}% MFU exceeds typical achieved utilization. Most production runs achieve 30-50%.`,
    );
  }
  warnings.push(...hwResult.warnings);
  warnings.push(...buildHardwareWarnings(gpu, scenario.precision));
  if (hwResult.memoryLowerBoundGpus > hwResult.requiredGpus) {
    warnings.push(
      `Memory bound exceeds compute bound: at least ${hwResult.memoryLowerBoundGpus} GPUs needed for model state memory alone (vs ${hwResult.requiredGpus} for compute).`,
    );
  }

  const trace = [...adapterResult.trace, ...hwResult.hardwareTrace];

  return {
    gpuId: gpu.id,
    tokens: adapterResult.effectiveTokens,
    baseFlops: adapterResult.baseFlops,
    totalFlops: adapterResult.totalFlops,
    sustainedFlopsPerGpu: hwResult.sustainedFlopsPerGpu,
    requiredGpus: hwResult.requiredGpus,
    h100Equivalents: hwResult.h100Equivalents,
    memoryLowerBoundGpus: hwResult.memoryLowerBoundGpus,
    warnings,
    trace,
  };
}
