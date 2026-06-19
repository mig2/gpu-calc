import type { GpuSku, TrainingScenario, EstimateResult } from './types';
import { estimateHardware } from './hardware-estimator';
import { computeLlmFlops } from './adapters/llm-adapter';

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
  if (gpu.id === 'h200-sxm') {
    warnings.push(
      'H200 has the same raw dense BF16 peak as H100. Its advantage is more HBM capacity and bandwidth, which can improve achieved MFU in memory-stressed runs.',
    );
  }
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
