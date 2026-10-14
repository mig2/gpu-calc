import type { GpuSku } from './types';
import type { ModelArchitecture, RlConfig, WeightPrecision } from './finetune-types';
import { WEIGHT_BYTES, resolveDims, USABLE_MEMORY_FRACTION } from './finetune-calculator';

/**
 * RL post-training rollout estimator (GRPO / PPO / RLVR).
 *
 * Research: docs/research/2026-10-14-rl-posttraining-research.md
 *
 * Central finding: RL post-training is an inference-serving problem wearing a
 * training problem's clothes. Rollout generation is 60-80% of step time, so the
 * primary output is a GPU *split* -- what fraction of the fleet should be
 * generating rather than training -- not a GPU count.
 *
 * Generation is decode-bound, so this composes the inference roofline rather
 * than using 2N x tokens, which would be optimistic by 10-30x.
 */

/** Decode roofline realization for a standalone serving engine. */
export const ETA_DECODE = 0.60;

/**
 * Additional realization loss specific to *RL* rollout, on top of ETA_DECODE.
 * Back-solved from two anchors: NeMo-RL 70B on 64x H100 (roofline 47.4 s vs
 * measured 84 s -> 0.56) and a 7B vLLM run (51 s vs ~180 s -> 0.28).
 * Causes: the effective batch decays as sequences finish so the tail sequence
 * gates the step; colocated vLLM gets a small memory budget (TRL's
 * `vllm_gpu_memory_utilization` defaults to 0.3 against 0.90 standalone);
 * CUDA graphs are re-captured after each weight refit.
 *
 * Applies ONLY to synchronous RL rollout, not to raw generation throughput.
 */
export const ETA_ROLLOUT = 0.45;

/**
 * Concurrency at which decode stops being bandwidth-bound and becomes
 * compute-bound (the roofline knee). Inference research doc sect. 1.4 puts it
 * near 280 on H100 BF16. Past this point extra concurrency buys no throughput,
 * so effective batch is capped here for timing purposes.
 */
export const BATCH_RIDGE = 280;

/** Inference engine fixed overhead per GPU (CUDA graphs, activation buffers). */
export const ENGINE_OVERHEAD_BYTES = 2e9;

/**
 * Effective weight-sync bandwidth for the per-step refit, back-solved from
 * NeMo-RL's measured figures (141 GB / 14 s at 70B on 64 GPUs). This covers the
 * whole refit -- re-sharding and CUDA-graph recapture -- not just the NCCL
 * collective, which alone is 20-500 ms. LoRA syncs only adapters, so ~0.
 *
 * At 70B this is ~14 s/step: on a 300-step run, 70 minutes of pure weight
 * shuffling, and the strongest numerical argument for LoRA-based RL.
 */
export const WEIGHT_SYNC_BYTES_PER_SEC = 10e9;

/**
 * Generation:training time ratio sanity bands, keyed on completion length
 * (research sect. 4.2). The engine computes `r` from the roofline; these are
 * the bands it should be warned against falling outside.
 */
export const R_SANITY_BANDS: { maxCompletionTokens: number; r: number }[] = [
  { maxCompletionTokens: 1024, r: 1.4 },
  { maxCompletionTokens: 4096, r: 2.5 },
  { maxCompletionTokens: 16384, r: 4.0 },
  { maxCompletionTokens: Number.POSITIVE_INFINITY, r: 6.0 },
];

/**
 * KV cache bytes per token.
 *
 * Standard GQA/MHA: 2 (K and V) x layers x kvHeads x headDim x bytes.
 * MLA (DeepSeek) compresses to a latent and is radically smaller; without the
 * published `kv_lora_rank` we approximate it at 1/8 of the GQA figure and flag
 * the estimate as unreliable.
 */
export function kvBytesPerToken(
  model: ModelArchitecture,
  kvPrecisionBytes = 2,
): { bytesPerToken: number; approximated: boolean } {
  const dims = resolveDims(model);
  const gqa = 2 * dims.layers * dims.kvDim * kvPrecisionBytes;
  if (model.attentionType === 'mla') {
    return { bytesPerToken: gqa / 8, approximated: true };
  }
  return { bytesPerToken: gqa, approximated: !dims.exact };
}

export type GenerationEstimate = {
  kvBytesPerToken: number;
  kvBytesPerSequence: number;
  /** Concurrency the KV pool can hold. */
  maxConcurrency: number;
  /** Concurrency actually used for timing, capped at the roofline knee. */
  effectiveConcurrency: number;
  /** How many sequential passes the requested sequence count needs. */
  waves: number;
  /** Time per output token at the effective concurrency, seconds. */
  tpotSeconds: number;
  /** Prefill time, seconds, after prefix-cache saving. */
  prefillSeconds: number;
  decodeSeconds: number;
  totalSeconds: number;
  warnings: string[];
};

/**
 * Raw generation time for `sequences` completions of `maxCompletionTokens`.
 *
 * `synchronous: false` models a throughput benchmark under continuous batching:
 * waves are fractional and no RL tail penalty applies.
 * `synchronous: true` models one RL step: every sequence in a wave must finish
 * before the trainer starts, so waves are whole and ETA_ROLLOUT applies. Using
 * the mean completion length instead of the max understates an RL step by the
 * tail ratio, routinely 3-10x (NeMo-RL's 70B row: mean 398 vs max 4096 = 10.3x).
 */
export function estimateGeneration(
  model: ModelArchitecture,
  gpu: GpuSku,
  opts: {
    inferenceGpus: number;
    sequences: number;
    promptTokens: number;
    maxCompletionTokens: number;
    weightPrecision: WeightPrecision;
    kvPrecisionBytes?: number;
    /** All completions in a GRPO group share the prompt prefix. */
    groupSize?: number;
    prefixCaching?: boolean;
    synchronous: boolean;
    /** Fraction of HBM the engine may use. Colocated TRL defaults to 0.3. */
    memoryUtilization?: number;
  },
): GenerationEstimate {
  const warnings: string[] = [];
  const bw = WEIGHT_BYTES[opts.weightPrecision];
  const { bytesPerToken: kvPerToken, approximated } = kvBytesPerToken(
    model,
    opts.kvPrecisionBytes ?? 2,
  );
  if (approximated) {
    warnings.push(
      model.attentionType === 'mla'
        ? 'MLA attention: KV cache approximated at 1/8 of the GQA equivalent. Supply kv_lora_rank for an accurate figure.'
        : 'KV cache derived from estimated architecture dimensions; concurrency and rollout time carry wide uncertainty.',
    );
  }

  const util = opts.memoryUtilization ?? USABLE_MEMORY_FRACTION;
  const seqTokens = opts.promptTokens + opts.maxCompletionTokens;
  const kvPerSequence = kvPerToken * seqTokens;

  const weightBytes = model.totalParameters * bw;
  const vramForKv =
    opts.inferenceGpus * gpu.memoryGb * 1e9 * util -
    weightBytes -
    opts.inferenceGpus * ENGINE_OVERHEAD_BYTES;

  if (vramForKv <= 0) {
    warnings.push(
      `Model weights (${(weightBytes / 1e9).toFixed(0)} GB) exceed the inference pool's usable memory. Add inference GPUs or quantize the serving copy.`,
    );
    return {
      kvBytesPerToken: kvPerToken,
      kvBytesPerSequence: kvPerSequence,
      maxConcurrency: 0,
      effectiveConcurrency: 0,
      waves: Infinity,
      tpotSeconds: Infinity,
      prefillSeconds: Infinity,
      decodeSeconds: Infinity,
      totalSeconds: Infinity,
      warnings,
    };
  }

  const maxConcurrency = Math.floor(vramForKv / kvPerSequence);
  const concurrency = Math.max(1, Math.min(maxConcurrency, opts.sequences));
  // Past the roofline knee, extra concurrency buys no throughput.
  const effectiveConcurrency = Math.min(concurrency, BATCH_RIDGE);

  const rawWaves = opts.sequences / Math.max(1, concurrency);
  const waves = opts.synchronous ? Math.ceil(rawWaves) : rawWaves;

  // Decode is bandwidth-bound: each step streams the weights plus the live KV.
  const sAvg = opts.promptTokens + opts.maxCompletionTokens / 2;
  const bytesPerStep = weightBytes + effectiveConcurrency * sAvg * kvPerToken;
  const bandwidth = (gpu.bandwidthTbps ?? 3.35) * 1e12 * opts.inferenceGpus;
  const tpotSeconds = bytesPerStep / (bandwidth * ETA_DECODE);

  const realization = opts.synchronous ? ETA_ROLLOUT : 1.0;
  const decodeSeconds = (waves * opts.maxCompletionTokens * tpotSeconds) / realization;

  // Prefill is compute-bound. With prefix caching, a GRPO group's shared prompt
  // is prefilled once per prompt rather than once per completion: a (G-1)/G saving.
  const g = opts.groupSize ?? 1;
  const prefillSequences =
    opts.prefixCaching !== false && g > 1 ? opts.sequences / g : opts.sequences;
  const prefillFlops = 2 * model.activeParameters * prefillSequences * opts.promptTokens;
  const prefillCompute = gpu.bf16DenseFlops * opts.inferenceGpus * 0.82;
  const prefillSeconds = prefillFlops / prefillCompute;

  if (maxConcurrency < opts.sequences) {
    warnings.push(
      `KV cache holds only ${maxConcurrency} concurrent sequences of ${seqTokens.toLocaleString()} tokens, so ${opts.sequences.toLocaleString()} rollouts run in ${Math.ceil(rawWaves)} waves. KV pool wants ${((kvPerSequence * opts.sequences) / 1e9).toFixed(0)} GB for full concurrency.`,
    );
  }
  if (concurrency > BATCH_RIDGE) {
    warnings.push(
      `Concurrency ${concurrency} exceeds the roofline knee (~${BATCH_RIDGE}); decode is compute-bound beyond that and extra concurrency buys no throughput.`,
    );
  }

  return {
    kvBytesPerToken: kvPerToken,
    kvBytesPerSequence: kvPerSequence,
    maxConcurrency,
    effectiveConcurrency,
    waves,
    tpotSeconds,
    prefillSeconds,
    decodeSeconds,
    totalSeconds: decodeSeconds + prefillSeconds,
    warnings,
  };
}

export type RlStepEstimate = {
  generation: GenerationEstimate;
  /** Training-side seconds per step. */
  trainSeconds: number;
  /** Per-step weight refit from trainer to inference engine. */
  weightSyncSeconds: number;
  /**
   * Total tokens generated across the run. Often more legible than GPU-hours,
   * and directly comparable to the Inference pane's per-million-token costs.
   */
  tokensGenerated: number;
  /** r = T_gen / T_train, the number the pane lives on. */
  genTrainRatio: number;
  /** Share of step time spent generating. */
  generationShare: number;
  /** Recommended fraction of the fleet on inference. */
  recommendedInferenceShare: number;
  stepSeconds: number;
  totalHours: number;
  /** KV pool bytes at full requested concurrency -- often the binding term. */
  kvPoolBytesAtFullConcurrency: number;
  warnings: string[];
};

/**
 * One GRPO/PPO step, and the GPU split it implies.
 *
 * PPO generates one completion per prompt rather than `G`, but roughly doubles
 * the training step (critic forward+backward plus a reward-model forward), so
 * its `r` is roughly GRPO's divided by two.
 */
export function estimateRlStep(
  model: ModelArchitecture,
  gpu: GpuSku,
  rl: RlConfig,
  opts: {
    method: 'grpo' | 'ppo';
    inferenceGpus: number;
    trainingGpus: number;
    promptTokens: number;
    weightPrecision: WeightPrecision;
    kvPrecisionBytes?: number;
    trainingMfu?: number;
    /** LoRA reduces the training-side FLOPs from 6N to 4N, and syncs ~0 bytes. */
    peft?: boolean;
    /**
     * How the synchronous step's cost is gated. Strict GRPO pays for the
     * slowest sequence, so `max` is correct -- but if completions cluster well
     * below the cap and the stack does in-flight replacement, the truth is
     * nearer `mean`. The same workload spans ~10x between these (NeMo-RL's 70B
     * row has mean/max = 398/4096). Default `max`; it is the conservative end.
     */
    tailGating?: 'max' | 'mean' | 'blend';
    /** Mean completion length as a fraction of the cap. Only used when gating != 'max'. */
    meanCompletionRatio?: number;
    /**
     * CALIBRATION. Observed seconds per step from the user's own stack, at this
     * configuration. One measured point collapses eta_rollout, policy MFU and
     * the tail factor into a single constant and moves this pane from `low` to
     * `medium` confidence -- the single highest-value input available.
     */
    observedStepSeconds?: number;
  },
): RlStepEstimate {
  const warnings: string[] = [];
  const isPpo = opts.method === 'ppo';
  // PPO samples one completion per prompt; GRPO samples G.
  const sequences = rl.promptsPerStep * (isPpo ? 1 : rl.groupSize);

  // Tail gating: what completion length the synchronous step actually pays for.
  const meanRatio = opts.meanCompletionRatio ?? 0.35;
  const gating = opts.tailGating ?? 'max';
  const gatedCompletionTokens =
    gating === 'mean'
      ? rl.maxCompletionTokens * meanRatio
      : gating === 'blend'
        ? rl.maxCompletionTokens * (0.5 + 0.5 * meanRatio)
        : rl.maxCompletionTokens;

  const generation = estimateGeneration(model, gpu, {
    inferenceGpus: opts.inferenceGpus,
    sequences,
    promptTokens: opts.promptTokens,
    maxCompletionTokens: gatedCompletionTokens,
    weightPrecision: opts.weightPrecision,
    kvPrecisionBytes: opts.kvPrecisionBytes,
    groupSize: isPpo ? 1 : rl.groupSize,
    prefixCaching: true,
    synchronous: true,
    // Colocated engines get a small memory budget (TRL default 0.3).
    memoryUtilization: rl.rolloutPlacement === 'colocated' ? 0.30 : USABLE_MEMORY_FRACTION,
  });

  // Training side: forward + backward over every generated token.
  const trainTokens = sequences * (opts.promptTokens + gatedCompletionTokens);
  const perToken = opts.peft ? 4 : 6;
  // PPO adds a critic forward+backward and a reward-model forward.
  const criticFactor = isPpo ? 2 + 2 / 6 : 1;
  const trainFlops = trainTokens * perToken * model.activeParameters * criticFactor;
  const trainMfu = opts.trainingMfu ?? 0.45;
  const trainSeconds =
    trainFlops / (gpu.bf16DenseFlops * opts.trainingGpus * trainMfu);

  // Per-step weight refit. LoRA syncs only adapters, which is negligible.
  const weightSyncSeconds = opts.peft
    ? 0
    : (model.totalParameters * WEIGHT_BYTES[opts.weightPrecision]) / WEIGHT_SYNC_BYTES_PER_SEC;

  const genTrainRatio = trainSeconds > 0 ? generation.totalSeconds / trainSeconds : Infinity;
  const generationShare = genTrainRatio / (1 + genTrainRatio);
  const modelledStepSeconds = generation.totalSeconds + trainSeconds + weightSyncSeconds;
  // A measured step overrides the model entirely.
  const stepSeconds = opts.observedStepSeconds ?? modelledStepSeconds;
  const tokensGenerated = sequences * gatedCompletionTokens * rl.steps;

  const kvPoolBytesAtFullConcurrency = generation.kvBytesPerSequence * sequences;

  if (Number.isFinite(genTrainRatio) && (genTrainRatio < 0.8 || genTrainRatio > 8)) {
    warnings.push(
      `Computed generation:training ratio r = ${genTrainRatio.toFixed(2)} falls outside the published 0.8-8 band. Check completion length, group size and the GPU split.`,
    );
  }
  warnings.push(
    `Rollout is ${(generationShare * 100).toFixed(0)}% of step time. Published runs put ${Math.round(generationShare * 100)}%-equivalent fleets on inference: Olmo 3 32B used 20 inference to 8 training nodes (71%), ROLL Flash 24/16 (60%), OpenRLHF 70B 16 vLLM / 16 train (50%).`,
  );
  if (kvPoolBytesAtFullConcurrency > model.totalParameters * 16) {
    warnings.push(
      `KV cache at full concurrency (${(kvPoolBytesAtFullConcurrency / 1e9).toFixed(0)} GB) exceeds the entire model state. For most RL configurations KV cache, not model state, is the binding memory term -- the opposite of supervised fine-tuning. Consider FP8 KV (halves it) or fewer prompts per step.`,
    );
  }
  if (rl.rolloutPlacement === 'colocated') {
    warnings.push(
      'Colocated placement: the inference engine holds a second weight copy and gets a small memory budget (TRL defaults vllm_gpu_memory_utilization to 0.3 against 0.90 standalone). Sleep/wake offload is what makes this viable.',
    );
  }

  if (opts.observedStepSeconds != null) {
    const delta = opts.observedStepSeconds / modelledStepSeconds;
    warnings.push(
      `Calibrated to an observed ${opts.observedStepSeconds.toFixed(0)} s/step; the model predicted ${modelledStepSeconds.toFixed(0)} s (${delta.toFixed(2)}x). Wall-clock and cost now derive from your measurement.`,
    );
  } else {
    warnings.push(
      'Uncalibrated estimate. Against the one published 70B RL run we could source (Tulu 3, 2,880 GPU-hours) this model runs ~4x optimistic -- treat the wall-clock as a FLOOR, not a prediction. Supplying one observed seconds-per-step figure collapses the three dominant uncertainties into a measured constant.',
    );
  }
  if (gating === 'max' && rl.maxCompletionTokens > 4096) {
    warnings.push(
      `Step cost assumes every sequence runs to the ${rl.maxCompletionTokens.toLocaleString()}-token cap. If completions cluster lower and your stack does in-flight replacement, the true cost can be up to 10x less -- switch tail gating to mean or blend.`,
    );
  }
  if (!opts.peft && weightSyncSeconds > 1) {
    warnings.push(
      `Weight refit costs ${weightSyncSeconds.toFixed(1)} s/step (${((weightSyncSeconds * rl.steps) / 60).toFixed(0)} min over ${rl.steps} steps). LoRA syncs only adapters and reduces this to roughly zero.`,
    );
  }

  return {
    generation,
    trainSeconds,
    weightSyncSeconds,
    tokensGenerated,
    genTrainRatio,
    generationShare,
    recommendedInferenceShare: generationShare,
    stepSeconds,
    totalHours: (stepSeconds * rl.steps) / 3600,
    kvPoolBytesAtFullConcurrency,
    warnings: [...generation.warnings, ...warnings],
  };
}

/** The published sanity band for `r` at a given completion length. */
export function expectedRatioBand(maxCompletionTokens: number): number {
  for (const band of R_SANITY_BANDS) {
    if (maxCompletionTokens <= band.maxCompletionTokens) return band.r;
  }
  return R_SANITY_BANDS[R_SANITY_BANDS.length - 1].r;
}
