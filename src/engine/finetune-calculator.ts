import type { ConfidenceLevel, GpuSku } from './types';
import type {
  FineTuneConfig,
  FineTuneEstimate,
  FineTuneMethod,
  FineTuneScenario,
  FitState,
  LoraTargets,
  MemoryBreakdown,
  ModelArchitecture,
  OptimizerKind,
  WeightPrecision,
} from './finetune-types';

/**
 * Fine-tuning estimator.
 *
 * Research:  docs/research/2026-10-14-finetuning-calculator-research.md
 * Decisions: docs/decisions/2026-10-14-finetuning-tab-decisions.md
 *
 * Central design point: fine-tuning is memory-capacity-bound. Published anchors
 * the memory model reproduces (research sect. 1.3), in GB for full / LoRA r=64 / QLoRA:
 *   7B   88 / 20 / 8      14B  174 / 35 / 14
 *   32B  394 / 76 / 28    70B  860 / 159 / 52
 */

// ---------------------------------------------------------------------------
// Constants, each traceable to the research document
// ---------------------------------------------------------------------------

/** Bytes per base weight by precision. NF4 is 4-bit plus quantization constants. */
export const WEIGHT_BYTES: Record<WeightPrecision, number> = {
  bf16: 2,
  fp8: 1,
  nf4: 0.5,
};

/** Gradient bytes per trainable parameter (BF16 gradients, the standard). */
export const GRADIENT_BYTES = 2;

/**
 * Optimizer moment bytes per trainable parameter.
 * AdamW fp32 keeps two fp32 moments = 8 B. 8-bit Adam block-wise quantizes
 * both to 1 B each (research sect. 1.2: "8 B -> 2 B/param").
 */
export const OPTIMIZER_MOMENT_BYTES: Record<OptimizerKind, number> = {
  adamw_fp32: 8,
  adamw_bf16: 4,
  adam_8bit: 2,
  adafactor: 2,
  sgd_momentum: 4,
  lomo: 0,
};

/**
 * Bytes for the fp32 master copy of trainable weights, per optimizer.
 * Mixed-precision AdamW keeps one; LOMO fuses the update and keeps none.
 * Full FT with adamw_fp32 therefore totals 2 (w) + 2 (g) + 4 + 8 = 16 B/param,
 * the standard Megatron accounting.
 */
export const OPTIMIZER_MASTER_BYTES: Record<OptimizerKind, number> = {
  adamw_fp32: 4,
  adamw_bf16: 2,
  adam_8bit: 2,
  adafactor: 2,
  sgd_momentum: 2,
  lomo: 0,
};

/**
 * LoRA adapter weights and gradients are fp32 in every mainstream
 * implementation, regardless of base precision (research sect. 1.4).
 */
export const ADAPTER_WEIGHT_BYTES = 4;
export const ADAPTER_GRADIENT_BYTES = 4;

/**
 * Activation memory coefficient per layer, per token, per hidden unit, with
 * FlashAttention assumed so the quadratic 5as/h attention-matrix term drops.
 * From Megatron's sbh(34 + 5as/h) (research sect. 1.5).
 */
export const ACTIVATION_COEFF = 34;
/** Bytes per stashed checkpoint boundary (one BF16 activation per layer). */
export const CHECKPOINT_STASH_BYTES = 2;

/** Fraction of advertised HBM actually usable, matching hardware-estimator. */
export const USABLE_MEMORY_FRACTION = 0.85;
/** Fixed per-GPU overhead: CUDA context, allocator fragmentation, comm buffers. */
export const PER_GPU_OVERHEAD_BYTES = 1.5e9;

/**
 * Achieved MFU by scale, back-solved from NVIDIA NeMo's published LoRA
 * throughputs (research sect. 2.2). Reproduces NeMo's own reported MFU to
 * within 2.4 percentage points. Lower than pretraining's 0.40-0.45 because
 * fine-tuning runs smaller batches and shorter sequences.
 */
export const SCALE_MFU = {
  singleGpu: 0.40,
  singleNode: 0.30,
  multiNode: 0.27,
} as const;

/**
 * Per-node-doubling scaling efficiency. Measured from NeMo's 8 -> 16 GPU rows:
 * 70B LoRA 1091.9 -> 984.9 tok/s/GPU = 90.2%; 32B 2250.3 -> 2072.9 = 92.1%.
 */
export const NODE_SCALING_EFFICIENCY = 0.90;

/**
 * QLoRA throughput penalty from on-the-fly NF4 dequantization. Benchmark
 * studies put it at 0.7-0.8x LoRA; sources disagree widely (some claim
 * 0.9-0.95 with recent bitsandbytes). Default mid-range, medium-low confidence.
 */
export const QLORA_THROUGHPUT_FACTOR = 0.75;
/** DoRA adds a magnitude-vector decomposition on top of LoRA. */
export const DORA_THROUGHPUT_FACTOR = 0.90;

/** GPU counts that are actually deployable. */
const DEPLOYABLE_COUNTS = [1, 2, 4, 8];

// ---------------------------------------------------------------------------
// Architecture fallback
// ---------------------------------------------------------------------------

/**
 * Estimate layers and hidden size from parameter count when the catalog lacks
 * them, using the empirical aspect ratio d ~= 128L observed across the Llama 3
 * family (8B 4096/32=128, 70B 8192/80=102, 405B 16384/126=130).
 *
 * With N = 12Ld^2 and d = 128L:  N = 196608 L^3.
 * Validation: 8B -> L=34.4 (actual 32); 70B -> L=70.9 (actual 80);
 *             405B -> L=127.2 (actual 126).
 */
export function estimateArchitecture(activeParameters: number): {
  layers: number;
  hidden: number;
  ffnDim: number;
  kvDim: number;
} {
  const layers = Math.max(1, Math.round(Math.cbrt(activeParameters / 196_608)));
  const hidden = Math.round(128 * layers);
  return {
    layers,
    hidden,
    // Llama 3 and Qwen 2.5 both sit at ffn/d = 3.5.
    ffnDim: Math.round(3.5 * hidden),
    // GQA ratios vary 0.125-0.25 of hidden; 0.2 is mid-range.
    kvDim: Math.round(0.2 * hidden),
  };
}

type ResolvedDims = {
  layers: number;
  hidden: number;
  ffnDim: number;
  kvDim: number;
  /** True when dimensions came from the catalog rather than the fallback. */
  exact: boolean;
  /**
   * True only when the catalog also confirms the dimensions were read from the
   * model's own config.json. Catalog entries may carry plausible dimensions
   * with `dimsVerified: false` (e.g. models whose config is unpublished);
   * those are still better than the fallback estimate, but must not earn
   * `high` confidence.
   */
  verified: boolean;
};

export function resolveDims(model: ModelArchitecture): ResolvedDims {
  const haveAll =
    model.layers != null &&
    model.hidden != null &&
    model.ffnDim != null &&
    model.kvHeads != null &&
    model.headDim != null;

  if (haveAll) {
    return {
      layers: model.layers as number,
      hidden: model.hidden as number,
      ffnDim: model.ffnDim as number,
      kvDim: (model.kvHeads as number) * (model.headDim as number),
      exact: true,
      // Absent flag means "not asserted"; only an explicit true counts.
      verified: model.dimsVerified === true,
    };
  }
  return {
    ...estimateArchitecture(model.activeParameters),
    exact: false,
    verified: false,
  };
}

// ---------------------------------------------------------------------------
// LoRA adapter parameters
// ---------------------------------------------------------------------------

/**
 * Exact adapter parameter count. For a linear layer d_in -> d_out, LoRA adds
 * A (r x d_in) and B (d_out x r), so r(d_in + d_out) parameters.
 *
 * Validation: Llama-3-8B (L=32, d=4096, kv=1024, ffn=14336) at r=16 all-linear
 * yields 41,943,040 -- PEFT's own reported figure for that configuration.
 */
export function loraAdapterParams(
  dims: Pick<ResolvedDims, 'layers' | 'hidden' | 'ffnDim' | 'kvDim'>,
  rank: number,
  targets: LoraTargets,
): number {
  const { layers, hidden: d, ffnDim, kvDim } = dims;

  const q = d + d;
  const k = d + kvDim;
  const v = d + kvDim;
  const o = d + d;
  const gate = d + ffnDim;
  const up = d + ffnDim;
  const down = ffnDim + d;

  let sum: number;
  switch (targets) {
    case 'attn_only':
      sum = q + v;
      break;
    case 'attn_plus_o':
      sum = q + k + v + o;
      break;
    case 'all_linear':
    default:
      sum = q + k + v + o + gate + up + down;
      break;
  }

  return layers * rank * sum;
}

/** Methods that freeze the base model and train only adapters. */
export function isPeftMethod(method: FineTuneMethod): boolean {
  return method === 'lora' || method === 'qlora' || method === 'dora';
}

/** Methods whose cost is dominated by rollout generation rather than gradients. */
export function isRlMethod(method: FineTuneMethod): boolean {
  return method === 'grpo' || method === 'ppo';
}

export function trainableParameters(
  model: ModelArchitecture,
  config: FineTuneConfig,
  dims: ResolvedDims,
): number {
  if (isPeftMethod(config.method)) {
    const base = loraAdapterParams(dims, config.loraRank, config.loraTargets);
    // DoRA adds one magnitude vector per targeted output dimension; small but
    // non-zero, approximated at 1% of adapter params.
    return config.method === 'dora' ? base * 1.01 : base;
  }
  // full, dpo, orpo, grpo, ppo all train the full parameter set by default.
  return model.totalParameters;
}

// ---------------------------------------------------------------------------
// Memory
// ---------------------------------------------------------------------------

/**
 * Resident model copies beyond the trainable policy.
 *
 * Multipliers on SFT full-FT model state (16 B/param), per the RL research
 * doc sect. 6.2 -- which corrects the looser figures in the fine-tuning doc:
 *
 *   SFT                        policy                              1.00x
 *   DPO, reference precomputed policy                              1.00x
 *   DPO, reference resident    policy + ref (2 B/param)            1.125x  (not "~2x")
 *   ORPO                       reference-free                      1.00x
 *   GRPO, beta=0 (TRL default) policy                              1.00x   (not "~2-3x")
 *   GRPO, beta>0               policy + ref                        1.125x
 *   PPO                        policy + critic + reward + ref      2.25x   (not "~4x")
 *
 * Only the policy and the PPO critic carry optimizer state; reference and
 * reward models are inference-only and quantizable or offloadable.
 */
export function auxiliaryCopies(config: FineTuneConfig): {
  frozenCopies: number;
  trainableCopies: number;
} {
  switch (config.method) {
    case 'dpo':
      // Precomputing reference logprobs removes the reference from memory.
      return {
        frozenCopies: config.precomputeReferenceLogprobs === false ? 1 : 0,
        trainableCopies: 0,
      };
    case 'grpo':
      // beta=0.0 is TRL's default and loads no reference model.
      return { frozenCopies: config.referenceModelResident === true ? 1 : 0, trainableCopies: 0 };
    case 'ppo':
      // Reward model always; reference only when the KL term is active.
      // Critic is trainable and carries full optimizer state.
      return {
        frozenCopies: config.referenceModelResident === false ? 1 : 2,
        trainableCopies: 1,
      };
    default:
      return { frozenCopies: 0, trainableCopies: 0 };
  }
}

export function activationBytes(
  dims: ResolvedDims,
  sequenceLength: number,
  microBatchSize: number,
  gradientCheckpointing: boolean,
): number {
  const { layers, hidden } = dims;
  const bs = microBatchSize * sequenceLength;

  if (gradientCheckpointing) {
    // Stashed layer boundaries, plus the peak working set of the one layer
    // being recomputed.
    return CHECKPOINT_STASH_BYTES * bs * hidden * layers + ACTIVATION_COEFF * bs * hidden;
  }
  return ACTIVATION_COEFF * bs * hidden * layers;
}

export function computeMemory(
  scenario: FineTuneScenario,
  dims: ResolvedDims,
  trainable: number,
): MemoryBreakdown {
  const { model, config, workload } = scenario;
  const bw = WEIGHT_BYTES[config.weightPrecision];

  // Memory follows TOTAL parameters: every MoE expert stays resident.
  const weightsBytes = model.totalParameters * bw;

  const gradientsBytes = isPeftMethod(config.method)
    ? trainable * ADAPTER_GRADIENT_BYTES
    : trainable * GRADIENT_BYTES;

  const optimizerBytes = isPeftMethod(config.method)
    ? trainable * (ADAPTER_WEIGHT_BYTES + OPTIMIZER_MOMENT_BYTES[config.optimizer])
    : trainable *
      (OPTIMIZER_MASTER_BYTES[config.optimizer] + OPTIMIZER_MOMENT_BYTES[config.optimizer]);

  const activationsBytes = activationBytes(
    dims,
    workload.sequenceLength,
    workload.microBatchSize,
    config.gradientCheckpointing,
  );

  const { frozenCopies, trainableCopies } = auxiliaryCopies(config);
  let auxiliaryBytes = frozenCopies * model.totalParameters * bw;
  // A trainable critic carries its own gradients and optimizer state.
  auxiliaryBytes +=
    trainableCopies *
    model.totalParameters *
    (bw +
      GRADIENT_BYTES +
      OPTIMIZER_MASTER_BYTES[config.optimizer] +
      OPTIMIZER_MOMENT_BYTES[config.optimizer]);

  const subtotal =
    weightsBytes + gradientsBytes + optimizerBytes + activationsBytes + auxiliaryBytes;
  // 8% allocator slack on top of the fixed per-GPU reservation.
  const overheadBytes = PER_GPU_OVERHEAD_BYTES + subtotal * 0.08;

  return {
    weightsBytes,
    gradientsBytes,
    optimizerBytes,
    activationsBytes,
    auxiliaryBytes,
    overheadBytes,
    totalBytes: subtotal + overheadBytes,
  };
}

// ---------------------------------------------------------------------------
// Compute
// ---------------------------------------------------------------------------

/**
 * FLOPs per token (research sect. 2.1).
 *
 * Full FT 6N: 2N forward + 4N backward.
 * LoRA   ~4N: 2N forward + ~2N backward. The frozen base needs no weight
 *             gradients -- standard backward is 4bmn per linear layer, LoRA's
 *             is 2bmn + 4br(m+n), and with r << m,n the adapter term vanishes.
 * Gradient checkpointing adds one extra forward: +2N.
 *
 * Uses ACTIVE parameters: compute follows active, memory follows total.
 */
export function flopsPerToken(
  activeParameters: number,
  method: FineTuneMethod,
  gradientCheckpointing: boolean,
): number {
  const base = isPeftMethod(method) ? 4 : 6;
  const recompute = gradientCheckpointing ? 2 : 0;
  return (base + recompute) * activeParameters;
}

/**
 * Objective multiplier on tokens processed, relative to SFT.
 *
 * Per-preference-pair, DPO processes two sequences (chosen + rejected):
 *   policy forward  2 x 2N
 *   policy backward 2 x 4N
 *   reference fwd   2 x 2N   (only when the reference is resident)
 * giving 8N per processed token against SFT's 6N, over 2 sequences:
 *   resident reference    8N/6N x 2 = 2.67x
 *   precomputed reference 6N/6N x 2 = 2.00x
 * ORPO is reference-free but still runs two completions: 2.00x.
 *
 * This replaces the fine-tuning doc's looser "~3-4x" (RL research doc sect. 8.1).
 */
export function objectiveMultiplier(config: FineTuneConfig): number {
  switch (config.method) {
    case 'dpo':
      return config.precomputeReferenceLogprobs === false ? 2.67 : 2.0;
    case 'orpo':
      return 2.0;
    default:
      return 1.0;
  }
}

export function tokensProcessed(scenario: FineTuneScenario): number {
  const { workload, config } = scenario;
  // Packing concatenates variable-length samples to eliminate padding.
  // Unpacked efficiency on a mixed-length corpus is typically 0.4-0.6.
  const packingEfficiency = workload.packing ? 0.95 : 0.55;
  const datasetTokens = workload.examples * workload.avgTokensPerExample;
  return (datasetTokens * workload.epochs * objectiveMultiplier(config)) / packingEfficiency;
}

/** Throughput penalty from the method's implementation overhead. */
export function methodThroughputFactor(method: FineTuneMethod): number {
  if (method === 'qlora') return QLORA_THROUGHPUT_FACTOR;
  if (method === 'dora') return DORA_THROUGHPUT_FACTOR;
  return 1.0;
}

export function scaleMfu(gpuCount: number): number {
  if (gpuCount <= 1) return SCALE_MFU.singleGpu;
  if (gpuCount <= 8) return SCALE_MFU.singleNode;
  return SCALE_MFU.multiNode;
}

/** Aggregate efficiency loss across node doublings. */
export function scalingEfficiency(gpuCount: number): number {
  if (gpuCount <= 8) return 1.0;
  const nodes = gpuCount / 8;
  return Math.pow(NODE_SCALING_EFFICIENCY, Math.log2(nodes));
}

// ---------------------------------------------------------------------------
// Main estimator
// ---------------------------------------------------------------------------

function usableBytesPerGpu(gpu: GpuSku): number {
  return gpu.memoryGb * 1e9 * USABLE_MEMORY_FRACTION - PER_GPU_OVERHEAD_BYTES;
}

/**
 * Minimum GPUs from the memory floor, rounded up to a deployable count.
 *
 * Sharding caveat: ZeRO-3/FSDP shards weights, gradients and optimizer state,
 * but activations are per-GPU and do not shard, so they are excluded from the
 * shardable pool. DDP replicates everything and cannot reduce the floor.
 */
export function minGpusForMemory(
  memory: MemoryBreakdown,
  gpu: GpuSku,
  sharding: FineTuneConfig['sharding'],
): { count: number; fitState: FitState } {
  const usable = usableBytesPerGpu(gpu);
  if (usable <= 0) return { count: Infinity, fitState: 'oom' };

  if (memory.totalBytes <= usable) {
    return { count: 1, fitState: 'fits' };
  }

  if (sharding === 'ddp') {
    // DDP replicates model state; one GPU must hold all of it.
    return { count: Infinity, fitState: 'oom' };
  }

  const shardable =
    memory.weightsBytes + memory.gradientsBytes + memory.optimizerBytes + memory.auxiliaryBytes;
  const perGpuFixed = memory.activationsBytes + PER_GPU_OVERHEAD_BYTES;

  for (const n of DEPLOYABLE_COUNTS) {
    if (shardable / n + perGpuFixed <= usable) {
      return { count: n, fitState: n === 1 ? 'fits' : 'sharded' };
    }
  }
  // Beyond one node, round up to whole nodes of 8.
  const needed = Math.ceil(shardable / Math.max(1, usable - perGpuFixed));
  if (!Number.isFinite(needed) || needed <= 0) {
    return { count: Infinity, fitState: 'oom' };
  }
  return { count: Math.ceil(needed / 8) * 8, fitState: 'sharded' };
}

function pickConfidence(
  scenario: FineTuneScenario,
  dims: ResolvedDims,
): ConfidenceLevel {
  const { method } = scenario.config;
  if (isRlMethod(method)) return 'low';
  if (method === 'qlora' || method === 'dpo' || method === 'orpo') return 'medium-low';
  // Fallback dims, or catalog dims that were never confirmed against a
  // published config, both cap confidence at medium.
  if (!dims.exact || !dims.verified) return 'medium';
  if (method === 'dora') return 'medium';
  return 'high';
}

export function estimateFineTune(
  scenario: FineTuneScenario,
  gpu: GpuSku,
): FineTuneEstimate {
  const { model, workload, config } = scenario;
  const warnings: string[] = [];
  const dims = resolveDims(model);
  const trainable = trainableParameters(model, config, dims);
  const memory = computeMemory(scenario, dims, trainable);
  const { count: minGpus, fitState } = minGpusForMemory(memory, gpu, config.sharding);

  const tokens = tokensProcessed(scenario);
  const fpt = flopsPerToken(model.activeParameters, config.method, config.gradientCheckpointing);
  const totalFlops = tokens * fpt;

  const gpuCount = Number.isFinite(minGpus) ? minGpus : 0;
  const baseMfu = scenario.mfuOverride ?? scaleMfu(gpuCount || 1);
  const effectiveMfu =
    baseMfu * methodThroughputFactor(config.method) * scalingEfficiency(gpuCount || 1);

  const sustainedPerGpu = gpu.bf16DenseFlops * effectiveMfu * scenario.availability;
  const wallClockHours =
    gpuCount > 0 && sustainedPerGpu > 0
      ? (totalFlops * scenario.overheadFactor) / (gpuCount * sustainedPerGpu) / 3600
      : Infinity;

  // --- warnings -----------------------------------------------------------
  if (fitState === 'oom') {
    warnings.push(
      config.sharding === 'ddp'
        ? `Does not fit on one ${gpu.label} and DDP cannot shard model state. Switch to FSDP/ZeRO-3 or use a lower-memory method.`
        : `Does not fit on ${gpu.label} at any practical GPU count. Use a lower-memory method (QLoRA), a smaller rank, or shorter sequences.`,
    );
  }
  if (fitState === 'sharded') {
    warnings.push(
      `Requires ${minGpus}x ${gpu.label} with ${config.sharding === 'fsdp_zero3' ? 'FSDP/ZeRO-3' : 'ZeRO-2'} sharding. Model state alone needs ${(memory.totalBytes / 1e9).toFixed(0)} GB.`,
    );
  }
  if (model.isMoe) {
    warnings.push(
      `MoE model: memory follows ${(model.totalParameters / 1e9).toFixed(0)}B total parameters, compute follows ${(model.activeParameters / 1e9).toFixed(0)}B active. Both experts and router stay resident.`,
    );
    if (config.weightPrecision === 'nf4') {
      warnings.push(
        'Do not use 4-bit QLoRA for MoE models: bitsandbytes lacks native MoE support and quantization interferes with expert routing. Use BF16 LoRA instead.',
      );
    }
  }
  if (!dims.exact) {
    warnings.push(
      'Architecture dimensions unknown for this model; layers and hidden size were estimated from parameter count (d ~= 128L). Adapter-parameter and activation estimates carry +/-40% uncertainty.',
    );
  } else if (!dims.verified) {
    warnings.push(
      "Architecture dimensions for this model are not confirmed against a published config.json. Adapter-parameter and activation estimates may be off; verify the model's layer count, hidden size and FFN dimension before committing hardware.",
    );
  }
  if (workload.epochs > 5) {
    warnings.push(
      `${workload.epochs} epochs risks memorization. Fine-tuning sets of 10^2-10^5 examples are tiny against a multi-trillion-token pretraining corpus; 1-3 epochs is standard.`,
    );
  }
  if (workload.examples < 100) {
    warnings.push(
      `${workload.examples} examples is below the useful floor. Published guidance puts the minimum for a narrow task at 100-500.`,
    );
  }
  if (workload.sequenceLength > 16_384) {
    warnings.push(
      `At sequence length ${workload.sequenceLength.toLocaleString()}, activations (${(memory.activationsBytes / 1e9).toFixed(1)} GB) dominate model state. Consider chunked or balance-packed long-context training.`,
    );
  }
  if (!workload.packing) {
    warnings.push(
      'Sequence packing disabled: padding waste assumed at 45%. Packing yields 3.8x at 32K and 7.3x at 128K sequence length versus naive batching.',
    );
  }
  if (config.method === 'qlora') {
    warnings.push(
      `QLoRA throughput factor ${QLORA_THROUGHPUT_FACTOR} applied for on-the-fly NF4 dequantization. Sources span 0.7-0.8x; some report far less penalty with recent bitsandbytes.`,
    );
  }
  if (config.method === 'full' && config.optimizer === 'adamw_fp32') {
    const qloraNote = (model.totalParameters * 0.5 + trainable * 0.02) / 1e9;
    warnings.push(
      `Full fine-tuning holds ${(memory.totalBytes / 1e9).toFixed(0)} GB of state. QLoRA on the same model would need roughly ${qloraNote.toFixed(0)} GB -- often a 1-2 order-of-magnitude cost difference.`,
    );
  }
  if (isRlMethod(config.method)) {
    warnings.push(
      'RL post-training cost is dominated by rollout generation (60-80% of step time), not gradient steps. This figure covers the training side only -- use the RL rollout estimator for the generation cost and the inference/training GPU split. Treat as a lower bound.',
    );
  }
  if (isPeftMethod(config.method) && config.loraRank > 128) {
    warnings.push(
      `LoRA rank ${config.loraRank} is unusually high; adapter parameters scale linearly with rank and the quality return flattens well below this.`,
    );
  }

  // --- trace --------------------------------------------------------------
  const gb = (b: number) => (b / 1e9).toFixed(2);
  const trace = [
    `Model: ${model.label} -- ${(model.totalParameters / 1e9).toFixed(1)}B total, ${(model.activeParameters / 1e9).toFixed(1)}B active`,
    `Dimensions: L=${dims.layers}, d=${dims.hidden}, ffn=${dims.ffnDim}, d_kv=${dims.kvDim}${dims.exact ? ' (from catalog)' : ' (estimated from N)'}`,
    `Trainable params = ${trainable.toExponential(3)} (${((trainable / model.totalParameters) * 100).toFixed(3)}% of total)`,
    `Weights = ${(model.totalParameters / 1e9).toFixed(1)}B x ${WEIGHT_BYTES[config.weightPrecision]} B = ${gb(memory.weightsBytes)} GB`,
    `Gradients = ${gb(memory.gradientsBytes)} GB`,
    `Optimizer (${config.optimizer}) = ${gb(memory.optimizerBytes)} GB`,
    `Activations (b=${workload.microBatchSize}, s=${workload.sequenceLength}, checkpointing ${config.gradientCheckpointing ? 'on' : 'off'}) = ${gb(memory.activationsBytes)} GB`,
    `Auxiliary copies = ${gb(memory.auxiliaryBytes)} GB`,
    `Overhead = ${gb(memory.overheadBytes)} GB`,
    `Total memory = ${gb(memory.totalBytes)} GB; usable per ${gpu.label} = ${gb(usableBytesPerGpu(gpu))} GB`,
    `Minimum GPUs (memory) = ${Number.isFinite(minGpus) ? minGpus : 'infeasible'} [${fitState}]`,
    `Dataset tokens = ${workload.examples.toLocaleString()} x ${workload.avgTokensPerExample} = ${(workload.examples * workload.avgTokensPerExample).toExponential(2)}`,
    `Tokens processed = x${workload.epochs} epochs x ${objectiveMultiplier(config)} objective / ${workload.packing ? 0.95 : 0.55} packing = ${tokens.toExponential(3)}`,
    `FLOPs/token = ${isPeftMethod(config.method) ? 4 : 6}N${config.gradientCheckpointing ? ' + 2N recompute' : ''} = ${fpt.toExponential(2)}`,
    `Total FLOPs = ${totalFlops.toExponential(3)}`,
    `Effective MFU = ${baseMfu} x ${methodThroughputFactor(config.method)} method x ${scalingEfficiency(gpuCount || 1).toFixed(3)} scaling = ${effectiveMfu.toFixed(3)}`,
    `Wall clock = ${totalFlops.toExponential(2)} x ${scenario.overheadFactor} / (${gpuCount} x ${sustainedPerGpu.toExponential(2)}) = ${Number.isFinite(wallClockHours) ? wallClockHours.toFixed(2) + ' h' : 'n/a'}`,
  ];

  return {
    gpuId: gpu.id,
    method: config.method,
    trainableParameters: trainable,
    trainableFraction: trainable / model.totalParameters,
    memory,
    minGpusMemory: minGpus,
    fitState,
    fitsOnSingleGpu: fitState === 'fits',
    tokensProcessed: tokens,
    flopsPerToken: fpt,
    totalFlops,
    wallClockHours,
    effectiveMfu,
    confidence: pickConfidence(scenario, dims),
    warnings,
    trace,
  };
}

// ---------------------------------------------------------------------------
// Hero output: the method x GPU comparison matrix
// ---------------------------------------------------------------------------

/**
 * Build the comparison matrix. The pane's headline finding is the
 * three-order-of-magnitude cost spread between full fine-tuning and QLoRA on
 * the same model and dataset, so the matrix -- not a single-config card -- is
 * the primary output (decisions doc, Decision 2).
 */
export function buildMethodMatrix(
  scenario: FineTuneScenario,
  gpus: GpuSku[],
  methods: FineTuneMethod[],
  gpuHourlyCost: Record<string, number>,
): {
  methods: FineTuneMethod[];
  gpus: GpuSku[];
  cells: Record<string, Record<string, { estimate: FineTuneEstimate; costUsd: number | null }>>;
} {
  const cells: Record<
    string,
    Record<string, { estimate: FineTuneEstimate; costUsd: number | null }>
  > = {};

  for (const method of methods) {
    cells[method] = {};
    for (const gpu of gpus) {
      const methodScenario: FineTuneScenario = {
        ...scenario,
        config: {
          ...scenario.config,
          method,
          // QLoRA implies a 4-bit base; full FT and the RL/preference
          // objectives keep the base at its selected precision.
          weightPrecision: method === 'qlora' ? 'nf4' : scenario.config.weightPrecision,
        },
      };
      const estimate = estimateFineTune(methodScenario, gpu);
      const rate = gpuHourlyCost[gpu.id];
      const costUsd =
        rate != null && Number.isFinite(estimate.wallClockHours) && Number.isFinite(estimate.minGpusMemory)
          ? rate * estimate.minGpusMemory * estimate.wallClockHours
          : null;
      cells[method][gpu.id] = { estimate, costUsd };
    }
  }

  return { methods, gpus, cells };
}
