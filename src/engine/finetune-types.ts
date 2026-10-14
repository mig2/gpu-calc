import type { ConfidenceLevel, GpuSku } from './types';

/**
 * Fine-tuning pane types.
 *
 * Research: docs/research/2026-10-14-finetuning-calculator-research.md
 * Decisions: docs/decisions/2026-10-14-finetuning-tab-decisions.md
 *
 * The fine-tuning question is memory-capacity-bound, unlike pretraining
 * (compute-bound) and inference (bandwidth-bound). These types are shaped
 * around "does the model state fit", not "how many GPUs for my deadline".
 */

/** Adaptation method. `full` through `dora` are SFT-family; the rest are objectives. */
export type FineTuneMethod =
  | 'full'
  | 'lora'
  | 'qlora'
  | 'dora'
  | 'dpo'
  | 'orpo'
  | 'grpo'
  | 'ppo';

export type OptimizerKind =
  | 'adamw_fp32'
  | 'adamw_bf16'
  | 'adam_8bit'
  | 'adafactor'
  | 'sgd_momentum'
  | 'lomo';

/** Precision of the (frozen or trainable) base weights. */
export type WeightPrecision = 'bf16' | 'fp8' | 'nf4';

/** Which linear projections LoRA adapters are injected into. */
export type LoraTargets = 'all_linear' | 'attn_plus_o' | 'attn_only';

/** How model state is distributed when it exceeds one GPU. */
export type ShardingStrategy = 'ddp' | 'zero2' | 'fsdp_zero3';

/** Architecture dimensions. Any may be absent; the engine falls back and downgrades confidence. */
export type ModelArchitecture = {
  id: string;
  label: string;
  /** All resident parameters, including every MoE expert. Memory follows this. */
  totalParameters: number;
  /** Parameters active per token. Compute follows this. Equals total for dense models. */
  activeParameters: number;
  layers?: number | null;
  hidden?: number | null;
  heads?: number | null;
  kvHeads?: number | null;
  headDim?: number | null;
  ffnDim?: number | null;
  vocab?: number | null;
  contextWindow?: number | null;
  isMoe: boolean;
  /** MLA (DeepSeek) has a radically smaller KV footprint than GQA. */
  attentionType?: 'mha' | 'gqa' | 'mqa' | 'mla' | null;
  /** False when any dimension was unverifiable; drives the confidence downgrade. */
  dimsVerified?: boolean;
};

export type FineTuneWorkload = {
  /** Number of training examples (prompt/response pairs). */
  examples: number;
  avgTokensPerExample: number;
  epochs: number;
  /** Max sequence length; drives activation memory linearly. */
  sequenceLength: number;
  /** Sequence packing eliminates padding waste. */
  packing: boolean;
  /** Per-GPU micro-batch size, in sequences. */
  microBatchSize: number;
};

export type FineTuneConfig = {
  method: FineTuneMethod;
  optimizer: OptimizerKind;
  weightPrecision: WeightPrecision;
  gradientCheckpointing: boolean;
  loraRank: number;
  loraTargets: LoraTargets;
  sharding: ShardingStrategy;
  /**
   * DPO only. Reference logprobs can be precomputed offline in one pass, which
   * removes the reference model from memory entirely and drops the 2N reference
   * forward from every step. TRL's `precompute_ref_log_probs` measures -31%
   * peak memory. Defaults to TRUE -- it is standard practice and the cheapest
   * win in the DPO path. Incompatible with moving-reference (`sync_ref_model`).
   */
  precomputeReferenceLogprobs?: boolean;
  /**
   * GRPO/PPO. Whether a frozen reference policy stays resident for the KL term.
   * Defaults to FALSE: TRL's GRPOConfig now defaults `beta=0.0` and does not
   * load a reference model, and DAPO removes the KL term outright. Set true
   * when optimizing against a *learned* reward model, where KL is the guard
   * against reward hacking. Under LoRA the reference is free regardless --
   * disabling the adapters recovers the base policy.
   */
  referenceModelResident?: boolean;
};

/** RL-specific inputs. Only read for `grpo` and `ppo`. */
export type RlConfig = {
  /** Completions sampled per prompt (GRPO group size). */
  groupSize: number;
  /** Max generated tokens per completion. Dominant cost driver for reasoning RL. */
  maxCompletionTokens: number;
  /** Prompts per optimization step. */
  promptsPerStep: number;
  /** Optimization steps. */
  steps: number;
  /**
   * Colocated: inference and training share GPUs (time-sliced, double-resident weights).
   * Disaggregated: separate GPU pools.
   */
  rolloutPlacement: 'colocated' | 'disaggregated';
};

export type FineTuneScenario = {
  model: ModelArchitecture;
  workload: FineTuneWorkload;
  config: FineTuneConfig;
  rl?: RlConfig;
  /** 0-1. Fraction of wall-clock the cluster is productive. */
  availability: number;
  /** >= 1.0. Multiplier for non-overlapped overhead. */
  overheadFactor: number;
  /** Optional MFU override; otherwise derived from GPU count per SCALE_MFU. */
  mfuOverride?: number;
};

export type MemoryBreakdown = {
  /** Resident base weights (total params, all experts). */
  weightsBytes: number;
  /** Gradients for trainable params. */
  gradientsBytes: number;
  /** Optimizer state for trainable params. */
  optimizerBytes: number;
  /** Batch x seqlen dependent. The only dynamic term. */
  activationsBytes: number;
  /** Reference / reward / critic copies and rollout KV cache. */
  auxiliaryBytes: number;
  /** CUDA context, fragmentation, comm buffers. */
  overheadBytes: number;
  totalBytes: number;
};

export type FitState = 'fits' | 'sharded' | 'oom';

export type FineTuneEstimate = {
  gpuId: string;
  method: FineTuneMethod;

  /** Trainable parameter count (adapter params for PEFT methods). */
  trainableParameters: number;
  trainableFraction: number;

  memory: MemoryBreakdown;
  /** Minimum GPUs from the memory floor, rounded to deployable granularity. */
  minGpusMemory: number;
  fitState: FitState;
  /** True when one GPU suffices without sharding. */
  fitsOnSingleGpu: boolean;

  tokensProcessed: number;
  flopsPerToken: number;
  totalFlops: number;

  /** Wall-clock on `minGpusMemory` GPUs. */
  wallClockHours: number;
  /** Effective MFU used, after scale and method factors. */
  effectiveMfu: number;

  confidence: ConfidenceLevel;
  warnings: string[];
  trace: string[];
};

/** One cell of the hero comparison matrix. */
export type MatrixCell = {
  gpu: GpuSku;
  method: FineTuneMethod;
  estimate: FineTuneEstimate | null;
  /** Null when the method is infeasible on this SKU at any count. */
  costUsd: number | null;
};
