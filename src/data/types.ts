export type ApiModel = {
  model: string;
  inputPer1M: number;    // $ per 1M input tokens
  outputPer1M: number;   // $ per 1M output tokens
  contextWindow: number;
};

export type ApiProvider = {
  provider: string;
  models: ApiModel[];
};

export type ApiPricingData = {
  lastUpdated: string;
  providers: ApiProvider[];
};

export type CloudGpuInstance = {
  instance: string;
  gpu: string;
  gpuCount: number;
  onDemandPerHr: number;
  reservedPerHr: number | null;
  spotPerHr: number | null;
};

export type CloudGpuProvider = {
  provider: string;
  instances: CloudGpuInstance[];
};

export type CloudGpuPricingData = {
  lastUpdated: string;
  providers: CloudGpuProvider[];
};

export type SelfHostEntry = {
  model: string;
  parameters: string;
  gpu: string;
  gpuCount: number;
  quantization: string;
  framework: string;
  outputTokensPerSec: number;
  maxConcurrentRequests: number;
};

export type SelfHostThroughputData = {
  lastUpdated: string;
  note: string;
  entries: SelfHostEntry[];
};

/** How the attention block is wired, which decides the KV-cache width. */
export type AttentionType = 'mha' | 'gqa' | 'mqa' | 'mla';

/**
 * Multi-head latent attention dimensions. MLA has no plain k/v projections —
 * it compresses to a latent of `kvLoraRank + qkRopeHeadDim` per token, which is
 * what the KV cache actually stores (far smaller than `kvHeads × headDim`).
 * Present only when `attentionType === 'mla'`.
 */
export type MlaDims = {
  qLoraRank: number;
  kvLoraRank: number;
  qkNopeHeadDim: number;
  qkRopeHeadDim: number;
  vHeadDim: number;
};

/**
 * One model in the fine-tuning catalog. Architecture dimensions are transcribed
 * from the model's own `config.json` (see `source`) so that adapter parameter
 * counts and activation memory can be computed exactly rather than estimated.
 *
 * Every dimension is nullable: a `null` means "not verifiable from the config"
 * and is a deliberate signal for the engine to fall back to its heuristic dims
 * at reduced confidence. A plausible-looking wrong number would be worse.
 */
export type FineTuneModelEntry = {
  id: string;
  label: string;
  family: string;
  /** All resident parameters. Memory follows this figure. */
  totalParameters: number | null;
  /** Parameters touched per token — equal to `totalParameters` when dense. Compute follows this figure. */
  activeParameters: number | null;
  layers: number | null;
  /** Residual stream width, `hidden_size`. */
  hidden: number | null;
  heads: number | null;
  kvHeads: number | null;
  headDim: number | null;
  /** SwiGLU intermediate width. For MoE this is the *per-expert* width. */
  ffnDim: number | null;
  vocab: number | null;
  contextWindow: number | null;
  /** True when embedding and output-head weights are shared. */
  tiedEmbeddings: boolean;
  isMoe: boolean;
  numExperts: number | null;
  expertsPerToken: number | null;
  attentionType: AttentionType;
  mla?: MlaDims;
  license: string;
  /** Exact URL of the config read, or of the mirror used when the repo is gated. */
  source: string;
  /**
   * True when the dims describe a uniform GQA/MHA + (optionally MoE) SwiGLU
   * stack, so the exact adapter and activation formulas apply. False for hybrid
   * or latent-attention stacks whose dims are recorded but whose per-layer shape
   * differs from the formula — the engine should use its fallback there.
   */
  dimsVerified: boolean;
  note?: string;
};

export type FineTuneModelData = {
  lastUpdated: string;
  note: string;
  models: FineTuneModelEntry[];
};
