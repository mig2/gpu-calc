/**
 * Whether a recorded rate was read from the vendor's own pricing page or from a
 * third-party price aggregator. Aggregator figures have been wrong by ~79% in
 * past research passes, so the distinction is worth carrying in the data.
 */
export type SourceQuality = 'vendor' | 'aggregator';

/**
 * One callable API model and its list rates.
 *
 * Every optional field is optional because the vendor may not publish it, not
 * because it is unimportant: an absent `cacheReadPer1M` means "this provider
 * does not publish a cache-hit rate", never "cache reads are free". A missing
 * figure is deliberate — a plausible-looking wrong price would be worse, and on
 * a buy-vs-build tool every pricing error lands on the buy side of the scale.
 *
 * Cache fields matter more than their optionality suggests. Cache reads cost
 * 2.5–10% of base input and agentic workloads run >85% cache hits by token
 * volume, so a model priced without them systematically overstates API cost.
 *
 * Tiered pricing (length tiers, time-of-day, promotional step-ups) is *not*
 * modeled — see issue #105. Rows record the base tier and name the tier in
 * `note`.
 */
export type ApiModel = {
  model: string;
  inputPer1M: number;    // $ per 1M input tokens
  outputPer1M: number;   // $ per 1M output tokens
  contextWindow: number;
  /** The provider's own API model id, where it publishes one. */
  modelId?: string;
  /** $ per 1M cached-input ("cache hit") tokens. */
  cacheReadPer1M?: number;
  /**
   * $ per 1M tokens to write a prefix into the cache. Absent where the provider
   * does not bill cache writes separately from base input (OpenAI, Google,
   * DeepSeek and the open-model hosts all fold the write into input).
   */
  cacheWritePer1M?: number;
  /** Fraction off list on the provider's batch/async API — 0.5 means 50% off. */
  batchDiscount?: number;
  /** Exact URL the rates were read from. */
  sourceUrl?: string;
  /**
   * Where the rates came from. Declared as the union widened to `string`
   * because `api-pricing.json` is imported directly and TypeScript widens JSON
   * string literals; narrowing this to bare `SourceQuality` would need a cast
   * at the import site, which is outside this data-only pass.
   */
  sourceQuality?: SourceQuality | (string & {});
  /** Which tier is recorded, announced increases, and other caveats. */
  note?: string;
};

export type ApiProvider = {
  provider: string;
  models: ApiModel[];
};

export type ApiPricingData = {
  lastUpdated: string;
  note: string;
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
