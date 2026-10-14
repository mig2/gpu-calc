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

/**
 * Where a row's three hourly rates came from.
 *
 * `vendor` means the numbers were read from the provider's own pricing page or
 * public pricing API. `aggregator` means they came from a third party because
 * the provider does not publish a fetchable rate — those rows are the ones to
 * distrust first, and the ones a refresh should re-check.
 */
export type CloudPriceSourceQuality = 'vendor' | 'aggregator';

export type CloudGpuInstance = {
  instance: string;
  gpu: string;
  gpuCount: number;
  onDemandPerHr: number;
  reservedPerHr: number | null;
  spotPerHr: number | null;
  /** Page or API the rates were read from, so a reader can re-check them. */
  sourceUrl?: string;
  /**
   * Whether that source was the provider itself. Always one of
   * `CloudPriceSourceQuality`, but typed `string` because TypeScript widens
   * string literals when importing the JSON and `InferenceCalculator` assigns
   * the imported row straight to this type. Narrow with `CloudPriceSourceQuality`
   * at the point of use.
   */
  sourceQuality?: string;
  /**
   * Region the rates apply to. GPU pricing varies materially by region, so a
   * rate without one is ambiguous. Providers that publish a single rate across
   * regions say so here rather than naming a region they do not price by.
   */
  region?: string;
  /** ISO date (YYYY-MM-DD) the rates were read from the source. */
  pricedOn?: string;
  /**
   * Anything needed to not misread the three rates — a rate that is a market
   * price rather than a quote, a tier that exists but is unpublished, a SKU
   * that cannot actually be bought on demand.
   */
  note?: string;
};

export type CloudGpuProvider = {
  provider: string;
  instances: CloudGpuInstance[];
};

export type CloudGpuPricingData = {
  lastUpdated: string;
  /** How the data was sourced, which region each provider was priced in. */
  note?: string;
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
