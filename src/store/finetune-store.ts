import { create } from 'zustand';
import { buildMethodMatrix } from '../engine/finetune-calculator';
import { GPU_SKUS, getGpuById } from '../engine/gpu-data';
import type {
  FineTuneConfig,
  FineTuneMethod,
  FineTuneScenario,
  FineTuneWorkload,
  LoraTargets,
  ModelArchitecture,
  OptimizerKind,
  ShardingStrategy,
  WeightPrecision,
} from '../engine/finetune-types';
import type { GpuSku } from '../engine/types';
import catalogData from '../data/finetune-models.json';
import pricingData from '../data/cloud-gpu-pricing.json';
import type {
  CloudGpuPricingData,
  FineTuneModelData,
  FineTuneModelEntry,
} from '../data/types';

/**
 * Fine-tuning pane store.
 *
 * Decisions: docs/decisions/2026-10-14-finetuning-tab-decisions.md
 *
 * Shaped like `scenario-store.ts`: raw inputs live in the store, and every
 * setter recomputes the derived matrix synchronously. Derivation deliberately
 * never happens in an effect — the matrix is a pure function of the inputs.
 *
 * The selected cell is not separate state: the method row is `config.method`
 * and the GPU column is `selectedGpuId`, so the Method control and the matrix
 * are two views of one selection.
 */

// ---------------------------------------------------------------------------
// Catalog
// ---------------------------------------------------------------------------

export const FINE_TUNE_MODELS: FineTuneModelEntry[] = (
  catalogData as FineTuneModelData
).models;

export const CATALOG_LAST_UPDATED = (catalogData as FineTuneModelData).lastUpdated;

/** Catalog id of the user-entered model; its dimensions come from the store. */
export const CUSTOM_MODEL_ID = 'custom';

/** Dimensions a user may supply for the Custom model. Nulls fall back to the engine's heuristic. */
export type CustomModelFields = {
  totalParameters: number;
  activeParameters: number;
  layers: number | null;
  hidden: number | null;
  ffnDim: number | null;
  kvHeads: number | null;
  headDim: number | null;
};

/**
 * Map a catalog entry onto the engine's `ModelArchitecture`.
 *
 * `overrides` carries the Custom-model fields. A `null` override means
 * "unknown", which is exactly the signal the engine wants: it then estimates
 * the dimension from the parameter count and downgrades confidence rather than
 * computing adapter parameters off a plausible-looking wrong number.
 */
export function toModelArchitecture(
  entry: FineTuneModelEntry,
  overrides?: Partial<CustomModelFields>,
): ModelArchitecture {
  const totalParameters = overrides?.totalParameters ?? entry.totalParameters ?? 0;
  const activeParameters =
    overrides?.activeParameters ?? entry.activeParameters ?? totalParameters;

  return {
    id: entry.id,
    label: entry.label,
    totalParameters,
    activeParameters,
    layers: overrides?.layers ?? entry.layers,
    hidden: overrides?.hidden ?? entry.hidden,
    heads: entry.heads,
    kvHeads: overrides?.kvHeads ?? entry.kvHeads,
    headDim: overrides?.headDim ?? entry.headDim,
    ffnDim: overrides?.ffnDim ?? entry.ffnDim,
    vocab: entry.vocab,
    contextWindow: entry.contextWindow,
    isMoe: entry.isMoe,
    attentionType: entry.attentionType,
    dimsVerified: entry.dimsVerified,
  };
}

export function getModelEntry(id: string): FineTuneModelEntry {
  return (
    FINE_TUNE_MODELS.find((m) => m.id === id) ??
    FINE_TUNE_MODELS[0]
  );
}

// ---------------------------------------------------------------------------
// Labels and option lists
// ---------------------------------------------------------------------------

/** Matrix rows, in the order the decisions doc tiers them by confidence. */
export const FINE_TUNE_METHODS: FineTuneMethod[] = [
  'full',
  'lora',
  'qlora',
  'dora',
  'dpo',
  'orpo',
  'grpo',
  'ppo',
];

export const METHOD_LABELS: Record<FineTuneMethod, string> = {
  full: 'Full fine-tune',
  lora: 'LoRA',
  qlora: 'QLoRA',
  dora: 'DoRA',
  dpo: 'DPO',
  orpo: 'ORPO',
  grpo: 'GRPO',
  ppo: 'PPO',
};

export const OPTIMIZER_LABELS: Record<OptimizerKind, string> = {
  adamw_fp32: 'AdamW (fp32 master)',
  adamw_bf16: 'AdamW (bf16 master)',
  adam_8bit: '8-bit Adam',
  adafactor: 'Adafactor',
  sgd_momentum: 'SGD + momentum',
  lomo: 'LOMO (fused, no state)',
};

export const PRECISION_LABELS: Record<WeightPrecision, string> = {
  bf16: 'BF16 (2 B/param)',
  fp8: 'FP8 (1 B/param)',
  nf4: 'NF4 4-bit (0.5 B/param)',
};

export const TARGET_LABELS: Record<LoraTargets, string> = {
  all_linear: 'All linear (q,k,v,o,gate,up,down)',
  attn_plus_o: 'Attention + output (q,k,v,o)',
  attn_only: 'Attention only (q,v)',
};

export const SHARDING_LABELS: Record<ShardingStrategy, string> = {
  ddp: 'DDP (replicated — cannot reduce the memory floor)',
  zero2: 'ZeRO-2 (gradients + optimizer sharded)',
  fsdp_zero3: 'FSDP / ZeRO-3 (weights too)',
};

// ---------------------------------------------------------------------------
// Goal presets (Decision 6)
// ---------------------------------------------------------------------------

/**
 * Presets fill the raw workload fields and nothing else — the fields stay
 * visible and editable at all times, following the precedent in
 * docs/decisions/2026-06-24-inference-ux-presets.md.
 */
export type GoalPreset = {
  id: string;
  label: string;
  description: string;
  basis: string;
  examples: number;
  avgTokensPerExample: number;
  epochs: number;
  sequenceLength: number;
  method: FineTuneMethod;
  loraRank: number;
};

export const GOAL_PRESETS: GoalPreset[] = [
  {
    id: 'style_tone',
    label: 'Style / tone adoption',
    description: 'Teach a voice, format or persona',
    basis: 'LIMA: 1,000 curated pairs sufficed',
    examples: 1_000,
    avgTokensPerExample: 600,
    epochs: 3,
    sequenceLength: 1024,
    method: 'lora',
    loraRank: 16,
  },
  {
    id: 'narrow_task',
    label: 'Narrow task (classification)',
    description: 'One label, one output shape',
    basis: '100-500 examples is the published floor',
    examples: 500,
    avgTokensPerExample: 400,
    epochs: 3,
    sequenceLength: 512,
    method: 'qlora',
    loraRank: 16,
  },
  {
    id: 'instruction',
    label: 'Instruction following',
    description: 'Broad, open-ended instruction coverage',
    basis: 'Broad tasks need 5-20K examples',
    examples: 10_000,
    avgTokensPerExample: 800,
    epochs: 2,
    sequenceLength: 2048,
    method: 'lora',
    loraRank: 32,
  },
  {
    id: 'tool_use',
    label: 'Tool use / function calling',
    description: 'Schema-faithful calls and arguments',
    basis: '~2K traces at r=32',
    examples: 2_000,
    avgTokensPerExample: 1_200,
    epochs: 3,
    sequenceLength: 4096,
    method: 'lora',
    loraRank: 32,
  },
  {
    id: 'reasoning_distill',
    label: 'Reasoning distillation',
    description: 'Long chains of thought from a teacher',
    basis: 'Long outputs drive sequence length, not example count',
    examples: 20_000,
    avgTokensPerExample: 4_000,
    epochs: 2,
    sequenceLength: 8192,
    method: 'lora',
    loraRank: 64,
  },
  {
    id: 'domain_adaptation',
    label: 'Domain adaptation',
    description: 'Continued pretraining on in-domain text',
    basis: 'Continued-pretraining regime: 10^8-10^10 tokens, 1 epoch',
    examples: 200_000,
    avgTokensPerExample: 2_000,
    epochs: 1,
    sequenceLength: 4096,
    method: 'full',
    loraRank: 16,
  },
];

// ---------------------------------------------------------------------------
// Pricing
// ---------------------------------------------------------------------------

export type PricingTier = 'onDemand' | 'reserved' | 'spot';

export const PRICING_TIER_LABELS: Record<PricingTier, string> = {
  onDemand: 'On-demand',
  reserved: 'Reserved',
  spot: 'Spot',
};

/**
 * Which `cloud-gpu-pricing.json` rows describe which `gpu-data.ts` SKU.
 *
 * Deliberately sparse. Most SKUs in the table — B300, GB300, MI355X, the TPUs,
 * Trainium, Gaudi — have no published instance price in the pricing file, and
 * the pane renders their cost as an em dash rather than borrowing a number from
 * a different accelerator. The A100 rows have no SKU to attach to.
 */
const PRICING_GPU_TO_SKU_ID: Record<string, string> = {
  'H100 SXM': 'h100-sxm',
  'H100 80GB': 'h100-sxm',
};

export type GpuRate = {
  usdPerGpuHour: number;
  provider: string;
  instance: string;
};

/**
 * Per-GPU hourly rate by SKU id, derived as `instancePrice / gpuCount`.
 *
 * Where several providers list the same accelerator the cheapest is used — the
 * rate a buyer could actually obtain — and the provider is kept so the UI can
 * name its source instead of presenting an anonymous number.
 */
export function buildGpuHourlyRates(tier: PricingTier): Record<string, GpuRate> {
  const rates: Record<string, GpuRate> = {};

  for (const provider of (pricingData as CloudGpuPricingData).providers) {
    for (const instance of provider.instances) {
      const skuId = PRICING_GPU_TO_SKU_ID[instance.gpu];
      if (!skuId || instance.gpuCount <= 0) continue;

      const total =
        tier === 'onDemand'
          ? instance.onDemandPerHr
          : tier === 'reserved'
            ? instance.reservedPerHr
            : instance.spotPerHr;
      if (total == null || !Number.isFinite(total)) continue;

      const usdPerGpuHour = total / instance.gpuCount;
      const existing = rates[skuId];
      if (!existing || usdPerGpuHour < existing.usdPerGpuHour) {
        rates[skuId] = {
          usdPerGpuHour,
          provider: provider.provider,
          instance: instance.instance,
        };
      }
    }
  }

  return rates;
}

export const PRICING_LAST_UPDATED = (pricingData as CloudGpuPricingData).lastUpdated;

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export type MethodMatrix = ReturnType<typeof buildMethodMatrix>;

export type FineTuneStore = {
  modelId: string;
  custom: CustomModelFields;
  workload: FineTuneWorkload;
  config: FineTuneConfig;
  selectedGpuIds: string[];
  /** The matrix column whose cell is open in the detail panel. */
  selectedGpuId: string;
  availability: number;
  overheadFactor: number;
  pricingTier: PricingTier;
  /** Which goal preset last filled the workload fields, or null once edited. */
  presetId: string | null;

  matrix: MethodMatrix;
  rates: Record<string, GpuRate>;

  setModelId: (id: string) => void;
  setCustomField: <K extends keyof CustomModelFields>(
    key: K,
    value: CustomModelFields[K],
  ) => void;
  setWorkloadField: <K extends keyof FineTuneWorkload>(
    key: K,
    value: FineTuneWorkload[K],
  ) => void;
  setConfigField: <K extends keyof FineTuneConfig>(key: K, value: FineTuneConfig[K]) => void;
  applyPreset: (presetId: string) => void;
  toggleGpu: (id: string) => void;
  setSelectedGpuId: (id: string) => void;
  selectCell: (method: FineTuneMethod, gpuId: string) => void;
  setAvailability: (value: number) => void;
  setOverheadFactor: (value: number) => void;
  setPricingTier: (tier: PricingTier) => void;
  reset: () => void;
};

const DEFAULT_WORKLOAD: FineTuneWorkload = {
  examples: 2_000,
  avgTokensPerExample: 800,
  epochs: 3,
  sequenceLength: 2048,
  packing: true,
  microBatchSize: 1,
};

const DEFAULT_CONFIG: FineTuneConfig = {
  method: 'lora',
  optimizer: 'adamw_fp32',
  weightPrecision: 'bf16',
  // Standard fine-tuning configuration; costs +2N FLOPs/token (Decision 4).
  gradientCheckpointing: true,
  loraRank: 16,
  loraTargets: 'all_linear',
  sharding: 'fsdp_zero3',
  precomputeReferenceLogprobs: true,
};

const DEFAULT_CUSTOM: CustomModelFields = {
  totalParameters: 8e9,
  activeParameters: 8e9,
  layers: null,
  hidden: null,
  ffnDim: null,
  kvHeads: null,
  headDim: null,
};

/** Three memory tiers — 80, 141 and 180 GB — so the fit column actually varies. */
const DEFAULT_GPU_IDS = ['h100-sxm', 'h200-sxm', 'b200-sxm'];

type Inputs = Pick<
  FineTuneStore,
  | 'modelId'
  | 'custom'
  | 'workload'
  | 'config'
  | 'selectedGpuIds'
  | 'availability'
  | 'overheadFactor'
  | 'pricingTier'
>;

export function buildScenario(inputs: Inputs): FineTuneScenario {
  const entry = getModelEntry(inputs.modelId);
  const model = toModelArchitecture(
    entry,
    entry.id === CUSTOM_MODEL_ID ? inputs.custom : undefined,
  );
  return {
    model,
    workload: inputs.workload,
    config: inputs.config,
    availability: inputs.availability,
    overheadFactor: inputs.overheadFactor,
  };
}

export function selectedGpus(selectedGpuIds: string[]): GpuSku[] {
  return selectedGpuIds
    .map((id) => getGpuById(id))
    .filter((gpu): gpu is GpuSku => gpu != null);
}

function derive(inputs: Inputs): { matrix: MethodMatrix; rates: Record<string, GpuRate> } {
  const rates = buildGpuHourlyRates(inputs.pricingTier);
  const hourly = Object.fromEntries(
    Object.entries(rates).map(([id, rate]) => [id, rate.usdPerGpuHour]),
  );
  const matrix = buildMethodMatrix(
    buildScenario(inputs),
    selectedGpus(inputs.selectedGpuIds),
    FINE_TUNE_METHODS,
    hourly,
  );
  return { matrix, rates };
}

/** The cell the detail panel is showing: method row x GPU column. */
export function selectCellResult(state: FineTuneStore) {
  return state.matrix.cells[state.config.method]?.[state.selectedGpuId] ?? null;
}

function initialInputs(): Inputs {
  return {
    modelId: 'llama-3.1-8b',
    custom: { ...DEFAULT_CUSTOM },
    workload: { ...DEFAULT_WORKLOAD },
    config: { ...DEFAULT_CONFIG },
    selectedGpuIds: [...DEFAULT_GPU_IDS],
    availability: 0.95,
    overheadFactor: 1.1,
    pricingTier: 'onDemand',
  };
}

function initialState() {
  const inputs = initialInputs();
  return {
    ...inputs,
    selectedGpuId: inputs.selectedGpuIds[0],
    presetId: null,
    ...derive(inputs),
  };
}

/** Recompute the matrix from current state plus whatever this setter changed. */
function recompute(state: FineTuneStore, overrides: Partial<Inputs> = {}) {
  return derive({
    modelId: overrides.modelId ?? state.modelId,
    custom: overrides.custom ?? state.custom,
    workload: overrides.workload ?? state.workload,
    config: overrides.config ?? state.config,
    selectedGpuIds: overrides.selectedGpuIds ?? state.selectedGpuIds,
    availability: overrides.availability ?? state.availability,
    overheadFactor: overrides.overheadFactor ?? state.overheadFactor,
    pricingTier: overrides.pricingTier ?? state.pricingTier,
  });
}

export const useFineTuneStore = create<FineTuneStore>((set) => ({
  ...initialState(),

  setModelId: (modelId) =>
    set((state) => ({ modelId, ...recompute(state, { modelId }) })),

  setCustomField: (key, value) =>
    set((state) => {
      const custom = { ...state.custom, [key]: value };
      return { custom, ...recompute(state, { custom }) };
    }),

  setWorkloadField: (key, value) =>
    set((state) => {
      const workload = { ...state.workload, [key]: value };
      // Editing a raw field means the preset no longer describes the workload.
      return { workload, presetId: null, ...recompute(state, { workload }) };
    }),

  setConfigField: (key, value) =>
    set((state) => {
      const config = { ...state.config, [key]: value };
      return { config, ...recompute(state, { config }) };
    }),

  applyPreset: (presetId) =>
    set((state) => {
      const preset = GOAL_PRESETS.find((p) => p.id === presetId);
      if (!preset) return {};
      const workload: FineTuneWorkload = {
        ...state.workload,
        examples: preset.examples,
        avgTokensPerExample: preset.avgTokensPerExample,
        epochs: preset.epochs,
        sequenceLength: preset.sequenceLength,
      };
      const config: FineTuneConfig = {
        ...state.config,
        method: preset.method,
        loraRank: preset.loraRank,
      };
      return { workload, config, presetId, ...recompute(state, { workload, config }) };
    }),

  toggleGpu: (id) =>
    set((state) => {
      const isSelected = state.selectedGpuIds.includes(id);
      if (isSelected && state.selectedGpuIds.length === 1) return {};
      const selectedGpuIds = isSelected
        ? state.selectedGpuIds.filter((g) => g !== id)
        : [...state.selectedGpuIds, id];
      const selectedGpuId = selectedGpuIds.includes(state.selectedGpuId)
        ? state.selectedGpuId
        : selectedGpuIds[0];
      return { selectedGpuIds, selectedGpuId, ...recompute(state, { selectedGpuIds }) };
    }),

  setSelectedGpuId: (selectedGpuId) => set({ selectedGpuId }),

  selectCell: (method, gpuId) =>
    set((state) => {
      const config = { ...state.config, method };
      return { config, selectedGpuId: gpuId, ...recompute(state, { config }) };
    }),

  setAvailability: (availability) =>
    set((state) => ({ availability, ...recompute(state, { availability }) })),

  setOverheadFactor: (overheadFactor) =>
    set((state) => ({ overheadFactor, ...recompute(state, { overheadFactor }) })),

  setPricingTier: (pricingTier) =>
    set((state) => ({ pricingTier, ...recompute(state, { pricingTier }) })),

  reset: () => set(initialState()),
}));

/** Every selectable GPU column. Custom SKUs belong to the training pane. */
export const FINE_TUNE_GPUS = GPU_SKUS;
