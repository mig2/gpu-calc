import { describe, it, expect } from 'vitest'
import {
  estimateFineTune,
  buildMethodMatrix,
  loraAdapterParams,
  resolveDims,
  estimateArchitecture,
  flopsPerToken,
  tokensProcessed,
  WEIGHT_BYTES,
  GRADIENT_BYTES,
  OPTIMIZER_MASTER_BYTES,
  OPTIMIZER_MOMENT_BYTES,
  SCALE_MFU,
} from './finetune-calculator'
import type {
  FineTuneScenario,
  ModelArchitecture,
  FineTuneMethod,
  WeightPrecision,
} from './finetune-types'
import type { GpuSku } from './types'

const H100: GpuSku = {
  id: 'h100-sxm',
  label: 'NVIDIA H100 SXM',
  bf16DenseFlops: 0.9895e15,
  fp8DenseFlops: 1.979e15,
  memoryGb: 80,
  defaultMfu: 0.40,
}

/** Dimensions transcribed from each model's config.json. */
const MODELS: Record<string, ModelArchitecture> = {
  '7B': { id: '7b', label: '7B dense', totalParameters: 7e9, activeParameters: 7e9, layers: 32, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128, ffnDim: 14336, vocab: 32000, isMoe: false, dimsVerified: true },
  '8B': { id: '8b', label: 'Llama-3-8B', totalParameters: 8.03e9, activeParameters: 8.03e9, layers: 32, hidden: 4096, heads: 32, kvHeads: 8, headDim: 128, ffnDim: 14336, vocab: 128256, isMoe: false, dimsVerified: true },
  '14B': { id: '14b', label: '14B dense', totalParameters: 14e9, activeParameters: 14e9, layers: 48, hidden: 5120, heads: 40, kvHeads: 8, headDim: 128, ffnDim: 17920, vocab: 100352, isMoe: false, dimsVerified: true },
  '32B': { id: '32b', label: '32B dense', totalParameters: 32e9, activeParameters: 32e9, layers: 64, hidden: 5120, heads: 40, kvHeads: 8, headDim: 128, ffnDim: 27648, vocab: 152064, isMoe: false, dimsVerified: true },
  '70B': { id: '70b', label: '70B dense', totalParameters: 70e9, activeParameters: 70e9, layers: 80, hidden: 8192, heads: 64, kvHeads: 8, headDim: 128, ffnDim: 28672, vocab: 128256, isMoe: false, dimsVerified: true },
}

const MOE: ModelArchitecture = {
  id: 'moe-30b-a3b', label: 'MoE 30B-A3B',
  totalParameters: 30e9, activeParameters: 3e9,
  layers: 48, hidden: 2048, heads: 32, kvHeads: 4, headDim: 128, ffnDim: 768,
  vocab: 151936, isMoe: true, dimsVerified: true,
}

function scenario(
  model: ModelArchitecture,
  method: FineTuneMethod,
  overrides: {
    rank?: number
    precision?: WeightPrecision
    seqLen?: number
    microBatch?: number
  } = {},
): FineTuneScenario {
  return {
    model,
    workload: {
      examples: 2000,
      avgTokensPerExample: 800,
      epochs: 3,
      sequenceLength: overrides.seqLen ?? 512,
      packing: true,
      microBatchSize: overrides.microBatch ?? 1,
    },
    config: {
      method,
      optimizer: 'adamw_fp32',
      weightPrecision: overrides.precision ?? (method === 'qlora' ? 'nf4' : 'bf16'),
      gradientCheckpointing: true,
      loraRank: overrides.rank ?? 64,
      loraTargets: 'all_linear',
      sharding: 'fsdp_zero3',
    },
    availability: 0.95,
    overheadFactor: 1.10,
  }
}

const GB = 1e9

describe('LoRA adapter parameter count', () => {
  it('reproduces PEFT ground truth for Llama-3-8B at r=16, all linear targets', () => {
    // PEFT reports exactly 41,943,040 trainable params for this configuration.
    const dims = resolveDims(MODELS['8B'])
    expect(loraAdapterParams(dims, 16, 'all_linear')).toBe(41_943_040)
  })

  it('scales linearly with rank', () => {
    const dims = resolveDims(MODELS['8B'])
    const r16 = loraAdapterParams(dims, 16, 'all_linear')
    expect(loraAdapterParams(dims, 64, 'all_linear')).toBe(r16 * 4)
    expect(loraAdapterParams(dims, 8, 'all_linear')).toBe(r16 / 2)
  })

  it('shrinks as target breadth narrows', () => {
    const dims = resolveDims(MODELS['8B'])
    const all = loraAdapterParams(dims, 16, 'all_linear')
    const attnO = loraAdapterParams(dims, 16, 'attn_plus_o')
    const attn = loraAdapterParams(dims, 16, 'attn_only')
    expect(attnO).toBeLessThan(all)
    expect(attn).toBeLessThan(attnO)
    // FFN projections dominate: attention-only is a small fraction of all-linear.
    expect(attn / all).toBeLessThan(0.2)
  })
})

describe('architecture fallback', () => {
  // d ~= 128L from the Llama 3 family; N = 196608 L^3.
  it('estimates Llama-3-family dims within 10% on layers', () => {
    for (const key of ['7B', '8B'] as const) {
      const m = MODELS[key]
      const est = estimateArchitecture(m.activeParameters)
      expect(Math.abs(est.layers - (m.layers as number)) / (m.layers as number)).toBeLessThan(0.10)
    }
  })

  it('degrades gracefully at larger scale but stays within 40% on hidden size', () => {
    // 32B is the worst case: estimated d=7040 vs actual 5120. This is why a
    // model without catalog dims is reported at `medium` confidence.
    for (const key of ['14B', '32B', '70B'] as const) {
      const m = MODELS[key]
      const est = estimateArchitecture(m.activeParameters)
      expect(Math.abs(est.hidden - (m.hidden as number)) / (m.hidden as number)).toBeLessThan(0.40)
    }
  })

  it('caps confidence at medium when catalog dims are unverified', () => {
    // The catalog carries plausible dims for some models with dimsVerified:false
    // (config.json unpublished). Those must not earn `high`.
    const unverified: ModelArchitecture = { ...MODELS['8B'], id: 'u', dimsVerified: false }
    const dims = resolveDims(unverified)
    expect(dims.exact).toBe(true)
    expect(dims.verified).toBe(false)
    const est = estimateFineTune(scenario(unverified, 'lora'), H100)
    expect(est.confidence).toBe('medium')
    expect(est.warnings.some((w) => w.includes('not confirmed against a published config'))).toBe(true)
  })

  it('still uses unverified catalog dims rather than the cruder fallback', () => {
    const unverified: ModelArchitecture = { ...MODELS['32B'], id: 'u32', dimsVerified: false }
    // 32B is where the fallback is worst (est d=7040 vs actual 5120), so the
    // catalog value must win even when unverified.
    expect(resolveDims(unverified).hidden).toBe(MODELS['32B'].hidden)
  })

  it('marks fallback dims as inexact and downgrades confidence', () => {
    const unknown: ModelArchitecture = {
      id: 'x', label: 'Unknown 13B', totalParameters: 13e9, activeParameters: 13e9, isMoe: false,
    }
    expect(resolveDims(unknown).exact).toBe(false)
    const est = estimateFineTune(scenario(unknown, 'lora'), H100)
    expect(est.confidence).toBe('medium')
    expect(est.warnings.some((w) => w.includes('dimensions unknown'))).toBe(true)
  })
})

describe('memory model vs published anchors', () => {
  // Anchors: docs/research/2026-10-14-finetuning-calculator-research.md sect. 1.3
  // LoRA r=64 and QLoRA columns, in GB.
  const anchors: Record<string, { lora: number; qlora: number }> = {
    '7B': { lora: 20, qlora: 8 },
    '14B': { lora: 35, qlora: 14 },
    '32B': { lora: 76, qlora: 28 },
    '70B': { lora: 159, qlora: 52 },
  }

  for (const [key, a] of Object.entries(anchors)) {
    it(`reproduces the LoRA r=64 anchor for ${key} within 10%`, () => {
      const got = estimateFineTune(scenario(MODELS[key], 'lora'), H100).memory.totalBytes / GB
      expect(Math.abs(got - a.lora) / a.lora).toBeLessThan(0.10)
    })

    it(`reproduces the QLoRA anchor for ${key} within 10%`, () => {
      const got = estimateFineTune(scenario(MODELS[key], 'qlora'), H100).memory.totalBytes / GB
      expect(Math.abs(got - a.qlora) / a.qlora).toBeLessThan(0.10)
    })
  }

  /**
   * The research's full-FT column (7B 88 GB, 70B 860 GB) uses a 12 bytes/param
   * convention that omits the FP32 master weight copy. We deliberately use the
   * standard mixed-precision AdamW accounting of 16 bytes/param:
   *   2 (BF16 weights) + 2 (BF16 grads) + 4 (FP32 master) + 8 (two FP32 moments)
   * This agrees with the research's second, independent anchor: 70B full FT is
   * "~1.12 TB of model state in theory", and 70e9 x 16 = 1.12e12 exactly.
   */
  it('uses 16 bytes/param for full fine-tuning with AdamW', () => {
    const perParam =
      WEIGHT_BYTES.bf16 +
      GRADIENT_BYTES +
      OPTIMIZER_MASTER_BYTES.adamw_fp32 +
      OPTIMIZER_MOMENT_BYTES.adamw_fp32
    expect(perParam).toBe(16)
  })

  it('reproduces the 1.12 TB model-state anchor for 70B full FT', () => {
    const est = estimateFineTune(scenario(MODELS['70B'], 'full'), H100)
    const modelState =
      est.memory.weightsBytes + est.memory.gradientsBytes + est.memory.optimizerBytes
    expect(Math.abs(modelState - 1.12e12) / 1.12e12).toBeLessThan(0.01)
  })

  it('shows 8-bit Adam cutting the 70B full-FT optimizer state by 3x', () => {
    const fp32 = estimateFineTune(scenario(MODELS['70B'], 'full'), H100)
    const s8 = scenario(MODELS['70B'], 'full')
    s8.config.optimizer = 'adam_8bit'
    const bit8 = estimateFineTune(s8, H100)
    const ratio = fp32.memory.optimizerBytes / bit8.memory.optimizerBytes
    expect(ratio).toBeCloseTo(3, 1)
    expect(bit8.minGpusMemory).toBeLessThan(fp32.minGpusMemory)
  })
})

describe('compute model', () => {
  it('charges 6N per token for full fine-tuning and 4N for LoRA, before recompute', () => {
    expect(flopsPerToken(7e9, 'full', false)).toBe(6 * 7e9)
    expect(flopsPerToken(7e9, 'lora', false)).toBe(4 * 7e9)
    expect(flopsPerToken(7e9, 'qlora', false)).toBe(4 * 7e9)
  })

  it('adds one forward pass for gradient checkpointing', () => {
    expect(flopsPerToken(7e9, 'full', true)).toBe(8 * 7e9)
    expect(flopsPerToken(7e9, 'lora', true)).toBe(6 * 7e9)
  })

  /**
   * Calibration against NVIDIA NeMo's published LoRA throughputs
   * (DGX-H100, BF16, seq 4096): Llama-3 8B on 1 GPU = 12,472.87 tok/s/GPU;
   * Llama-3 70B on 8 GPUs = 1,091.85 tok/s/GPU.
   * Under the 4N model with our scale-keyed MFU defaults, the implied
   * throughput should land within 5% of those figures.
   */
  it('implies NeMo-published throughput within 5% at 8B on one GPU', () => {
    const implied = (H100.bf16DenseFlops * SCALE_MFU.singleGpu) / (4 * MODELS['8B'].activeParameters)
    expect(Math.abs(implied - 12_472.87) / 12_472.87).toBeLessThan(0.05)
  })

  it('implies NeMo-published throughput within 5% at 70B on eight GPUs', () => {
    const implied = (H100.bf16DenseFlops * SCALE_MFU.singleNode) / (4 * MODELS['70B'].activeParameters)
    expect(Math.abs(implied - 1_091.85) / 1_091.85).toBeLessThan(0.05)
  })

  it('derives tokens processed from examples, epochs and packing', () => {
    const s = scenario(MODELS['8B'], 'lora')
    // 2000 x 800 x 3 epochs / 0.95 packing
    expect(tokensProcessed(s)).toBeCloseTo((2000 * 800 * 3) / 0.95, 0)
  })

  it('penalises disabled packing', () => {
    const packed = tokensProcessed(scenario(MODELS['8B'], 'lora'))
    const s = scenario(MODELS['8B'], 'lora')
    s.workload.packing = false
    expect(tokensProcessed(s)).toBeGreaterThan(packed)
  })
})

describe('MoE handling', () => {
  it('bills memory on total params and compute on active params', () => {
    const est = estimateFineTune(scenario(MOE, 'lora'), H100)
    // Weights follow the 30B total, not the 3B active.
    expect(est.memory.weightsBytes).toBe(30e9 * 2)
    // LoRA 4N + 2N recompute (checkpointing on), billed on ACTIVE params.
    expect(est.flopsPerToken).toBe(6 * 3e9)
    // Conflating total with active would overstate FLOPs 10x on this model.
    expect(est.flopsPerToken).toBeLessThan(6 * 30e9)
  })

  it('warns that experts stay resident', () => {
    const est = estimateFineTune(scenario(MOE, 'lora'), H100)
    expect(est.warnings.some((w) => w.includes('memory follows') && w.includes('compute follows'))).toBe(true)
  })

  it('warns against 4-bit QLoRA on MoE models', () => {
    const est = estimateFineTune(scenario(MOE, 'qlora'), H100)
    expect(est.warnings.some((w) => w.includes('bitsandbytes lacks native MoE support'))).toBe(true)
  })
})

describe('activation memory', () => {
  it('scales linearly with sequence length', () => {
    const short = estimateFineTune(scenario(MODELS['8B'], 'lora', { seqLen: 2048 }), H100)
    const long = estimateFineTune(scenario(MODELS['8B'], 'lora', { seqLen: 8192 }), H100)
    expect(long.memory.activationsBytes / short.memory.activationsBytes).toBeCloseTo(4, 1)
  })

  it('scales linearly with micro-batch size', () => {
    const b1 = estimateFineTune(scenario(MODELS['8B'], 'lora', { microBatch: 1 }), H100)
    const b8 = estimateFineTune(scenario(MODELS['8B'], 'lora', { microBatch: 8 }), H100)
    expect(b8.memory.activationsBytes / b1.memory.activationsBytes).toBeCloseTo(8, 1)
  })

  it('gradient checkpointing substantially reduces activation memory', () => {
    const on = scenario(MODELS['8B'], 'lora', { seqLen: 8192, microBatch: 4 })
    const off = scenario(MODELS['8B'], 'lora', { seqLen: 8192, microBatch: 4 })
    off.config.gradientCheckpointing = false
    const withCkpt = estimateFineTune(on, H100).memory.activationsBytes
    const without = estimateFineTune(off, H100).memory.activationsBytes
    expect(withCkpt).toBeLessThan(without * 0.3)
  })

  it('warns when long context makes activations dominant', () => {
    const est = estimateFineTune(scenario(MODELS['8B'], 'lora', { seqLen: 32_768 }), H100)
    expect(est.warnings.some((w) => w.includes('activations'))).toBe(true)
  })
})

describe('GPU count and fit state', () => {
  it('fits 7B QLoRA on a single H100', () => {
    const est = estimateFineTune(scenario(MODELS['7B'], 'qlora'), H100)
    expect(est.minGpusMemory).toBe(1)
    expect(est.fitState).toBe('fits')
    expect(est.fitsOnSingleGpu).toBe(true)
  })

  it('requires sharding for 70B LoRA', () => {
    // Research: the non-activation floor of ~149 GB already exceeds one H200.
    const est = estimateFineTune(scenario(MODELS['70B'], 'lora'), H100)
    expect(est.minGpusMemory).toBeGreaterThan(1)
    expect(est.fitState).toBe('sharded')
  })

  it('reports OOM when DDP cannot shard oversized state', () => {
    const s = scenario(MODELS['70B'], 'full')
    s.config.sharding = 'ddp'
    const est = estimateFineTune(s, H100)
    expect(est.fitState).toBe('oom')
    expect(est.warnings.some((w) => w.includes('DDP cannot shard'))).toBe(true)
  })

  it('rounds multi-node counts up to whole nodes of eight', () => {
    const est = estimateFineTune(scenario(MODELS['70B'], 'full'), H100)
    expect(est.minGpusMemory % 8).toBe(0)
  })
})

describe('preference-optimization objectives', () => {
  /**
   * Defaults follow current practice (RL research doc sect. 8.2): TRL's
   * `precompute_ref_log_probs` removes the reference model from memory
   * entirely (-31% peak), so it is ON by default and DPO carries no
   * auxiliary copy unless explicitly disabled.
   */
  it('holds no reference copy by default, since logprobs are precomputed', () => {
    const dpo = estimateFineTune(scenario(MODELS['8B'], 'dpo'), H100)
    expect(dpo.memory.auxiliaryBytes).toBe(0)
  })

  it('holds a frozen reference copy when precomputation is disabled', () => {
    const s = scenario(MODELS['8B'], 'dpo')
    s.config.precomputeReferenceLogprobs = false
    const dpo = estimateFineTune(s, H100)
    expect(dpo.memory.auxiliaryBytes).toBeCloseTo(MODELS['8B'].totalParameters * 2, -6)
  })

  /**
   * The fine-tuning doc's "DPO ~= 2x SFT memory" counted activations, not
   * model state. With a resident reference the true model-state multiplier is
   * 1.125x (16 -> 18 B/param), per RL research doc sect. 6.2.
   */
  it('costs 1.125x SFT model state with a resident reference, not 2x', () => {
    const sft = estimateFineTune(scenario(MODELS['8B'], 'full'), H100)
    const s = scenario(MODELS['8B'], 'dpo')
    s.config.precomputeReferenceLogprobs = false
    const dpo = estimateFineTune(s, H100)
    const state = (e: typeof sft) =>
      e.memory.weightsBytes + e.memory.gradientsBytes + e.memory.optimizerBytes + e.memory.auxiliaryBytes
    expect(state(dpo) / state(sft)).toBeCloseTo(1.125, 2)
  })

  it('ORPO is reference-free and so carries no auxiliary copy', () => {
    const orpo = estimateFineTune(scenario(MODELS['8B'], 'orpo'), H100)
    expect(orpo.memory.auxiliaryBytes).toBe(0)
  })

  it('charges 2.67x SFT for a resident reference and 2.0x when precomputed', () => {
    const precomputed = estimateFineTune(scenario(MODELS['8B'], 'dpo'), H100)
    const s = scenario(MODELS['8B'], 'dpo')
    s.config.precomputeReferenceLogprobs = false
    const resident = estimateFineTune(s, H100)
    expect(resident.totalFlops).toBeGreaterThan(precomputed.totalFlops)
    // 8N/6N over two sequences vs 6N/6N over two sequences.
    expect(resident.totalFlops / precomputed.totalFlops).toBeCloseTo(2.67 / 2.0, 2)
  })

  /**
   * PPO holds policy + critic + reward + reference = 36 B/param against SFT's
   * 16, i.e. 2.25x -- not the "~4x" in the fine-tuning doc, because only the
   * policy and critic carry optimizer state (RL research doc sect. 6.2).
   */
  it('costs 2.25x SFT model state, not 4x', () => {
    const sft = estimateFineTune(scenario(MODELS['8B'], 'full'), H100)
    const ppo = estimateFineTune(scenario(MODELS['8B'], 'ppo'), H100)
    const state = (e: typeof sft) =>
      e.memory.weightsBytes + e.memory.gradientsBytes + e.memory.optimizerBytes + e.memory.auxiliaryBytes
    expect(state(ppo) / state(sft)).toBeCloseTo(2.25, 2)
  })

  /**
   * GRPO's reference model is absent by default: TRL's GRPOConfig defaults
   * beta=0.0 and loads no reference policy; DAPO removes the KL term outright.
   * The old "GRPO = 2-3x SFT memory" heuristic overstates the modern default.
   */
  it('loads no reference model for GRPO by default', () => {
    const grpo = estimateFineTune(scenario(MODELS['8B'], 'grpo'), H100)
    expect(grpo.memory.auxiliaryBytes).toBe(0)
  })

  it('loads a reference model for GRPO when the KL term is active', () => {
    const s = scenario(MODELS['8B'], 'grpo')
    s.config.referenceModelResident = true
    expect(estimateFineTune(s, H100).memory.auxiliaryBytes).toBeGreaterThan(0)
  })

  it('reports RL methods at low confidence with a rollout caveat', () => {
    for (const method of ['grpo', 'ppo'] as const) {
      const est = estimateFineTune(scenario(MODELS['8B'], method), H100)
      expect(est.confidence).toBe('low')
      expect(est.warnings.some((w) => w.includes('rollout generation'))).toBe(true)
    }
  })
})

describe('workload guardrails', () => {
  it('warns above five epochs', () => {
    const s = scenario(MODELS['8B'], 'lora')
    s.workload.epochs = 10
    expect(estimateFineTune(s, H100).warnings.some((w) => w.includes('memorization'))).toBe(true)
  })

  it('warns below one hundred examples', () => {
    const s = scenario(MODELS['8B'], 'lora')
    s.workload.examples = 40
    expect(estimateFineTune(s, H100).warnings.some((w) => w.includes('useful floor'))).toBe(true)
  })
})

describe('confidence tiering', () => {
  it('rates exact-dim LoRA high and QLoRA medium-low', () => {
    expect(estimateFineTune(scenario(MODELS['8B'], 'lora'), H100).confidence).toBe('high')
    expect(estimateFineTune(scenario(MODELS['8B'], 'qlora'), H100).confidence).toBe('medium-low')
  })
})

describe('method comparison matrix', () => {
  const methods: FineTuneMethod[] = ['full', 'lora', 'qlora']

  it('produces a cell per method per GPU', () => {
    const m = buildMethodMatrix(scenario(MODELS['70B'], 'lora'), [H100], methods, { 'h100-sxm': 3.0 })
    for (const method of methods) {
      expect(m.cells[method]['h100-sxm'].estimate).toBeTruthy()
    }
  })

  it('forces a 4-bit base for the QLoRA row regardless of the selected precision', () => {
    const m = buildMethodMatrix(scenario(MODELS['70B'], 'full'), [H100], methods, { 'h100-sxm': 3.0 })
    const qlora = m.cells.qlora['h100-sxm'].estimate
    const lora = m.cells.lora['h100-sxm'].estimate
    expect(qlora.memory.weightsBytes).toBe(70e9 * 0.5)
    expect(lora.memory.weightsBytes).toBe(70e9 * 2)
  })

  /**
   * What the matrix actually reveals, on one fixed workload.
   *
   * Cost is close to method-invariant when billing by the GPU-hour, because
   * cost ~= totalFLOPs / (MFU x availability) and GPU count cancels against
   * wall-clock. Method moves FLOPs/token only 8N vs 6N. The research's
   * "$2.76 vs $1,785" headline compares a 7B QLoRA run on an RTX 5090 against
   * a 70B full-FT run on 11x H100 -- different models, hardware and datasets --
   * so it is NOT a like-for-like method spread and must not be asserted as one.
   *
   * What does vary by more than an order of magnitude is FEASIBILITY: memory
   * footprint and the minimum deployable GPU count. That is the pane's subject.
   */
  it('reveals an order-of-magnitude spread in memory and minimum GPU count', () => {
    const m = buildMethodMatrix(scenario(MODELS['70B'], 'lora'), [H100], methods, { 'h100-sxm': 3.0 })
    const full = m.cells.full['h100-sxm'].estimate
    const qlora = m.cells.qlora['h100-sxm'].estimate

    expect(full.memory.totalBytes / qlora.memory.totalBytes).toBeGreaterThan(10)
    expect(full.minGpusMemory / qlora.minGpusMemory).toBeGreaterThan(10)
    expect(qlora.fitState).toBe('fits')
    expect(full.fitState).toBe('sharded')
  })

  it('keeps total cost within a small factor across methods on one workload', () => {
    const m = buildMethodMatrix(scenario(MODELS['70B'], 'lora'), [H100], methods, { 'h100-sxm': 3.0 })
    const costs = methods.map((k) => m.cells[k]['h100-sxm'].costUsd as number)
    for (const c of costs) expect(c).toBeGreaterThan(0)
    const spread = Math.max(...costs) / Math.min(...costs)
    // Driven only by the 8N/6N FLOP ratio and the MFU differences.
    expect(spread).toBeGreaterThan(1.1)
    expect(spread).toBeLessThan(3)
  })

  it('trades GPU count against wall-clock at roughly constant GPU-hours', () => {
    const m = buildMethodMatrix(scenario(MODELS['70B'], 'lora'), [H100], methods, { 'h100-sxm': 3.0 })
    const full = m.cells.full['h100-sxm'].estimate
    const qlora = m.cells.qlora['h100-sxm'].estimate
    // Full FT finishes far sooner -- because it is spread over many more GPUs.
    expect(full.wallClockHours).toBeLessThan(qlora.wallClockHours)
    expect(full.minGpusMemory).toBeGreaterThan(qlora.minGpusMemory)
  })

  it('costs scale with GPU count and wall-clock', () => {
    const m = buildMethodMatrix(scenario(MODELS['7B'], 'qlora'), [H100], ['qlora'], { 'h100-sxm': 3.0 })
    const cell = m.cells.qlora['h100-sxm']
    expect(cell.costUsd).toBeCloseTo(3.0 * cell.estimate.minGpusMemory * cell.estimate.wallClockHours, 4)
  })
})
