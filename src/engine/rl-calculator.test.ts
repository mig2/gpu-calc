import { describe, it, expect } from 'vitest'
import {
  estimateGeneration,
  estimateRlStep,
  kvBytesPerToken,
  expectedRatioBand,
  WEIGHT_SYNC_BYTES_PER_SEC,
  ETA_ROLLOUT,
} from './rl-calculator'
import { WEIGHT_BYTES } from './finetune-calculator'
import type { ModelArchitecture, RlConfig } from './finetune-types'
import type { GpuSku } from './types'

const H100: GpuSku = {
  id: 'h100-sxm', label: 'NVIDIA H100 SXM',
  bf16DenseFlops: 0.9895e15, fp8DenseFlops: 1.979e15,
  memoryGb: 80, bandwidthTbps: 3.35, defaultMfu: 0.40,
}

const QWEN_7B: ModelArchitecture = {
  id: 'qwen2.5-7b', label: 'Qwen2.5-7B', totalParameters: 7.6e9, activeParameters: 7.6e9,
  layers: 28, hidden: 3584, heads: 28, kvHeads: 4, headDim: 128, ffnDim: 18944,
  vocab: 152064, isMoe: false, attentionType: 'gqa', dimsVerified: true,
}
const QWEN_32B: ModelArchitecture = {
  id: 'qwen2.5-32b', label: 'Qwen2.5-32B', totalParameters: 32.8e9, activeParameters: 32.8e9,
  layers: 64, hidden: 5120, heads: 40, kvHeads: 8, headDim: 128, ffnDim: 27648,
  vocab: 152064, isMoe: false, attentionType: 'gqa', dimsVerified: true,
}
const LLAMA_70B: ModelArchitecture = {
  id: 'llama-70b', label: 'Llama-3.1-70B', totalParameters: 70.6e9, activeParameters: 70.6e9,
  layers: 80, hidden: 8192, heads: 64, kvHeads: 8, headDim: 128, ffnDim: 28672,
  vocab: 128256, isMoe: false, attentionType: 'gqa', dimsVerified: true,
}
const DEEPSEEK: ModelArchitecture = {
  id: 'deepseek-v3', label: 'DeepSeek-V3', totalParameters: 671e9, activeParameters: 37e9,
  layers: 61, hidden: 7168, heads: 128, kvHeads: 128, headDim: 128, ffnDim: 2048,
  vocab: 129280, isMoe: true, attentionType: 'mla', dimsVerified: false,
}

const KIB = 1024

describe('KV cache sizing', () => {
  // Research sect. 2.4 anchors: 7B (28L, 4 kv, 128) = 56 KiB/token;
  //                             70B (80L, 8 kv, 128) = 320 KiB/token.
  it('reproduces the published per-token KV figures exactly', () => {
    expect(kvBytesPerToken(QWEN_7B).bytesPerToken / KIB).toBe(56)
    expect(kvBytesPerToken(LLAMA_70B).bytesPerToken / KIB).toBe(320)
  })

  it('halves under FP8 KV quantization', () => {
    expect(kvBytesPerToken(LLAMA_70B, 1).bytesPerToken / KIB).toBe(160)
  })

  it('reproduces the published per-sequence figures at L_out=8192', () => {
    // 7B: 0.465 GiB/seq; 70B: 2.656 GiB/seq (at 8192 tokens).
    const giB = 1024 ** 3
    expect((kvBytesPerToken(QWEN_7B).bytesPerToken * 8192) / giB).toBeCloseTo(0.4375, 2)
    expect((kvBytesPerToken(LLAMA_70B).bytesPerToken * 8192) / giB).toBeCloseTo(2.5, 1)
  })

  it('treats MLA as radically smaller and flags the approximation', () => {
    const gqa = kvBytesPerToken({ ...DEEPSEEK, attentionType: 'gqa' })
    const mla = kvBytesPerToken(DEEPSEEK)
    expect(mla.bytesPerToken).toBeLessThan(gqa.bytesPerToken / 4)
    expect(mla.approximated).toBe(true)
  })

  it('surfaces the KV capacity wall: 70B at G=8, L_out=8192, P=64', () => {
    // Research: the KV pool alone wants ~1.36 TB -- 20 H100s of usable VRAM,
    // before a single byte of weights or optimizer state.
    const perSeq = kvBytesPerToken(LLAMA_70B).bytesPerToken * (512 + 8192)
    const pool = perSeq * 64 * 8
    expect(pool / 1e12).toBeGreaterThan(1.2)
    expect(pool / 1e12).toBeLessThan(1.6)
  })
})

/**
 * The research's stated regression suite: vLLM on 1x H100, 512 rollouts
 * (HF async-RL survey). These are throughput benchmarks under continuous
 * batching, so ETA_ROLLOUT -- which exists to capture the *synchronous* RL
 * tail -- must NOT be applied. Hence `synchronous: false`.
 *
 * FINDING: the published targets omit the quantization used, and that omission
 * decides whether the model matches. A 32.8B model at BF16 is 65.6 GB of
 * weights, which leaves ~0.4 GB for KV on an 80 GB H100 -- B_max = 0, i.e. the
 * stated configuration is not physically runnable. The 32B rows are therefore
 * tested at FP8 weights + FP8 KV, which is what makes them feasible and which
 * brings them into range. We assert this explicitly rather than silently
 * widening the tolerance.
 */
describe('generation throughput vs the published vLLM regression suite', () => {
  function minutes(
    model: ModelArchitecture,
    maxCompletionTokens: number,
    weightPrecision: 'bf16' | 'fp8',
    kvPrecisionBytes: number,
  ): number {
    return (
      estimateGeneration(model, H100, {
        inferenceGpus: 1,
        sequences: 512,
        promptTokens: 512,
        maxCompletionTokens,
        weightPrecision,
        kvPrecisionBytes,
        synchronous: false,
      }).totalSeconds / 60
    )
  }

  it('matches 7B at 8K within 2x (measured ~11 min)', () => {
    const got = minutes(QWEN_7B, 8192, 'bf16', 2)
    expect(got / 11).toBeGreaterThan(0.5)
    expect(got / 11).toBeLessThan(2.0)
  })

  it('matches 32B at 8K within 2x at FP8 weights and FP8 KV (measured ~56 min)', () => {
    const got = minutes(QWEN_32B, 8192, 'fp8', 1)
    expect(got / 56).toBeGreaterThan(0.5)
    expect(got / 56).toBeLessThan(2.0)
  })

  /**
   * KNOWN MISSES, recorded rather than hidden.
   *
   * 7B @2K: model ~1.1 min vs measured ~3 min (0.36x -- optimistic). At 2K the
   * run is short enough that engine startup, sampling and detokenization are a
   * large share of wall-clock, and the roofline sees none of that.
   *
   * 7B @32K: model ~216 min vs measured ~45 min (4.8x -- pessimistic). B_max
   * collapses to 26 at BF16 KV, so the model pays 20 waves. The published row
   * is almost certainly FP8 KV and/or a mean completion well below the cap.
   *
   * Both bound the model's honest accuracy at the extremes of completion
   * length. The assertions below pin the CURRENT behaviour so a future change
   * that improves or worsens them is visible, and are deliberately not dressed
   * up as passes.
   */
  /**
   * The 2K rows miss by the SAME factor at both model sizes -- 7B/2K lands at
   * 0.36x and 32B/2K (at FP8, which is what makes it runnable at all) lands at
   * 0.36x too. That consistency is diagnostic: it is not a model-size or
   * quantization artefact but a missing FIXED per-run overhead -- engine
   * startup, CUDA-graph capture, sampling and detokenization -- which the
   * roofline cannot see and which dominates when a run is only minutes long.
   *
   * Recorded as a known limitation rather than patched with a fudge factor,
   * because the correct fix needs a sourced figure for that overhead and the
   * research found none.
   */
  it('documents a consistent ~0.36x short-completion optimism at 2K', () => {
    const r7 = minutes(QWEN_7B, 2048, 'bf16', 2) / 3
    const r32 = minutes(QWEN_32B, 2048, 'fp8', 1) / 14
    for (const r of [r7, r32]) {
      expect(r).toBeLessThan(0.5)
      expect(r).toBeGreaterThan(0.2)
    }
    // The two miss by closely matching factors -- hence a fixed, not scaling, term.
    expect(Math.abs(r7 - r32)).toBeLessThan(0.1)
  })

  it('documents the long-completion pessimism at 7B/32K under BF16 KV', () => {
    const ratio = minutes(QWEN_7B, 32768, 'bf16', 2) / 45
    expect(ratio).toBeGreaterThan(2.0) // known: model is ~4.8x
    expect(ratio).toBeLessThan(8.0)
  })

  it('comes back into range at 7B/32K with FP8 KV', () => {
    const ratio = minutes(QWEN_7B, 32768, 'fp8', 1) / 45
    expect(ratio).toBeLessThan(3.0)
  })
})

describe('weight sync', () => {
  it('reproduces the NeMo-RL 70B refit anchor of ~14 s', () => {
    const seconds = (LLAMA_70B.totalParameters * WEIGHT_BYTES.bf16) / WEIGHT_SYNC_BYTES_PER_SEC
    expect(seconds).toBeGreaterThan(12)
    expect(seconds).toBeLessThan(16)
  })

  it('charges roughly zero for LoRA, which syncs only adapters', () => {
    const rl: RlConfig = {
      groupSize: 8, maxCompletionTokens: 4096, promptsPerStep: 64,
      steps: 100, rolloutPlacement: 'disaggregated',
    }
    const full = estimateRlStep(LLAMA_70B, H100, rl, {
      method: 'grpo', inferenceGpus: 16, trainingGpus: 8,
      promptTokens: 512, weightPrecision: 'bf16',
    })
    const lora = estimateRlStep(LLAMA_70B, H100, rl, {
      method: 'grpo', inferenceGpus: 16, trainingGpus: 8,
      promptTokens: 512, weightPrecision: 'bf16', peft: true,
    })
    expect(full.weightSyncSeconds).toBeGreaterThan(12)
    expect(lora.weightSyncSeconds).toBe(0)
    expect(full.warnings.some((w) => w.includes('Weight refit costs'))).toBe(true)
  })
})

describe('generation:training split', () => {
  const rl: RlConfig = {
    groupSize: 8, maxCompletionTokens: 4096, promptsPerStep: 64,
    steps: 100, rolloutPlacement: 'disaggregated',
  }

  it('reports a generation share, which is the pane\'s primary output', () => {
    const e = estimateRlStep(QWEN_7B, H100, rl, {
      method: 'grpo', inferenceGpus: 8, trainingGpus: 8,
      promptTokens: 512, weightPrecision: 'bf16',
    })
    expect(e.generationShare).toBeGreaterThan(0)
    expect(e.generationShare).toBeLessThan(1)
    expect(e.recommendedInferenceShare).toBe(e.generationShare)
    expect(e.warnings.some((w) => w.includes('Olmo 3'))).toBe(true)
  })

  it('makes r rise monotonically with completion length', () => {
    const rs = [1024, 4096, 16384].map(
      (L) =>
        estimateRlStep(QWEN_7B, H100, { ...rl, maxCompletionTokens: L }, {
          method: 'grpo', inferenceGpus: 8, trainingGpus: 8,
          promptTokens: 512, weightPrecision: 'bf16',
        }).genTrainRatio,
    )
    expect(rs[1]).toBeGreaterThan(rs[0])
    expect(rs[2]).toBeGreaterThan(rs[1])
  })

  it('warns when computed r leaves the published 0.8-8 band', () => {
    // Short completions drive r below the band: the model omits the fixed
    // per-step overheads that keep real short-form RL near r=1.4.
    const e = estimateRlStep(QWEN_7B, H100, { ...rl, maxCompletionTokens: 512 }, {
      method: 'grpo', inferenceGpus: 8, trainingGpus: 8,
      promptTokens: 512, weightPrecision: 'bf16',
    })
    expect(e.genTrainRatio).toBeLessThan(0.8)
    expect(e.warnings.some((w) => w.includes('outside the published 0.8-8 band'))).toBe(true)
  })

  it('exposes the published sanity bands for comparison', () => {
    expect(expectedRatioBand(512)).toBe(1.4)
    expect(expectedRatioBand(4096)).toBe(2.5)
    expect(expectedRatioBand(16384)).toBe(4.0)
    expect(expectedRatioBand(65536)).toBe(6.0)
  })

  /**
   * PPO samples one completion per prompt where GRPO samples G, so PPO
   * generates G-fold fewer tokens. But rollout WALL-CLOCK only falls with it
   * when generation is concurrency-limited.
   *
   * The research's "PPO's r is roughly GRPO's r / 2" holds at equal SEQUENCE
   * count. At equal PROMPT count -- which is what RlConfig models -- the
   * behaviour splits by regime, and the engine reproduces both:
   */
  it('generates G-fold fewer tokens under PPO at equal prompt count', () => {
    const base = {
      inferenceGpus: 8, trainingGpus: 8, promptTokens: 512,
      weightPrecision: 'bf16' as const,
    }
    const grpo = estimateRlStep(QWEN_7B, H100, rl, { ...base, method: 'grpo' })
    const ppo = estimateRlStep(QWEN_7B, H100, rl, { ...base, method: 'ppo' })
    expect(grpo.tokensGenerated / ppo.tokensGenerated).toBe(rl.groupSize)
  })

  it('when NOT concurrency-limited, GRPO\'s extra completions are nearly free', () => {
    // 512 sequences of ~4.6K tokens fit one wave on 8 H100s, so GRPO's 8x
    // sequence count costs little extra wall-clock -- while its training side
    // scales fully. PPO\'s r is therefore HIGHER here, not lower.
    const base = {
      inferenceGpus: 8, trainingGpus: 8, promptTokens: 512,
      weightPrecision: 'bf16' as const,
    }
    const grpo = estimateRlStep(QWEN_7B, H100, rl, { ...base, method: 'grpo' })
    const ppo = estimateRlStep(QWEN_7B, H100, rl, { ...base, method: 'ppo' })
    expect(grpo.generation.waves).toBe(1)
    expect(ppo.generation.waves).toBe(1)
    expect(ppo.genTrainRatio).toBeGreaterThan(grpo.genTrainRatio)
  })

  it('when concurrency-limited, GRPO rollout scales with group size', () => {
    // Long completions on few GPUs collapse B_max, so waves -- and rollout
    // time -- do scale with the sequence count. This is the regime the
    // research\'s "GRPO rollout = G x PPO rollout" describes.
    const longRl: RlConfig = { ...rl, maxCompletionTokens: 32768 }
    const base = {
      inferenceGpus: 2, trainingGpus: 8, promptTokens: 512,
      weightPrecision: 'bf16' as const,
    }
    const grpo = estimateRlStep(QWEN_7B, H100, longRl, { ...base, method: 'grpo' })
    const ppo = estimateRlStep(QWEN_7B, H100, longRl, { ...base, method: 'ppo' })
    expect(grpo.generation.waves).toBeGreaterThan(ppo.generation.waves)
    expect(grpo.generation.decodeSeconds).toBeGreaterThan(ppo.generation.decodeSeconds * 3)
  })
})

describe('tail gating', () => {
  const rl: RlConfig = {
    groupSize: 8, maxCompletionTokens: 16384, promptsPerStep: 64,
    steps: 100, rolloutPlacement: 'disaggregated',
  }
  const base = {
    method: 'grpo' as const, inferenceGpus: 16, trainingGpus: 8,
    promptTokens: 512, weightPrecision: 'bf16' as const,
  }

  it('defaults to max gating, the conservative end', () => {
    const d = estimateRlStep(QWEN_7B, H100, rl, base)
    const max = estimateRlStep(QWEN_7B, H100, rl, { ...base, tailGating: 'max' })
    expect(d.stepSeconds).toBeCloseTo(max.stepSeconds, 6)
  })

  it('makes mean gating substantially cheaper than max', () => {
    const max = estimateRlStep(QWEN_7B, H100, rl, { ...base, tailGating: 'max' })
    const mean = estimateRlStep(QWEN_7B, H100, rl, {
      ...base, tailGating: 'mean', meanCompletionRatio: 0.1,
    })
    expect(mean.stepSeconds).toBeLessThan(max.stepSeconds / 3)
  })

  it('warns that long-cap runs assume every sequence hits the cap', () => {
    const e = estimateRlStep(QWEN_7B, H100, rl, base)
    expect(e.warnings.some((w) => w.includes('up to 10x less'))).toBe(true)
  })
})

describe('honest uncertainty surfacing', () => {
  const rl: RlConfig = {
    groupSize: 8, maxCompletionTokens: 4096, promptsPerStep: 64,
    steps: 500, rolloutPlacement: 'disaggregated',
  }
  const base = {
    method: 'grpo' as const, inferenceGpus: 16, trainingGpus: 8,
    promptTokens: 512, weightPrecision: 'bf16' as const,
  }

  /**
   * The research validates its own 70B GRPO estimate against Tulu 3's measured
   * 2,880 GPU-hours and finds it ~3.8x optimistic. Rather than hide that behind
   * a confidence badge, the engine states it.
   */
  it('labels an uncalibrated estimate a floor, citing the Tulu 3 miss', () => {
    const e = estimateRlStep(LLAMA_70B, H100, rl, base)
    expect(e.warnings.some((w) => w.includes('FLOOR') && w.includes('Tulu 3'))).toBe(true)
  })

  it('accepts one observed step time and derives everything from it', () => {
    const uncal = estimateRlStep(LLAMA_70B, H100, rl, base)
    const cal = estimateRlStep(LLAMA_70B, H100, rl, { ...base, observedStepSeconds: 300 })
    expect(cal.stepSeconds).toBe(300)
    expect(cal.totalHours).toBeCloseTo((300 * 500) / 3600, 6)
    expect(cal.warnings.some((w) => w.includes('Calibrated to an observed'))).toBe(true)
    expect(cal.warnings.some((w) => w.includes('FLOOR'))).toBe(false)
    expect(uncal.stepSeconds).not.toBe(300)
  })

  it('reports tokens generated, the most legible headline number', () => {
    const e = estimateRlStep(QWEN_7B, H100, rl, base)
    // 500 steps x 64 prompts x 8 completions x 4096 tokens = 1.05 billion.
    expect(e.tokensGenerated).toBe(500 * 64 * 8 * 4096)
    expect(e.tokensGenerated).toBeGreaterThan(1e9)
  })

  it('warns when the KV pool exceeds the entire model state', () => {
    const e = estimateRlStep(LLAMA_70B, H100, { ...rl, maxCompletionTokens: 8192 }, base)
    expect(e.warnings.some((w) => w.includes('binding memory term'))).toBe(true)
  })

  it('warns about the colocated memory penalty', () => {
    const e = estimateRlStep(QWEN_7B, H100, { ...rl, rolloutPlacement: 'colocated' }, base)
    expect(e.warnings.some((w) => w.includes('vllm_gpu_memory_utilization'))).toBe(true)
  })

  it('reports infeasible when weights exceed the inference pool', () => {
    const g = estimateGeneration(LLAMA_70B, H100, {
      inferenceGpus: 1, sequences: 64, promptTokens: 512,
      maxCompletionTokens: 4096, weightPrecision: 'bf16', synchronous: true,
    })
    expect(g.totalSeconds).toBe(Infinity)
    expect(g.warnings.some((w) => w.includes('exceed'))).toBe(true)
  })
})

describe('prefix caching across a GRPO group', () => {
  it('prefills the shared prompt once per prompt, not once per completion', () => {
    const common = {
      inferenceGpus: 8, sequences: 512, promptTokens: 16384,
      maxCompletionTokens: 1024, weightPrecision: 'bf16' as const,
      groupSize: 8, synchronous: true,
    }
    const cached = estimateGeneration(QWEN_7B, H100, { ...common, prefixCaching: true })
    const uncached = estimateGeneration(QWEN_7B, H100, { ...common, prefixCaching: false })
    // Saving is (G-1)/G = 87.5% at G=8.
    expect(cached.prefillSeconds / uncached.prefillSeconds).toBeCloseTo(0.125, 3)
  })
})

describe('eta_rollout application', () => {
  it('applies only to synchronous RL steps, not throughput benchmarks', () => {
    const common = {
      inferenceGpus: 8, sequences: 64, promptTokens: 512,
      maxCompletionTokens: 2048, weightPrecision: 'bf16' as const,
    }
    const sync = estimateGeneration(QWEN_7B, H100, { ...common, synchronous: true })
    const async_ = estimateGeneration(QWEN_7B, H100, { ...common, synchronous: false })
    // Same wave count here, so the ratio is exactly 1/ETA_ROLLOUT.
    expect(sync.decodeSeconds / async_.decodeSeconds).toBeCloseTo(1 / ETA_ROLLOUT, 2)
  })
})
