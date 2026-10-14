import { describe, it, expect } from 'vitest'
import catalogData from './finetune-models.json'
import type { FineTuneModelData, FineTuneModelEntry } from './types'

const catalog = catalogData as FineTuneModelData
const MODELS = catalog.models

/** Entries the engine is allowed to compute adapter and activation figures from exactly. */
const VERIFIED = MODELS.filter((m) => m.dimsVerified)

/**
 * Parameter count implied by the transcribed dimensions, for a uniform
 * GQA/MHA + SwiGLU stack:
 *
 *   embeddings  vocab x d, counted twice unless the output head is tied
 *   attention   q and o are d x (heads x headDim); k and v are d x (kvHeads x headDim)
 *   mlp         gate, up, down at d x ffn, once per expert when sparse
 *
 * Norms, biases and routers are sub-1% and deliberately omitted.
 */
function impliedParameters(m: FineTuneModelEntry): number {
  const d = m.hidden!
  const qDim = m.heads! * m.headDim!
  const kvDim = m.kvHeads! * m.headDim!
  const embeddings = m.vocab! * d * (m.tiedEmbeddings ? 1 : 2)
  const attention = d * qDim + 2 * d * kvDim + qDim * d
  const mlp = 3 * d * m.ffnDim! * (m.isMoe ? m.numExperts! : 1)
  return embeddings + m.layers! * (attention + mlp)
}

/** Trainable parameters for LoRA rank `r` targeting all seven linear projections. */
function adapterParameters(m: FineTuneModelEntry, r: number): number {
  const d = m.hidden!
  const dKv = m.kvHeads! * m.headDim!
  const ffn = m.ffnDim!
  const perLayer =
    r *
    (d + d + // q
      (d + dKv) + // k
      (d + dKv) + // v
      (d + d) + // o
      (d + ffn) + // gate
      (d + ffn) + // up
      (ffn + d)) // down
  return m.layers! * perLayer
}

describe('finetune-models catalog integrity', () => {
  it('has a unique id and a cited source for every entry', () => {
    const ids = new Set(MODELS.map((m) => m.id))
    expect(ids.size).toBe(MODELS.length)
    for (const m of MODELS) {
      expect(m.source).not.toBe('')
      expect(m.label).not.toBe('')
      expect(m.family).not.toBe('')
    }
  })

  it('cites a real config URL for every entry but the custom placeholder', () => {
    for (const m of MODELS.filter((e) => e.id !== 'custom')) {
      expect(m.source, m.id).toMatch(/^https:\/\/huggingface\.co\/.+\/config\.json$/)
    }
  })

  it('leaves every dimension null on the custom placeholder so the engine falls back', () => {
    const custom = MODELS.find((m) => m.id === 'custom')
    expect(custom).toBeDefined()
    expect(custom!.dimsVerified).toBe(false)
    for (const dim of [
      custom!.totalParameters,
      custom!.layers,
      custom!.hidden,
      custom!.heads,
      custom!.kvHeads,
      custom!.headDim,
      custom!.ffnDim,
      custom!.vocab,
    ]) {
      expect(dim).toBeNull()
    }
  })

  it('carries a complete dimension set on every verified entry', () => {
    expect(VERIFIED.length).toBeGreaterThan(0)
    for (const m of VERIFIED) {
      for (const [name, dim] of Object.entries({
        totalParameters: m.totalParameters,
        activeParameters: m.activeParameters,
        layers: m.layers,
        hidden: m.hidden,
        heads: m.heads,
        kvHeads: m.kvHeads,
        headDim: m.headDim,
        ffnDim: m.ffnDim,
        vocab: m.vocab,
        contextWindow: m.contextWindow,
      })) {
        expect(dim, `${m.id}.${name}`).not.toBeNull()
        expect(dim, `${m.id}.${name}`).toBeGreaterThan(0)
      }
    }
  })
})

describe('transcribed dimensions reproduce the stated parameter count', () => {
  // The guard against a mis-transcribed dimension: a wrong layer count, hidden
  // size or FFN width moves the implied total well outside 15%, while the
  // deliberately omitted norms, biases and routers move it under 1%.
  it.each(VERIFIED)('$id implied parameters are within 15% of totalParameters', (m) => {
    const implied = impliedParameters(m)
    const error = Math.abs(implied - m.totalParameters!) / m.totalParameters!
    expect(error, `${m.id}: implied ${implied} vs stated ${m.totalParameters}`).toBeLessThan(0.15)
  })

  it('reproduces dense parameter counts to better than 0.1% for plain GQA stacks', () => {
    // These have no shared experts and no second head-dim regime, so the
    // formula should be near-exact. A regression here is a transcription bug.
    const exact = ['llama-3-8b', 'llama-3.1-70b', 'qwen2.5-72b', 'mistral-7b-v0.3', 'phi-4-14b']
    for (const id of exact) {
      const m = MODELS.find((e) => e.id === id)!
      const error = Math.abs(impliedParameters(m) - m.totalParameters!) / m.totalParameters!
      expect(error, id).toBeLessThan(0.001)
    }
  })
})

describe('LoRA adapter parameter counts', () => {
  it('reproduces PEFT\'s 41,943,040 for Llama-3-8B at r=16 on all linear projections', () => {
    const llama3 = MODELS.find((m) => m.id === 'llama-3-8b')!
    expect(adapterParameters(llama3, 16)).toBe(41_943_040)
  })

  it('scales linearly in rank', () => {
    const llama3 = MODELS.find((m) => m.id === 'llama-3-8b')!
    expect(adapterParameters(llama3, 32)).toBe(2 * adapterParameters(llama3, 16))
    expect(adapterParameters(llama3, 64)).toBe(4 * adapterParameters(llama3, 16))
  })
})

describe('MoE total-versus-active split', () => {
  // Compute follows active, memory follows total. Conflating them overstates
  // FLOPs by 20-34x on the frontier sparse models, so the split is asserted
  // rather than assumed.
  it.each(MODELS.filter((m) => m.isMoe))('$id activates strictly fewer parameters than it resides', (m) => {
    expect(m.activeParameters).not.toBeNull()
    expect(m.totalParameters).not.toBeNull()
    expect(m.activeParameters!).toBeLessThan(m.totalParameters!)
    expect(m.numExperts, `${m.id}.numExperts`).toBeGreaterThan(1)
    expect(m.expertsPerToken, `${m.id}.expertsPerToken`).toBeGreaterThan(0)
    expect(m.expertsPerToken!).toBeLessThanOrEqual(m.numExperts!)
  })

  it.each(MODELS.filter((m) => !m.isMoe && m.id !== 'custom'))(
    '$id is dense, so active equals total and no expert fields are set',
    (m) => {
      expect(m.activeParameters).toBe(m.totalParameters)
      expect(m.numExperts).toBeNull()
      expect(m.expertsPerToken).toBeNull()
    },
  )
})

describe('attention typing', () => {
  it('labels every entry with an attention type consistent with its head counts', () => {
    for (const m of MODELS) {
      if (m.attentionType === 'mha') expect(m.kvHeads, m.id).toBe(m.heads)
      if (m.attentionType === 'mqa') expect(m.kvHeads, m.id).toBe(1)
      if (m.attentionType === 'gqa' && m.id !== 'custom') {
        expect(m.kvHeads!, m.id).toBeLessThan(m.heads!)
      }
    }
  })

  it('carries the latent dimensions for every MLA model, since kvHeads x headDim is not its KV width', () => {
    const mla = MODELS.filter((m) => m.attentionType === 'mla')
    expect(mla.length).toBeGreaterThan(0)
    for (const m of mla) {
      expect(m.mla, m.id).toBeDefined()
      expect(m.mla!.kvLoraRank, m.id).toBeGreaterThan(0)
      expect(m.mla!.qkRopeHeadDim, m.id).toBeGreaterThan(0)
      // The whole point: the latent is far narrower than a GQA cache would be.
      const latent = m.mla!.kvLoraRank + m.mla!.qkRopeHeadDim
      expect(latent, m.id).toBeLessThan(m.kvHeads! * m.headDim!)
      // Latent attention is not describable by the 7-projection adapter formula.
      expect(m.dimsVerified, m.id).toBe(false)
    }
  })
})
