export type ReferenceModel = {
  name: string;
  /** Total resident parameters. For MoE this is the full expert bank, not the per-token cost. */
  parameters: number;
  /**
   * Parameters actually multiplied per token. Equal to `parameters` for dense models.
   * This — not `parameters` — is what drives FLOPs, so `estimatedFlops` is 6 · activeParameters · trainingTokens.
   */
  activeParameters?: number;
  trainingTokens: number;
  estimatedFlops: number;
  /**
   * Published (or explicitly derived) accelerator-hours for the run. This is the quantity
   * directly comparable to the calculator's own output, which makes these rows usable as
   * regression targets rather than just UI decoration. Absent when nobody published it.
   */
  gpuHours?: number;
  gpuType?: string;
  /** Numeric format the training run used, where stated. Matters because FLOP peaks differ per format. */
  precision?: string;
  /**
   * What the GPU-hours figure covers. Runs disagree: Llama 2 counts productive time only,
   * Llama 3.1 counts everything including restarts and idle. A 2x spread hides here.
   */
  hoursConvention?: string;
  source: string;
};

/**
 * Known training runs for comparison, oldest first.
 *
 * `estimatedFlops` is always 6 · N_active · D so the whole table stays internally
 * consistent and machine-checkable (see reference-models.test.ts). Where a lab publishes
 * its own FLOP figure and it differs, the difference is noted in a comment rather than
 * stored — the published figures use varying accounting (embeddings, attention, logits).
 *
 * Research basis: docs/research/2026-10-14-llm-training-research.md §1.2, §11, §12.
 * Every row carries a URL in `source`; rows that could only be sourced to an aggregator
 * or to arithmetic say so in that field.
 */
export const REFERENCE_MODELS: ReferenceModel[] = [
  {
    name: 'GPT-3 175B',
    parameters: 175e9,
    activeParameters: 175e9,
    trainingTokens: 300e9,
    estimatedFlops: 3.15e23,
    // Not a published figure. OpenAI never released GPU-hours; this is 10,000 V100 x 14.8 days,
    // the duration implied by the paper's compute table. Treat as an order-of-magnitude anchor.
    gpuHours: 3.552e6,
    gpuType: 'V100',
    hoursConvention: 'derived (10,000 V100 x 14.8 d) — not published',
    source: 'Brown et al. 2020, arXiv:2005.14165 — https://arxiv.org/abs/2005.14165',
  },
  {
    name: 'Chinchilla 70B',
    parameters: 70e9,
    activeParameters: 70e9,
    trainingTokens: 1.4e12,
    estimatedFlops: 5.88e23,
    gpuType: 'TPU v3/v4',
    hoursConvention: 'not published — FLOPs only',
    source: 'Hoffmann et al. 2022, arXiv:2203.15556 — https://arxiv.org/abs/2203.15556',
  },
  {
    name: 'PaLM 540B',
    parameters: 540e9,
    activeParameters: 540e9,
    trainingTokens: 780e9,
    estimatedFlops: 2.5272e24,
    gpuType: 'TPU v4 (6,144)',
    precision: 'BF16',
    // Chip-hours not published; the paper reports 46.2% MFU / 57.8% HFU instead, which is
    // the public ceiling for pretraining utilization and is why calculator.ts warns above 0.60.
    hoursConvention: 'not published — reported 46.2% MFU instead',
    source: 'Chowdhery et al. 2022, PaLM — https://arxiv.org/pdf/2204.02311',
  },
  {
    name: 'Llama 2 7B',
    parameters: 7e9,
    activeParameters: 7e9,
    trainingTokens: 2e12,
    estimatedFlops: 8.4e22,
    gpuHours: 184320,
    gpuType: 'A100 80GB',
    precision: 'BF16',
    hoursConvention: 'productive GPU-time only (excludes restarts/idle)',
    source: 'Touvron et al. 2023, Llama 2, Table 2 — https://arxiv.org/pdf/2307.09288',
  },
  {
    name: 'Llama 2 13B',
    parameters: 13e9,
    activeParameters: 13e9,
    trainingTokens: 2e12,
    estimatedFlops: 1.56e23,
    gpuHours: 368640,
    gpuType: 'A100 80GB',
    precision: 'BF16',
    hoursConvention: 'productive GPU-time only',
    source: 'Touvron et al. 2023, Llama 2, Table 2 — https://arxiv.org/pdf/2307.09288',
  },
  {
    name: 'Llama 2 34B',
    parameters: 34e9,
    activeParameters: 34e9,
    trainingTokens: 2e12,
    estimatedFlops: 4.08e23,
    gpuHours: 1038336,
    gpuType: 'A100 80GB',
    precision: 'BF16',
    // Weights were never released, but the GPU-hours were, which is what makes it useful here.
    hoursConvention: 'productive GPU-time only',
    source: 'Touvron et al. 2023, Llama 2, Table 2 — https://arxiv.org/pdf/2307.09288',
  },
  {
    name: 'Llama 2 70B',
    parameters: 70e9,
    activeParameters: 70e9,
    trainingTokens: 2e12,
    estimatedFlops: 8.4e23,
    gpuHours: 1720320,
    gpuType: 'A100 80GB',
    precision: 'BF16',
    hoursConvention: 'productive GPU-time only',
    source: 'Touvron et al. 2023, Llama 2, Table 2 — https://arxiv.org/pdf/2307.09288',
  },
  {
    name: 'Falcon 180B',
    parameters: 180e9,
    activeParameters: 180e9,
    trainingTokens: 3.5e12,
    estimatedFlops: 3.78e24,
    // Weak figure: "~7,000,000 GPU hours" is a rounded number from the HuggingFace launch post,
    // not the technical report. It back-solves to 48% of A100 dense peak, above anything
    // independently measured for A100-class pretraining, so the real number is probably higher.
    gpuHours: 7.0e6,
    gpuType: 'A100 40GB (up to 4,096)',
    hoursConvention: 'rounded aggregate — low confidence',
    source:
      'TII 2023, arXiv:2311.16867 (N, D) — https://arxiv.org/abs/2311.16867; GPU-hours from the HuggingFace launch post (weaker) — https://github.com/huggingface/blog/blob/main/falcon-180b.md',
  },
  {
    name: 'Llama 3.1 8B',
    parameters: 8e9,
    activeParameters: 8e9,
    trainingTokens: 15.6e12,
    estimatedFlops: 7.488e23,
    gpuHours: 1.46e6,
    gpuType: 'H100 80GB',
    precision: 'BF16',
    hoursConvention: 'all-inclusive (cumulative cluster time)',
    source:
      'Llama 3.1 model card — https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md; 15.6T tokens from arXiv:2407.21783 — https://arxiv.org/html/2407.21783v3',
  },
  {
    name: 'Llama 3.1 70B',
    parameters: 70e9,
    activeParameters: 70e9,
    trainingTokens: 15.6e12,
    estimatedFlops: 6.552e24,
    gpuHours: 7.0e6,
    gpuType: 'H100 80GB',
    precision: 'BF16',
    hoursConvention: 'all-inclusive',
    source:
      'Llama 3.1 model card — https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md; 15.6T tokens from arXiv:2407.21783 — https://arxiv.org/html/2407.21783v3',
  },
  {
    name: 'Llama 3.1 405B',
    parameters: 405e9,
    activeParameters: 405e9,
    // 15.6T, not the 15T of the Llama 3 blog post (issue #77). Meta's own stated total is
    // 3.8e25 FLOPs, which 6ND reproduces as 3.79e25 — agreement to within 0.3%.
    trainingTokens: 15.6e12,
    estimatedFlops: 3.7908e25,
    gpuHours: 30.84e6,
    gpuType: 'H100 80GB (16,384)',
    precision: 'BF16',
    hoursConvention: 'all-inclusive — 466 interruptions, >90% effective training time',
    source:
      'Llama 3 Herd, arXiv:2407.21783 — https://arxiv.org/html/2407.21783v3; GPU-hours from the Llama 3.1 model card — https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/MODEL_CARD.md',
  },
  {
    name: 'DeepSeek-V3 671B-A37B',
    parameters: 671e9,
    activeParameters: 37e9,
    trainingTokens: 14.8e12,
    estimatedFlops: 3.2856e24,
    // 2.664M is pretraining only; full training including context extension and post-training
    // is 2.788M. Pretraining is the comparable figure for this calculator.
    gpuHours: 2.664e6,
    gpuType: 'H800',
    precision: 'FP8',
    hoursConvention: 'pretraining only (2.788M for full training)',
    source: 'DeepSeek-V3 Technical Report, arXiv:2412.19437 — https://arxiv.org/abs/2412.19437',
  },
  {
    name: 'Qwen3-30B-A3B',
    parameters: 30e9,
    activeParameters: 3e9,
    trainingTokens: 36e12,
    estimatedFlops: 6.48e23,
    hoursConvention: 'not published — FLOPs only',
    source: 'Qwen3 Technical Report, arXiv:2505.09388 — https://arxiv.org/pdf/2505.09388',
  },
  {
    name: 'Qwen3-235B-A22B',
    parameters: 235e9,
    activeParameters: 22e9,
    trainingTokens: 36e12,
    estimatedFlops: 4.752e24,
    hoursConvention: 'not published — FLOPs only',
    source: 'Qwen3 Technical Report, arXiv:2505.09388 — https://arxiv.org/pdf/2505.09388',
  },
  {
    name: 'Kimi K2 1T-A32B',
    parameters: 1e12,
    activeParameters: 32e9,
    trainingTokens: 15.5e12,
    estimatedFlops: 2.976e24,
    hoursConvention: 'not published — FLOPs only',
    source: 'Kimi K2 Technical Report, arXiv:2507.20534 — https://arxiv.org/abs/2507.20534',
  },
  {
    name: 'GLM-4.5 355B-A32B',
    parameters: 355e9,
    activeParameters: 32e9,
    // 23T is the multi-stage total. The report's first general-domain stage alone is 15T;
    // sources that quote 15T are describing that stage, not the run.
    trainingTokens: 23e12,
    estimatedFlops: 4.416e24,
    hoursConvention: 'not published — FLOPs only',
    source: 'GLM-4.5 Technical Report, arXiv:2508.06471 — https://arxiv.org/abs/2508.06471',
  },
  {
    name: 'Llama 4 Maverick 400B-A17B',
    parameters: 400e9,
    activeParameters: 17e9,
    trainingTokens: 22e12,
    estimatedFlops: 2.244e24,
    gpuHours: 2.38e6,
    gpuType: 'H100 80GB',
    hoursConvention: 'all-inclusive (Meta convention, as for Llama 3.1)',
    source:
      'Llama 4 model card — https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E (the card covers both Scout and Maverick)',
  },
  {
    name: 'Llama 4 Scout 109B-A17B',
    parameters: 109e9,
    activeParameters: 17e9,
    trainingTokens: 40e12,
    estimatedFlops: 4.08e24,
    gpuHours: 5.0e6,
    gpuType: 'H100 80GB',
    hoursConvention: 'all-inclusive',
    source:
      'Llama 4 model card — https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E',
  },
  {
    name: 'OLMo 2 32B',
    parameters: 32e9,
    activeParameters: 32e9,
    // 6T tokens over ~1.5 epochs of a 4T-token corpus. Ai2's own stated training compute is
    // 1.3e24, ~13% above 6ND; their figure includes terms 6ND omits. 6ND kept for consistency.
    trainingTokens: 6e12,
    estimatedFlops: 1.152e24,
    gpuType: 'H100 (1,280 — 160 nodes x 8)',
    hoursConvention: 'not published — ~38% MFU reported instead',
    source:
      'Ai2, OLMo 2 32B — https://allenai.org/blog/olmo2-32b; model card — https://huggingface.co/allenai/OLMo-2-0325-32B',
  },
  {
    name: 'OLMo 3 32B',
    parameters: 32e9,
    activeParameters: 32e9,
    // Sources range 5.5T–5.93T. 5.93T is the paper's own Table 4 figure for the final
    // training mix (the Dolma 3 Mix is a 6T-token pool it draws from); the lower numbers
    // circulating come from secondary write-ups. Paper figure taken.
    trainingTokens: 5.93e12,
    estimatedFlops: 1.13856e24,
    // Derived, not published: 1,024 H100 x ~47 days covering pretraining, midtraining and the
    // long-context stage. Independent write-ups land at 1.05M–1.38M, bracketing this.
    gpuHours: 1.155e6,
    gpuType: 'H100 (1,024)',
    hoursConvention: 'derived (1,024 H100 x ~47 d, incl. midtraining + long-context)',
    source: 'Olmo 3, arXiv:2512.13961 — https://arxiv.org/html/2512.13961v2',
  },
  // Deliberately absent: Mistral Large 3 (675B total / 41B active, Dec 2025) and Kimi K3
  // (2.8T total / 104B active, Jul 2026). Both publish parameter counts but no token count,
  // so there is no honest `trainingTokens` and therefore no FLOP figure. Mistral states only
  // "trained from scratch on 3,000 H200s" with no duration. Omitted rather than guessed.
  {
    name: 'GLM-5 744B-A40B',
    parameters: 744e9,
    activeParameters: 40e9,
    // 28.5T is the base-model total across stages; the initial pretraining corpus alone is 27T.
    // Active-parameter reports vary 40B–44B; the technical report's own figure (40B) is used.
    trainingTokens: 28.5e12,
    estimatedFlops: 6.84e24,
    hoursConvention: 'not published — FLOPs only',
    source: 'GLM-5, arXiv:2602.15763 — https://arxiv.org/html/2602.15763v2',
  },
  {
    name: 'MiniMax-M2 230B-A10B',
    parameters: 229.9e9,
    activeParameters: 9.8e9,
    // 29.2T total = 19.9T constant phase + 9.3T decay phase. Both are stated in the report;
    // quoting 29.2T as the constant phase alone (and so summing to 38.5T) is a misreading.
    trainingTokens: 29.2e12,
    estimatedFlops: 1.71696e24,
    hoursConvention: 'not published — FLOPs only',
    source: 'MiniMax-M2 Series, arXiv:2605.26494 — https://arxiv.org/html/2605.26494v1',
  },
  {
    name: 'Nemotron 3 Ultra 550B-A55B',
    parameters: 550e9,
    activeParameters: 55e9,
    trainingTokens: 20e12,
    // Hybrid Mamba-Attention MoE, so 6ND is a looser fit here than for a pure transformer:
    // Mamba layers have a different per-token cost. Treat this row's FLOPs as indicative.
    estimatedFlops: 6.6e24,
    precision: 'NVFP4',
    hoursConvention: 'not published — FLOPs only',
    source:
      'Nemotron 3 Ultra Technical Report, arXiv:2606.15007 — https://arxiv.org/html/2606.15007v1',
  },
  {
    name: 'DeepSeek-V4-Flash 284B-A13B',
    parameters: 284e9,
    activeParameters: 13e9,
    trainingTokens: 32e12,
    estimatedFlops: 2.496e24,
    precision: 'FP8 (routed experts FP4)',
    hoursConvention: 'not published — FLOPs only',
    source: 'DeepSeek-V4, arXiv:2606.19348 — https://arxiv.org/html/2606.19348v1',
  },
  {
    name: 'DeepSeek-V4-Pro 1.6T-A49B',
    parameters: 1.6e12,
    activeParameters: 49e9,
    trainingTokens: 33e12,
    estimatedFlops: 9.702e24,
    precision: 'FP8 (routed experts FP4)',
    hoursConvention: 'not published — FLOPs only',
    source: 'DeepSeek-V4, arXiv:2606.19348 — https://arxiv.org/html/2606.19348v1',
  },
  {
    name: 'Marin 8B',
    parameters: 8e9,
    activeParameters: 8e9,
    // 12.7T, not 1.6e12 (issue #77). 1.6T was an early-checkpoint figure; the deeper-starling
    // release is 12.7T, which puts Marin 8B at ~1,590 tokens/parameter.
    trainingTokens: 12.7e12,
    estimatedFlops: 6.096e23,
    gpuType: 'TPU v4/v5e',
    hoursConvention: 'not published — FLOPs only',
    source:
      'Marin 8B Retro report — https://marin.readthedocs.io/en/latest/reports/marin-8b-retro/',
  },
  {
    name: 'Marin 32B',
    parameters: 32e9,
    activeParameters: 32e9,
    // 6.437T, not 3.2e12 (issue #77).
    trainingTokens: 6.437e12,
    estimatedFlops: 1.235904e24,
    gpuType: 'TPU v5p/v4',
    hoursConvention: 'not published — FLOPs only',
    source:
      'Marin 32B Retro report — https://github.com/marin-community/marin/blob/main/docs/reports/marin-32b-retro.md',
  },
];

/**
 * Find the closest reference models to a given FLOP count.
 * Returns the nearest smaller and nearest larger model.
 */
export function findNearestReferences(totalFlops: number): {
  smaller: ReferenceModel | null;
  larger: ReferenceModel | null;
  closest: ReferenceModel;
} {
  const sorted = [...REFERENCE_MODELS].sort((a, b) => a.estimatedFlops - b.estimatedFlops);

  let smaller: ReferenceModel | null = null;
  let larger: ReferenceModel | null = null;

  for (const model of sorted) {
    if (model.estimatedFlops <= totalFlops) {
      smaller = model;
    } else if (!larger) {
      larger = model;
    }
  }

  const closest = sorted.reduce((prev, curr) =>
    Math.abs(curr.estimatedFlops - totalFlops) < Math.abs(prev.estimatedFlops - totalFlops)
      ? curr : prev
  );

  return { smaller, larger, closest };
}

export function formatFlopsShort(n: number): string {
  if (n >= 1e24) return `${(n / 1e24).toFixed(1)}e24`;
  if (n >= 1e23) return `${(n / 1e23).toFixed(1)}e23`;
  if (n >= 1e22) return `${(n / 1e22).toFixed(1)}e22`;
  if (n >= 1e21) return `${(n / 1e21).toFixed(1)}e21`;
  if (n >= 1e20) return `${(n / 1e20).toFixed(1)}e20`;
  if (n >= 1e19) return `${(n / 1e19).toFixed(1)}e19`;
  return n.toExponential(1);
}
