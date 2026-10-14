# Deep Research: Alternative Model Family Panes (Time-Series, Tabular FM, Classical Tabular)

**Date:** 2026-10-14
**Purpose:** Re-ground the three non-LLM panes of the calculator on 2026 evidence, and audit the shipped adapters (`src/engine/adapters/time-series-adapter.ts`, `tabular-adapter.ts`, `classical-tabular-adapter.ts`) against it.
**Supersedes as research input:** `scaling_laws_time_series_tabular_models.md` (2026-06-19).
**Question each pane answers:** different in each case — and that is the finding.

---

## 0. The central findings (one per family)

These three panes look like siblings in the UI. They are not siblings in the physics. Each answers a structurally different question, and only one of them is the LLM question in a new costume.

### 0.1 Time-series foundation models — *the LLM question, but the token is a design choice and the models are tiny*

Time-series pretraining **is** a `factor × N × D` compute problem, and the `6ND` machinery transfers. Two things break:

1. **`D` is not given, it is constructed.** Chronos emits one token per timestep (quantized into a 4,096-word vocabulary); Chronos-Bolt emits one token per 16-timestep patch; Time-MoE is point-wise; TTM is an MLP-Mixer over 8 patches. For the *same* raw corpus, token counts differ by 16–64×. The repo is right to make tokenization a first-class input.
2. **The models are 2–4 orders of magnitude smaller than LLMs,** and at that scale the run stops being FLOP-bound. Granite TTM — 1M params, ~1B samples — took **24–30 hours on 6× A100** ([IBM](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2)). A `6ND` estimate at MFU 0.3 predicts **1.4 minutes**. That is a **~1,100× underestimate**, because a 1M-parameter model achieves ~0.03% MFU: the run is bound by data loading, normalization, and kernel launch, not by matmul.

So the time-series pane's dominant failure mode is not a wrong formula — it is a wrong MFU. The useful product decision is **MFU-by-model-scale**, not a better architecture factor.

Also: the tokens-per-parameter analogue exists, and it is *sublinear*. Edwards et al. derive **`D ∝ N^0.8`** (`αN/αD`), i.e. "doubling the model size requires roughly 1.7× more data" — not 2× as Chinchilla implies ([arXiv:2405.13867](https://arxiv.org/abs/2405.13867), corroborated by [arXiv:2410.12360](https://arxiv.org/html/2410.12360v3)). Published time-series TPP ratios span **~40 to ~2,000** — every one of them *above* Chinchilla-20 (see §1.3) — so TPP is a terrible default and a good *diagnostic*.

And the scale of the hardware is small throughout: Lag-Llama was pretrained on **one Tesla P100-12GB**; the entire 30-dataset break-even study of TSFM fine-tuning and zero-shot use consumed **~8 GPU-hours on one RTX 5090**. The only published multi-node TSFM run is Time-MoE (128× A100).

### 0.2 Tabular foundation models — *there is no training question for ~all users; the cost is inference*

TabPFN and its descendants do no gradient step at adaptation time. "Fitting" is a forward pass over the support set. Consequences:

- **The repo's pane models pretraining** (`numberOfPretrainingTasks × rowsPerTask`), which approximately nobody does. The three organizations that do it (Prior Labs, Soda/Inria, AWS AutoGluon) publish enough to confirm it is a *single-node* job: the community **speedrun** reaches a TabArena quality target in **0.92 minutes on one L40S** (down from a 74.32-minute baseline) ([arXiv:2606.03681](https://arxiv.org/abs/2606.03681)). TabPFN-3 itself is **53M/58M parameters** trained on **>8 trillion tokens** of synthetic data ([arXiv:2605.13986](https://arxiv.org/html/2605.13986v2)) — a TPP of ~150,000, the most extreme over-training ratio in any model family.
- **The real user question is "how long will my 200k × 300 table take, and does it fit in VRAM?"** That has a published closed form: TabPFN-2.5's cost scales as **`O(r²·min(c,500) + r·min(c,500)²)`** ([arXiv:2511.08667](https://arxiv.org/html/2511.08667v2)) — quadratic in rows. 50,000 rows × 500 columns is **23–30 s on one H100**. TabPFN-3's KV cache is **7 GiB per estimator at 1M rows**.
- `testTimeComputeMultiplier` as a scalar on *pretraining* FLOPs (`tabular-adapter.ts:64`) multiplies the wrong base and cannot express the quadratic. It is the single weakest construct in the three adapters.

**Direct answer: the tabular FM pane should be an inference/feasibility calculator, not a training calculator.**

### 0.3 Classical tabular / GBDT — *the question is not "how many GPUs", it is "do I need a GPU at all"*

The evidence supports stating this bluntly.

- A **28–64-core CPU server trains 500 boosting rounds of LightGBM on Higgs (10.5M × 28) in 130 seconds** ([LightGBM Experiments](https://lightgbm.readthedocs.io/en/latest/Experiments.html)). The repo's own default scenario (1M × 100 × 1,000 rounds) is a ~2–5 minute CPU job.
- GPU helps, but only above roughly **1M rows of dense numeric features**, and the honest range is **4–23×** for XGBoost/CatBoost and **0.5–1.5× (i.e. often slower)** for LightGBM's OpenCL backend. LightGBM-GPU was *slower than LightGBM-CPU on 5 of 6 datasets* in the reference suite.
- On small data GPU is a net loss: a 100K-row dataset can train **2–3× slower on GPU** ([LightGBM GPU Performance](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html)).
- VRAM is essentially **never** the binding constraint any more: XGBoost 3.0's External-Memory QuantileDMatrix trains a **1 TB** dataset (2^29 rows × 512 features) on a **single GH200** (80 GB HBM3 + 480 GB LPDDR5X) ([NVIDIA](https://developer.nvidia.com/blog/train-with-terabyte-scale-datasets-on-a-single-nvidia-grace-hopper-superchip-using-xgboost-3-0)).

So the GBDT pane should lead with a **"GPU: yes / no / doesn't matter" verdict plus a wall-clock number**, and should be honest that 1 GPU is the answer whenever the answer is a GPU at all. Multi-GPU GBDT is a niche.

---

## 1. Time-series foundation models

### 1.1 Model catalog (transcribable into `src/data/time-series-models.json`)

Fields marked **—** are genuinely not published; do not invent them.

| Model | Params (total / active) | Layers | Hidden | Tokenization | Max context | Pretraining corpus | Training hardware / budget | License | Source |
|---|---|---|---|---|---|---|---|---|---|
| Chronos-T5 (Mini/Small/Base/Large) | 20M / 46M / 200M / 710M | T5 stack (not restated per size) | — | **per-timestep, quantized to 4,096-token vocab**, uniform bins on [−15, 15], mean scaling | 512 in + 64 out | ~890K real series + 10M TSMixup + 1M KernelSynth ≈ **84B tokens** | **8× A100-40GB**, 200K steps, effective batch 256 | Apache-2.0 | [arXiv:2403.07815](https://arxiv.org/abs/2403.07815) |
| Chronos-Bolt (Tiny/Mini/Small/Base) | 9M / 21M / 48M / 205M | — | — | **patch = 16**, instance-norm, T5-style encoder, direct quantile head | 2048 | same lineage as Chronos | — | Apache-2.0 | [AWS](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/) |
| **Chronos-2** (Oct 2025) | 120M (family cited 9M–710M) | — (not published) | — | non-overlapping patches per variate + **group attention** across series in a group | 2048 pretrain → **8192** post-train; horizon 1024 | ~900K+ real series over 22 sources; multivariate tasks **entirely synthetic** via "multivariatizers" | — | Apache-2.0 | [HF](https://huggingface.co/amazon/chronos-2), [arXiv:2510.15821](https://arxiv.org/html/2510.15821v1) |
| TimesFM 1.0 / 2.0 / **2.5** | 200M → 500M → **200M** (+30M optional quantile head) | — | — | patch-based decoder | 512 → 2048 → **16,000** | **~100B real time points** | — | Apache-2.0 | [MarkTechPost](https://www.marktechpost.com/2025/09/16/google-ai-ships-timesfm-2-5-smaller-longer-context-foundation-model-that-now-leads-gift-eval-zero-shot-forecasting/) |
| Moirai 1.0 (S/B/L) | 14M / 91M / 311M | — | — | patch ∈ {8,16,32,64,128}, **flattened channels** + relative channel encodings (all variates in one sequence) | any-variate | **LOTSA, 27.6B observations**, 9 domains | **A100-40G**, 10^6 steps, batch 256 | Apache-2.0 | [arXiv:2402.02592](https://arxiv.org/abs/2402.02592) |
| Moirai-MoE (S/B) | 117M? total / **11M, 86M active**; 32 experts, top-2 | — | — | patch = **16** fixed; masking ratio 0.3 | — | LOTSA | — | Apache-2.0 | [arXiv:2410.10469](https://arxiv.org/pdf/2410.10469) |
| **Moirai 2.0** (Aug 2025) | **11M** | — | — | single patch length **16**, decoder-only, 9-quantile head, multi-token prediction | — | new corpus of **36M series** | — | Apache-2.0 | [arXiv:2511.11698](https://arxiv.org/html/2511.11698v1), [Salesforce](https://www.salesforce.com/blog/moirai-2-0/) |
| Time-MoE (base/large/ultra) | 113M/50M · 453M/200M · **2.4B / 1.1B active**; 8 experts top-2 | 12 / 12 / **36** | 384 / — / **1024** (FFN 1536/—/4096, expert dim 192/—/512) | **point-wise (1 token per timestep)** | 4096 | **Time-300B, >300B points**; run consumed ~117B (paper) | **128× A100-80G**, 100K steps, batch 1024, BF16, **4M points/iteration** | Apache-2.0 | [arXiv:2409.16040](https://arxiv.org/html/2409.16040v2) |
| Sundial (S/B/L) | 32M / 128M / **444M** | 6 / 12 / 24 | 512 / 768 / 1024 | **patch = 16**, decoder-only | 2880 | **TimeBench, 1 trillion time points** (0.05% synthetic) | — (not published) | — | [arXiv:2502.00816](https://arxiv.org/html/2502.00816v1) |
| **Toto 2.0** (May 2026) | **4M / 22M / 313M / 1B / 2.5B**, u-μP-scaled | proxy: L=12, d=256, h=4, d_head=64 constant | see proxy | patch-based decoder; mask c_max=16 patches, p_max=0.4 | **4096 training context** | **5.04 trillion data points** (Datadog observability + synthetic; **zero public forecasting data**) | — (not published) | Apache-2.0 | [arXiv:2605.20119](https://www.alphaxiv.org/abs/2605.20119), [Datadog](https://www.datadoghq.com/blog/ai/toto-2/) |
| **Xihe** (tiny→max) | **9.5M → 1.5B** | — | — | Hierarchical Interleaved Block Attention (HIBA): intra-/inter-block alternation | — | **325B time points** | — | — | [arXiv:2510.21795](https://arxiv.org/abs/2510.21795) |
| **Timer-S1** | **8.3B total / 0.75B active** | — | — | — | — | **~1T points** | — | — | via [TSFM.ai](https://tsfm.ai/blog/tsfm-scaling-laws-2026) (aggregator) |
| Granite TTM (TinyTimeMixer) R1/R2 | **1M–5M** | MLP-Mixer, no attention | — | context ∈ {512, 1024, 1536}, **patch = 64** → 8–24 tokens; horizons {96,192,336,720} | 1536 | **~1B samples** (Monash + LibCity) | **6× A100, 24–30 h (R1) / 12–24 h (R2)** | Apache-2.0 | [HF](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2), [arXiv:2401.03955](https://arxiv.org/pdf/2401.03955) |
| Lag-Llama | **~2.4M** (derived: L=8, 9 heads × 16 = d_model 144; **the paper does not state a total**) | **8** | **144** | lag covariates, per-timestep windows | 32 | **7,965 series ⇒ ~352M data windows (tokens)** | **one Tesla P100-12GB**, batch 256, lr 1e-4 | Apache-2.0 | [arXiv:2310.08278](https://arxiv.org/html/2310.08278v3) |
| Moment (S/B/L) | 40M / 125M / 385M | — | — | patch-based encoder, masked modeling | 512 | — | — | MIT | [arXiv:2402.03885](https://arxiv.org/abs/2402.03885) |
| TiRex / **TiRex-2** (Jul 2026) | 35M / **38.4M active (+44.1M multivariate)** | xLSTM (recurrent) | — | **linear in sequence length**, streaming state updates | unbounded (recurrent) | — | — | NXAI license | [arXiv:2607.01204](https://arxiv.org/html/2607.01204v1) |
| TimeGPT | **undisclosed** (closed) | — | — | — | — | — | — | commercial | Nixtla |

**Catalog caveats worth encoding as data-quality flags:** Chronos-2, TimesFM, Sundial, Toto 2.0, and Xihe all publish parameter counts and corpus sizes but **not** layer/hidden dimensions per size, and **none of the 2026 releases publish training hardware or GPU-hours**. The only three families with a published training budget are Chronos (8× A100, 200K steps), Moirai (A100, 10^6 steps × 256), and TTM (6× A100, 24–30 h). Those are the only usable regression targets.

### 1.2 Tokenization: verifying the repo's two modes against real architectures

The repo offers `channel_compressed | channel_expanded | custom` (`src/engine/types.ts:56`), computing `patchesPerWindow = ceil(lookback / patchSize)` and then either `patchesPerWindow` or `variables × patchesPerWindow` (`time-series-adapter.ts:51-66`).

Mapped onto what 2026 models actually do:

| Real design | Token count per window | Repo mode | Verdict |
|---|---|---|---|
| **Channel-independent** (Chronos, Chronos-Bolt, TimesFM, Time-MoE, Sundial, Toto) — each variate is its own univariate sequence | `V × ceil(L/P)`, but as **V separate sequences of length `ceil(L/P)`** | `channel_expanded` | ✅ token count correct; ❌ attention cost over-counted (see note) |
| **Channel-flattened** (Moirai 1.0 — flattens variates into one sequence with relative channel encodings) | `V × ceil(L/P)` in **one** sequence | `channel_expanded` | ✅ correct, and here the attention really is quadratic in `V × ceil(L/P)` |
| **Group attention** (Chronos-2) — per-variate time attention + cross-series group attention | `V × ceil(L/P)` tokens, plus a group-attention term over group members at each patch index | none | ❌ **gap**: the extra cross-series term is unrepresented |
| **Channel-mixing at embedding** (TTM / TSMixer) — all `V` channels folded into one token per patch | `ceil(L/P)` | `channel_compressed` | ✅ correct, and `channel_compressed` is the *right* default for Mixer-family models |
| **Inverted / variate tokens** (iTransformer family) — one token per variate covering the whole window | `V` (independent of patch size) | none | ❌ **gap** |
| **Per-timestep quantization** (Chronos, Time-MoE) | `L` tokens | `channel_compressed` with `patchSize = 1` | ✅ expressible, but non-obvious; deserves a named preset |
| **Recurrent** (TiRex-2, xLSTM) | `ceil(L/P)` but **linear** cost, no quadratic attention term | any | ⚠️ warnings about long sequences are wrong for this family |

**Conclusion on the repo's modes:** both existing modes map onto real architectures and the 4× ratio tested in `time-series-adapter.test.ts:42-46` is correct for a 4-variate channel-expanded model. The decision to default to `channel_compressed` ([decisions doc](../decisions/2026-06-19-model-family-extension-decisions.md)) is defensible but is now the *minority* design — every 2026 frontier TSFM is channel-independent or channel-flattened, i.e. `channel_expanded`. **Recommend flipping the default to `channel_expanded`** and renaming the modes to the architecture names practitioners use (`channel_independent`, `channel_mixed`, `variate_tokens`).

One subtlety the repo cannot currently express and should warn about: channel-independent and channel-flattened produce the *same token count* but very different attention cost (`V` sequences of length `T` → `V·T²`; one sequence of length `V·T` → `V²·T²`). Since v1 deliberately does not model attention FLOPs, a warning is the right instrument.

### 1.3 Scaling laws: what actually exists

**A power law in parameters, data, and compute exists and is well replicated. A compute-optimal allocation law exists in exactly one paper and is sublinear.**

| Finding | Value | Source | Confidence |
|---|---|---|---|
| Power-law scaling in `N`, `D`, `C` for decoder-only TSFMs over 5 orders of magnitude | functional forms `L(N) ≈ (N_c/N)^α_N`, `L(C) ≈ (C_c/C)^α_C`, `L(D) ≈ (D_c/D)^α_D` | [Edwards et al., arXiv:2405.13867](https://arxiv.org/abs/2405.13867) | high |
| **Compute-optimal data/parameter relation** | **`D ∝ N^(α_N/α_D) ≈ N^0.8`** — "doubling the model size requires roughly 1.7× more data" | [arXiv:2410.12360](https://arxiv.org/html/2410.12360v3) | medium |
| Compute convention used in that derivation | **`C = 6·N·B·S`** (Kaplan convention — identical to the repo's `6ND`) | same | high |
| Numeric values of `α_N`, `α_D`, `α_C` | **not published in consolidated form.** The paper reports *relative* sensitivities (doubling data → OOD MAPE ×0.90, ID MAPE ×0.97) but no exponent table | same | — (do not invent) |
| Encoder-only vs decoder-only scalability | "similar scalability"; encoder-only has a "marginally higher power-law exponent on ID data"; nearly identical OOD. Architectural details (aspect ratio, head count) have "minimal effect over broad ranges" | [arXiv:2410.12360](https://arxiv.org/html/2410.12360v3), [arXiv:2405.13867](https://arxiv.org/abs/2405.13867) | medium |
| Lookback horizon is a third axis | longer lookback can *hurt*; optimal lookback **grows with dataset size** | [arXiv:2405.15124](https://arxiv.org/abs/2405.15124) | medium |
| MoE breaks parameter-count-as-capacity | Moirai-MoE's **11M activated** params beat Moirai's **311M dense** by 3–14% CRPS. Time-MoE sparse vs compute-matched dense: **78% training-cost reduction, 39% inference-cost reduction** | [arXiv:2410.10469](https://arxiv.org/pdf/2410.10469), [arXiv:2409.16040](https://arxiv.org/html/2409.16040v2) | medium |
| **No universal compute-optimal TSFM law** | "the right size is the smallest size whose curve has bent on your data" | [TSFM.ai](https://tsfm.ai/blog/tsfm-scaling-laws-2026) (aggregator; flagged) | — |

**Published tokens-per-parameter ratios — the reason TPP must not be a default:**

| Model | Params | Corpus | TPP |
|---|---|---|---|
| Lag-Llama | ~2.4M (derived) | 352M windows | **~147** |
| Granite TTM | 1–5M | ~1B samples | **200–1,000** (per-sample; ×8 as patch tokens) |
| Chronos-Large | 710M | 84B tokens (29.5B actually seen) | **41.5** seen / 118 available |
| Moirai-Large | 311M | 27.6B observations | **89** |
| TimesFM 2.5 | 200M | ~100B points | **500** |
| Time-MoE-ultra | 2.4B / 1.1B active | ~400B points seen | **167 total / 364 active** |
| Xihe-max | 1.5B | 325B points | **217** |
| Timer-S1 | 8.3B / 0.75B active | ~1T points | **~1,300 active** |
| Toto 2.0-2.5B | 2.5B | 5.04T points | **~2,000** |

Range: **~40 to ~2,000**, i.e. nearly two orders of magnitude, with every single entry *above* Chinchilla-20. (An earlier draft of this table listed Lag-Llama at TPP 1.8 on the strength of a survey's "200 million parameter" claim; the paper itself specifies 8 layers and `d_model = 144`, so the real figure is ~2.4M params and TPP ≈ 147. The aggregator was wrong — corrected here, and a caution about this class of error.)

**Recommend: no TPP input in the time-series pane at all** — the pane already derives `D` from domain geometry, which is the correct design. Surface TPP as a *derived diagnostic* with the `D ∝ N^0.8` curve as a reference band, and note that **TSFMs are universally over-trained relative to Chinchilla**: the lowest observed ratio is ~40, the highest ~2,000.

### 1.4 FLOP accounting: where `6ND` needs adjustment

The repo uses `baseFlops = architectureFactor × N × effectiveTokens` (`time-series-adapter.ts:69`), `architectureFactor` default 6. Four corrections, in descending order of magnitude:

**(a) Encoder-decoder double-counting — ~2.0–2.2×, the largest systematic error.**
Chronos is a T5. The encoder sees 512 tokens; the decoder sees 64. Naive `6 × N_total × (512+64)` charges every parameter for all 576 tokens. Correct form:

```
FLOPs = 6 × (N_encoder × L_in + N_decoder × L_out) × batch × steps
```

Worked for Chronos-Large (710M, 200K steps, batch 256):

| Accounting | Total FLOPs | Hours on 8× A100 @ MFU 0.3 |
|---|---|---|
| naive `6 × N_total × 576` | **1.26e20** | 46.6 |
| split, `N_enc` = 50% | 6.28e19 | 23.3 |
| split, `N_enc` = 43% (T5 cross-attn weighting) | **5.60e19** | **20.8** |

**Ratio 2.0–2.2×.** Recommend an `encoder_decoder` architecture type that takes `inputTokens` / `outputTokens` and an encoder parameter fraction (default 0.45, `medium` confidence).

**(b) MoE — compute on active, memory on total.** Same split as the LLM pane. Moirai-MoE: 11M active / ~117M total. Time-MoE-ultra: 1.1B active / 2.4B total. The June doc deferred MoE ([decision #3](../decisions/2026-06-19-model-family-extension-decisions.md)); two of the strongest 2026 TSFMs are MoE, so this is now a real gap.

**(c) Recurrent / linear-attention architectures.** TiRex-2 (xLSTM) is linear in sequence length with constant-cost state updates. `6ND` is still the right first-order model, but the long-sequence attention warnings are false positives.

**(d) Encoder-only with a heavy output head.** Chronos-2 emits 21 quantiles and TimesFM 2.5 adds a 30M-param quantile head; Moirai 2.0 emits 9 quantiles with multi-token prediction. For small backbones the head is a non-trivial parameter fraction and is applied once per *forecast*, not per token. Below ~50M params this matters; above it, it does not.

### 1.5 The small-model MFU collapse (the most actionable time-series finding)

Back-solving achieved FLOP/s from the three published training budgets:

| Run | Params | Tokens `D` | `6ND` | GPU-seconds | Achieved FLOP/s/GPU | Implied MFU vs A100 (312 TF BF16) |
|---|---|---|---|---|---|---|
| **Granite TTM R1** | 1e6 | 8.0e9 (1B samples × 8 patches) | 4.8e16 | 6 × 27 h = **5.83e5** | **8.2e10** | **0.00026** |
| **Chronos-Large** | 7.1e8 | 2.95e10 | 1.26e20 (naive) | 8 × ? | — | back-solve gives MFU 0.134 at 46.6 h, 0.30 at 20.8 h with split accounting |
| **Time-MoE-ultra** | 1.1e9 active | 4.0e11 | 2.64e21 | 128 × ? | — | at MFU 0.3 → **61 hours** on 128× A100 (plausible; not independently confirmed) |

The TTM row is not a rounding error — it is a 1,100× discrepancy, and it is *expected*: a 1M-parameter model cannot saturate a tensor core. Recommended MFU ladder, as an explicit, editable default keyed on parameter count:

```
MFU defaults for time-series foundation model pretraining:
  N < 10M            0.005   (confidence: low;    anchored on TTM's 0.00026 plus headroom)
  10M ≤ N < 100M     0.08    (confidence: low)
  100M ≤ N < 1B      0.25    (confidence: medium; Chronos back-solve)
  N ≥ 1B             0.30    (confidence: medium; LLM-regime assumption)
```

The 10M–100M band is **interpolated, not measured** — no published run in that range reports wall-clock. Mark it `low` and say so. The honest headline for the pane: *"below ~100M parameters, your estimate is a data-pipeline estimate, not a FLOP estimate."*

### 1.6 Where the repo's golden case sits

The shipped golden case (`time-series-adapter.test.ts:34`) — 1B params, 10M series × 1,000 timesteps, 4 variates, lookback 256, horizon 64, stride 64, patch 16, channel-expanded — yields 7.04e9 tokens and 4.224e19 FLOPs. Checks:

- **Arithmetic: correct.** 11 windows/series, 64 tokens/window, 7.04e9 tokens, 4.224e19 FLOPs all verify.
- **Plausibility: the implied TPP is 7.04**, below every published TSFM except Lag-Llama. Against `D ∝ N^0.8`, a 1B-param model calibrated to Chronos-Large's point would want ~5e10 tokens — **7× more**. The golden case is a legitimate *unit test* but a misleading *default scenario*.
- **Scale: 4.2e19 FLOPs is ~1/3 of Chronos-Large's entire pretraining run** at 1.4× its parameter count. It is the right order of magnitude for a real TSFM pretraining job, which is the useful thing about it.

---

## 2. Tabular foundation models

### 2.1 Model catalog (transcribable into `src/data/tabular-models.json`)

| Model | Released | Params | Architecture | Max rows | Max features | Pretraining corpus | Training compute | License | Source |
|---|---|---|---|---|---|---|---|---|---|
| TabPFN v1 | 2022 | ~7M | row-sequence transformer, ICL | 1,000 | 100 | synthetic SCM/BNN priors | — | Apache-2.0 | [arXiv:2207.01848](https://arxiv.org/pdf/2207.01848) |
| **TabPFN v2** | Jan 2025 (Nature) | **11M** | **two-way / alternating attention** over rows *and* features | 10,000 | 500 | **~100–130M synthetic datasets** | — (not published) | Prior Labs License | [Nature](https://www.nature.com/articles/s41586-024-08328-6), [Hutter](https://x.com/FrankRHutter/status/1878864024055533671) |
| **TabPFN-2.5** | Nov 2025 | **not published** | alternating attention, **24 layers (clf) / 18 (reg)**, feature group size 3, +64 "thinking" rows | 50,000 design / **100,000** validated | **2,000** | synthetic | — | TabPFN-2.5 License v1.1 | [arXiv:2511.08667](https://arxiv.org/html/2511.08667v2) |
| TabPFN-2.6 | 2026 | — | — | 100,000 | 2,000 | synthetic | — | TabPFN-2.6 License | [Prior Labs docs](https://docs.priorlabs.ai/models) |
| **TabPFN-3** | **12 May 2026** | **53M (clf) / 58M (reg)** | 3 stages: inducing-point attention → non-causal attention → MHSA + multi-query cross-attn; row-chunking; FlashAttention-3 | **1,000,000** | 200/estimator (2,000 w/ subsampling), ≤160 classes | **>8 trillion tokens**, SCM prior, exclusively synthetic | — (not published) | TabPFN-3 License v1.0 | [arXiv:2605.13986](https://arxiv.org/html/2605.13986v2) |
| **TabPFN-3.5** (+Fast/Plus/Thinking) | **15 Sep 2026** | — | — | **1,000,000** | **20,000** | synthetic | — | TabPFN-3.5 License v1.0 | [report](https://priorlabs.ai/technical-reports/tabpfn-3-5) (cites arXiv:2609.17895 — **not independently verified here**) |
| TabICL | Feb 2025 | — | two-stage: column-then-row ICL | 500,000 | 500 | synthetic, up to 60K samples/task | — | BSD-3 | [arXiv:2502.05564](https://arxiv.org/pdf/2502.05564) |
| **TabICLv2** | **Feb 2026** | — | scalable softmax, **Muon** optimizer, new synthetic engine | **million-scale under 50 GB VRAM** | — | synthetic (engine released) | "only moderate pretraining compute" | open (code released) | [arXiv:2602.11139](https://arxiv.org/abs/2602.11139) |
| TabDPT | ICLR 2025 | — | ICL on **real** data + retrieval | — | — | real tabular corpora | — | open | [arXiv:2410.18164](https://arxiv.org/html/2410.18164) |
| **Mitra** (AWS AutoGluon) | NeurIPS 2025 | **72M** | mixed synthetic priors (SCM + tree generators) | strongest <5K rows | ~100 | mixed synthetic | — | Apache-2.0 | [arXiv:2510.21204](https://arxiv.org/pdf/2510.21204) |
| LimiX | 2025–26 | — | unified predict / impute / causal | — | — | — | — | — | — |
| **LimiX-2** | **Sep 2026** | **400M** | contextual mechanism network | — | 2–10 classes + regression | — | — | — | [arXiv:2609.17488](https://arxiv.org/pdf/2609.17488) |
| **LimiX-2M** | **Jun 2026** | **2M** | RaBEL + reordered sample-attn → FFN → feature-attn; fixes low-rank collapse | — | — | — | — | — | [arXiv:2606.04485](https://arxiv.org/html/2606.04485) |
| **TabSwift** | **Jun 2026** | — | row-wise attention only (efficiency-first) | — | — | — | — | — | [arXiv:2606.07345](https://arxiv.org/html/2606.07345) |
| nanoTabPFN | Nov 2025 | small | educational reimplementation; the **speedrun** target | — | — | synthetic | **0.92 min on 1× L40S** to hit TabArena target | open | [arXiv:2511.03634](https://arxiv.org/pdf/2511.03634), [arXiv:2606.03681](https://arxiv.org/abs/2606.03681) |
| TabuLa-8B, CARTE, XTab | 2024–25 | 8B / — / — | LLM-tokenized rows / graph-attentional / cross-table | — | — | — | — | varies | — |

**The headline catalog fact: tabular foundation models are small.** 2M to 400M parameters, with 11M–58M the dominant band. Nothing in this family is an LLM-scale object, and TabPFN-2.5's parameter count is simply **not disclosed** — do not guess it.

### 2.2 The ICL regime: cost is inference-side, and it is quadratic in rows

The published complexity is explicit:

```
TabPFN-2.5 cost  ∝  r² · min(c, 500)  +  r · min(c, 500)²
                     ^^^^^^^^^^^^^^^^     ^^^^^^^^^^^^^^^^
                     row attention        feature attention
```
— [arXiv:2511.08667](https://arxiv.org/html/2511.08667v2). The `min(c, 500)` cap is the feature-subsampling ceiling. Time scales **linearly in test rows** but **quadratically in support rows**.

Calibrating that against the one published timing anchor (50,000 rows × 500 columns ⇒ **23–30 s on one H100**, midpoint 26.5 s) gives `k ≈ 4.8e10` cost-units/s:

| Support rows × cols | Cost units | Predicted H100 seconds |
|---|---|---|
| 1,000 × 20 | 2.0e7 | < 0.01 |
| 10,000 × 100 | 1.0e10 | 0.21 |
| **50,000 × 500** | **1.26e12** | **26.5 (anchor)** |
| 100,000 × 500 | 5.0e12 | 105 |
| 200,000 × 500 | 2.0e13 | 421 |
| 1,000,000 × 500 | 5.0e14 | 10,500 (2.9 h) |

Cross-check against TabPFN-3, whose architecture changes the exponent: **1,000,000 rows × 10 features = ~107 s cold fit+predict on H100**, versus 210 s predicted by the v2.5 quadratic form — i.e. TabPFN-3's inducing-point stage really does beat the quadratic. Other TabPFN-3 anchors: 50,000 × 10 ⇒ ~5–10 ms, 50,000 × 100 ⇒ ~10–30 ms (warm, with KV cache), and **1.5–1.7× speedup from FlashAttention-3 on H100**.

**Memory, not time, is the feasibility wall.** TabPFN-3's KV cache is **7 GiB per estimator at 1M rows**; row-chunking (enabled when `n_train + n_test > 2048`) gives a **~5× peak-memory reduction**. Practitioner reports put TabPFN v2 at ~8 GB VRAM for small data and ~16 GB for the 10K×500 ceiling, with OOM around 3,000 samples at batch size 1 on a 48 GB card in one unbatched configuration ([GH issue #250](https://github.com/PriorLabs/TabPFN/issues/250)). TabPFN-2.5 **does not run on CPU above 1,000 rows by default**. TabICLv2 targets "million-scale under 50 GB VRAM."

**This is what the pane should compute.** Rows × columns in, seconds and GB out, with a GPU-fits verdict — exactly the memory-feasibility-ladder shape the fine-tuning research recommended for its own tab.

### 2.3 Pretraining compute: real but small, and no one needs it

If the pane keeps a pretraining mode, these are the anchors:

| Run | Params | Corpus | Compute | Source |
|---|---|---|---|---|
| **nanoTabPFN speedrun** | small | 22× fewer synthetic datasets than baseline | **0.92 min on 1× L40S** (from 74.32 min) | [arXiv:2606.03681](https://arxiv.org/abs/2606.03681) |
| TabPFN-3 | 53M | **>8e12 tokens** | not published; `6ND` ⇒ **2.54e21 FLOPs** ≈ **298 h on 8× H100 @ MFU 0.3** (derived, not confirmed) | [arXiv:2605.13986](https://arxiv.org/html/2605.13986v2) |
| TabPFN v2 | 11M | ~100–130M synthetic datasets | not published | [Nature](https://www.nature.com/articles/s41586-024-08328-6) |

Note the enormous TPP: **8e12 / 53e6 ≈ 151,000 tokens per parameter**. Tabular FMs are the most over-trained family in existence, by a factor of ~7,500 over Chinchilla. This is coherent — synthetic data is free, so there is no reason to stop.

The repo's golden cases (`tabular-adapter.test.ts:31,38`) give 6.14e17 FLOPs (row mode) and 6.14e19 (cell mode): **0.58 h and 57.5 h on a single H100 at MFU 0.3**. Both are single-GPU jobs. That is directionally right and worth saying out loud in the UI.

### 2.4 Scaling laws for tabular FMs

| Finding | Value | Source | Confidence |
|---|---|---|---|
| Power laws in model size and pretraining cells | "Increasing model or pre-training data size (number of cells) leads to consistent improvements predictable by power laws" — **the data unit is the cell** | [TabDPT, NeurIPS 2025](https://arxiv.org/html/2410.18164) | medium |
| Fitted exponents | **not published** | — | — (do not invent) |
| Depth saturation | gains "saturate beyond 12 layers" | via [MDPI critical review](https://www.mdpi.com/2504-4990/8/8/244) (403 on fetch; via search snippet — flagged) | medium-low |
| Synthetic-task saturation | diminishing returns after **~18K steps × 2,048 datasets/step ≈ 37M unique synthetic datasets** | same search snippet — flagged | low |
| Real vs synthetic | TabDPT shows real data carries useful signal; Toto 2.0's analogous finding in time series was that the optimal *pretraining* mix excluded public data entirely while the optimal *fine-tuning* mix was 45% public | [TabDPT](https://arxiv.org/html/2410.18164), [Datadog](https://www.datadoghq.com/blog/ai/toto-2/) | medium-low |
| Compute-optimal law | **none exists** | — | — |

**Honest statement for the doc and the UI: there is no compute-optimal scaling law for tabular foundation models.** The cell is the agreed data unit; the exponents are unpublished. The repo's `low`/`medium-low` confidence labels (`tabular-adapter.ts:9-16`) are correctly calibrated, and that is the right call.

### 2.5 Benchmark results on compute (TabArena and the TabPFN reports)

[TabArena](https://arxiv.org/html/2506.16791v2) — 51 curated datasets, 500–250,000 train samples:

- Protocol: **200 random hyperparameter configurations**, **8-fold inner CV**, outer = 10×3-fold repeated CV (<2,500 samples) or 3 repeats. That is **1,600 fits per model per outer fold**, 14,400–48,000 total.
- Tuned+ensembled normalized scores: RealMLP 0.537 > TabM 0.502 > **TabPFNv2 0.476** > LightGBM 0.452 > CatBoost 0.433.
- Per-1,000-sample times: RealMLP train 6,564.71 s / infer 10.26 s; TabM 3,285.87 / 1.47; CatBoost 1,658.41 / **0.65**; LightGBM 416.98 / 2.64.
- 2026 leaderboard state: **TabPFN-2.5** then **TabPFN-3** then **TabPFN-3.5** take first place, with TabPFN-3 gaining **+72 Elo over 2.5** and TabPFN-3.5 ranking first on TabArena *and* BeyondArena.

Compute comparisons from the TabPFN reports:
- TabPFN-2.5 default: **100% win rate vs default XGBoost** on ≤10K rows / 500 features; **87%** up to 100K × 2K. Matches AutoGluon 1.4 extreme (a **4-hour** tuned ensemble) in a forward pass.
- TabPFN-3: beats **8-hour-tuned GBDT baselines** on datasets up to 1M rows; **up to 20× faster than TabPFN-2.5**; TabPFN-3-Plus (Thinking) beats AutoGluon 1.5 extreme "in less than a tenth of its runtime."
- TabICL: 0.6–6 s per dataset (fit+predict) vs tuned XGBoost 0.7–27.2 s over 25 configurations.

**The compute comparison is real but asymmetric: TFMs move the cost from a training-time search to an inference-time forward pass.** That is the economically interesting claim and the pane should make it.

---

## 3. Classical tabular / GBDT

### 3.1 What the cost actually is

Histogram-based GBDT training cost per boosting round is, to first order:

```
per_round_work ≈ rows × features × depth × (1 − histogram_subtraction_saving)
```

At each tree level the union of nodes covers all rows, so building that level's histograms costs `rows × features` bin-increments; the subtraction trick means only the smaller sibling is built, saving ~40–50%. Over `depth` levels, cost is **linear in depth**. `max_bin` barely affects the increment count but strongly affects histogram footprint and GPU efficiency.

Order-of-magnitude sanity check: LightGBM on Higgs (10.5M × 28), `num_leaves=31` (~5 effective levels), 500 rounds ⇒ ~7.4e11 bin-increments in 130 s on 56 threads ⇒ **~5.7e9 increments/s**, or ~1e8 per core-second. At ~15 ops per increment (bin gather + two atomic adds) and ~3 GHz that is plausible. **The physical model holds.**

The repo's work-unit formula (`classical-tabular-adapter.ts:44-45`) is:

```
totalWorkUnits = rows × columns × boostingRounds × cvFolds × hyperparameterTrials
```

**`maxDepth` and `bins` are declared in the config (`src/engine/types.ts:134-135`) and never read** — they are not in the destructure at `classical-tabular-adapter.ts:33-40`. So the formula silently assumes a fixed depth, and the fixed depth it was implicitly calibrated against is unknown.

### 3.2 Measured throughput — the data that adjudicates `throughputCoefficient`

**Primary source A: [GBM-Benchmarks](https://github.com/RAMitchell/GBM-Benchmarks) (XGBoost maintainer's suite).** AWS p3.16xlarge (8× V100-16GB, 64 vCPU Xeon E5-2686 v4), **500 boosting rounds, `max_depth = 6`**, full datasets, run 7 Jun 2018. Dataset dimensions from [ml_dataset_loader](https://github.com/RAMitchell/ml_dataset_loader). Throughput = `rows × cols × 500 / seconds`, i.e. **exactly the repo's work unit**.

| Dataset | Rows | Cols | Work units | xgb-cpu | xgb-gpu | lgbm-cpu | lgbm-gpu | cat-cpu | cat-gpu |
|---|---|---|---|---|---|---|---|---|---|
| YearPredictionMSD | 515,345 | 90 | 2.32e10 | 5.84e7 | 6.77e8 | 6.08e8 | 2.90e8 | 6.02e8 | 2.37e9 |
| CoverType | 581,012 | 54 | 1.57e10 | 3.38e7 | 1.52e8 | 1.87e8 | 1.24e8 | 5.44e7 | N/A |
| Bosch | 1,183,747 | 968 | 5.73e11 | 7.61e8 | **1.78e10** | 2.29e9 | 1.18e9 | 2.36e9 | N/A |
| Synthetic | 10,000,000 | 100 | 5.00e11 | 8.85e8 | **1.30e10** | 1.19e9 | 8.20e8 | 1.15e9 | **1.41e10** |
| Higgs | 11,000,000 | 28 | 1.54e11 | 3.28e8 | **4.52e9** | 5.02e8 | 2.91e8 | 3.88e8 | **5.11e9** |
| Airline | 115,069,017 | 13 | 7.48e11 | 3.84e8 | **5.16e9** | 8.17e8 | 1.22e9 | 2.54e8 | 2.47e9 |

*(units: work units / second)*

**Geometric means over the four datasets with ≥1M rows:**

| Library | GPU (V100) | CPU (64 vCPU) | Speedup |
|---|---|---|---|
| XGBoost `hist` | **8.57e9** | 5.40e8 | **13.4×** (range 4.5–23.4×) |
| CatBoost | **5.62e9** | 7.19e8 | **9.7×** (range 3.9–13.2×) |
| LightGBM (OpenCL backend) | **7.66e8** | 1.03e9 | **0.69×** — *GPU is slower* |

**Primary source B: [LightGBM official single-machine experiments](https://lightgbm.readthedocs.io/en/latest/Experiments.html).** 2× E5-2690 v4 (28 cores), 448 GB, 500 iterations, CPU only.

| Dataset | Rows | Cols | Seconds | Work units/s |
|---|---|---|---|---|
| Higgs | 10,500,000 | 28 | 130.094 | **1.13e9** |
| MS LTR | 2,270,296 | 137 | 70.417 | **2.21e9** |
| Yahoo LTR | 473,134 | 700 | 76.229 | **2.17e9** |
| Expo | 11,000,000 | 700 | 62.607 | 6.15e10 ⚠️ |
| Allstate | 13,184,290 | 4,228 | 148.231 | 1.88e11 ⚠️ |

The two flagged rows are **the work-unit model breaking down on sparse/one-hot data**: Expo and Allstate have 700 and 4,228 one-hot columns that LightGBM's Exclusive Feature Bundling collapses, so `rows × cols` over-counts by ~100×. On the three dense datasets the geometric mean is **1.76e9 work units/s on CPU**.

**Primary source C: [CatBoost GPU-vs-CPU benchmarks](https://github.com/catboost/catboost/tree/master/catboost/benchmarks/gpu_vs_cpu_training_speed).** Epsilon (400K × 2,000, dense numeric):

| Hardware | 128 bins | Speedup | 32 bins | Speedup |
|---|---|---|---|---|
| CPU, 32 threads | 713 s | 1.00× | 653 s | 1.00× |
| Tesla K40 | 547 s | 1.30× | 248 s | 2.63× |
| GTX 1080 | 194 s | 3.68× | 120 s | 5.44× |
| P40 | 162 s | 4.40× | 91 s | 7.18× |
| GTX 1080 Ti | 145 s | 4.92× | 88 s | 7.42× |
| P100-PCI | 127 s | 5.61× | 70 s | 9.33× |
| V100-PCI | **77 s** | **9.26×** | **49 s** | **13.33×** |

Criteo (36M rows, 26 categorical + 13 numeric), 128 bins: CPU 1,060 s → V100 69.8 s = **15×**. CatBoost's own mean tree-construction comparison on Epsilon (8,000 trees): **CatBoost 17.9 ms, LightGBM 40 ms, XGBoost 488 ms** — the iteration count behind the GPU table is not stated, so that row is not usable as a throughput anchor.

**Primary source D: [cuML accelerator benchmarks](https://docs.nvidia.com/cuml/latest/cuml-accel/benchmarks/)** (RTX PRO 6000 Blackwell vs Threadripper PRO 7975WX, 32 cores) — Random Forest, 100 estimators:

| Rows | Cols | CPU fit | GPU fit | Speedup | GPU work units/s |
|---|---|---|---|---|---|
| 50,000 | 128 | 439.4 ms | 140.3 ms | 3.13× | 4.56e9 |
| 1,562,500 | 16 | 7.81 s | 778.6 ms | 10.0× | 3.21e9 |
| 250,000 | 128 | 2.25 s | 284.9 ms | 7.90× | 1.12e10 |
| 244,140 | 1,024 | 7.95 s | 791.2 ms | 10.0× | 3.16e10 |
| **1,953,125** | **128** | **28.6 s** | **1.12 s** | **25.6×** | **2.23e10** |

Broader cuML reports put RF `cuml.accel` speedups at **15–80×**, rising with dataset size (≈8× at 1K rows, ≈24× at 10M rows) — but the baseline is scikit-learn, which is a much weaker CPU implementation than LightGBM.

**Modern-hardware adjustment.** All GBDT GPU anchors above are V100-era (2018). XGBoost 3.0 reports "roughly 2× speedups" in GPU histogram building on mostly-dense data, and H100/Blackwell have ~3.5× the memory bandwidth of V100. GBDT histogram building is bandwidth-bound, so **2–3× above the V100 numbers is the right expectation for H100**, giving XGBoost GPU ≈ 2e10 work units/s. That is extrapolation, not measurement — mark it `medium-low`.

### 3.3 Recommended `throughputCoefficient` defaults

| Algorithm | Repo GPU default (`classical-tabular-adapter.ts:8-14`) | Measured V100 (≥1M rows, depth 6) | Recommended GPU default | Recommended CPU default | Error in repo |
|---|---|---|---|---|---|
| XGBoost | 3e8 | **8.6e9** | **8e9** | **5e8** | GPU **29× low**; CPU (3e7) **17× low** |
| LightGBM | 5e8 | 7.7e8 (OpenCL, often < CPU) | **1.5e9** (CUDA backend) | **1.5e9** | GPU ~3× low; CPU (5e7) **30× low** |
| CatBoost | 4e8 | **5.6e9** | **5e9** | **7e8** | GPU **14× low**; CPU (4e7) **18× low** |
| Random Forest | 2e8 | 3e9–3e10 (cuML, 100 trees) | **5e9** | **1e9** | GPU **25× low**; CPU (2e7) **50× low** |

Two separate defects, and they compound in opposite directions:

1. **Absolute level: all four GPU defaults are 3–29× too pessimistic.** The repo's default scenario (1M × 100 × 1,000 rounds = 1e11 work units) reports **200 s** on LightGBM GPU; the measured rate puts it at **~12–60 s**.
2. **`CPU_THROUGHPUT_FACTOR = 0.1` (`classical-tabular-adapter.ts:16`) is defensible as a *ratio* for XGBoost and CatBoost on large dense data** (measured 13.4× and 9.7×) — but it is badly wrong for LightGBM (measured **0.69×**, GPU *slower*) and wrong for all libraries below ~1M rows. Combined with the low GPU base, the repo's CPU numbers end up **17–50× too pessimistic**, which is the worse of the two errors because CPU is the common case.

The warning text at `classical-tabular-adapter.ts:81` — "GPU implementations of histogram-based GBDTs are typically 5-10x faster" — is accurate for XGBoost and CatBoost on large dense data, and **false for LightGBM and for small data**. It should be conditioned on algorithm and row count.

Also recommend making `maxDepth` live:

```
workUnits = rows × columns × boostingRounds × (maxDepth / 6) × cvFolds × hyperparameterTrials
```

with coefficients re-expressed at the depth-6 calibration point. This costs nothing, uses an input the UI already collects, and removes a silent assumption.

### 3.4 When GPU does *not* help

| Condition | Effect | Source |
|---|---|---|
| Small data (<~1M rows) | transfer + kernel-launch overhead dominates; **100K rows can be 2–3× slower on GPU** | [LightGBM GPU Performance](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html) |
| LightGBM OpenCL backend | **slower than CPU on 5 of 6** reference datasets | GBM-Benchmarks (table §3.2) |
| High `max_bin` | GPU is much faster at 63 bins than 255; CPU is nearly flat in `max_bin`. CatBoost Epsilon: 9.26× at 128 bins → **13.33× at 32 bins** | LightGBM docs; CatBoost benchmarks |
| High-cardinality categoricals | one-hot encoding destroys the dense-GPU advantage; use native categorical handling or target encoding instead. Target encoding is preferred at high cardinality with deep trees | [LightGBM docs](https://lightgbm.readthedocs.io/en/stable/Features.html), [arXiv:2307.02071](https://arxiv.org/pdf/2307.02071) |
| Disk-backed external memory | "using the disk is not recommended as it's likely to make the GPU slower than the CPU" | [XGBoost external memory](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) |
| PCIe-attached GPU on huge data | PCIe4 x16 ≈ 24 GB/s, PCIe5 ≈ 42 GB/s vs NVLink-C2C 900 GB/s. XGBoost external memory effectively needs NVLink-C2C | same |
| Budget/laptop GPUs | ~0.9–1.2× on an RTX 3050 (bandwidth-limited); 2–4× on A100/H100 | aggregator (theneuralbase) — flagged, low confidence |

### 3.5 Memory: VRAM is (almost) never the constraint

XGBoost GPU memory has two components ([docs](https://xgboost.readthedocs.io/en/stable/gpu/index.html)):

```
M_data       ≈ rows × features × bits_per_bin / 8        (compressed quantized ELLPACK,
                                                          ~1/4 of float32 CSR)
M_working    ∝ rows                                       (gradients, row→node map, per-row stats)
M_histograms ∝ bins × features × nodes_cached             (previous nodes retained until a
                                                          memory threshold, then dropped)
```

LightGBM stores binned data as `uint8` when `max_bin ≤ 255` (and supports 4-bit), so the data matrix is roughly **`rows × features` bytes**. A 1M × 100 dataset is ~100 MB binned. One practitioner report: 120M rows reduced from **~120 GB to ~38 GB** by downcasting, native categoricals, lower `max_bin`, and lower `num_leaves`.

And since XGBoost 3.0, external memory makes a **1 TB dataset (2^29 rows × 512 features × float32)** trainable on **one GH200** (80 GB HBM3 + 480 GB LPDDR5X, 900 GB/s NVLink-C2C), with 128 batches of 8 GB and `max_quantile_batches=32`. NVIDIA reports **up to 8× faster than a 112-core dual-socket Xeon 8480CL**, "a 1 TB model trained in minutes," and for one customer pipeline **16× end-to-end speedup and 94% TCO reduction** (no dollar figures published).

**Recommendation: do not add a VRAM-feasibility ladder to the GBDT pane.** Add an informational note that a binned matrix is ~`rows × features` bytes, and that beyond GPU memory the answer is external memory on one coherent-memory device, not more GPUs. If the pane needs a memory warning at all, the threshold is *host* RAM, not VRAM.

### 3.6 CV × hyperparameter search: the multiplier is the whole cost

The repo warns when `cvFolds × hyperparameterTrials > 50` (`classical-tabular-adapter.ts:73`). Against published practice that threshold is far too low to be informative:

| Protocol | Multiplier | Source |
|---|---|---|
| **TabArena** (the current reference benchmark) | **200 configs × 8 inner folds = 1,600 fits** per model per outer fold; 14,400 (large) to 48,000 (small) in total | [arXiv:2506.16791](https://arxiv.org/html/2506.16791v2) |
| Common published tabular protocol | 100 trials × 5 folds = 500 | multiple (see §3.6 notes) |
| Typical applied Optuna run | 20–100 trials; 100 standard in benchmarks (often 20 random + 80 TPE) | [xgboosting.com](https://xgboosting.com/xgboost-hyperparameter-optimization-with-optuna/) and survey of papers |
| Light tuning | 5–25 trials × 5 folds = 25–125 | TabICL comparison used 25 XGBoost configurations |

**Published guidance on a diminishing-returns point does not exist in a citable form.** The practical literature converges on **50–100 trials** as the band where TPE-style samplers have extracted most of the gain, and recommends pruning (Hyperband / early stopping) over more trials. Mark any specific number `medium-low`.

Recommended changes: raise the warning threshold to **>500** (where the search genuinely dominates and pruning is the answer), add a second informational note at **>100**, default `hyperparameterTrials` to **50** and `cvFolds` to **5** (currently both 1 in the test fixture), and surface the multiplier as a first-class output — because `5 × 50 = 250×` turns the repo's 12–60 s single fit into a 50-minute to 4-hour job, which *is* the real wall-clock answer and the only place a GPU starts to matter for mid-size data.

### 3.7 Do GBDTs still beat deep learning and foundation models in 2026?

The answer changed during 2026, and it changed in a way that matters for which pane a user should open.

**What is now established:**
- On TabArena with full tuning and ensembling, **the best deep models match or beat GBDTs**: RealMLP 0.537 and TabM 0.502 vs LightGBM 0.452 and CatBoost 0.433. This directly contradicts the long-standing "GBDTs universally dominate" claim.
- **The best single models by accuracy on TabArena are now tabular foundation models**, not GBDTs — TabPFN-2.5, then TabPFN-3 (+72 Elo), then TabPFN-3.5 (first on TabArena *and* BeyondArena, and reported to beat the winning Otto Kaggle solution with default settings).
- TabPFN-3 beats **8-hour-tuned GBDT baselines** on datasets up to **1M rows** — the size ceiling that used to be the GBDT redoubt has moved.

**What is still true:**
- GBDTs win decisively on **cost per prediction and operational simplicity**. TabArena per-1,000-sample inference: CatBoost **0.65 s**, LightGBM 2.64 s, RealMLP 10.26 s. CatBoost and EBM "offer strong performance with relatively low inference costs."
- **Latency is the hard wall.** "A fraud scorer needs to answer in under 2 ms. The best tabular foundation models take **151–1,275 ms on GPU**." Distilling a TFM into a CPU XGBoost student recovers **96.5% of teacher AUC at 1.9 ms on CPU** — a **38–860× speedup** — and still beats a tuned CatBoost baseline (Wilcoxon p = 0.0008, 51% win rate) ([arXiv:2605.18654](https://arxiv.org/abs/2605.18654)).
- GBDTs need **no GPU**, no license, no API, and no 7 GiB KV cache.
- On high-cardinality categoricals and very wide sparse data, GBDT native categorical handling remains the pragmatic choice.

**Decision rule the calculator can state:**

```
rows ≲ 100K and you have a GPU        → tabular foundation model (TabPFN-3.5 / TabICLv2);
                                        a forward pass beats hours of GBDT tuning
rows ≳ 1M, or latency budget < 50 ms,
  or no GPU, or hard licensing limits → GBDT (CPU is fine; GPU if rows ≳ 1M and dense)
best accuracy, cost no object          → TFM, or TFM distilled into a GBDT student for serving
```

---

## 4. Discrepancies with the current implementation

| # | Quantity | Repo's current value | Researched current value | Impact | Recommended action |
|---|---|---|---|---|---|
| 1 | GBDT GPU `throughputCoefficient` defaults (`classical-tabular-adapter.ts:8-14`) | lightgbm 5e8, xgboost 3e8, catboost 4e8, rf 2e8 | XGBoost **8.6e9**, CatBoost **5.6e9**, LightGBM-CUDA ~1.5e9, cuML RF 3e9–3e10 (V100-era, depth 6, ≥1M rows) | **Times are 3–29× too long.** Worst single defect in the three adapters. | Replace with §3.3 table; cite GBM-Benchmarks + LightGBM Experiments + CatBoost benchmarks in the UI |
| 2 | `CPU_THROUGHPUT_FACTOR = 0.1` (`classical-tabular-adapter.ts:16`) | GPU always 10× CPU | 13.4× (XGBoost), 9.7× (CatBoost), **0.69× (LightGBM OpenCL)**, <1× for any library below ~1M rows | Combined with #1, CPU estimates are **17–50× too long** — and CPU is the common case | Make the factor per-algorithm and row-count-dependent; add explicit CPU coefficients rather than a single global ratio |
| 3 | `maxDepth`, `bins` unused | declared (`types.ts:134-135`), never destructured (`classical-tabular-adapter.ts:33-40`) | cost is **linear in depth**; `max_bin` drives GPU efficiency (CatBoost Epsilon 9.26× @128 → 13.33× @32) | Two collected inputs are dead; a silent depth-6 assumption is baked into the coefficients | Multiply work units by `maxDepth / 6`; use `bins` to adjust the GPU factor only |
| 4 | Provenance of the GBDT defaults | comment claims "Benchmark-derived defaults" (`classical-tabular-adapter.ts:6-7`) | **No benchmark source exists.** The spec asked for "required/custom" (`model_family_extensions_functional_spec.md:546`); the numbers appear nowhere in spec, decisions, or plan docs | Comment overstates confidence; nothing to re-derive from | Replace the comment with the real citation + hardware + depth/rounds the number was calibrated at |
| 5 | Sparse / one-hot data | `rows × columns` assumed to be real work | breaks by ~100× on sparse data (Expo 6.15e10, Allstate 1.88e11 vs ~1.8e9 dense) | Massive over-estimate for wide categorical tables | Add a `featureDensity` input or a warning when `columns > 500`; note that EFB/one-hot collapse invalidates the model |
| 6 | CV × HPO warning threshold | `> 50` (`classical-tabular-adapter.ts:73`) | TabArena uses **1,600 fits per model per outer fold** | Warning fires on every realistic configuration → warning fatigue | Raise to >500 for the "dominates" warning; informational note at >100; default trials 50, folds 5 |
| 7 | GPU-speedup warning text | "typically 5-10x faster" (`classical-tabular-adapter.ts:81`) | true for XGBoost/CatBoost on ≥1M dense rows; **false** for LightGBM-OpenCL and for small data | Actively misleading in the two cases where the user most needs guidance | Condition on algorithm and row count; add a "GPU will be *slower*" warning below ~500K rows |
| 8 | `testTimeComputeMultiplier` (`tabular-adapter.ts:28,64`) | scalar multiplier on **pretraining** FLOPs | TabPFN cost is `O(r²·min(c,500) + r·min(c,500)²)` at **inference**; 50K×500 ⇒ 23–30 s on H100; KV cache 7 GiB/estimator @1M rows | Multiplies the wrong base; cannot express quadratic row growth; the quantity users actually care about is unreachable | **Replace with a real inference model** taking support rows, test rows, columns, estimators → seconds + VRAM. Keep the multiplier only for ensemble/"thinking" repeats |
| 9 | Tabular pane's framing | pretraining calculator (`numberOfPretrainingTasks × rowsPerTask`) | ~all users do zero-shot ICL; pretraining is a single-node job (**0.92 min on 1 L40S** for the speedrun target) | Pane answers a question three organizations in the world have | Add an **Inference** sub-mode as the default; keep pretraining as "advanced" |
| 10 | Time-series MFU | inherits shared `mfuByGpuId` default 0.40 | TTM (1M params) achieves **MFU 0.00026**; naive `6ND` under-predicts its 24–30 h run by **~1,100×** | Any model under ~100M params gets a wildly optimistic time | Add the §1.5 parameter-keyed MFU ladder; warn hard below 100M params |
| 11 | Encoder-decoder FLOPs | single `architectureFactor × N × tokens` (`time-series-adapter.ts:69`) | T5-style models charge encoder params for input tokens only and decoder params for output tokens only — **2.0–2.2× difference** for Chronos-Large | 2× over-estimate for the single best-documented TSFM family | Add `encoder_decoder` handling with `inputTokens`/`outputTokens` and an encoder-param fraction (default 0.45) |
| 12 | MoE in time series | deferred ([decision #3](../decisions/2026-06-19-model-family-extension-decisions.md)) | Moirai-MoE (11M active / 32 experts) and Time-MoE-ultra (1.1B active / 2.4B total) are among the strongest 2026 TSFMs; **78% training-cost reduction vs compute-matched dense** | Cannot model two leading architectures | Reuse the LLM pane's active/total split; it is the same machinery |
| 13 | Default tokenization mode | `channel_compressed` ([decision #1](../decisions/2026-06-19-model-family-extension-decisions.md)) | every 2026 frontier TSFM is channel-independent or channel-flattened (= `channel_expanded`); `channel_compressed` is the Mixer/TTM minority | Default under-counts tokens ~`V`× for the models users name | Flip the default to `channel_expanded`; rename modes to `channel_independent` / `channel_mixed`; add `variate_tokens` |
| 14 | Missing tokenization modes | 3 modes (`types.ts:56`) | no mode for Chronos-2 group attention, iTransformer variate tokens, or per-timestep quantization as a named option | Three real designs unexpressible or non-obvious | Add `variate_tokens`; add a `per_timestep_quantized` preset (patch = 1, vocab 4,096) |
| 15 | Tabular `axial` mode | `rowsPerTask + columnsPerTask`, `seqLength = max(rows, cols)` (`tabular-adapter.ts:48-52`) | TabPFN's actual alternating attention is `O(r²c + rc²)` — **multiplicative, not additive** | Under-estimates the dominant family by orders of magnitude | Keep `axial` as the token count, but compute cost from the published complexity formula |
| 16 | Long-sequence warnings in time series | none for recurrent models | TiRex-2 (xLSTM) is **linear** in sequence length | False positives for the recurrent family | Suppress attention warnings when `architectureType` is recurrent/linear |
| 17 | Time-series golden-case default | TPP = 7.04 (`time-series-adapter.test.ts`) | published TSFM TPP spans 1.8–2,000; `D ∝ N^0.8` at Chronos's calibration point wants ~7× more data for 1B params | Shipping default is an under-trained scenario | Keep as a unit test; ship a different *default scenario* nearer a real run |
| 18 | Classical pane GPU count | pane feeds the shared hardware estimator | **1 GPU is the answer whenever GPU is the answer**; 1 TB trains on one GH200; multi-GPU GBDT is niche | Framing invites a question that has no interesting answer | Replace the GPU-count output with a **"GPU: yes / no / doesn't matter" verdict + wall-clock + CPU-alternative time** |

---

## 5. What's new since the June 2026 input doc

`scaling_laws_time_series_tabular_models.md` is dated 2026-06-19 and cites five references, the newest from 2025. Four months on:

### 5.1 What it got right (and should be preserved)

| Claim | Status |
|---|---|
| "Time-series and tabular models do not yet have an equally canonical Chinchilla law" | ✅ **Still exactly true.** No compute-optimal law for either family. The closest is `D ∝ N^0.8` for time series, in one paper |
| "The data unit changes by domain" and must be a first-class input | ✅ **Vindicated.** Chronos (per-timestep) vs Chronos-Bolt (patch 16) vs TTM (patch 64) differ by 64× on the same corpus |
| Channel-compressed vs channel-expanded is the decisive multivariate choice | ✅ Correct, and both modes map onto real architectures (§1.2) |
| Cell- vs row-tokenized tabular differs by ~100× and attention is the wall | ✅ Correct in direction; the published complexity `O(r²c + rc²)` confirms it |
| "GBDTs remain extremely competitive and often dominant for ordinary supervised tabular tasks" | ⚠️ **Half-outdated** — see §5.2 |
| "For classical tabular, the LLM-style FLOPs calculator is the wrong abstraction; use empirical throughput coefficients" | ✅ **Correct, and the most important call in the doc.** The failure was in the numbers chosen, not the method |
| Confidence tiering (high LLM → low cell-tokenized/uncalibrated GBDT) | ✅ Well calibrated; the shipped `getConfidence` functions match the evidence |

### 5.2 What is now outdated

| June claim | Now |
|---|---|
| TabPFN "performs strongly on datasets up to 10,000 samples and 500 features" | **Off by 100×.** TabPFN-2.5 (Nov 2025) → 100K × 2K; **TabPFN-3 (12 May 2026) → 1M rows**; **TabPFN-3.5 (15 Sep 2026) → 1M × 20,000**. TabPFN-3 was already released when the June doc was written |
| GBDTs "often dominant for ordinary supervised tabular tasks" | TabArena shows tuned+ensembled **RealMLP and TabM beat LightGBM and CatBoost**, and the best *single* models are now TFMs. TabPFN-3 beats 8-hour-tuned GBDTs up to 1M rows. The GBDT case is now **cost, latency, and operational simplicity** — not accuracy |
| Cites TabPFN (Nature 2025) and TabDPT (ICLR 2025) as the landscape | Misses TabICL/TabICLv2, Mitra, LimiX/LimiX-2/LimiX-2M, TabSwift, TabPFN-2.5/2.6/3/3.5 |
| Time-series landscape not enumerated at all | Misses Chronos-2, TimesFM 2.5, Moirai 2.0, Moirai-MoE, Toto 2.0, Xihe, Timer-S1, Time-MoE, Sundial, TiRex-2 |
| `architecture_factor ≈ 6` as "a default for dense decoder-style transformer training" | Correct for decoder-only, but **2.0–2.2× too high for encoder-decoder** (Chronos) and silent about MoE |
| "estimate training time empirically using benchmark-derived coefficients" | Right instruction; the implementation's coefficients are 3–29× off (§3.3). The doc never supplied numbers, so the gap was filled at implementation time |
| No mention of MFU breaking down at small scale | The dominant error source for this family (§1.5) |

### 5.3 New results that landed after the June doc

| Date | What | Why it matters here |
|---|---|---|
| **May 2026** | **TabPFN-3** — 53M/58M params, **8T tokens**, 1M rows, inducing-point attention, 20× faster than 2.5, beats 8-hour-tuned GBDTs | Redefines the tabular pane's scope; gives the first real inference-cost anchors |
| **May 2026** | **Toto 2.0** — 4M→2.5B u-μP family, **5.04 trillion data points**, no public forecasting data in pretraining | Largest TSFM corpus published; demonstrates clean scaling over 3 orders of magnitude |
| **May 2026** | **Pocket Foundation Models** — TFM→CPU-GBDT distillation: 96.5% of teacher AUC at **1.9 ms on CPU**, 38–860× speedup; TFM teachers take **151–1,275 ms on GPU** | The strongest available argument for the GBDT pane, and a concrete latency number for the TFM pane's warnings |
| **Jun 2026** | **TFM pretraining speedrun** — TabArena target in **0.92 min on one L40S** (from 74.32 min), 22× fewer synthetic datasets | Settles that tabular FM pretraining is a single-GPU job |
| **Jun 2026** | **LimiX-2M** (2M params beating TabPFN-v2/TabICL), **TabSwift** (row-wise attention) | Tabular FMs are getting *smaller*, not larger — the opposite of the LLM trend |
| **Jul 2026** | **TiRex-2** — recurrent xLSTM TSFM, multivariate + streaming, linear in sequence length | Breaks the attention-cost assumptions in the time-series warnings |
| **Jul 2026** | **"When Do Foundation Models Pay Off?"** — 30-dataset break-even analysis; TSFMs win unconditionally on **15/30**; rule *"if n_train < 700 and seasonality is non-negligible, use FM zero-shot"*; **all FM experiments = ~8 GPU-hours on one RTX 5090**; XGBoost best classical on 14/30 | A usable "which pane should I open" rule, and proof that TSFM *application* is single-GPU work |
| **Sep 2026** | **TabPFN-3.5** — 1M × 20,000; first on TabArena and BeyondArena; available via SAP AI Core and SageMaker | Current state of the art; enterprise availability changes the buy-vs-build framing |
| **Sep 2026** | **LimiX-2** (400M) | Largest open tabular FM |
| **Feb 2026** | **TabICLv2** — million-scale under 50 GB VRAM, Muon optimizer, open pretraining + synthetic-data code | The only tabular FM with a fully open pretraining path — the one case where the pretraining mode is genuinely useful |
| **2025–26** | **XGBoost 3.0 external memory** — 1 TB on one GH200; up to 8× vs 112-core Xeon; 16× end-to-end and 94% TCO reduction at one customer | Removes VRAM from the GBDT feasibility question entirely |

---

## 6. Regression targets (test fixtures)

Published runs usable as calibration tests. The right assertion is usually a *range*, not a point.

### 6.1 Time-series foundation model pretraining

| Fixture | Inputs | Expected | Assertion | Confidence |
|---|---|---|---|---|
| `chronos-large` | N = 710M, encoder-decoder, 200K steps × batch 256, in 512 / out 64 | split-accounting FLOPs **5.60e19**; naive 6ND **1.26e20**; 20.8 h on 8× A100 @ MFU 0.3 | FLOPs within 5%; hours 15–47 | medium |
| `moirai-large` | N = 311M, 10^6 steps × batch 256, patch 32 | 6.1e19 (seq 128) to 2.45e20 (seq 512) FLOPs | order of magnitude; flag that Moirai's per-step token count is unpublished | medium-low |
| `time-moe-ultra` | N_active = 1.1B, N_total = 2.4B, 100K steps × 4M points/iter, point-wise | D = **4.0e11**; FLOPs(active) **2.64e21**; **61 h on 128× A100 @ MFU 0.3** | FLOPs within 10%; hours 46–92 | medium |
| `granite-ttm-r1` | N = 1M, 1B samples, context 512, patch 64, channel-mixed | measured **24–30 h on 6× A100**; `6ND` = 4.8e16; **requires MFU ≈ 0.00026** | the adapter must reach 24–30 h **only** with the small-model MFU ladder; this is the ladder's regression test | high (measurement), low (model) |
| `repo-golden-ts` | existing fixture (`time-series-adapter.test.ts:34`) | 7.04e9 tokens, 4.224e19 FLOPs | unchanged — protects the arithmetic | high |

### 6.2 Tabular foundation models

| Fixture | Inputs | Expected | Assertion | Confidence |
|---|---|---|---|---|
| `tabpfn25-inference` | 50,000 support rows × 500 cols, 1 estimator, H100 | **23–30 s** | predicted seconds in [20, 35] | high |
| `tabpfn3-million` | 1,000,000 rows × 10 cols, H100, cold fit+predict | **~107 s** | [70, 160] s — and must NOT follow the pure `r²` law (inducing points break it) | high |
| `tabpfn3-warm` | 50,000 × 100, H100, KV cache warm | **10–30 ms** | order of magnitude only | medium |
| `tabpfn3-kvcache` | 1,000,000 rows, 1 estimator | **7 GiB** KV cache; chunking gives ~5× peak reduction | VRAM estimate in [5, 10] GiB | high |
| `nanotabpfn-speedrun` | pretraining to TabArena target, 1× L40S | **0.92 min** (baseline 74.32 min) | asserts pretraining mode produces a single-GPU, sub-hour answer | high |
| `tabpfn3-pretrain` | N = 53M, D = 8e12 tokens | `6ND` = **2.54e21**; ~298 h on 8× H100 @ MFU 0.3 | FLOPs exact; hours unvalidated — **mark low** | low |
| `repo-golden-tab-row` / `-cell` | existing fixtures (`tabular-adapter.test.ts:31,38`) | 1.024e9 / 102.4e9 tokens; 6.144e17 / 6.144e19 FLOPs | unchanged | high |

### 6.3 Classical tabular / GBDT

All at **500 rounds, `max_depth = 6`**, full dataset, single GPU.

| Fixture | Rows × cols | Hardware | Measured seconds | Implied work units/s | Confidence |
|---|---|---|---|---|---|
| `xgb-gpu-higgs` | 11,000,000 × 28 | V100-16GB | **34.08** | 4.52e9 | high |
| `xgb-cpu-higgs` | 11,000,000 × 28 | 64 vCPU E5-2686v4 | **470.09** | 3.28e8 | high |
| `xgb-gpu-airline` | 115,069,017 × 13 | V100 | **144.86** | 5.16e9 | high |
| `xgb-gpu-synthetic` | 10,000,000 × 100 | V100 | **38.49** | 1.30e10 | high |
| `cat-gpu-higgs` | 11,000,000 × 28 | V100 | **30.15** | 5.11e9 | high |
| `lgbm-gpu-higgs` | 11,000,000 × 28 | GTX 1080 / OpenCL | **529.54** | 2.91e8 — **slower than CPU (306.98 s)** | high |
| `lgbm-cpu-higgs-official` | 10,500,000 × 28 | 2× E5-2690v4 (28c) | **130.094** | 1.13e9 | high |
| `lgbm-cpu-msltr` | 2,270,296 × 137 | 2× E5-2690v4 | **70.417** | 2.21e9 | high |
| `cat-epsilon-v100-128bin` | 400,000 × 2,000 | V100-PCI, 128 bins | **77** (CPU 713 ⇒ 9.26×) | iteration count unpublished — use for **speedup ratio only** | medium |
| `cat-epsilon-v100-32bin` | 400,000 × 2,000 | V100-PCI, 32 bins | **49** (CPU 653 ⇒ 13.33×) | speedup ratio only | medium |
| `cuml-rf-large` | 1,953,125 × 128, 100 trees | RTX PRO 6000 Blackwell vs TR PRO 7975WX | **1.12** (CPU 28.6 ⇒ 25.6×) | 2.23e10 | high |
| `xgb-1tb-gh200` | 2^29 × 512 (1 TB float32) | 1× GH200 (80 GB + 480 GB) | "minutes"; up to 8× vs 112-core Xeon 8480CL | **no exact seconds published** — assert only that it is feasible on 1 device | medium-low |

**Sparse-data counter-fixtures (the model must be shown to fail here):**

| Fixture | Rows × cols | Seconds | Implied work units/s | Why it is a counter-test |
|---|---|---|---|---|
| `lgbm-cpu-expo` | 11,000,000 × 700 (one-hot) | 62.607 | **6.15e10** | 35× the dense rate — EFB collapses the one-hot columns |
| `lgbm-cpu-allstate` | 13,184,290 × 4,228 (one-hot) | 148.231 | **1.88e11** | 100× the dense rate; `rows × cols` is meaningless here |

---

## 7. Confidence ratings (reusing `ConfidenceLevel`)

| Quantity | Current repo | Recommended | Why |
|---|---|---|---|
| Time-series token derivation (windows, patches, tokens) | `medium` | **high** | pure arithmetic from user inputs; verified against five architectures |
| Time-series FLOPs, decoder-only, N ≥ 100M | `medium` | **medium** | `6ND` convention confirmed as `C = 6NBS` in the scaling-law literature |
| Time-series FLOPs, encoder-decoder | `medium` | **medium-low** until split accounting lands, then `medium` | currently 2.0–2.2× high |
| Time-series wall-clock, N ≥ 100M | inherits 0.40 MFU | **medium** (@ MFU 0.25–0.30) | Chronos back-solve |
| Time-series wall-clock, 10M ≤ N < 100M | — | **low** | **no published run in this band reports wall-clock** |
| Time-series wall-clock, N < 10M | — | **low** | TTM is the only anchor; `6ND` is off by ~1,100× without the MFU ladder |
| Time-series MoE | unsupported | **medium-low** | active/total split is well understood; no wall-clock anchors |
| Tabular FM inference time (TabPFN 2.5 family) | n/a | **medium** | one published anchor + a published complexity formula |
| Tabular FM inference time (TabPFN-3 family) | n/a | **medium-low** | architecture changes the exponent; three anchors, no formula |
| Tabular FM inference VRAM | n/a | **medium** | the 7 GiB/estimator @1M-rows figure is published |
| Tabular FM pretraining FLOPs, `row` mode | `medium-low` | **medium-low** | keep |
| Tabular FM pretraining FLOPs, `axial` mode | `medium-low` | **low** | additive model contradicts the published `O(r²c + rc²)` |
| Tabular FM pretraining FLOPs, `cell` mode | `low` | **low** | keep |
| Tabular FM pretraining wall-clock | — | **low** | only the speedrun is published, and it is not a from-scratch frontier run |
| GBDT time, calibrated coefficient | `medium` | **medium** | keep — user-supplied benchmark is the right mechanism |
| GBDT time, default coefficient, **dense numeric, ≥1M rows** | `low` | **medium** | with §3.3 values this is now backed by two primary benchmark suites |
| GBDT time, default coefficient, <1M rows or sparse/categorical | `low` | **low** | per-dataset variation reaches 100× |
| GBDT GPU-vs-CPU speedup | implicit 10× | **medium** for XGBoost/CatBoost ≥1M dense; **low** otherwise | measured 0.69×–23.4× |
| GBDT memory / VRAM feasibility | not modelled | **medium** (as an informational note) | binned matrix ≈ `rows × features` bytes; external memory removes the ceiling |

---

## 8. Warnings worth emitting

### Time-series
- `N < 100M` → **"At this scale the run is data-pipeline bound, not FLOP bound. Granite TTM (1M params, ~1B samples) took 24–30 h on 6× A100 — about 1,100× longer than a 6ND estimate at 30% MFU predicts. Times below 100M parameters are low confidence."**
- `architectureType = encoder_decoder` and no input/output token split → "Encoder-decoder models charge encoder parameters only for input tokens. A single 6ND term over-estimates Chronos-class models by about 2.2×."
- channel-expanded with `variables > 20` → keep existing warning (`time-series-adapter.ts:87-91`); add: "channel-*independent* models pay `V × T²` attention; channel-*flattened* models (Moirai) pay `V² × T²`."
- recurrent / linear architecture selected → suppress long-sequence attention warnings; note "cost is linear in sequence length (xLSTM/TiRex-2 class)."
- implied TPP outside [40, 2000] → "Published TSFM tokens-per-parameter ratios span ~40 (Chronos-Large) to ~2,000 (Toto 2.0) — every published TSFM is *over*-trained relative to Chinchilla-20. Your value of X is outside the observed band."
- MoE selected → "Compute follows active parameters; memory follows total. Time-MoE reports a 78% training-cost reduction vs a compute-matched dense model."
- `stride ≤ 0.1 × lookback` → keep existing warning.

### Tabular foundation
- always, in pretraining mode → **"Almost no one pretrains a tabular foundation model. If you want to *use* one, switch to inference mode. The community pretraining speedrun reaches a TabArena quality target in 0.92 minutes on one L40S."**
- support rows > 100,000 → "TabPFN-2.5's cost is quadratic in support rows. Above ~100K rows use TabPFN-3/3.5 (1M rows) or TabICLv2, which break the quadratic with inducing points and chunking."
- support rows > 10,000 and target latency < 50 ms → **"Tabular foundation models take 151–1,275 ms per batch on GPU. For sub-10 ms serving, distil into a CPU GBDT: 96.5% of teacher AUC at 1.9 ms."**
- columns > 500 → "Cost saturates at `min(columns, 500)` under TabPFN-2.5's feature subsampling; beyond that, columns are subsampled across estimators rather than attended jointly."
- `tokenizationMode = axial` → "This pane's additive axial token count under-states real alternating attention, which is `O(r²c + rc²)`."
- `tokenizationMode = cell` → keep existing severe-attention warnings (`tabular-adapter.ts:79-91`).
- rows < 10,000 → "At this size a tabular foundation model in a single forward pass beats hours of GBDT tuning (TabPFN-2.5: 100% win rate vs default XGBoost below 10K rows / 500 features)."

### Classical tabular
- **lead output, not a warning** → "GPU: yes / no / doesn't matter," with the CPU time alongside.
- `rows < 500,000` and `cpuOrGpu = gpu` → **"GPU will likely be *slower* than CPU at this size. Transfer and kernel-launch overhead dominates; a 100K-row dataset can train 2–3× slower on GPU."**
- `algorithm = lightgbm` and `cpuOrGpu = gpu` → **"LightGBM's OpenCL GPU backend was slower than its CPU backend on 5 of 6 reference datasets. Use `device=cuda` (not `gpu`), or stay on CPU."**
- `columns > 500` → "If these are one-hot encoded categoricals, the `rows × columns` work-unit model over-states cost by up to 100× (feature bundling collapses them). Use native categorical handling and set the column count to the pre-encoding width."
- `cvFolds × hyperparameterTrials > 100` → informational: "This multiplier is the dominant cost. TabArena uses 200 configs × 8 folds = 1,600 fits per model."
- `cvFolds × hyperparameterTrials > 500` → "Search dominates total time. Prefer pruning (Hyperband / early stopping) over more trials; published practice converges around 50–100 trials."
- `bins > 128` and `cpuOrGpu = gpu` → "GPU GBDT is markedly faster at lower `max_bin`: CatBoost on Epsilon went from 9.26× to 13.33× speedup moving from 128 to 32 bins."
- binned data size > ~60 GB → "Beyond GPU memory the answer is external memory on one coherent-memory device (XGBoost 3.0 trains 1 TB on a single GH200), not more GPUs."
- `throughputCoefficient = 0` → keep the existing calibration warning, but soften it: the new defaults are benchmark-derived for dense numeric data at ≥1M rows and depth 6.

---

## 9. Open design questions

1. **Does the classical tabular pane belong in a GPU calculator at all?** The honest answer for most inputs is "use a CPU." Options: (a) keep it and make the "no GPU needed" verdict the hero output — the most useful and most honest framing; (b) rename it "Tabular training time" and drop GPU-count framing; (c) cut it. **Recommend (a)** — a calculator that tells you *not* to buy a GPU earns trust, and it mirrors the Inference pane's buy-vs-build breakeven.
2. **Split the tabular FM pane into Inference (default) and Pretraining (advanced)?** The inference question is the one users have, has published anchors and a published complexity formula, and produces a memory-feasibility ladder. Pretraining is a three-organization question. **Recommend splitting**, retiring `testTimeComputeMultiplier` in favour of a real inference model.
3. **Where does MFU-by-scale live?** The small-model MFU collapse is not time-series-specific — it would hit any sub-100M-parameter model. Make it a shared hardware-estimator concern keyed on parameter count, or a time-series-pane default? **Recommend shared**, since tabular FMs (2M–400M) land in exactly the same danger band.
4. **Is a model catalog worth it here?** §1.1 and §2.1 are transcribable, but note what is *missing*: no 2026 TSFM publishes layer/hidden dims per size, and TabPFN-2.5's parameter count is undisclosed. A catalog with honest gaps is still valuable (it fixes tokenization, patch size, context length, and corpus size — the inputs users get wrong) but it cannot be complete, and the UI must render "not published" rather than a plausible number.
5. **Should the three panes share a "which pane should I use?" router?** The evidence now supports a concrete decision rule (§3.7, plus the TSFM `n_train < 700` rule from the break-even paper). This may be more valuable than any individual pane's precision.
6. **Add `featureDensity` to the GBDT pane, or just warn?** The sparse-data breakdown is 100×, which is larger than every other error in the pane combined. A `density` multiplier is cheap; the risk is that users cannot estimate it. **Recommend: warn at `columns > 500`, and offer a "pre-encoding column count" input instead** — users do know how many raw categorical fields they have.
7. **How should unsourced numbers be rendered?** This domain has far thinner published data than LLMs. Several recommended defaults here are extrapolations (modern-hardware GBDT scaling, the 10M–100M MFU band). The existing `ConfidenceBadge` carries the level, but not the *reason*. Consider a tooltip that distinguishes "measured," "back-solved from a published run," and "interpolated."
8. **Should the GBDT pane model the depth/`num_leaves` distinction?** XGBoost is depth-wise; LightGBM is leaf-wise with `num_leaves`. A `maxDepth / 6` multiplier is right for XGBoost and CatBoost but approximate for LightGBM (`num_leaves = 31` ≈ 5 effective levels). Probably acceptable; worth naming in the trace.

---

## Sources

**Time-series foundation models — primary**
- [Chronos: Learning the Language of Time Series (arXiv:2403.07815)](https://arxiv.org/abs/2403.07815) — 8× A100-40GB, 200K steps, batch 256, context 512 / prediction 64, 4,096-token vocab on [−15, 15], ~84B tokens
- [Chronos-2 technical report (arXiv:2510.15821)](https://arxiv.org/html/2510.15821v1) and [HF model card](https://huggingface.co/amazon/chronos-2) — 120M, encoder-only, group attention, 2048→8192 context, >300 forecasts/s on A10G
- [Fast and accurate zero-shot forecasting with Chronos-Bolt and AutoGluon (AWS)](https://aws.amazon.com/blogs/machine-learning/fast-and-accurate-zero-shot-forecasting-with-chronos-bolt-and-autogluon/) — patch 16, context 2048, 250× faster / 20× more memory-efficient than Chronos
- [Unified Training of Universal Time Series Forecasting Transformers (Moirai, arXiv:2402.02592)](https://arxiv.org/abs/2402.02592) — LOTSA 27.6B observations, 10^6 steps × batch 256 on A100-40G
- [Moirai-MoE (arXiv:2410.10469)](https://arxiv.org/pdf/2410.10469) — 11M/86M activated, 32 experts top-2, patch 16, masking 0.3
- [Moirai 2.0: When Less Is More (arXiv:2511.11698)](https://arxiv.org/html/2511.11698v1) and [Salesforce blog](https://www.salesforce.com/blog/moirai-2-0/) — 11M decoder-only, 36M series, patch 16, 9 quantiles
- [Time-MoE (arXiv:2409.16040)](https://arxiv.org/html/2409.16040v2) — 128× A100-80G, 100K steps, batch 1024, 4M points/iteration, Time-300B, point-wise tokenization, 78%/39% cost reductions
- [Sundial (arXiv:2502.00816)](https://arxiv.org/html/2502.00816v1) — 32M/128M/444M, patch 16, context 2880, TimeBench 1T points
- [Toto 2.0 (arXiv:2605.20119)](https://www.alphaxiv.org/abs/2605.20119) and [Datadog blog](https://www.datadoghq.com/blog/ai/toto-2/) — 4M–2.5B u-μP family, **5.04T data points**, 4,096 training context, proxy model d=256/h=4/L=12
- [Xihe (arXiv:2510.21795)](https://arxiv.org/abs/2510.21795) — 9.5M–1.5B, HIBA attention, 325B points
- [Granite TinyTimeMixer R2 (HF)](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2) and [TTM paper (arXiv:2401.03955)](https://arxiv.org/pdf/2401.03955) — **1M params, ~1B samples, 24–30 h on 6× A100**; context {512,1024,1536}, patch 64
- [TiRex-2 (arXiv:2607.01204)](https://arxiv.org/html/2607.01204v1) — recurrent xLSTM, 38.4M active (+44.1M multivariate), linear in sequence length
- [Lag-Llama (arXiv:2310.08278v3)](https://arxiv.org/html/2310.08278v3) — 8 layers, 9 heads × 16 dim/head, context 32, 7,965 series ⇒ ~352M windows, batch 256, **a single Tesla P100-12GB**. Note: a widely-repeated survey figure of "200M parameters" is inconsistent with the paper's own architecture table
- [MOMENT: A Family of Open Time-series Foundation Models (arXiv:2402.03885)](https://arxiv.org/abs/2402.03885) — 40M/125M/385M, patch-based encoder, masked modeling
- [TimesFM 2.5 coverage (MarkTechPost)](https://www.marktechpost.com/2025/09/16/google-ai-ships-timesfm-2-5-smaller-longer-context-foundation-model-that-now-leads-gift-eval-zero-shot-forecasting/) — 200M params, 16K context, 30M quantile head (vendor-adjacent press; primary repo at [google-research/timesfm](https://github.com/google-research/timesfm))

**Time-series scaling laws**
- [Scaling-laws for Large Time-series Models (arXiv:2405.13867)](https://arxiv.org/abs/2405.13867) — power laws over five orders of magnitude; aspect ratio and head count have minimal effect
- [Towards Neural Scaling Laws for Time Series Foundation Models (arXiv:2410.12360)](https://arxiv.org/html/2410.12360v3) — **`D ∝ N^0.8`**, `C = 6NBS`, encoder-only vs decoder-only comparison, ID/OOD shift
- [Scaling Law for Time Series Forecasting (arXiv:2405.15124)](https://arxiv.org/abs/2405.15124) — look-back horizon as a third axis; optimal lookback grows with dataset size
- [TSFM Scaling Laws in 2026 (TSFM.ai)](https://tsfm.ai/blog/tsfm-scaling-laws-2026) — **aggregator**, used only for the cross-family TPP table (Toto 15,000:1 point-level, Timer-S1 1,300:1, Granite TTM, corpus sizes)
- [Towards Long-Context Time Series Foundation Models (arXiv:2409.13530)](https://arxiv.org/html/2409.13530v1) — channel-independence vs channel-mixing taxonomy; Moirai's channel flattening

**Time-series application economics**
- [When Do Foundation Models Pay Off? A Break-Even Analysis (arXiv:2607.04919)](https://arxiv.org/html/2607.04919) — 15/30 FM-dominant, `n_train < 700` rule, **~8 GPU-hours total on one RTX 5090**, XGBoost best classical on 14/30
- [Assessing the Operational Viability of Foundation Models for Time Series Forecasting (arXiv:2605.24381)](https://arxiv.org/pdf/2605.24381) — 200–1,764 ms inference latencies; qualitative on cost
- [Deploy Time Series Foundation Models on GPU Cloud (Spheron)](https://www.spheron.network/blog/deploy-time-series-foundation-models-gpu-cloud/) — **aggregator**; used for model-size recaps and fine-tuning hardware (RTX 5090 32 GB, 4× A100 DDP)

**Tabular foundation models — primary**
- [Accurate predictions on small data with a tabular foundation model (TabPFN v2, Nature 2025)](https://www.nature.com/articles/s41586-024-08328-6) — 11M params, two-way attention, 10K × 500 ceiling
- [TabPFN-2.5 (arXiv:2511.08667)](https://arxiv.org/html/2511.08667v2) — 24/18 layers, feature group size 3, 64 thinking rows, **`O(r²·min(c,500) + r·min(c,500)²)`**, **50K × 500 ⇒ 23–30 s on 1 H100**, 100% / 87% win rates vs default XGBoost
- [TabPFN-3 technical report (arXiv:2605.13986)](https://arxiv.org/html/2605.13986v2) and [Prior Labs summary](https://priorlabs.ai/technical-reports/tabpfn-3) — **53M/58M params, >8T tokens, 1M rows, 7 GiB KV cache, 1M×10 ⇒ ~107 s on H100, 20× faster than 2.5, beats 8-h-tuned GBDTs**, FlashAttention-3 1.5–1.7×
- [TabPFN-3.5 technical report](https://priorlabs.ai/technical-reports/tabpfn-3-5) and [Prior Labs model matrix](https://docs.priorlabs.ai/models) — 1M rows × 20,000 features; first on TabArena and BeyondArena; the full version/limit table
- [TabICL (arXiv:2502.05564)](https://arxiv.org/pdf/2502.05564) — two-stage column-then-row ICL, 500K samples / 500 features
- [TabICLv2 (arXiv:2602.11139)](https://arxiv.org/abs/2602.11139) — million-scale under 50 GB VRAM, scalable softmax, Muon, open pretraining code
- [TabDPT: Scaling Tabular Foundation Models on Real Data (NeurIPS 2025)](https://arxiv.org/html/2410.18164) — power laws in model size and **pretraining cells**; real-data retrieval
- [Mitra (arXiv:2510.21204)](https://arxiv.org/pdf/2510.21204) — 72M, mixed SCM + tree synthetic priors, strongest below 5K rows
- [LimiX-2 (arXiv:2609.17488)](https://arxiv.org/pdf/2609.17488) — 400M contextual mechanism network
- [LimiX-2M (arXiv:2606.04485)](https://arxiv.org/html/2606.04485) — 2M params beating TabPFN-v2/TabICL; RaBEL, attention-bottleneck analysis
- [TabSwift (arXiv:2606.07345)](https://arxiv.org/html/2606.07345) — row-wise-attention-only efficiency design
- [nanoTabPFN (arXiv:2511.03634)](https://arxiv.org/pdf/2511.03634) and [Speedrunning TFM Pretraining (arXiv:2606.03681)](https://arxiv.org/abs/2606.03681) — **0.92 min on one L40S** vs a 74.32 min baseline, 22× fewer synthetic datasets

**Tabular benchmarks and the GBDT-vs-deep debate**
- [TabArena: A Living Benchmark (arXiv:2506.16791)](https://arxiv.org/html/2506.16791v2) — 51 datasets, 500–250K samples, **200 configs × 8 inner folds**, 16 models, normalized scores and per-1,000-sample train/infer times
- [Pocket Foundation Models: Distilling TFMs into CPU-Ready GBDTs (arXiv:2605.18654)](https://arxiv.org/abs/2605.18654) — **TFM teachers 151–1,275 ms on GPU; distilled XGBoost 1.9 ms on CPU at 96.5% of teacher AUC; 38–860× speedup**; Wilcoxon p = 0.0008
- [Light-Weight Benchmarks Reveal the Hidden Hardware Cost of Zero-Shot Tabular Foundation Models (arXiv:2512.00888)](https://arxiv.org/pdf/2512.00888) — framing source; numeric tables were not machine-extractable from the PDF, so no numbers are cited from it
- [The Current Generation of Tabular Foundation Models: A Critical Review (MDPI MAKE 8(8):244)](https://www.mdpi.com/2504-4990/8/8/244) — **403 on direct fetch**; the 12-layer saturation and 37M-synthetic-dataset points come from search snippets of it and are marked low/medium-low
- [TabPFN (Wikipedia)](https://en.wikipedia.org/wiki/TabPFN) — **aggregator**; used only for release dates
- [Frank Hutter on TabPFN v2's 11M parameters](https://x.com/FrankRHutter/status/1878864024055533671) — author statement, used for the parameter count the Nature paper does not foreground

**GBDT throughput — primary benchmark suites**
- [GBM-Benchmarks (RAMitchell)](https://github.com/RAMitchell/GBM-Benchmarks) — the §3.2 table. AWS p3.16xlarge (8× V100, 64 vCPU E5-2686v4), **500 rounds, `max_depth=6`**, 7 Jun 2018; parameters confirmed from [`benchmark.py`](https://github.com/RAMitchell/GBM-Benchmarks/blob/master/benchmark.py); dimensions from [ml_dataset_loader](https://github.com/RAMitchell/ml_dataset_loader)
- [LightGBM Experiments](https://lightgbm.readthedocs.io/en/latest/Experiments.html) — Higgs 130.094 s, MS LTR 70.417 s, Yahoo LTR 76.229 s, Expo 62.607 s, Allstate 148.231 s on 2× E5-2690v4, 500 iterations; Criteo 1.7B rows × 67 features scaling 627.8 s → 42 s per tree over 16 machines
- [LightGBM GPU Tuning Guide and Performance Comparison](https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html) — six datasets with dimensions; RX 480 / GTX 1080 vs 28-core Haswell; `max_bin` 255/63/15 effects; "if a dataset is too small, computing it on GPU is inefficient"
- [CatBoost GPU vs CPU training speed benchmarks](https://github.com/catboost/catboost/tree/master/catboost/benchmarks/gpu_vs_cpu_training_speed) — Epsilon and Criteo tables; 1.30×–13.33× by GPU and bin count
- [XGBoost: Scalable GPU Accelerated Learning (arXiv:1806.11248)](https://arxiv.org/pdf/1806.11248) and [Out-of-Core GPU Gradient Boosting (arXiv:2005.09148)](https://arxiv.org/pdf/2005.09148) — the 1M × 50 / 500-round anchor (13.87 s gpu_hist vs 63.55 s CPU hist) and Titan V in-core numbers
- [cuML accelerator benchmarks (NVIDIA)](https://docs.nvidia.com/cuml/latest/cuml-accel/benchmarks/) — Random Forest fit times, RTX PRO 6000 Blackwell vs Threadripper PRO 7975WX, 3.13×–27.4×
- [Quantized Training of Gradient Boosting Decision Trees (arXiv:2207.09682)](https://arxiv.org/pdf/2207.09682) — bin-width / quantization effects on GBDT speed

**GBDT memory, external memory, and GPU support**
- [XGBoost GPU Support](https://xgboost.readthedocs.io/en/stable/gpu/index.html) — ELLPACK compressed quantized storage (~1/4 of float32 CSR), working memory ∝ rows, histogram memory ∝ bins × features × cached nodes
- [XGBoost External Memory tutorial](https://xgboost.readthedocs.io/en/stable/tutorials/external_memory.html) — **1 TB = 2^29 × 512 float32**, 128 × 8 GB batches, `max_quantile_batches=32`, ~350 GB/s host-to-device on GH200 LPDDR5-480G, PCIe4 ≈ 24 GB/s / PCIe5 ≈ 42 GB/s, "using the disk is not recommended"
- [Train with Terabyte-Scale Datasets on a Single GH200 using XGBoost 3.0 (NVIDIA)](https://developer.nvidia.com/blog/train-with-terabyte-scale-datasets-on-a-single-nvidia-grace-hopper-superchip-using-xgboost-3-0) — up to 8× vs 112-core Xeon 8480CL, `max_bin=256` default, ~2× GPU histogram speedup, 16× end-to-end and 94% TCO at one customer
- [LightGBM Features](https://lightgbm.readthedocs.io/en/stable/Features.html) — histogram/EFB/GOSS, 8-bit and 4-bit bin storage, sparse `O(2·#non_zero)` histogram construction
- [ScalaGBM: Memory Efficient GBDT Training for High-Dimensional Data on GPU (ACM)](https://dl.acm.org/doi/pdf/10.1145/3690624.3709271) — **403 on fetch; no numbers cited from it.** Listed as the relevant primary source if the memory model is ever implemented
- [TabPFN GPU requirements (GH issue #250)](https://github.com/PriorLabs/TabPFN/issues/250) — practitioner VRAM reports (8 GB small, 16 GB at the 10K×500 ceiling)

**Hyperparameter search and categorical handling**
- [XGBoost Hyperparameter Optimization with Optuna (xgboosting.com)](https://xgboosting.com/xgboost-hyperparameter-optimization-with-optuna/) — **aggregator**; typical 20–100 trial counts, pruning guidance
- [Learning From High-Cardinality Categorical Features (arXiv:2307.02071)](https://arxiv.org/pdf/2307.02071) — target encoding preferred at high cardinality with complex trees
- [CatBoost: gradient boosting with categorical features support (arXiv:1810.11363)](https://arxiv.org/pdf/1810.11363) — oblivious trees, fixed-bin discretization

**Repo documents cross-referenced**
- `scaling_laws_time_series_tabular_models.md` (2026-06-19) — the input doc this supersedes
- `model_family_extensions_functional_spec.md` — §13 confirms the spec never specified GBDT throughput values (line 546: "Rows × columns × rounds per second or similar", "required/custom")
- `docs/decisions/2026-06-19-model-family-extension-decisions.md` — decisions #1 (channel-compressed default), #2 (warn-only on attention), #3 (defer MoE), #5 (fold test-time compute into training), #7 (no benchmark library)
- `docs/plans/model-family-extension-phases.md` — Phase 4 task 2 ("benchmark-derived throughput defaults") is where the unsourced coefficients entered
- `docs/research/2026-10-14-finetuning-calculator-research.md` — the memory-feasibility-ladder pattern reused in §2.2, and the format of this document
