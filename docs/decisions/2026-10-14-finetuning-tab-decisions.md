# Fine-Tuning Tab — Design Decisions

**Date:** 2026-10-14
**Issue:** #138 (epic)
**Research:** [`docs/research/2026-10-14-finetuning-calculator-research.md`](../research/2026-10-14-finetuning-calculator-research.md), [`…-rl-posttraining-research.md`](../research/2026-10-14-rl-posttraining-research.md)
**Context:** The calculator answers "how many GPUs to *pretrain*" and "API vs self-host for *inference*". It cannot answer the most common practical question: *how many of GPU X do I need to fine-tune model Y with method Z.*

---

## Decision 1 — A separate mode, not a new `TrainingMode` value

`TrainingMode` already enumerates `SFT | LORA | RLHF | DISTILLATION` (`src/engine/types.ts:3-10`), but the Training pane's formulas are pretraining-oriented (`6ND` driven by tokens-per-parameter) and its question is deadline-driven.

The research found the binding constraint differs by pane:

| Pane | Binding constraint | Primary unknown |
|---|---|---|
| Training | compute × deadline | achieved MFU at scale |
| **Fine-tuning** | **memory capacity** | **bytes/param per method** |
| Inference | bandwidth, then KV cache | concurrency under an SLO |

Decisive evidence: in pretraining, `memoryLowerBoundGpus` is **~470× slacker** than the compute bound and never binds. In fine-tuning it *is* the question. Same arithmetic, inverted dominance — so a third top-level mode in `ModeSelector`, reusing `GpuSku`, `hardware-estimator`, `FormulaTraceDrawer`, `AssumptionChips`, `ConfidenceBadge`, and `src/data/cloud-gpu-pricing.json`.

The existing `TrainingMode` enum values stay where they are; they describe the *run type* for a pretraining-shaped estimate and are not load-bearing for this pane.

## Decision 2 — Hero output is a method × GPU comparison matrix

Each cell: GPU count, wall-clock, cost, and a fit state (`fits` / `sharded (FSDP/ZeRO-3)` / `OOM`).

**Rationale, corrected after implementation (2026-10-14).** The original rationale cited the research's "three-order-of-magnitude cost spread" ($2.76 for 7B QLoRA vs $1,785 for 70B full FT). Validating the engine against a *fixed* workload showed that framing is wrong: those two anchors differ in model, hardware and dataset, so they are not a like-for-like method comparison. See the caveat added to research §6.

What the engine actually produces for 70B, 2,000 examples × 800 tokens × 3 epochs, H100 at $3/GPU-hr:

| Method | Memory | Min GPUs | Wall-clock | GPU-hours | Cost |
|---|---|---|---|---|---|
| Full FT | 1,212 GB | 24 | 0.17 h | 4.0 | $12.08 |
| LoRA r=16 | 157 GB | 4 | 0.57 h | 2.3 | $6.90 |
| QLoRA | 44 GB | 1 | 2.30 h | 2.3 | $6.90 |

Cost varies **1.75×**, not 1000×. The reason is structural: billed by the GPU-hour, `cost ≈ totalFLOPs / (MFU × availability)` — GPU count cancels against wall-clock, and method moves FLOPs/token only `8N` vs `6N`.

What varies by **24–27×** is memory footprint and minimum deployable GPU count. That is the feasibility question this pane exists to answer, and it is the real justification for the matrix: a single-config card cannot show that full FT needs three nodes while QLoRA needs one GPU.

Two consequences for the UI:

1. **The matrix must show GPU count, wall-clock *and* cost together.** Showing wall-clock alone inverts the ranking — full FT looks 13× *faster* than QLoRA, because it is spread over 24× more GPUs.
2. **Near-constant cost is itself a finding worth stating**: for a fixed workload, the method choice is about what hardware you can get and how long you are willing to wait, not about the bill. Surface that as an assumption chip rather than leaving users to infer it.

A single-config feasibility panel (stacked memory breakdown vs capacity, with the `FormulaTraceDrawer`) sits below the matrix for the selected cell.

## Decision 3 — All methods ship in v1, with honest confidence tiering

| Method | FLOPs/token | Model state | Confidence |
|---|---|---|---|
| Full FT | `6N` | `N × 16 B` (AdamW) | high |
| LoRA | `4N` | `2N + 16 × N_adapter` | high |
| QLoRA | `4N × 0.75` | `0.5N + 16 × N_adapter` | medium-low (throughput factor spans 0.7–0.8) |
| DoRA | `4N × 0.9` | LoRA + magnitude vectors | medium |
| DPO | ~3.5× SFT | 2× (policy + frozen ref) | medium-low |
| ORPO | ~2× SFT | 1× (reference-free) | medium-low |
| **GRPO / PPO** | rollout-dominated | see RL research | **low** |

`4N` for LoRA is not a guess: standard backward costs `4bmn` per linear layer, LoRA's is `2bmn + 4br(m+n)`, and with `r ≪ m,n` the backward pass costs the same as the forward. Back-solving MFU from NVIDIA NeMo's published LoRA throughputs under `4N` reproduces NeMo's own reported MFU to within 2.4 points across four configurations.

GRPO/PPO were **recommended for deferral** by the research, because their cost is dominated by rollout *generation* rather than gradient steps — a different engine wearing the same UI. Shipping them anyway is a deliberate accepted trade: they render with a `low` `ConfidenceBadge` and an explicit caveat naming what would pin the estimate down. Better a sourced wide interval than a blank pane.

## Decision 4 — Memory is componentised, not a single bytes/param constant

```
M_total = N_total × b_w          // weights; TOTAL params (all MoE experts resident)
        + N_train × b_g          // gradients
        + N_train × b_opt        // optimizer state
        + M_activations          // the only batch × seqlen term
        + M_aux                  // reference / reward / critic copies; rollout KV cache
        + M_overhead             // ~1.5 GB/GPU + 8%
```

Published anchors the engine must reproduce (regression fixtures): 7B 88/20/8 GB, 14B 174/35/14, 32B 394/76/28, 70B 860/159/52 (full / LoRA r=64 / QLoRA), MoE-30B-A3B 105/69/21.

Optimizer bytes are a named preset, not a hardcoded constant: AdamW-fp32 12, AdamW-bf16 8, 8-bit Adam 4, Adafactor 4, SGD-momentum 4, LOMO ~0. The two conventions in the literature (12 vs 16 bytes/param total for full FT) differ only on whether the FP32 master copy is counted separately — the trace must show the breakdown so the number is auditable rather than asserted.

Activations, with FlashAttention assumed (so the `5as/h` quadratic term drops):
```
checkpointed:  M_act ≈ 2·b·s·d·L  +  34·b·s·d     // boundary stash + one-layer recompute peak
unckpt:        M_act ≈ 34·b·s·d·L
```
Gradient checkpointing defaults **on** — it is the standard fine-tuning configuration — at `+2N` FLOPs/token.

## Decision 5 — MoE is first-class from day one

Compute follows `N_active`, memory follows `N_total`. Every frontier 2026 open-weight model is sparse (DeepSeek V4 1T/37B, Qwen3-235B-A22B, GLM-5.2 744B/40B, Llama 4 Scout 109B/17B, Mistral Small 4 119B/6B). Conflating the two overstates FLOPs 20–34×.

This is the same `activeParameters` split the Training pane needs (its own discrepancy table, item 3), so the field is added to a shared location rather than duplicated.

Hard warning: **do not 4-bit QLoRA an MoE** — bitsandbytes lacks native MoE support and quantization interferes with expert routing. Use BF16 LoRA.

## Decision 6 — Goal presets over raw token counts

Following the precedent set in [`2026-06-24-inference-ux-presets.md`](2026-06-24-inference-ux-presets.md): users have "2,000 support transcripts", not a token count. Raw fields (`examples`, `avgTokensPerExample`, `epochs`, `sequenceLength`, `packing`) stay visible and editable at all times; presets are a convenience layer on top.

| Preset | Examples | Epochs | Method | Basis |
|---|---|---|---|---|
| Style / tone adoption | 1,000 | 3 | LoRA r=16 | LIMA: 1,000 curated pairs sufficed |
| Narrow task (classification) | 500 | 3 | QLoRA r=16 | 100–500 is the published floor |
| Instruction following | 10,000 | 2 | LoRA r=32 | broad tasks need 5–20K |
| Tool use / function calling | 2,000 | 3 | LoRA r=32 | |
| Reasoning distillation | 20,000 | 2 | LoRA r=64 | long outputs; high seqlen |
| Domain adaptation | — | 1 | Full FT | continued-pretraining regime, 10^8–10^10 tokens |

Epochs default 1–3; warn above 5 (memorization). Packing defaults on at 0.95 efficiency; unpacked 0.55.

## Decision 7 — Model catalog as a data file

`src/data/finetune-models.json`, following the `src/data/*.json` + `src/data/types.ts` pattern. Needs per-model `N_total`, `N_active`, `layers`, `hidden`, `kvHeads`, `headDim`, `ffnDim`, `vocab`, `contextWindow`, `isMoe`, `license` — none of which `reference-models.ts` carries (it holds only params/tokens/FLOPs for pretraining comparison).

12–15 curated models plus a `custom` entry. Architecture dims must come from each model's `config.json`, not from survey articles — the surveys report parameter counts reliably and architecture dims only sporadically.

Adapter params are computed exactly when catalog dims are present:
```
per_layer = r × [2(d+d) + 2(d+d_kv) + 2(d+d_ff) + (d_ff+d)]   // q,o,k,v,gate,up,down
```
falling back to `N_total × 0.0026 × (r/16) × target_factor` (calibrated on 8B r=16 → ~21M and 70B r=16 → ~150M trainable) at `medium` confidence when they are not.

## Decision 8 — Fine-tuning MFU defaults are lower than pretraining's

Back-solved from NeMo's published throughputs: **0.40** single GPU, **0.30** sharded single node, **0.27** multi-node — against the Training pane's 0.40–0.45. Fine-tuning runs smaller batches and shorter sequences and so utilizes the hardware less well. Multi-GPU scaling efficiency `0.90^log2(nodes)` (measured: 70B LoRA 8→16 GPUs = 90.2%, 32B = 92.1%).

## Deferred

- **Multimodal / vision fine-tuning** — different activation accounting (image patches), no good published anchors found.
- **Embedding / reranker fine-tuning** — contrastive losses with large in-batch negatives; different memory shape.
- **Multi-adapter serving economics** — belongs in the Inference pane, not here.
- **Spot-instance interruption modeling** — relevant but orthogonal; the Inference pane's pricing tiers already carry a spot warning.

## Open

- Whether the matrix's GPU columns are user-selected (like `GpuSelector`) or fixed to a sensible four. Leaning user-selected, reusing the existing component.
- Whether the deadline inverse-solve ("to finish by Friday, use N GPUs") is a panel here or a reuse of `ReverseSolve`.

## References

See the two research docs linked above; every number in this document is sourced there.
