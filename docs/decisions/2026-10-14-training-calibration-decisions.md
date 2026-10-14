# Training Pane — Calibration Decisions

**Date:** 2026-10-14
**Issues:** #74 (TPP default), #64–#70 (precision and GPU data), #73 (reference FLOPs), #81 (reference runs)
**Research:** [`docs/research/2026-10-14-llm-training-research.md`](../research/2026-10-14-llm-training-research.md)
**Context:** Validation against twelve sourced runs found the pane accurate to ±7% on large dense BF16 runs — the case it was built for — and structurally wrong outside it.

---

## The framing: one number carries all the accuracy

In pretraining the memory floor is **~470× slacker** than the compute bound and never binds. Everything except achieved utilization is exact arithmetic, so the pane's accuracy lives almost entirely in `mfu × availability / overheadFactor`.

Across eleven sourced runs that product lands in **0.26–0.40** of dense BF16 peak, median ~0.34. The repo's defaults multiply to `0.400 × 0.90 / 1.10 = 0.327` — within 5.4% of the Llama 3.1 405B anchor. **The central estimate was already well calibrated.** The errors were structural: FP8, small models, MoE, long context.

## Decision 1 — FP8 and FP4 get a precision-relative MFU factor

`getPeakFlops` swapped in the FP8 peak while `estimateHardware` reused the BF16 MFU, asserting a full 2× speedup. Measured end-to-end FP8 gain is **1.3–1.5×**, because default recipes quantize only transformer-block linears while softmax, layernorm, residuals, optimizer and all-reduce stay higher precision.

```
effectiveMfu = baseMfu × precisionFactor × moeFactor
precisionFactor:  BF16 1.00   FP8 0.70   FP4 0.50
moeFactor:        dense 1.00  MoE 0.85
```

Validated against DeepSeek-V3, the one sourced FP8 run: `0.70 × 0.85 = 0.595` against an observed 0.50 ratio.

**This changed every FP8 number the pane had ever produced** — required GPU counts at FP8 are ~43% higher than the old model gave. Documented prominently in the help rather than shipped silently, because users may hold saved or shared FP8 scenarios.

## Decision 2 — Default tokens-per-parameter moves from 20 to 150

FLOPs are `6 × TPP × N²`, so TPP scales the answer linearly. The default of 20 — Chinchilla-optimal — understated tokens, FLOPs, GPUs and cost by **7.5×** for any realistic 2026 run, and it sat on the default screen.

Chinchilla is not wrong; it answers a different question. It minimises loss for a fixed *training* budget. Once a model will be served, lifetime compute optimises differently and favours a smaller model trained longer. No 2025–26 production run trains compute-optimal.

Presets are keyed to published runs rather than round numbers:

| Preset | TPP | Evidence |
|---|---|---|
| Chinchilla | 20 | Hoffmann et al. 2022 |
| Replicated | 26 | Epoch AI's re-fit (25.6); DeepMind's own parametric form reads ~70, and the paper's three methods disagree |
| **Frontier dense** | **150** | Median of Llama 3.1 70B (223), OLMo 3 32B (172), Marin 32B (201), Qwen3-235B total (153) |
| MoE active | 400 | DeepSeek-V3 400, GLM-5 712 on active — both 22–38 on total |
| Small, served | 1,000 | Llama 3.1 8B 1,950; Marin 8B 1,588 |

The old 50 and 100 entries were dropped; neither had evidence behind it.

**The MoE denominator is a separate preset, not a blend.** "Is this over-trained?" depends entirely on which parameter count you divide by, and averaging the two would mislead in both directions.

## Decision 3 — Warnings must stay silent on legitimate choices

`D = TPP × N` has to come from a real corpus. About 4 epochs of repetition is near-free with returns decaying sharply after, and a 2026 refinement finds even that holds only near 1× Chinchilla at ~3B scale. A warning now fires when the implied corpus exceeds ~30T tokens, past the largest published datasets.

It is deliberately **silent on every shipped preset** — 150 on 70B, 1,000 on 8B, 20 on anything. Tests assert the silence as well as the firing case.

This rule came from the GBDT work in the same sweep, where the CV×HPO warning fired above 50 and therefore on every realistic configuration, while TabArena itself uses 1,600 fits. **A warning that fires on a legitimate choice is fatigue, not information**, and trains users to ignore the ones that matter.

## Decision 4 — Reference runs exist to validate the engine, not to be recognisable

The list was ten dense entries ending at Llama 3 (2024), with `estimatedFlops` and nothing comparable to the calculator's own output.

Now 27 entries, 12 MoE with `activeParameters`, and **13 carrying published `gpuHours`** plus `gpuType`, `precision` and `hoursConvention`. GPU-hours are the quantity directly comparable to what the pane computes, so **models with published training compute were preferred over models that are merely famous.**

Two supporting rules:
- Rows where sources disagree say so (GLM-5's active-parameter reports span 40–44B; the technical report's own 40B is used).
- Rows without published GPU-hours state that explicitly rather than leaving the field quietly absent.

Also fixed here: Llama 2 7B and 13B `estimatedFlops` were **1000× too low** — an `e19/e20`-for-`e22/e23` typo that corrupted nearest-reference ordering and the IsoFLOP filter window. A regression test now asserts `6ND` for every row, branching on sparsity so MoE rows check against active parameters.

## Known-unfixed, deliberately

**The three sliders are one degree of freedom.** `requiredGpus ∝ overheadFactor / (mfu × availability)` — they are perfectly degenerate, and three controls for one parameter invites double-counting. Left open as #79 because the fix is a UX decision, not a calculation one, and the current defaults are well calibrated in aggregate.

Also open: MoE active-parameter input (#71), the long-context attention correction to the flat `6` architecture factor (#75), scale-dependent availability (#78), and the small-batch MFU-collapse warning (#72).

## References

See the research document linked above. Every figure here is sourced there, including the twelve-run accuracy table behind the framing.
