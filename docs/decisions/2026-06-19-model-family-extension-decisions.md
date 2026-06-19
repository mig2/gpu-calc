# Model Family Extension — Design Decisions

**Date:** 2026-06-19
**Context:** Decisions made before implementing the model family extensions spec (`model_family_extensions_functional_spec.md`).

---

## Open Questions from Spec (Section 20) — Resolved

| # | Question | Decision | Rationale |
|---|----------|----------|-----------|
| 1 | Default tokenization for time-series mode? | **Channel-compressed** | Safer default (lower compute). User opts into channel-expanded explicitly. |
| 2 | Model attention FLOPs explicitly for long sequences? | **No for v1** — warn only | Keep the calculator simple. Warn when sequence lengths imply infeasible dense attention, but don't compute quadratic attention cost. |
| 3 | MoE as cross-cutting architecture option? | **Defer** | Keep model families clean. MoE can be a future extension across LLM and time-series modes. |
| 4 | Tabular: distinguish synthetic vs real-data tasks? | **No for v1** | Treat all tasks uniformly. User can note this in scenario name. |
| 5 | TabPFN test-time compute: training calc or separate? | **Fold into training calc** via `testTimeComputeMultiplier` | Simpler than a separate inference calculator. Multiplier defaults to 1. |
| 6 | Classical tabular: support CPU cluster cost estimation? | **No for v1** | Focus on GPU. CPU is a future addition. |
| 7 | Support empirical benchmark libraries? | **No for v1** — user-provided calibration presets only | Ship with rough benchmark-derived defaults, but no curated library. |

## Implementation Decisions

| Topic | Decision | Rationale |
|-------|----------|-----------|
| **Model family selector UI** | **Tabs** across top of input rail | Only 4 modes — tabs are more discoverable than a dropdown. |
| **Classical tabular defaults** | Ship with **benchmark-derived throughput defaults** for LightGBM, XGBoost, CatBoost, Random Forest | Users won't know what to enter otherwise. Defaults are rough estimates, clearly labeled as such. |
| **URL hash / export schema** | **Version the schema** (`schema_version: "2.0"`). Fall back to LLM mode if no `modelFamily` param present. | Backwards compatible with existing shared URLs. |
| **Testing strategy** | All **64 existing LLM tests must pass** throughout the refactor. New modes get their own test files with golden cases from the spec. | No regressions. |
| **LLM mode additions** | Add `architectureFactor` (default 6) and `trainingTokensOverride` as advanced LLM inputs | Per spec. These are new fields that don't exist today. |
| **Full refactor** | Yes — discriminated union types, domain adapter pattern, refactored store | Per spec Section 16. Work done on a **feature branch**. |
| **Branch strategy** | Feature branch `feature/model-family-extensions` off main | Keep main stable. |

## Deferred Decisions

- MoE support across model families
- CPU cluster cost estimation for classical tabular
- Curated benchmark library for throughput coefficients
- Explicit attention FLOP modeling for long sequences
- Separate inference/serving calculator
