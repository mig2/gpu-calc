# Model Family Extension — Implementation Phases

**Date:** 2026-06-19
**Source specs:** `model_family_extensions_functional_spec.md`, `scaling_laws_time_series_tabular_models.md`
**Decisions:** `docs/decisions/2026-06-19-model-family-extension-decisions.md`
**Branch:** `feature/model-family-extensions`

---

## Phase 1: Model Family Infrastructure

Refactor the existing codebase to support multiple model families through a domain adapter pattern.

### Tasks

1. **Discriminated union types** — Replace `TrainingScenario` with `BaseScenario` + `LlmScenario | TimeSeriesScenario | TabularFoundationScenario | ClassicalTabularScenario`. Add `ModelFamily` type. Add `ConfidenceLevel` type.

2. **Domain adapter pattern** — Create adapter interfaces: each model family has a function that converts domain-specific inputs into `{ effectiveTrainingTokens, estimatedFlops }`. The existing `estimateTrainingRun` becomes the hardware estimator that takes FLOPs + window + GPU → GPU count.

3. **Refactor engine** — Split `calculator.ts` into:
   - `hardware-estimator.ts` — shared GPU count calculation from FLOPs
   - `llm-adapter.ts` — LLM-specific token/FLOP derivation (wraps existing 6ND logic)
   - `calculator.ts` — orchestrator that calls adapter → hardware estimator

4. **Refactor store** — Update Zustand store to hold a discriminated union scenario. Add `modelFamily` field. Keep all existing LLM setters working. Add `setModelFamily()` action that switches input schema while preserving shared fields.

5. **Model family selector UI** — Tabs at top of input rail: LLM, Time Series, Tabular Foundation, Classical Tabular. Switching tab changes the input panel. Shared fields (window, GPU, MFU, availability, overhead) persist across switches.

6. **Confidence labels** — Add confidence badge to result cards. LLM full pretraining = High. Other LLM modes = Medium. Time series = Medium. Tabular = Medium-low. Classical = depends on calibration.

7. **Extended warning system** — Generalize warnings to support model-family-specific rules. Keep existing LLM warnings. Add shared warnings (>10K GPUs hyperscale, MFU >60% optimistic).

8. **Add `architectureFactor` and `trainingTokensOverride` to LLM mode** — New advanced inputs for LLM panel per spec.

9. **All 64 existing LLM tests must still pass** — No regressions. Refactored code produces identical results.

---

## Phase 2: Time-Series Foundation Mode

Add the first non-LLM model family.

### Tasks

1. **Time-series adapter** — `time-series-adapter.ts`: derives windows_per_series, patches_per_window, tokens_per_window, effective_training_tokens from series/window/patch/stride/horizon/tokenization inputs. Returns effective tokens + FLOPs.

2. **Time-series adapter tests** — Golden test case from spec:
   - 1B params, 10M series, 1000 timesteps, 4 variables, lookback 256, horizon 64, stride 64, patch 16, channel-expanded → 11 windows, 64 tokens/window, 7.04B effective tokens, 4.224e19 FLOPs
   - Channel-compressed variant
   - Edge cases: invalid window geometry, stride=1 overlap warning

3. **Time-series input panel** — `TimeSeriesForm.tsx`: number of series, avg timesteps, variables, lookback, horizon, stride, patch size, tokenization mode (channel-compressed/expanded/custom), epochs, architecture type, architecture factor. With sensible defaults from spec.

4. **Time-series data breakdown card** — Shows: raw timesteps, windows generated, patches/window, tokens/window, effective training tokens. Displayed before GPU results.

5. **Time-series sensitivity** — Sensitivity tables for lookback, stride, patch size, tokenization mode in addition to MFU/window/SKU.

6. **Time-series warnings** — Invalid window geometry, small stride overlap, high variables + channel-expanded, long lookback + dense attention.

7. **Time-series trace** — Calculation trace showing full derivation from raw series → windows → patches → tokens → FLOPs → GPUs.

8. **Update export/URL encoding** — Add `schema_version: "2.0"`, `modelFamily` field. Graceful fallback: missing family = LLM. Time-series fields encoded in URL hash.

---

## Phase 3: Tabular Foundation Mode (future)

### Tasks
1. Tabular foundation adapter — row/cell/axial/custom tokenization derivation
2. Tabular adapter tests — golden cases: row-tokenized (1.024B tokens, 6.144e17 FLOPs), cell-tokenized (102.4B tokens, 6.144e19 FLOPs, severe attention warning)
3. Tabular input panel — tasks, rows, columns, tokenization mode, epochs, architecture factor, test-time compute multiplier
4. Dense attention feasibility warnings — thresholds at 16K, 65K, 100K sequence length
5. Tabular data breakdown card
6. Tabular sensitivity tables — rows, columns, tokenization mode, tasks
7. Tabular trace

---

## Phase 4: Classical Tabular / GBDT Mode (future)

### Tasks
1. GBDT work-unit estimator — rows × columns × rounds × folds × trials / throughput coefficient
2. Benchmark-derived throughput defaults — LightGBM, XGBoost, CatBoost, Random Forest (GPU)
3. GBDT input panel — algorithm, rows, columns, rounds, depth, bins, CV folds, HP trials, CPU/GPU, throughput coefficient
4. GBDT output card — work units, throughput, estimated time (not FLOPs)
5. GBDT warnings — no calibration, CV/HP search dominance, sparse data
6. GBDT calibration — enter benchmark, back-solve throughput coefficient, save preset

---

## Phase 5: Calibration & Presets (future)

### Tasks
1. Generalized calibration across model families — back-solve MFU or architecture factor for transformer modes, throughput coefficient for GBDT
2. Preset management — save/load calibration presets by model family, label as empirical
3. Scenario bundles — import/export full scenario with model family, all inputs, calibration data
4. Help documentation update — confidence levels, model family explanations, limitations
