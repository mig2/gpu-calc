# LLM GPU Calculator — Functional Spec Extension: Model Family Extensions

**Document type:** Functional specification extension  
**Related documents:**

- `llm_gpu_calculator_functional_spec.md`
- `llm_gpu_calculator_technical_quant_spec.md`
- `llm_gpu_calculator_spa_design_spec.md`
- `scaling_laws_time_series_tabular_models.md`

**Prepared for:** Matt Greenwood  
**Date:** 2026-06-19  
**Status:** Draft v1

---

## 1. Purpose

This document extends the original LLM Training GPU Calculator functional specification to support additional model families beyond dense language-model pretraining. The original calculator was designed around Chinchilla-style language-model scaling, where the user can begin with a model parameter count, derive a training-token target, estimate FLOPs, and convert that compute requirement into GPU count for a given training or retraining window.

That formulation is powerful for LLMs, but it should not be applied blindly to time-series or tabular models. Time-series and tabular foundation models have emerging scaling-law behavior, but they do not share a single canonical data unit analogous to the text token, and they do not yet have a broadly accepted equivalent of the Chinchilla `tokens ≈ 20 × parameters` rule.

The purpose of this extension is to add a **model family layer** to the SPA so that the calculator can support:

```text
1. LLM / token transformer models
2. Time-series foundation models
3. Tabular foundation models
4. Classical tabular / GBDT-style models
```

The design goal is to preserve the shared GPU/FLOPs machinery while making each model family’s data-unit and architecture assumptions explicit.

---

## 2. Product Goal for the Extension

The extended calculator should answer a broader question:

> Given a model family, data shape, model size, hardware target, and training window, what is the approximate accelerator requirement, and how sensitive is the estimate to assumptions about tokenization, architecture, MFU, and overhead?

The extension should allow a user to compare LLM, time-series, and tabular training regimes while clearly distinguishing between high-confidence LLM estimates and more empirical, lower-confidence estimates for emerging model families.

---

## 3. Non-Goals

The extension should not pretend that all model families have equally mature scaling laws. Specifically, it is not intended to:

- Provide a universal Chinchilla-style law for time-series models.
- Provide a universal Chinchilla-style law for tabular models.
- Predict model quality, downstream accuracy, loss, or business value.
- Replace empirical benchmarking for a particular architecture or codebase.
- Model every detail of distributed training, compiler behavior, memory layout, or data-pipeline bottlenecks.
- Treat classical tabular models, such as XGBoost or LightGBM, as transformer-style FLOP-dominated training workloads unless the user explicitly selects a transformer-like model.

---

## 4. Core Architectural Change

The original LLM calculator can be thought of as:

```text
parameters → training tokens → FLOPs → GPU count
```

The extended calculator should use a layered architecture:

```text
Domain adapter:
  Converts raw domain-specific data into effective training units.

Compute estimator:
  Converts effective training units and model parameters into estimated FLOPs.

Hardware estimator:
  Converts estimated FLOPs and training window into GPU count by SKU.

Presentation layer:
  Shows estimates, traces, sensitivity ranges, confidence labels, and warnings.
```

This preserves the original LLM path while allowing time-series and tabular modes to introduce their own data-unit calculations.

---

## 5. Model Family Selector

The SPA should add a top-level model family selector. This selector changes the input panel, formulas, warnings, and explanatory text.

```text
Model family:
  - LLM / language transformer
  - Time-series foundation model
  - Tabular foundation model
  - Classical tabular / GBDT
```

The selected model family should be visible at all times, preferably in the main scenario header.

### 5.1 Acceptance Criteria

- The user can switch between model families without losing previously entered scenario values where fields overlap.
- Each model family has a distinct input schema.
- The output panel always includes GPU count, training-window sensitivity, hardware comparison, and a calculation trace.
- The confidence level changes based on model family and architecture choices.
- Warnings update when the selected model family implies low-confidence or high-risk assumptions.

---

## 6. Shared Inputs Across Model Families

The following inputs should remain common across all modes where applicable:

| Input            | Type          | Default              | Notes                                              |
| ---------------- | -------------:| --------------------:| -------------------------------------------------- |
| Model parameters | number + unit | family-dependent     | M/B/T units supported.                             |
| Training window  | duration      | 30 days              | Accept hours, days, weeks.                         |
| GPU SKU          | multi-select  | H100 SXM             | H100, H200, B200, GB200, custom.                   |
| Precision        | enum          | BF16 dense           | BF16 default; FP8 optional with warning.           |
| MFU              | percentage    | family/SKU-dependent | Editable per scenario and SKU.                     |
| Availability     | percentage    | 90%                  | Useful wall-clock fraction.                        |
| Overhead factor  | number        | 1.10                 | Checkpoints, evals, stalls, retries.               |
| Hardware profile | object        | selected GPU         | Peak FLOPs, memory, bandwidth, interconnect notes. |

### 6.1 Shared Output Fields

Every model family should produce:

```text
effective_training_units
effective_training_tokens_or_equivalent
estimated_base_FLOPs
overhead_adjusted_FLOPs
sustained_GPU_FLOP/s
required_GPU_count_by_SKU
H100_equivalent_count
training_time_for_fixed_GPU_count
sensitivity_table
warnings
confidence_label
calculation_trace
```

---

## 7. Mode 1: LLM / Language Transformer

The existing LLM mode remains the cleanest and highest-confidence mode.

### 7.1 Inputs

| Input                    | Type          | Default          | Notes                                                     |
| ------------------------ | -------------:| ----------------:| --------------------------------------------------------- |
| Model parameters         | number + unit | 70B              | Core driver.                                              |
| Tokens per parameter     | number        | 20               | Chinchilla-style default.                                 |
| Training tokens override | number + unit | blank            | Optional override.                                        |
| Training mode            | enum          | Full pretraining | Full pretraining, continued pretraining, SFT, LoRA, RLHF. |
| Architecture factor      | number        | 6                | Default dense transformer training approximation.         |

### 7.2 Formula

```text
N = model parameters
D = training tokens
D = tokens_per_parameter × N, unless overridden
training_FLOPs = architecture_factor × N × D
total_FLOPs = training_FLOPs × overhead_factor
```

With defaults:

```text
D = 20N
training_FLOPs ≈ 6ND ≈ 120N²
```

### 7.3 Confidence

```text
High confidence:
  Dense decoder-only transformer pretraining with known N and D.

Medium confidence:
  Continued pretraining or domain-adaptive pretraining.

Low confidence:
  SFT, LoRA, RLHF, MoE, retrieval-augmented training, or nonstandard architectures.
```

### 7.4 Warnings

- If training mode is not full pretraining, state that the full-pretraining formula may overestimate compute.
- If MoE is selected in a future version, require active parameters and total parameters separately.
- If FP8 is selected, warn that end-to-end FP8 training may not achieve a simple peak-based estimate.

---

## 8. Mode 2: Time-Series Foundation Model

Time-series mode estimates compute from raw series structure, window generation, patching, and tokenization assumptions.

### 8.1 Inputs

| Input                        | Type          | Default             | Notes                                                              |
| ---------------------------- | -------------:| -------------------:| ------------------------------------------------------------------ |
| Model parameters             | number + unit | 1B                  | Time-series foundation models may be smaller than LLMs.            |
| Number of series             | number        | 10M                 | Count of independent time series.                                  |
| Average timesteps per series | number        | 1,000               | After resampling and cleaning.                                     |
| Variables per series         | number        | 1                   | Channels, features, or variates.                                   |
| Lookback window              | number        | 256                 | Input context length in timesteps.                                 |
| Forecast horizon             | number        | 64                  | Output horizon in timesteps.                                       |
| Stride                       | number        | 64                  | Window step size.                                                  |
| Patch size                   | number        | 16                  | Timesteps per token/patch.                                         |
| Tokenization mode            | enum          | Channel-compressed  | Channel-compressed, channel-expanded, custom.                      |
| Epochs / repeats             | number        | 1                   | Number of passes or equivalent augmentation factor.                |
| Architecture type            | enum          | Decoder transformer | Encoder, decoder, encoder-decoder, patch transformer, MoE, custom. |
| Architecture factor          | number        | 6                   | Editable; default LLM-like dense transformer factor.               |

### 8.2 Derived Quantities

```text
windows_per_series = floor(
  (average_timesteps_per_series - lookback_window - forecast_horizon) / stride
) + 1
```

If the value is less than or equal to zero, the app should display an error and withhold the GPU estimate until the user corrects the inputs.

For patch-based tokenization:

```text
patches_per_window = ceil(lookback_window / patch_size)
```

For channel-compressed tokenization:

```text
tokens_per_window = patches_per_window
```

For channel-expanded tokenization:

```text
tokens_per_window = variables_per_series × patches_per_window
```

For custom tokenization:

```text
tokens_per_window = user_supplied_tokens_per_window
```

Effective training tokens:

```text
effective_training_tokens =
    number_of_series
  × windows_per_series
  × tokens_per_window
  × epochs_or_repeats
```

Estimated training FLOPs:

```text
training_FLOPs = architecture_factor × model_parameters × effective_training_tokens
```

### 8.3 Output Additions

Time-series mode should include a data-unit breakdown:

```text
raw timesteps observed
windows generated
patches per window
tokens per window
effective training tokens
```

The calculation trace should show each of these steps before the FLOP estimate.

### 8.4 Sensitivity Requirements

Time-series mode should add sensitivity tables for:

```text
lookback window
forecast horizon
stride
patch size
tokenization mode
MFU
GPU SKU
training window
```

### 8.5 Warnings

The app should display warnings when:

- `lookback_window + forecast_horizon > average_timesteps_per_series`.
- `stride` is very small relative to the lookback window, creating highly overlapping windows and possibly inflated data volume.
- `patch_size` is larger than the lookback window.
- `variables_per_series` is high and channel-expanded tokenization is selected.
- The result relies on a Chinchilla-like tokens/parameters analogy rather than an empirically calibrated time-series scaling law.
- The architecture is MoE or custom and the user has not supplied active parameters or an architecture factor.

### 8.6 Confidence

```text
Medium confidence:
  Transformer-style time-series models with explicit patch/window tokenization.

Medium-low confidence:
  Encoder-decoder, masked, diffusion-style, or heavily augmented objectives.

Low confidence:
  MoE, retrieval-based, or custom architectures without empirical calibration.
```

### 8.7 Acceptance Criteria

- A user can enter the canonical example from the companion note:

```text
model_parameters = 1B
number_of_series = 10M
average_timesteps_per_series = 1,000
variables_per_series = 4
lookback_window = 256
forecast_horizon = 64
stride = 64
patch_size = 16
tokenization_mode = channel-expanded
epochs = 1
architecture_factor = 6
```

- The app computes:

```text
windows_per_series = 11
tokens_per_window = 64
effective_training_tokens = 7.04B
training_FLOPs ≈ 4.224e19
```

- The app then converts the FLOP estimate into GPU counts using the same hardware estimator as LLM mode.

---

## 9. Mode 3: Tabular Foundation Model

Tabular foundation model mode estimates compute from task/table structure, rows, columns, tokenization, and architecture type.

### 9.1 Inputs

| Input                        | Type          | Default         | Notes                                                                          |
| ---------------------------- | -------------:| ---------------:| ------------------------------------------------------------------------------ |
| Model parameters             | number + unit | 100M            | Foundation model parameter count.                                              |
| Number of pretraining tasks  | number        | 1M              | A task may be a table, dataset, synthetic problem, or supervised split.        |
| Rows per task                | number        | 1,024           | Context rows or training rows per task.                                        |
| Columns per task             | number        | 100             | Feature count; label handled separately if needed.                             |
| Tokenization mode            | enum          | Row-tokenized   | Row-tokenized, cell-tokenized, axial/row-column, custom.                       |
| Epochs / repeats             | number        | 1               | Passes, augmentation repeats, or generated tasks multiplier.                   |
| Architecture type            | enum          | Row transformer | Row transformer, cell transformer, axial transformer, TabPFN-like ICL, custom. |
| Architecture factor          | number        | 6               | Editable; not a universal constant.                                            |
| Label/task type              | enum          | supervised      | Classification, regression, mixed, self-supervised.                            |
| Feature type mix             | enum/object   | mixed           | Numeric, categorical, text, date/time, missingness.                            |
| Context rows at inference    | number        | optional        | For inference/test-time compute view.                                          |
| Test-time compute multiplier | number        | 1               | For ensembles, repeated inference, adaptation, or TabPFN-style scaling.        |

### 9.2 Tokenization Formulas

For row-tokenized models:

```text
sequence_length_per_task = rows_per_task
```

For cell-tokenized models:

```text
sequence_length_per_task = rows_per_task × columns_per_task
```

For axial or row-column models:

```text
row_tokens = rows_per_task
column_tokens = columns_per_task
sequence_length_equivalent = custom or architecture-specific
```

The first version should let the user choose one of two approximations for axial models:

```text
Conservative token count:
  effective_tokens_per_task = rows_per_task + columns_per_task

Cell-equivalent token count:
  effective_tokens_per_task = rows_per_task × columns_per_task
```

For custom models:

```text
effective_tokens_per_task = user_supplied_tokens_per_task
```

Effective training tokens:

```text
effective_training_tokens =
    number_of_pretraining_tasks
  × effective_tokens_per_task
  × epochs_or_repeats
```

Training FLOPs:

```text
training_FLOPs = architecture_factor × model_parameters × effective_training_tokens
```

### 9.3 Attention Feasibility Warning

For transformer-style tabular models, the app should check dense attention risk.

```text
attention_risk_score = sequence_length_per_task²
```

The app does not need to present this score as a precise FLOP estimate, but it should warn when dense attention over the implied sequence is likely infeasible.

Suggested warning thresholds:

```text
sequence_length_per_task > 16,384:
  Show medium warning for dense attention.

sequence_length_per_task > 65,536:
  Show high warning for dense attention.

sequence_length_per_task > 100,000:
  Show severe warning unless sparse/factorized attention is selected.
```

### 9.4 Output Additions

Tabular foundation mode should show:

```text
pretraining tasks
tables/tasks processed
rows per task
columns per task
tokenization mode
effective tokens per task
effective training tokens
attention feasibility warning
```

### 9.5 Sensitivity Requirements

Tabular mode should include sensitivity tables for:

```text
rows per task
columns per task
tokenization mode
number of tasks
epochs / repeats
architecture factor
MFU
GPU SKU
training window
```

### 9.6 Confidence

```text
Medium-low confidence:
  Row-tokenized or axial tabular foundation models with explicit tokenization.

Low confidence:
  Cell-tokenized dense transformers over long tables.

Low confidence:
  Retrieval-augmented, test-time-scaling, TabPFN-like, or custom systems without calibration data.
```

### 9.7 Acceptance Criteria

For the row-tokenized example:

```text
model_parameters = 100M
number_of_pretraining_tasks = 1M
rows_per_task = 1,024
columns_per_task = 100
tokenization_mode = row-tokenized
epochs = 1
architecture_factor = 6
```

The app computes:

```text
effective_tokens_per_task = 1,024
effective_training_tokens = 1.024B
training_FLOPs ≈ 6.144e17
```

For the cell-tokenized variant with the same task dimensions, the app computes:

```text
effective_tokens_per_task = 102,400
effective_training_tokens = 102.4B
training_FLOPs ≈ 6.144e19
```

The app should also display a severe dense-attention warning for the cell-tokenized case.

---

## 10. Mode 4: Classical Tabular / GBDT

Classical tabular models should not use transformer-style `6 × parameters × tokens` formulas. For gradient-boosted trees and related algorithms, the relevant axes are algorithmic and empirical rather than parameter/token scaling.

### 10.1 Inputs

| Input                            | Type   | Default         | Notes                                               |
| -------------------------------- | ------:| ---------------:| --------------------------------------------------- |
| Algorithm                        | enum   | LightGBM        | XGBoost, LightGBM, CatBoost, Random Forest, custom. |
| Rows                             | number | 1M              | Training rows.                                      |
| Columns                          | number | 100             | Feature count.                                      |
| Boosting rounds / trees          | number | 1,000           | Algorithm-dependent.                                |
| Max depth / leaves               | number | 8 / custom      | Tree complexity.                                    |
| Histogram bins                   | number | 256             | For histogram-based algorithms.                     |
| CV folds                         | number | 1               | Multiplies training work.                           |
| Hyperparameter trials            | number | 1               | Multiplies training work.                           |
| CPU/GPU implementation           | enum   | GPU             | Affects throughput coefficient.                     |
| Empirical throughput coefficient | number | required/custom | Rows × columns × rounds per second or similar.      |

### 10.2 Estimation Approach

The first version should use an empirical throughput model rather than a theoretical FLOP model.

```text
work_units = rows × columns × boosting_rounds × CV_folds × hyperparameter_trials
estimated_seconds = work_units / empirical_throughput_coefficient
```

The app may later support algorithm-specific models for histogram construction, split search, categorical handling, sparsity, and GPU implementation.

### 10.3 Outputs

```text
estimated training time
work units
throughput assumption
CPU/GPU implementation note
calibration prompt
```

If the user wants GPU count for GBDT training, the app should frame this as a throughput-calibration problem rather than a scaling-law problem.

### 10.4 Warnings

- Warn that classical tabular estimates require empirical calibration.
- Warn that theoretical FLOPs are not the right abstraction for many GBDT implementations.
- Warn that cross-validation and hyperparameter search often dominate wall-clock cost.

### 10.5 Confidence

```text
Medium confidence:
  If the user supplies an empirical throughput coefficient from a benchmark on similar data.

Low confidence:
  Without calibration data.
```

---

## 11. Calibration Mode Extension

The existing calibration mode should be generalized across model families.

### 11.1 Calibration Inputs

| Input                     | Applies To        | Notes                                          |
| ------------------------- | ----------------- | ---------------------------------------------- |
| Known wall-clock time     | all               | Actual elapsed training time.                  |
| GPU count and SKU         | all               | Hardware used.                                 |
| Precision                 | transformer modes | BF16, FP8, etc.                                |
| Model parameters          | transformer modes | Used to back-solve architecture factor or MFU. |
| Effective training tokens | transformer modes | Derived or supplied.                           |
| Work units                | GBDT mode         | Rows × columns × rounds × trials, etc.         |
| Observed throughput       | all               | Optional direct override.                      |

### 11.2 Calibration Outputs

For LLM, time-series, and tabular foundation modes:

```text
back_solved_MFU
back_solved_architecture_factor
scenario-specific throughput
reusable preset
```

For classical tabular mode:

```text
empirical_throughput_coefficient
estimated scaling to larger row/column/trial counts
reusable benchmark preset
```

### 11.3 Acceptance Criteria

- User can enter a known time-series training run and save the resulting MFU/architecture-factor preset.
- User can enter a known GBDT benchmark and save a throughput coefficient.
- Saved presets are marked as local empirical calibrations, not general laws.

---

## 12. Confidence Labels

Every result should include a confidence label. This is especially important once the calculator supports model families beyond LLMs.

| Confidence | Meaning                                                                     | Example                                                                          |
| ---------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| High       | Formula is widely used and inputs are well-defined.                         | Dense LLM pretraining with known N and D.                                        |
| Medium     | Formula is plausible but architecture/data-unit choices matter.             | Time-series transformer with explicit patching.                                  |
| Medium-low | Estimate is useful for comparison but not procurement.                      | Row-tokenized tabular foundation model.                                          |
| Low        | Estimate depends heavily on custom implementation or empirical calibration. | Cell-tokenized tabular model, TabPFN-style test-time scaling, uncalibrated GBDT. |

The confidence label should appear near the top-line GPU estimate and in exported summaries.

---

## 13. Warning System Extension

The warning system should be expanded from LLM-specific guardrails to model-family-specific guardrails.

### 13.1 Shared Warnings

- GPU count exceeds 1,024: distributed-systems warning.
- GPU count exceeds 10,000: hyperscale warning.
- MFU exceeds 60%: optimistic-utilization warning.
- FP8 selected: recipe/quality warning.
- Memory lower bound exceeds available GPU memory: sharding/parallelism warning.
- Custom architecture factor missing: require explicit assumption.

### 13.2 Time-Series Warnings

- Invalid window geometry: lookback + horizon exceeds series length.
- Very small stride creates many overlapping windows.
- Channel-expanded tokenization with many variables may inflate compute.
- Long lookback plus dense attention may become infeasible.
- Forecast horizon is long relative to lookback or series length.

### 13.3 Tabular Warnings

- Cell-tokenized mode produces very long sequence length.
- Dense attention over large tables is likely infeasible.
- Very high column count may require specialized feature embeddings or column attention.
- Test-time compute multiplier is high and may dominate serving cost.
- Tabular foundation estimate is not comparable to GBDT estimate without calibration.

### 13.4 Classical Tabular Warnings

- No empirical throughput coefficient supplied.
- CV folds or hyperparameter trials dominate total work.
- Sparse/categorical-heavy data may not scale like dense numeric data.
- CPU/GPU implementation choice materially changes performance.

---

## 14. Output Design Requirements

The output panel should adapt to the selected model family but preserve a common structure.

### 14.1 Main Estimate Card

Always show:

```text
required GPU count
selected GPU SKU
training window
confidence label
major warning count
```

### 14.2 Data-Unit Breakdown Card

LLM:

```text
parameters
tokens per parameter
training tokens
```

Time series:

```text
series
timesteps
windows per series
patches per window
tokens per window
effective training tokens
```

Tabular foundation:

```text
pretraining tasks
rows per task
columns per task
tokenization mode
effective tokens per task
effective training tokens
```

Classical tabular:

```text
rows
columns
boosting rounds
CV folds
hyperparameter trials
work units
empirical throughput
```

### 14.3 FLOPs / Throughput Card

Transformer modes show:

```text
base FLOPs
overhead-adjusted FLOPs
sustained FLOP/s per GPU
H100-equivalent count
```

Classical tabular mode shows:

```text
work units
throughput coefficient
estimated time
scaling with trials/folds
```

### 14.4 Sensitivity Panel

All model families should support:

```text
MFU sensitivity
GPU SKU sensitivity
training-window sensitivity
```

Additional domain-specific sensitivities:

```text
LLM:
  tokens per parameter

Time series:
  lookback, stride, patch size, tokenization mode

Tabular foundation:
  rows, columns, tokenization mode, tasks

Classical tabular:
  rounds, folds, trials, throughput coefficient
```

---

## 15. Export Requirements

Exports should include the selected model family and all domain-specific assumptions.

### 15.1 JSON Export Schema

At minimum:

```json
{
  "schema_version": "2.0",
  "model_family": "time_series_foundation",
  "scenario_name": "Example TSFM run",
  "inputs": {},
  "derived_quantities": {},
  "hardware": {},
  "results": {},
  "warnings": [],
  "confidence": "medium",
  "calculation_trace": []
}
```

### 15.2 Human-Readable Export

The text/Markdown export should include:

```text
scenario name
model family
input assumptions
data-unit breakdown
FLOP or work-unit estimate
GPU-count estimate
sensitivity summary
warnings
confidence label
```

---

## 16. State and Data Model Changes

The app should introduce a discriminated-union scenario model.

```typescript
type ModelFamily =
  | "llm"
  | "time_series_foundation"
  | "tabular_foundation"
  | "classical_tabular";

interface BaseScenario {
  id: string;
  name: string;
  modelFamily: ModelFamily;
  trainingWindowSeconds: number;
  hardwareSelections: HardwareSelection[];
  precision: PrecisionMode;
  availability: number;
  overheadFactor: number;
}

interface LlmScenario extends BaseScenario {
  modelFamily: "llm";
  parameters: number;
  tokensPerParameter: number;
  trainingTokensOverride?: number;
  architectureFactor: number;
}

interface TimeSeriesScenario extends BaseScenario {
  modelFamily: "time_series_foundation";
  parameters: number;
  numberOfSeries: number;
  averageTimestepsPerSeries: number;
  variablesPerSeries: number;
  lookbackWindow: number;
  forecastHorizon: number;
  stride: number;
  patchSize: number;
  tokenizationMode: "channel_compressed" | "channel_expanded" | "custom";
  customTokensPerWindow?: number;
  epochs: number;
  architectureFactor: number;
}

interface TabularFoundationScenario extends BaseScenario {
  modelFamily: "tabular_foundation";
  parameters: number;
  numberOfPretrainingTasks: number;
  rowsPerTask: number;
  columnsPerTask: number;
  tokenizationMode: "row" | "cell" | "axial" | "custom";
  customTokensPerTask?: number;
  epochs: number;
  architectureFactor: number;
  testTimeComputeMultiplier?: number;
}

interface ClassicalTabularScenario extends BaseScenario {
  modelFamily: "classical_tabular";
  algorithm: "xgboost" | "lightgbm" | "catboost" | "random_forest" | "custom";
  rows: number;
  columns: number;
  boostingRounds: number;
  maxDepth?: number;
  bins?: number;
  cvFolds: number;
  hyperparameterTrials: number;
  throughputCoefficient?: number;
}
```

---

## 17. Testing Requirements

### 17.1 Unit Tests

Add unit tests for:

- LLM token derivation and FLOP calculation.
- Time-series window generation.
- Time-series channel-compressed vs channel-expanded tokenization.
- Tabular row-tokenized vs cell-tokenized token counts.
- Dense-attention warning thresholds.
- Classical tabular work-unit calculation.
- GPU count conversion shared across transformer modes.

### 17.2 Golden Test Cases

#### Time-Series Golden Case

Input:

```text
parameters = 1B
series = 10M
timesteps = 1,000
variables = 4
lookback = 256
horizon = 64
stride = 64
patch = 16
tokenization = channel-expanded
epochs = 1
architecture_factor = 6
```

Expected:

```text
windows_per_series = 11
tokens_per_window = 64
effective_training_tokens = 7.04B
training_FLOPs = 4.224e19
```

#### Tabular Row-Token Golden Case

Input:

```text
parameters = 100M
tasks = 1M
rows = 1,024
columns = 100
tokenization = row
epochs = 1
architecture_factor = 6
```

Expected:

```text
effective_tokens_per_task = 1,024
effective_training_tokens = 1.024B
training_FLOPs = 6.144e17
```

#### Tabular Cell-Token Golden Case

Input:

```text
parameters = 100M
tasks = 1M
rows = 1,024
columns = 100
tokenization = cell
epochs = 1
architecture_factor = 6
```

Expected:

```text
effective_tokens_per_task = 102,400
effective_training_tokens = 102.4B
training_FLOPs = 6.144e19
warning = severe dense-attention warning
```

---

## 18. Rollout Plan

### Phase 1: Model Family Infrastructure

- Add model family selector.
- Refactor existing LLM logic into an `llm` domain adapter.
- Add shared compute and hardware estimator interfaces.
- Add confidence labels and generalized warning system.

### Phase 2: Time-Series Foundation Mode

- Add time-series input panel.
- Implement window/patch/token derivation.
- Add time-series sensitivity table.
- Add golden tests and trace output.

### Phase 3: Tabular Foundation Mode

- Add tabular foundation input panel.
- Implement row/cell/axial/custom tokenization.
- Add dense-attention warnings.
- Add tabular sensitivity table.

### Phase 4: Classical Tabular Mode

- Add GBDT-style work-unit model.
- Require or encourage empirical throughput calibration.
- Add benchmark preset support.

### Phase 5: Calibration and Presets

- Allow users to save calibration presets by model family.
- Add import/export of scenario bundles.
- Add documentation pages explaining confidence levels and limitations.

---

## 19. Updated User Stories

| User Story                                                                                              | Acceptance Criteria                                                                |
| ------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| As a user, I want to select time-series mode and estimate training GPUs from series/window assumptions. | The app derives windows, patches, effective tokens, FLOPs, and GPU count.          |
| As a user, I want to see how patch size affects a time-series estimate.                                 | The sensitivity panel shows GPU count across patch sizes.                          |
| As a user, I want to compare row-tokenized and cell-tokenized tabular models.                           | The app shows token count, FLOPs, and dense-attention warnings for each.           |
| As a user, I want the app not to overclaim precision for tabular scaling.                               | The result includes a medium-low or low confidence label and explanatory warning.  |
| As a user, I want to enter a known benchmark run.                                                       | Calibration mode back-solves MFU, architecture factor, or throughput coefficient.  |
| As a user, I want to export a scenario.                                                                 | Export includes model family, domain assumptions, trace, confidence, and warnings. |

---

## 20. Open Questions

- Should time-series mode default to channel-compressed or channel-expanded tokenization?
- Should the calculator include attention FLOPs explicitly for very long sequence lengths?
- Should MoE be added as a cross-cutting architecture option or deferred to a separate extension?
- Should tabular foundation mode distinguish synthetic pretraining tasks from real-data tasks?
- Should TabPFN-style test-time compute be modeled as a separate inference calculator rather than folded into training?
- Should classical tabular mode support cost estimation for CPU clusters as well as GPUs?
- Should the app support empirical benchmark libraries or only user-provided calibration presets?

---

## 21. References

[1] Thomas D. P. Edwards, James Alvey, Justin Alsing, Nam H. Nguyen, Benjamin D. Wandelt, “Scaling-laws for Large Time-series Models,” arXiv:2405.13867.  
https://arxiv.org/abs/2405.13867

[2] Qingren Yao et al., “Towards Neural Scaling Laws for Time Series Foundation Models,” arXiv:2410.12360 / ICLR 2025.  
https://arxiv.org/abs/2410.12360

[3] Jingzhe Shi, Qinwei Ma, Huan Ma, Lei Li, “Scaling Law for Time Series Forecasting,” arXiv:2405.15124.  
https://arxiv.org/abs/2405.15124

[4] Noah Hollmann et al., “Accurate predictions on small data with a tabular foundation model,” Nature, 2025.  
https://www.nature.com/articles/s41586-024-08328-6

[5] Junwei Ma et al., “TabDPT: Scaling Tabular Foundation Models,” arXiv:2410.18164 / ICLR 2025.  
https://arxiv.org/abs/2410.18164

---

## 22. Summary

The model family extension should make the calculator more general without weakening the precision of the original LLM mode. The key is to introduce domain adapters that convert raw domain quantities into effective training units, then pass those units through the existing compute and hardware machinery.

For LLMs, the default data unit remains the text token and Chinchilla-20 remains a useful default.

For time-series foundation models, the data unit should be derived from series, windows, variables, patches, stride, and horizon.

For tabular foundation models, the data unit should be derived from tasks, rows, columns, and tokenization architecture.

For classical tabular models, the calculator should shift away from transformer FLOPs and toward empirical throughput calibration.

The SPA should make these differences explicit, show confidence labels, and expose all assumptions in the calculation trace.
