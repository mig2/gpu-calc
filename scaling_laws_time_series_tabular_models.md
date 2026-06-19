# Scaling Laws Beyond LLMs: Time-Series and Tabular Models

**Companion note for the LLM GPU Calculator project**  
**Date:** 2026-06-19  
**Purpose:** Extend the conceptual and quantitative framing of the LLM-oriented GPU calculator to time-series and tabular model families.

---

## 1. Executive Summary

The LLM calculator is unusually clean because the Chinchilla-style abstraction gives a simple relationship between model size, training tokens, compute, and expected loss. A first-order LLM training calculator can reasonably start from:

```text
training tokens ≈ 20 × parameters
training FLOPs ≈ 6 × parameters × training tokens
```

Time-series and tabular models do not yet have an equally canonical “Chinchilla law.” There are emerging scaling-law results, especially for time-series foundation models and tabular foundation models, but they are more architecture-dependent and more sensitive to the definition of the data unit.

For a practical calculator, the right extension is not to force all model families into the LLM token/parameter mold. Instead, the calculator should support multiple model-family modes:

```text
1. LLM / token-transformer mode
2. Time-series foundation model mode
3. Tabular foundation model mode
4. Optional classical tabular mode for GBDT-style models
```

The calculator can still estimate GPU requirements by reducing each mode to:

```text
effective training units → approximate FLOPs → GPU count
```

But the definition of “training unit” changes by domain:

```text
LLM:         text token
Time series: patch, window, timestep, or channel-time token
Tabular:     task, table, row, column, or cell
```

The resulting calculator should present estimates as sensitivity ranges rather than precise point estimates.

---

## 2. Why LLM Scaling Is Cleaner

The Chinchilla result is powerful because it turns a messy pretraining problem into a relatively compact planning model. For dense transformer LLMs, the planner can use:

```text
N = number of parameters
D = number of training tokens
training FLOPs ≈ 6ND
```

Chinchilla-style compute-optimality then gives a simple default:

```text
D ≈ 20N
```

This means that, under the Chinchilla-20 assumption:

```text
training FLOPs ≈ 120N²
```

That is why doubling the model size is not merely twice as expensive: if one also scales data compute-optimally, cost grows roughly quadratically.

For time-series and tabular models, there is no single universal analogue of `D ≈ 20N`. The data unit, task structure, and architecture choices are too varied.

---

## 3. Time-Series Models

### 3.1 Do Time-Series Scaling Laws Exist?

Yes, but they are less settled and less universal than LLM scaling laws.

Recent work has shown that large time-series models can exhibit power-law scaling with parameter count, dataset size, and training compute. Edwards et al. report analogous scaling behavior for foundational decoder-only time-series transformer models trained on heterogeneous time-series corpora, with power-law behavior over multiple orders of magnitude of scale. [1]

Other work complicates the story. Yao et al. study neural scaling laws for time-series foundation models across in-distribution and out-of-distribution data, comparing encoder-only and decoder-only transformer architectures. Their framing is important for a calculator because it emphasizes that architecture and distribution shift affect scaling behavior. [2]

Shi et al. argue that time-series forecasting has an additional variable that is not central in the same way for text: the look-back horizon. They explicitly address the observation that more data and larger models can help, but longer input horizons can sometimes hurt, depending on data characteristics and model behavior. [3]

### 3.2 Why Time-Series Is Different from LLMs

Text has a fairly natural unit: the token. Time-series data has several plausible units:

```text
timestep
channel-time cell
window
patch
series segment
forecasting task
```

The right unit depends on preprocessing and architecture. A model may tokenize time as individual timesteps, or it may patch a long numerical sequence into coarser segments. A multivariate time series may be treated as several independent channels, as a dense channel-time grid, or as a structured multivariate object.

In forecasting, additional variables also matter:

```text
lookback length
forecast horizon
sampling frequency / granularity
seasonality
trend structure
noise level
missingness
number of domains
number of independent series
```

This means that a Chinchilla-style rule such as `tokens = 20 × parameters` is not a safe default for time-series models.

### 3.3 Proposed Calculator Model for Time-Series Foundation Models

The calculator should expose time-series-specific data construction choices.

A first-pass effective-token estimate could be:

```text
effective_training_tokens =
    number_of_series
  × windows_per_series
  × variables_per_series
  × tokens_per_window
```

Where:

```text
windows_per_series ≈ floor((timesteps_per_series - lookback_window - forecast_horizon) / stride) + 1
```

If using patching:

```text
tokens_per_window ≈ lookback_window / patch_size
```

For multivariate series, there are two plausible choices:

```text
channel-compressed tokenization:
    tokens_per_window ≈ lookback_window / patch_size

channel-expanded tokenization:
    tokens_per_window ≈ variables_per_series × lookback_window / patch_size
```

The calculator should let the user choose this explicitly, because the compute difference can be large.

### 3.4 Approximate FLOPs for Transformer-Style Time-Series Models

For dense transformer-style training, a rough LLM-like approximation may still be useful:

```text
training_FLOPs ≈ architecture_factor × parameters × effective_training_tokens
```

With:

```text
architecture_factor ≈ 6
```

as a default for dense decoder-style transformer training.

But the calculator should expose this as an assumption, not a law. Time-series models may use encoder-only, decoder-only, encoder-decoder, masked reconstruction, diffusion-style objectives, patch-based transformers, or mixture-of-experts architectures.

Recommended first-pass parameterization:

```text
Inputs:
  model_parameters
  number_of_series
  average_timesteps_per_series
  variables_per_series
  lookback_window
  forecast_horizon
  stride
  patch_size
  tokenization_mode
  epochs_or_repeats
  architecture_type
  architecture_factor
  MFU
  availability
  overhead
  GPU type
```

### 3.5 Time-Series Worked Example

Suppose:

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

Then:

```text
windows_per_series = floor((1000 - 256 - 64) / 64) + 1
                   = floor(680 / 64) + 1
                   = 10 + 1
                   = 11
```

Tokens per window:

```text
tokens_per_window = variables × lookback_window / patch_size
                  = 4 × 256 / 16
                  = 64
```

Effective training tokens:

```text
effective_training_tokens = 10M × 11 × 64
                          = 7.04B tokens
```

Training FLOPs:

```text
training_FLOPs ≈ 6 × 1B × 7.04B
               ≈ 4.224e19 FLOPs
```

This is dramatically smaller than LLM-scale pretraining. But the example is also modest: many foundation-model settings would use far more series, longer histories, more domains, multiple epochs, synthetic augmentation, or larger models.

### 3.6 Time-Series Calculator Guidance

The calculator should communicate the following warning:

> Time-series scaling estimates depend strongly on how raw observations are converted into model tokens. Lookback length, patch size, stride, horizon, and multivariate tokenization can change compute by orders of magnitude.

The calculator should therefore show:

```text
raw observations
windows generated
effective tokens / patches
FLOPs
GPU count
sensitivity to lookback, stride, patch size, and MFU
```

---

## 4. Tabular Models

### 4.1 Do Tabular Scaling Laws Exist?

Yes, but again they are not yet as canonical as LLM scaling laws.

Tabular foundation models such as TabPFN and TabDPT demonstrate that transformer-like or in-context-learning approaches can work well in tabular settings. TabPFN is presented as a foundation model for small-to-medium tabular data and is reported to perform strongly on datasets up to 10,000 samples and 500 features. [4]

TabDPT explicitly frames itself as scaling tabular foundation models on real data. The authors report scaling with both model size and amount of available data, suggesting that larger tabular pretraining corpora and larger models can improve future systems. [5]

However, tabular ML remains different from LLMs because classical models, especially gradient-boosted decision trees, remain extremely competitive and often dominant for ordinary supervised tabular tasks. Tabular foundation models are not simply “LLMs for rows.” They usually rely on task-level pretraining, synthetic data generation, real-data retrieval, in-context learning, or specialized row/column attention.

### 4.2 Why Tabular Is Different from LLMs

A tabular dataset has multiple possible units:

```text
table
task
row
column
cell
label
feature type
```

The difference between row-tokenized and cell-tokenized architectures is enormous.

For a table with:

```text
rows = 10,000
columns = 500
```

A row-tokenized model may see a sequence length of roughly:

```text
sequence_length ≈ 10,000
```

A cell-tokenized model may see:

```text
sequence_length ≈ 10,000 × 500 = 5,000,000
```

That distinction dominates compute. A calculator must therefore treat tabular architecture as a first-class input.

### 4.3 Proposed Calculator Model for Tabular Foundation Models

A first-pass tabular pretraining-unit estimate could be:

```text
effective_training_units =
    number_of_pretraining_tasks
  × rows_per_task
  × columns_per_task
```

But the conversion from units to transformer tokens depends on architecture.

#### Cell-token transformer

```text
sequence_length_per_task ≈ rows_per_task × columns_per_task
```

This can become infeasible very quickly unless attention is sparse, factorized, or otherwise constrained.

#### Row-token transformer

```text
sequence_length_per_task ≈ rows_per_task
```

The feature vector is embedded into a row representation before attention across rows.

#### Column-row / axial attention model

```text
row_attention_cost    ≈ f(rows)
column_attention_cost ≈ g(columns)
combined_cost         ≈ row_attention_cost + column_attention_cost
```

This is often much more plausible than dense attention over all cells.

#### In-context tabular model

Many tabular foundation models are best thought of as learning a distribution over supervised learning tasks. In this case, the relevant scaling variables include:

```text
number_of_pretraining_tasks
rows_per_task
columns_per_task
feature-type diversity
label diversity
synthetic task diversity
real-data task diversity
context rows at inference
test-time compute multiplier
```

### 4.4 Approximate FLOPs for Tabular Transformer Models

A simple approximation analogous to the LLM formula can be used only after choosing a tokenization strategy:

```text
training_FLOPs ≈ architecture_factor × parameters × effective_training_tokens
```

Where:

```text
effective_training_tokens = tasks × sequence_length_per_task × epochs
```

But for tabular models, attention cost may be more important than in the simplified LLM formula when sequence lengths are large:

```text
attention_cost_per_layer ≈ O(sequence_length² × hidden_size)
```

Therefore, the calculator should warn when implied sequence length is too large for dense attention.

### 4.5 Tabular Worked Example: Row-Tokenized Foundation Model

Suppose:

```text
model_parameters = 100M
number_of_pretraining_tasks = 1M
rows_per_task = 1,024
columns_per_task = 100
tokenization_mode = row-tokenized
epochs = 1
architecture_factor = 6
```

Then:

```text
sequence_length_per_task = rows_per_task
                         = 1,024
```

Effective training tokens:

```text
effective_training_tokens = 1M × 1,024
                          = 1.024B tokens
```

Training FLOPs:

```text
training_FLOPs ≈ 6 × 100M × 1.024B
               ≈ 6.144e17 FLOPs
```

This is small compared with LLM pretraining. But it ignores the cost of embedding columns, task construction, preprocessing, retrieval, and any expensive test-time compute.

### 4.6 Tabular Worked Example: Cell-Tokenized Variant

Using the same table dimensions:

```text
number_of_pretraining_tasks = 1M
rows_per_task = 1,024
columns_per_task = 100
tokenization_mode = cell-tokenized
```

Then:

```text
sequence_length_per_task = 1,024 × 100
                         = 102,400
```

Effective training tokens:

```text
effective_training_tokens = 1M × 102,400
                          = 102.4B tokens
```

Training FLOPs:

```text
training_FLOPs ≈ 6 × 100M × 102.4B
               ≈ 6.144e19 FLOPs
```

This is 100× the row-tokenized estimate before accounting for attention scaling. With dense attention over 102,400 tokens, the attention cost may become prohibitive. This illustrates why architecture matters so much in tabular models.

### 4.7 Classical Tabular Models

For non-foundation tabular models, especially GBDTs, the LLM-style FLOPs calculator is the wrong abstraction. For XGBoost, LightGBM, CatBoost, random forests, and similar systems, the more relevant quantities are:

```text
number_of_rows
number_of_columns
number_of_trees
tree_depth
number_of_bins
number_of_boosting_rounds
sparsity
categorical handling
CPU vs GPU implementation
```

A classical tabular mode should therefore use algorithmic cost models rather than transformer-style `6ND` formulas.

A rough GBDT-oriented calculator might expose:

```text
rows
columns
boosting_rounds
max_depth
histogram_bins
GPU implementation
cross-validation folds
hyperparameter trials
```

And estimate training time empirically using benchmark-derived coefficients rather than theoretical FLOPs.

---

## 5. Proposed Extension to the Existing SPA

### 5.1 Add a Model Family Selector

The SPA should begin with a mode selector:

```text
Model family:
  - LLM / language transformer
  - Time-series foundation model
  - Tabular foundation model
  - Classical tabular / GBDT
```

Each family should have a different input panel, but all should feed into a shared compute backend.

### 5.2 Shared Compute Backend

All modes should eventually produce:

```text
effective_training_units
effective_training_tokens_or_equivalent
estimated_training_FLOPs
required_sustained_FLOPs
GPU count by SKU
sensitivity ranges
warnings / feasibility checks
```

Shared hardware assumptions remain useful:

```text
GPU type
precision
peak FLOPs
MFU
availability
overhead
memory capacity
memory bandwidth
interconnect assumptions
```

### 5.3 Domain-Specific Input Panels

#### LLM panel

```text
parameters
tokens_per_parameter
training tokens override
retraining window
precision
GPU type
MFU
availability
overhead
```

#### Time-series panel

```text
parameters
number of series
average timesteps per series
variables per series
lookback window
forecast horizon
stride
patch size
tokenization mode
epochs / repeats
architecture factor
retraining window
GPU type
MFU
availability
overhead
```

#### Tabular foundation model panel

```text
parameters
number of pretraining tasks
rows per task
columns per task
tokenization mode: row, cell, axial, custom
epochs / repeats
architecture factor
context rows at inference
test-time compute multiplier
retraining window
GPU type
MFU
availability
overhead
```

#### Classical tabular panel

```text
algorithm: XGBoost, LightGBM, CatBoost, Random Forest, custom
rows
columns
boosting rounds
tree depth
bins
CV folds
hyperparameter trials
CPU/GPU choice
empirical throughput coefficient
```

---

## 6. Recommended Output Design

For each model family, the calculator should show:

```text
1. Main estimate
   - GPUs required for selected retraining window
   - Training time for selected GPU count
   - H100-equivalent count

2. Data-unit breakdown
   - Raw data units
   - Effective tokens / patches / rows / cells
   - Architecture-specific expansion factor

3. FLOPs breakdown
   - Base FLOPs
   - Overhead-adjusted FLOPs
   - Sustained FLOPs per GPU

4. Sensitivity table
   - MFU: 30%, 40%, 50%
   - Window: 7, 14, 30, 60 days
   - GPU SKU: H100, H200, B200, GB200

5. Feasibility warnings
   - Dense attention too large
   - Memory lower bound exceeded
   - Sequence length extreme
   - GPU count implies hyperscale cluster
   - Model family estimate is empirical / low-confidence
```

---

## 7. Confidence Levels

The calculator should label estimates by confidence tier.

```text
High confidence:
  Dense LLM transformer pretraining with known parameter count and token count.

Medium confidence:
  Transformer-style time-series foundation models where tokenization is explicit.

Medium-low confidence:
  Tabular foundation models with row-token or axial architectures.

Low confidence:
  Cell-tokenized tabular transformers with very long sequences, MoE tabular systems,
  retrieval-augmented tabular systems, and classical GBDT runtime estimates without
  calibration benchmarks.
```

The calculator should not pretend these are all equally grounded.

---

## 8. Key Design Principle

The most important design principle is:

> Preserve the shared GPU/FLOPs machinery, but make the data unit and architecture assumptions explicit for each model family.

For LLMs, the data unit is usually a token and the Chinchilla-20 default is a reasonable starting point.

For time-series models, the data unit might be a timestep, patch, channel-time cell, or forecasting window.

For tabular models, the data unit might be a task, row, column, cell, or in-context example.

The SPA should therefore guide the user from raw domain quantities to effective model-training units before estimating GPU count.

---

## 9. Implications for the Existing LLM GPU Calculator

The existing calculator should not be generalized by simply renaming “tokens” to “data points.” That would be misleading.

Instead, extend the calculator with a layered architecture:

```text
Domain adapter:
  converts domain-specific raw data into effective training units

Compute estimator:
  converts effective training units and model parameters into FLOPs

Hardware estimator:
  converts FLOPs and retraining window into GPU count

Presentation layer:
  shows estimates, sensitivity ranges, and warnings
```

This architecture allows the LLM mode to remain clean while adding time-series and tabular modes without overclaiming.

---

## 10. References

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

## 11. Suggested Next Step

The next useful artifact would be an updated functional spec section called **Model Family Extensions**, containing:

```text
- new SPA mode selector
- domain-specific input schemas
- formulas for each family
- warning logic
- confidence labels
- sample outputs
```

This can be incorporated into the existing functional and SPA design specs without disturbing the core LLM calculator logic.
