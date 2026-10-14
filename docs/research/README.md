# Research

Derived research backing each pane of the calculator. Distinct from the root-level
`*_spec.md` files, which are the **input specs** that drove the features — these
documents are the **evidence layer**: calibration anchors, back-solved defaults,
regression targets, and per-quantity confidence ratings.

Each doc follows the same structure:

- **Section 0 — central finding:** what makes this pane's question structurally different
- **Numbered sections:** formulas and dense tables of sourced numbers
- **Calibration anchors:** published figures the engine must reproduce
- **Discrepancies with current implementation:** `quantity | repo value | researched value | impact | action`, cited to `file:line`
- **Confidence ratings:** per quantity, on the repo's `ConfidenceLevel` scale (`high | medium | medium-low | low`)
- **Warnings worth emitting:** candidates for `WarningsPanel`
- **Open design questions**
- **Sources:** categorized markdown links

## Documents

| Pane | Doc | Central finding |
|---|---|---|
| LLM Training | [llm-training](2026-10-14-llm-training-research.md) | **Compute-bound by ~470×** — memory never binds. All accuracy lives in one number: `mfu × availability / overheadFactor`. Accurate to ±7% on large dense BF16 runs, **−47% on FP8**, **−56% at 8B** |
| Inference | [inference-serving](2026-10-14-inference-serving-research.md) | **Bandwidth-bound, then KV-cache-capacity-bound.** Measured cost spans **$0.21–$15.25/MTok on identical hardware** — a 24× spread driven by concurrency and SLO, neither of which the engine models |
| Fine-tuning *(new pane)* | [fine-tuning](2026-10-14-finetuning-calculator-research.md) | **Memory-capacity-bound.** Lead with a feasibility ladder, not a GPU count. `6N`/token full FT vs **`4N` LoRA**, validated against NeMo throughputs to within 2.4 pts of their reported MFU |
| Time-series FM | [alt-model-families §1](2026-10-14-alt-model-families-research.md) | The LLM question, but **`D` is constructed, not given** (16–64× swing by tokenization), and models are so small that MFU collapses — TTM hits **MFU 0.00026**, making naive `6ND` **~1,100× optimistic** |
| Tabular FM | [alt-model-families §2](2026-10-14-alt-model-families-research.md) | **There is no training question for ~all users.** Cost is inference-side and quadratic in rows: `O(r²c + rc²)`. The pane models pretraining, which ~3 organizations do |
| Classical tabular | [alt-model-families §3](2026-10-14-alt-model-families-research.md) | The question is **"do I need a GPU at all"**, not how many. 1 GPU is the answer whenever a GPU is the answer; 1 TB trains on one GH200 |
| RL post-training *(GRPO/PPO)* | [rl-posttraining](2026-10-14-rl-posttraining-research.md) | **An inference-serving problem wearing a training problem's clothes.** Rollout is 60–80% of step time, so the output is a GPU *split*, not a count. Corrects the fine-tuning doc's DPO/GRPO/PPO memory multipliers |

### Cross-pane synthesis

The four panes are not variations on one question — each has a different binding constraint, which is why each needs its own primary output:

| | Training | Fine-tuning | Inference | RL post-training | Classical tabular |
|---|---|---|---|---|---|
| Binding constraint | compute × deadline | memory capacity | bandwidth, then KV cache | decode bandwidth, then KV | CPU wall-clock |
| Primary unknown | achieved MFU at scale | bytes/param per method | concurrency under an SLO | rollout:train ratio | library × data shape |
| Memory's role | never binds (470× slack) | the whole question | co-binding | KV pool usually exceeds model state | never binds |
| Right hero output | GPU count for a date | does it fit → time → cost | cost/MTok vs load | inference/training GPU split | GPU yes/no + wall-clock |
| Bad-estimate failure | 2× cluster capex | job OOMs | p99 SLO breach | idle trainer GPUs | wasted GPU rental |

Note the ordering reversal between the two training-side panes: in supervised
fine-tuning, **model state** binds and KV cache does not exist; in RL
post-training, the **KV cache pool** is routinely larger than the entire model
state, because rollout holds `P × G` concurrent sequences. The same model and
the same method give different answers depending on which pane you are in, and
that is correct rather than inconsistent.

## Relationship to the root specs

| Root spec | Pane | Research doc |
|---|---|---|
| `llm_gpu_calculator_functional_spec.md` | Training | `2026-10-14-llm-training-research.md` |
| `llm_gpu_calculator_technical_quant_spec.md` | Training | ″ |
| `llm_gpu_calculator_spa_design_spec.md` | all | ″ |
| `model_family_extensions_functional_spec.md` | Time-series / Tabular / Classical | `2026-10-14-alt-model-families-research.md` |
| `scaling_laws_time_series_tabular_models.md` | ″ | ″ |
| *(none — gap)* | Inference | `2026-10-14-inference-serving-research.md` |
| *(none yet)* | Fine-tuning | `2026-10-14-finetuning-calculator-research.md` |
| *(none yet)* | RL post-training | `2026-10-14-rl-posttraining-research.md` |

## Convention

Research docs are dated and additive — supersede rather than edit in place when the
underlying facts change, so the calibration history stays auditable. Design decisions
derived from this research belong in `docs/decisions/`.
