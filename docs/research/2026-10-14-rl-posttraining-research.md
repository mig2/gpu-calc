# Deep Research: RL Post-Training (GRPO / PPO / RLVR) Cost and GPU Sizing

**Date:** 2026-10-14
**Purpose:** Close the research gap left by `2026-10-14-finetuning-calculator-research.md` §1.6, which recommended **deferring** GRPO/PPO because "rollout generation dominates compute — a materially different cost shape." The decision has been made to ship them in v1 at `low` confidence. This document makes that as defensible as possible, and says precisely where it is not.
**Question the mode answers:** *How many of GPU X do I need to run GRPO/PPO on model Y — how should I split them between rollout and training, how long will it take, and what will it cost?*

---

## 0. The central finding (shapes the whole design)

**RL post-training is an inference-serving problem wearing a training problem's clothes.**

The three existing panes each have one binding constraint. RL has two, in series, and the *inference* one is bigger:

| Pane | Binding constraint | The question |
|---|---|---|
| Training | Compute | `GPUs = total_FLOPs / (deadline × sustained_FLOPs)` |
| Fine-tuning | Memory capacity | *Does the model state fit at all?* |
| Inference | Memory bandwidth, then KV capacity | *How many concurrent sequences under an SLO?* |
| **RL post-training** | **Decode bandwidth first, memory capacity second** | ***What fraction of my GPUs should be generating rather than training?*** |

Every credible measurement says the same thing:

> "The rollout stage typically accounts for **over 70%** of the total training time."
> — [ROLL Flash (arXiv 2510.11345)](https://arxiv.org/html/2510.11345)

> "The rollout stage dominates runtime, accounting for approximately **70%** of each training step across three tasks, while the training stage accounts for **21–23%**."
> — [RollPacker (arXiv 2509.21009)](https://arxiv.org/pdf/2509.21009)

> "In a typical GRPO or PPO run, sampling the rollouts can eat **60–80%** of each step's time, with the actual backward pass a relative afterthought."
> — [The LLM Stack, ch. 6.2](https://prakashkagitha.github.io/llm-stack-book/06-rl-infra/02-generation-training-loop.html)

The mechanism is MFU asymmetry, not FLOP count. Policy training runs at **40–55% MFU** on a tuned Megatron/FSDP setup; autoregressive decode runs at **low single-digit percent** of the compute roof ([LLM Stack 6.2](https://prakashkagitha.github.io/llm-stack-book/06-rl-infra/02-generation-training-loop.html); corroborated by the inference research doc §1.4, which measures decode at B=32 as 11% of the H100 compute roof). The same FLOP gets ~10× less hardware out of decode than out of a backward pass.

**Three design consequences, in order of importance:**

1. **The primary output is a GPU *split*, not a GPU *count*.** If rollout is 3× training time, then at the optimum ~75% of the fleet should be generating. Published splits confirm: Olmo 3 32B used **20 inference nodes to 8 training nodes (71% inference)** ([Olmo 3](https://arxiv.org/html/2512.13961v2)); ROLL Flash's measured optimum was **24 inference / 16 training (60%)**; OpenRLHF's 70B recipe on 32 A100s is **16 vLLM / 8 actor / 8 critic (50%)** ([OpenRLHF](https://arxiv.org/pdf/2405.11143)).
2. **Max completion length is the dominant input, and it enters twice.** Decode time is linear in `L_out`; and because a synchronous step is gated by its *slowest* sequence, you pay for `L_out,max` even when the *mean* completion is 10× shorter. This is why reasoning RL at 16K–32K is 10–50× costlier per step than instruction-following RL at 512.
3. **The reference model is mostly gone.** TRL's `GRPOConfig` now defaults `beta=0.0`, and "if 0.0 (default), the reference model is not loaded" ([TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)); DAPO removes the KL term outright ([DAPO](https://arxiv.org/pdf/2503.14476)). The old "GRPO = 2–3× SFT memory" heuristic in §1.6 of the fine-tuning doc **overstates the modern default**.

---

## 1. The per-step cost anatomy

One synchronous GRPO/RLVR step, with `P` prompts, `G` completions per prompt, prompt length `L_in`, max completion `L_out`:

```
T_step = T_prefill + T_decode + T_logprob + T_update + T_reward + T_sync
```

| Phase | Work | Bound by | Typical share |
|---|---|---|---|
| **Prefill** | `2·N_active·P·L_in` FLOPs (with prefix caching) | compute | 0–5% |
| **Decode** | `P·G·L_out` tokens through the decode roofline | **HBM bandwidth** | **45–80%** |
| **Logprob forward(s)** | `2N` × tokens, ×1 (old policy) or ×2 (+ reference) | compute | 10–17% |
| **Policy update** | `6N` (full) or `4N` (LoRA) × tokens | compute | 10–45% |
| **Reward / verifier** | rule-based ≈ free; reward *model* = `2N_rm` × tokens | varies | ~3% |
| **Weight sync (refit)** | broadcast `N·b_w` bytes to the engine | network | 2–10% |

The reference breakdown for a single shared GPU pool, from [LLM Stack 6.2](https://prakashkagitha.github.io/llm-stack-book/06-rl-infra/02-generation-training-loop.html): **generation ~70%, experience prep ~15%, backward/update ~10%, reward ~3%, weight sync ~2%.**

### 1.1 The measured ground truth: NeMo-RL GRPO step breakdowns

**This is the single most valuable table in this research.** NVIDIA publishes per-phase step times for GRPO, averaged over steps 22–29, all colocated, `max_seq_len 4096`, rollout batch 2048, global batch 512, sequence packing on ([NeMo-RL v0.3 blog](https://developer.nvidia.com/blog/reinforcement-learning-with-nvidia-nemo-rl-megatron-core-support-for-optimized-training-throughput/), [discussion #1161](https://github.com/NVIDIA-NeMo/RL/discussions/1161)):

| Model | Backend | GPUs | Total (s) | Policy train (s) | Refit (s) | **Generation (s)** | Logprobs (s) | Avg gen tok/sample | **Gen share** |
|---|---|---|---|---|---|---|---|---|---|
| Llama 3.1-8B Instruct | Megatron | 8 | 112 | 28 | 5 | **58** | 18 | 795 | **52%** |
| Llama 3.1-8B Instruct | DTensor | 8 | 122 | 38 | 4 | 57 | 19 | 777 | 47% |
| Llama 3.1-70B Base | Megatron | 64 | 147 | 28 | 14 | **84** | 18 | 398 | **57%** |
| Llama 3.1-70B Base | DTensor | 64 | 230 | 97 | 15 | 82 | 28 | 395 | 36% |
| Qwen3 32B | Megatron | 64 | 213 | 68 | 7 | **96** | 40 | 3,283 | **45%** |
| Qwen3 30B-A3B | Megatron | 64 | 167 | 50 | 12 | **78** | 23 | 3,251 | **47%** |
| Llama 3.3-70B Instruct (seq 16384, CP=4) | Megatron | 128 | 445 | 64 | 17 | **287** | 75 | 749 | **64.5%** |

Four things fall straight out of this table:

- **Generation share rises with sequence length**: 45–57% at `max_seq 4096`, **64.5% at 16384**. Extrapolating toward 32K lands on the 70–80% the systems papers report.
- **Generation is backend-invariant** (84 vs 82 s; 58 vs 57 s). Only the *training* side changed between Megatron and DTensor. So generation share is mostly a property of (model, `L_out`, concurrency) and **not** of the training framework — which is exactly what makes it modelable.
- **Logprob forwards are 12–17% of the step** and must be a named term. They are not "the backward pass."
- **Refit (weight sync) is 5–17 s**, i.e. 3–10% — not negligible, and it scales with parameter bytes: 141 GB of BF16 70B params in 14 s ⇒ **~10 GB/s effective broadcast**. Use that constant.

### 1.2 Back-solved constants from this table

```
MFU_policy(Megatron, sharded)  ≈ 0.44     # Llama 70B: 2048 seqs × ~900 tok × 6N / (64 × 989.5 TF × 28 s)
MFU_policy(DTensor/FSDP2)      ≈ 0.13     # same arithmetic, 97 s
weight_sync_bw_effective       ≈ 10 GB/s  # 141 GB / 14 s
```

The 3.4× Megatron-vs-DTensor policy-training gap is the largest single controllable factor on the *training* side — and it barely moves total step time at long `L_out`, which is the whole point.

---

## 2. Rollout generation: the time model

Generation is **inference**, so reuse the inference doc's roofline rather than inventing a FLOP model. `2N × tokens` is the wrong shape — it will be optimistic by 10–30× because it ignores that decode sits at ~1 FLOP/byte.

### 2.1 Composition with the inference research doc

Taking `§1.2` of `2026-10-14-inference-serving-research.md` verbatim:

```
# concurrency actually achievable (inference doc §2.3)
kv_bytes_per_token = 2 · L · H_kv · d_head · b_kv            # MLA: L · (kv_lora_rank + qk_rope_head_dim) · b_kv
vram_for_kv        = G_inf · vram_per_gpu · util  −  N_total · b_w  −  engine_overhead
B_max              = floor(vram_for_kv / (kv_bytes_per_token · (L_in + L_out)))
B                  = min(B_max, P · G)
waves              = ceil(P · G / B)

# decode step time (inference doc §1.2)
s_avg              = L_in + L_out / 2
bytes_per_step     = N_total · b_w + B · s_avg · kv_bytes_per_token
TPOT               = bytes_per_step / (G_inf · BW_peak · η_decode)        η_decode = 0.60

# rollout wall-clock for the step  ← the RL-specific part
T_decode           = waves · L_out · TPOT / η_rollout
```

**The `L_out · TPOT` form — not `tokens / throughput` — is the key RL-specific correction.** In a synchronous step, every sequence in a wave must finish before the trainer starts, so the wave costs `L_out,max` decode steps regardless of the mean completion length. Using mean length instead understates the step by the tail ratio, which on reasoning RL is routinely 3–10× (NeMo-RL's 70B row has mean 398 against max 4096 ⇒ **10.3×**).

### 2.2 `η_rollout`: the roofline realization factor

The roofline is an upper bound and real RL rollout misses it substantially. Two back-solves:

| Anchor | Roofline prediction | Measured | `η_rollout` |
|---|---|---|---|
| Llama 3.1-70B, 64× H100, B=2048, `L_out≈3600` ([NeMo-RL](https://github.com/NVIDIA-NeMo/RL/discussions/1161)) | TPOT 13.2 ms ⇒ 47.4 s | **84 s** | **0.56** |
| 7B, 1× H100, 512 rollouts × 2K tokens, vLLM ([HF async-RL survey](https://huggingface.co/blog/async-rl-training-landscape)) | 1.3 waves × 2048 × 19.1 ms ⇒ 51 s | **~180 s** | **0.28** |

```
η_rollout default 0.45    (range 0.25–0.60, confidence: medium-low)
```

Causes of the gap, all documented: the effective batch decays as sequences finish so the tail sequence gates the step ([LLM Stack 6.2](https://prakashkagitha.github.io/llm-stack-book/06-rl-infra/02-generation-training-loop.html)); colocated vLLM gets a *small* memory budget (TRL's `vllm_gpu_memory_utilization` default is **0.3**, versus 0.90 for standalone serving — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)); CUDA graphs are re-captured after each refit; and sampling/detokenization is on the critical path.

Independent corroboration of raw rollout throughput, vLLM on 1× H100 ([HF async-RL survey](https://huggingface.co/blog/async-rl-training-landscape)):

| Model | 512 rollouts @ 2K | @ 8K | @ 32K |
|---|---|---|---|
| 7B | ~3 min | ~11 min | ~45 min |
| 32B | ~14 min | ~56 min | **~3.7 h** |

"During all of that time, training GPUs sit completely idle." These six numbers are the **regression test suite for the generation model** — any engine that cannot reproduce them within ~2× is wrong.

### 2.3 Prefill and prefix caching — quantified

All `G` completions in a group share the prompt prefix, so the prompt is prefilled **once per prompt, not once per completion**. This is a structural property of GRPO's access pattern and vLLM/SGLang automatic prefix caching captures it for free:

```
prefill_flops(no cache) = 2 · N_active · P · G · L_in
prefill_flops(cached)   = 2 · N_active · P · L_in
saving                  = (G − 1) / G        →  87.5% at G=8,  93.75% at G=16
```

| Model | `L_in` | Prefill on 8× H100, no cache | cached | saved |
|---|---|---|---|---|
| 7B | 512 | 0.62 s | 0.08 s | 87.5% |
| 7B | 16,384 | 19.7 s | 2.5 s | 87.5% |
| 70B | 4,096 | 45.6 s | 5.7 s | 87.5% |
| 70B | 16,384 | **182.5 s** | **22.8 s** | 87.5% |

At short prompts this is a rounding error; at agentic/RAG prompt lengths it is the difference between prefill being 3% and 30% of the step. [Prefix Grouper (arXiv 2506.05433)](https://arxiv.org/html/2506.05433v1) proves the same `1/G` reduction applies to the *training* forward/backward over grouped sequences (asymptotically "1/G of the Repeated-Prefix Forward approach" as `L_p/L_r → ∞`) — but that is a research technique, not a default; **model the inference-side prefix cache as on by default, the training-side one as off.**

### 2.4 KV cache during rollout — the capacity wall

Composing the inference doc §2.1 formula with GRPO's `P × G` concurrency. BF16 KV, `L_in = 512`, `L_out = 8192`:

| Model | KV/token | KV/sequence | 1 group (G=8) | P=64 × G=8 (512 seqs) | P=64 × G=16 | P=512 × G=8 |
|---|---|---|---|---|---|---|
| 7B (28L, 4 kv, 128) | 56 KiB | 0.465 GiB | 3.7 GiB | **238 GiB** | 476 GiB | 1,904 GiB |
| 70B (80L, 8 kv, 128) | 320 KiB | 2.656 GiB | 21.2 GiB | **1,360 GiB** | 2,720 GiB | 10,880 GiB |

**At G=8, L_out=8192, P=64 the 70B KV pool alone wants 1.36 TB — 20 H100s' worth of usable VRAM, before a single byte of weights or optimizer state.** This is the term that binds. Two responses, both standard:

- **Run in waves.** You do not need `P·G` concurrent; `B_max` caps concurrency and `waves = ceil(P·G / B)` multiplies the time. The calculator must do this, not declare infeasible.
- **Quantize the KV.** FP8 KV halves it ("safe for production across virtually all tasks", inference doc §2.4). [KV4-RL](https://arxiv.org/html/2610.14448) reports **1.80× cumulative rollout speedup / 1.27× full-iteration speedup** from INT4 KV specifically in RLVR.

---

## 3. Published hyperparameters — what people actually run

### 3.1 Group size `G`

| Source | `G` | Context |
|---|---|---|
| [DeepSeekMath (GRPO's origin)](https://arxiv.org/pdf/2402.03300) | **64** | 7B, max len 1024, train batch 1024 (= 16 prompts × 64), lr 1e-6, KL coef 0.04 |
| [TRL `GRPOConfig` default](https://huggingface.co/docs/trl/main/en/grpo_trainer) | **8** | with `max_completion_length=512`, `beta=0.0` |
| [HF async-RL survey](https://huggingface.co/blog/async-rl-training-landscape) reference config | **8** | × 64 prompts = 512 rollouts/step |
| [ROLL Flash](https://arxiv.org/html/2510.11345) experiments | **16–32** | 0.6B–8B, seq 2K–32K |
| [verl AMD GRPO demo](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html) | 4 | Qwen2.5-VL-7B, Geometry3k |
| [verl AMD DAPO demo](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html) | 16 | Qwen2.5-Math-7B, max response 8,192 |
| [Olmo 3 / OlmoRL-adjacent configs](https://arxiv.org/html/2512.13961v2) | 8 | 64 prompts × 8 rollouts = 512 |
| [Union.ai 8-GPU GRPO](https://www.union.ai/blog-post/we-ran-multi-node-grpo-on-8-gpus-and-the-trainer-cost-us-nothing) | 6 | Qwen 8B + LoRA, 32 prompts, 512-token completions |

**Recommended default `G = 8`** (TRL's default, the modal published value, and the HF survey's reference config). Offer 4 / 8 / 16 / 32 / 64. Warn above 16: cost is linear in `G` and the marginal advantage-estimate quality is not.

### 3.2 Prompts per step `P`, and the stability floor

| Source | `P` | Note |
|---|---|---|
| DeepSeekMath | 16 | 16 × 64 = 1,024 sequences |
| HF survey / Olmo-class configs | 64 | 64 × 8 = 512 sequences |
| ROLL Flash | 32–512 | |
| verl config examples | `rollout_batch_size: 512` | with `ppo_mini_batch_size: 32–64` ([Spheron, aggregator](https://www.spheron.network/blog/rlhf-training-infrastructure-verl-openrlhf-trl-gpu-cloud/)) |
| NeMo-RL benchmarks | rollout batch 2048, global batch 512 | |

The consistent rule is on the **product**, not on `P`: `P × G ≥ 512` unique sequences for stable advantage normalization, 1,024+ preferred ([Spheron, aggregator — treat as folklore-grade but widely repeated](https://www.spheron.network/blog/rlhf-training-infrastructure-verl-openrlhf-trl-gpu-cloud/)). **Default `P = 64`, `G = 8` ⇒ 512 sequences/step.**

### 3.3 Max completion length `L_out` — the dominant cost driver

| Regime | `L_out` | Evidence |
|---|---|---|
| Instruction following / tool use / short-form | 512–1,024 | TRL default 512; DeepSeekMath 1,024; verl default `max_response_length: 1024` |
| General reasoning | 4,096–8,192 | verl DAPO demo 8,192; NeMo-RL benchmarks `max_seq 4096` |
| Long-CoT reasoning RL | **16,384–32,768** | NeMo-RL long-seq run at 16,384; ROLL Flash to 32K; HF survey notes "up to 64K for long CoT" |

**Default 4,096**, with a preset ladder (short 1,024 / reasoning 8,192 / long-CoT 32,768). Emit a hard warning above 16,384: this is where the step becomes 70–80% generation and the long tail dominates.

Observed *mean* completions run far below the cap: NeMo-RL measured 398–3,283 mean against a 4,096 cap. **Expose `mean_completion_ratio` (default 0.35) so the user can separate the compute they pay for (`L_out,max`, because of tail gating) from the tokens they train on (mean).**

### 3.4 Steps to convergence `S`

| Source | Steps |
|---|---|
| [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388) — Qwen3-235B-A22B, AIME'24 70.1 → 85.1 | **170** |
| [DAPO](https://arxiv.org/pdf/2503.14476) — Qwen2.5-32B to 50 on AIME'24 | ~50% of GRPO's step count for +3 points |
| [SimpleRL](https://github.com/hkust-nlp/simpleRL-reason) — 7B/14B | ~100 |
| [Mini-R1 / Open-R1 tutorial](https://www.philschmid.de/mini-deepseek-r1) — 3B | 450 |
| [DeepSeek-R1, Epoch AI reconstruction](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1) | **~8,000 gradient steps total** |
| GRPO-LEAD / assorted RLVR papers | 100–500, peak often at 160–500 |

**Default `S = 300`** (range 100–1,000). Flag that this is the least knowable input: it is a *research outcome*, not a config value, and a 3× error here is a 3× error in the answer. Offer "steps" and "total sequences generated" as alternate entry points.

---

## 4. The generation:training split — the number the calculator lives on

### 4.1 Published measurements

| Source | Setup | Generation share | Train share | Implied `r = T_gen/T_train` |
|---|---|---|---|---|
| [RollPacker](https://arxiv.org/pdf/2509.21009) | Qwen2.5 family, ≤128 H800 | **~70%** | 21–23% | **~3.1** |
| [ROLL Flash](https://arxiv.org/html/2510.11345) | 0.6–8B, 2K–32K, G=16–32 | **>70%** | — | >2.3 |
| [LLM Stack 6.2](https://prakashkagitha.github.io/llm-stack-book/06-rl-infra/02-generation-training-loop.html) | generic GRPO/PPO | **60–80%** (ref. breakdown 70%) | ~10% + 15% prep | ~2.8 |
| [NeMo-RL](https://github.com/NVIDIA-NeMo/RL/discussions/1161) Llama-70B Megatron, seq 4096 | colocated, 64 H100 | 57% | 19% (+12% logprob) | **1.8** (3.0 vs policy-only) |
| NeMo-RL Llama-3.3-70B, **seq 16384** | colocated, 128 H100 | **64.5%** | 14% (+17% logprob) | **2.1** (4.5 vs policy-only) |
| NeMo-RL Llama-8B Megatron, seq 4096 | 8 H100 | 52% | 25% (+16%) | 1.4 |
| NeMo-RL Llama-70B **DTensor** | 64 H100 | 36% | 42% | **0.85** |
| [Union.ai](https://www.union.ai/blog-post/we-ran-multi-node-grpo-on-8-gpus-and-the-trainer-cost-us-nothing) Qwen-8B LoRA, L40S, 512-tok completions | 2–4 gen GPUs / 8 train | 294 s gen / 140 s train | | **2.1** |
| [AReaL-Hex](https://arxiv.org/html/2511.00796) | 1.5–14B distilled Qwen | "rollout generation is the primary system bottleneck" on homogeneous H800 | | — |

**Where sources conflict, the conflict is explicable and is itself the model:** `r` falls when the training backend is slow (DTensor: 0.85) or `L_out` is short (8B at 795 tokens: 1.4), and rises when the training backend is fast and `L_out` is long (70B Megatron at 16K: 4.5 against policy time alone). The systems papers quoting 70–80% are measuring long-CoT workloads on tuned trainers; NeMo-RL's lower shares are short-completion benchmarks. **Both are right.**

### 4.2 Recommended defaults

```
r_default(L_out)  =  1.4   at L_out ≤ 1024          # short-form RL
                     2.5   at L_out = 4096          # reasoning (default)
                     4.0   at L_out = 16384         # long-CoT
                     6.0   at L_out ≥ 32768         # extrapolated, UNVERIFIED
gen_share = r / (1 + r)   →  58% / 71% / 80% / 86%
```

The engine should **compute** `r` from §2's roofline plus §1.2's MFU rather than look it up — but these are the sanity bands, and the calculator should warn if its computed `r` falls outside 0.8–8.

### 4.3 PPO vs GRPO on this axis

PPO's generation is **one** completion per prompt, not `G`. Its *training* side is roughly doubled (critic forward+backward ≈ another `6N_critic`, plus a reward-model forward `2N_rm`). So:

```
At equal prompt count:   GRPO rollout = G × PPO rollout        → GRPO is far more generation-heavy
At equal sequence count: PPO train step ≈ 2× GRPO train step   → PPO's r is roughly GRPO's r / 2
```

Modern RLVR practice is GRPO-family precisely because the critic was the expensive, unstable part and group-relative advantage replaces it with more sampling — trading a compute-bound component for a bandwidth-bound one. **That trade is the reason this pane has to be bandwidth-first.**

---

## 5. Colocated vs disaggregated — what sets the GPU count

### 5.1 The two architectures

| | **Colocated** (verl HybridEngine, TRL `vllm_mode="colocate"`, Unsloth) | **Disaggregated** (OpenRLHF, NeMo-RL separate, AReaL, StreamRL, ROLL Flash) |
|---|---|---|
| GPU pools | One; time-sliced | Two; `G_inf` + `G_train` |
| Weight sync | "essentially free … at most an in-place resharding on the same GPU, not a network transfer" | NCCL broadcast across pools |
| Overlap | **None** — `T_step = T_gen + T_train` | Yes — `T_step ≈ max(T_gen, T_train)` |
| Memory | Engine weights + KV **coexist** with trainer state (or sleep/wake) | Each pool sized for its own job |
| Min GPUs | Set by the **sum** of both memory footprints | Set by the **larger** of the two, independently |

Sources: [verl v0.9 notes / HF async-RL survey](https://huggingface.co/blog/async-rl-training-landscape), [OpenRLHF hybrid engine docs](https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html), [verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html).

### 5.2 Published GPU splits — the regression targets

| Setup | Inference GPUs | Training GPUs | % inference |
|---|---|---|---|
| [Olmo 3 Think 32B](https://arxiv.org/html/2512.13961v2) | **20 nodes** | **8 nodes** | **71%** |
| [ROLL Flash measured optimum](https://arxiv.org/html/2510.11345) | 24 | 16 | 60% |
| [OpenRLHF 70B / 32 A100](https://arxiv.org/pdf/2405.11143) | 16 vLLM | 8 actor + 8 critic | 50% |
| OpenRLHF 70B / 16 A100 (4-way each) | 4 vLLM | 4 ref + 4 RM + 4 critic + 4 actor | 25% (critic-era PPO) |
| [verl on MI355X, GRPO + DAPO](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html) | 4 | 4 | 50% |
| [Union.ai 8× L40S](https://www.union.ai/blog-post/we-ran-multi-node-grpo-on-8-gpus-and-the-trainer-cost-us-nothing) | 2–4 | 8 (shared) | ~33% |

**The split should be derived, not guessed.** Equalizing the two pools' wall-clock gives

```
inference_fraction*  =  r / (1 + r)          # r measured at equal GPU counts
```

| `r` | 0.5 | 1 | 2 | **3** | 4 | 5 |
|---|---|---|---|---|---|---|
| inference GPUs | 33% | 50% | 67% | **75%** | 80% | 83% |

At the modal `r ≈ 3` this predicts **75% inference**, which brackets Olmo 3's 71% and sits above ROLL Flash's 60%. ROLL Flash gives the more careful form, `β* = E·N·μ_train / [N·μ_gen + K·L_gen/(α+1) + E·N·μ_train]`, where the extra terms capture async staleness tolerance `α`. **Recommendation: implement `r/(1+r)`, cite ROLL Flash's `β*` as the refinement, round to node granularity (multiples of 8), and default to 2/3 inference.**

### 5.3 Weight sync overhead per step

| Mechanism | Latency | Source |
|---|---|---|
| Colocated in-place reshard | ~0 (no network) | [HF async-RL survey](https://huggingface.co/blog/async-rl-training-landscape) |
| Standard NCCL broadcast | **100–500 ms** | same |
| verl NCCL + bucketing | **~20 ms** | same |
| **Measured NeMo-RL "refit"** | **5 s (8B/8 GPU) → 14 s (70B/64) → 17 s (70B/128)** | [NeMo-RL #1161](https://github.com/NVIDIA-NeMo/RL/discussions/1161) |
| Mooncake Transfer Engine, Kimi-K2 trillion-param, 256× H20 | 16–17 s | [HF async-RL survey](https://huggingface.co/blog/async-rl-training-landscape) |
| AReaL-Hex, share of step | **1.9% homogeneous / 13.54% heterogeneous** | [AReaL-Hex](https://arxiv.org/html/2511.00796) |
| LoRA adapter sync | "a few megabytes", <0.3 s | [Union.ai](https://www.union.ai/blog-post/we-ran-multi-node-grpo-on-8-gpus-and-the-trainer-cost-us-nothing), [SparseRL-Sync claims ~100× less comm](https://arxiv.org/pdf/2605.07330) |

The sub-100 ms figures describe the *collective*; NeMo-RL's 5–17 s measures the whole refit including re-sharding and CUDA-graph recapture. **Model the realistic one:**

```
T_sync = N_total · b_w / 10 GB/s              # full-param, back-solved from NeMo-RL (141 GB / 14 s)
T_sync = N_adapter · b_w / 10 GB/s  ≈ 0       # LoRA
```
⇒ 7B: ~1.5 s. 70B: ~14 s. 405B: ~81 s. On a 300-step run at 70B that is **70 minutes of pure weight shuffling**, and it is the strongest numerical argument for LoRA-based RL in a disaggregated topology.

### 5.4 Colocated memory: sleep/wake is the whole game

vLLM sleep mode releases "up to 90%+ of GPU memory": **level 1** offloads engine weights to CPU and discards the KV cache (same model resumes); **level 2** discards weights too, for the RLHF weight-update path ([vLLM sleep mode docs](https://docs.vllm.ai/en/latest/features/sleep_mode/)). With sleep, colocated peak memory is `max(trainer_peak, rollout_peak)`; without it, it is the sum.

Two further colocated facts with large effects:

- **TRL's colocated default `vllm_gpu_memory_utilization = 0.3`** ([TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)) versus 0.90 for standalone vLLM. The engine gets less than a third of the memory, so `B_max` — and therefore rollout throughput — is roughly 3× worse than a dedicated inference pool at the same GPU count. **This is the quantitative cost of colocation, and it is bigger than the weight-sync saving.**
- verl guidance: use 0.5–0.7 for the rollout engine, drop to 0.80-of-default in colocated setups, and "if rollout is colocated with training, consider disaggregating rollout to a separate node where the KV cache has the full GPU VRAM budget" ([verl perf tuning](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html), [verl config](https://verl.readthedocs.io/en/latest/examples/config.html)).

**Recommendation for the engine:** default to **disaggregated** for ≥2 nodes and **colocated + sleep** for single-node, and show the memory and throughput consequence of each.

---

## 6. Memory accounting by method

### 6.1 Bytes per parameter per resident copy

| Copy | Trainable? | Optimizer state? | Bytes/param | Notes |
|---|---|---|---|---|
| **Policy (actor)**, full FT | yes | yes | **16** = 2 w + 2 g + 12 (fp32 master + m + v) | matches fine-tuning doc §1.2 |
| Policy, LoRA | adapters only | adapters only | `2·N_total + 16·N_adapter` | |
| **Critic / value** (PPO only) | **yes** | **yes** | **16** | "the critic needs the same architecture as the actor, carrying the same weight count and full AdamW optimizer state" ([Spheron, aggregator](https://www.spheron.network/blog/rlhf-training-infrastructure-verl-openrlhf-trl-gpu-cloud/)) |
| **Reward model** (PPO/RLHF) | no | no | **2** (BF16 inference) | quantizable to 1 (FP8) or 0.5 (NF4); offloadable |
| **Reference policy** | no | no | **2**, or **0** | 0 when `beta=0` (TRL default) or under LoRA via adapter-disable |
| **Inference engine weight copy** | no | no | **2** (or 1 at FP8) | colocated: additional, unless sleep-mode offloaded or memory-pooled |
| KV cache pool | — | — | §2.4 | |

### 6.2 Resident copies and multiplier vs SFT

| Method | Copies | Model-state multiplier vs SFT full-FT | Replaces fine-tuning doc §1.6's |
|---|---|---|---|
| SFT | policy | 1.00× | 1.0× |
| **DPO, ref resident** | policy + ref | **1.125×** (16 → 18 B/param) | "~2×" — **the old figure counted activations, not state** |
| **DPO, ref precomputed** | policy | **1.00×** | — |
| ORPO | policy | 1.00× | ~1.0× ✓ |
| **GRPO, `beta=0`, disaggregated** | policy (+ engine copy on the inference pool) | **1.00×** on the training pool | "~2–3×" — **overstated for the modern default** |
| GRPO, `beta>0`, colocated, no sleep | policy + ref + engine | **1.25×** + KV pool | |
| **PPO** | policy + **critic** + reward + ref | **2.25×** (36 vs 16 B/param) + KV | "~4×" — too high; only 2 of the 4 copies carry optimizer state |

**The fine-tuning doc's "GRPO = 2–3× SFT memory" should be corrected to "1.0–1.25× of SFT *model state*, plus a KV-cache pool that is usually larger than the entire model state."** The multiplier framing hides where the memory actually goes.

### 6.3 Worked memory floors (GRPO, colocated, 20 GiB aggregate activations)

| Model | Method | Trainer | Ref | Engine copy | Total | Min H100-80 (0.85 usable) |
|---|---|---|---|---|---|---|
| 7B | full | 113.5 GiB | 14.2 | 14.2 | **161.9 GiB** | **3** |
| 7B | LoRA r=16 | 14.5 GiB | 0 | 14.2 | **48.7 GiB** | **1** |
| 70B | full | 1,052 GiB | 131.5 | 131.5 | **1,335 GiB** | **20** |
| 70B | LoRA r=16 | 134.2 GiB | 0 | 131.5 | **285.7 GiB** | **5** |

Corroborating anchor: a 70B PPO actor+optimizer is quoted at **980 GB unsharded, ~122 GB/GPU when 8-way sharded on B200**, with the reward model alone at 140 GB ([Spheron, aggregator](https://www.spheron.network/blog/rlhf-training-infrastructure-verl-openrlhf-trl-gpu-cloud/)) — consistent with the 1,052 GiB trainer row above.

**But note the ordering:** the 70B LoRA memory floor is 5 GPUs while its KV pool at `G=8, L_out=8192, P=64` wants 20 GPUs (§2.4). **For every LoRA configuration and most full-FT ones, KV cache — not model state — is the binding memory term.** That inverts the fine-tuning pane's whole framing and must be surfaced in the UI.

### 6.4 Reference and reward models can be cheapened

- `beta = 0.0` removes the reference model entirely — TRL's default, and the DAPO prescription: "during training the long-CoT reasoning model, the model distribution can diverge significantly from the initial model, thus this restriction is not necessary" ([DAPO](https://arxiv.org/pdf/2503.14476)). Valid for **verifiable rewards**; *not* valid when optimizing against a learned reward model, where KL is the guard against reward hacking ([Interconnects](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo)).
- Under LoRA, "using PEFT in GRPO allows one to forgo loading a separate reference model during training, as we can obtain the original, unmodified model during training by simply disabling the LoRA adapters" ([TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer)). **Zero-cost reference model.**
- Reference and reward models are inference-only and can be FP8/NF4-quantized or CPU-offloaded; only policy and critic need optimizer state.

---

## 7. LoRA + RL — viable, and unusually well-justified

**Yes, and the theoretical case is stronger for RL than for SFT.** [Thinking Machines, *LoRA Without Regret*](https://thinkingmachines.ai/blog/lora/) (replicated in [TRL's docs](https://huggingface.co/docs/trl/lora_without_regret)):

> "LoRA fully matches the learning performance of FullFT when running policy gradient algorithms for reinforcement learning, **even with ranks as low as 1**."

The argument is information-theoretic: policy gradient absorbs ~1 bit per episode (the reward signal) against supervised learning's O(tokens), so the adapter's capacity is never the bottleneck. Practical prescriptions from the same work: **rank 1–32 for RL** (vs 256 for SFT-scale post-training), apply LoRA to **all** layers including MLP/MoE (attention-only is consistently worse), use **10× the full-FT learning rate**, and expect less tolerance of very large batches.

What it saves, in this pane's terms:

| Lever | Saving | Source |
|---|---|---|
| Model state | 70B: 1,052 → 134 GiB (**7.8×**) | §6.3 arithmetic |
| Reference model | 2 B/param → **0** (adapter-disable) | [TRL](https://huggingface.co/docs/trl/main/en/grpo_trainer) |
| Policy-update FLOPs | `6N → 4N` (**−33%**) | fine-tuning doc §2.1 |
| Weight sync per step | 141 GB → ~0.4 GB (**>300×**) | [Union.ai](https://www.union.ai/blog-post/we-ran-multi-node-grpo-on-8-gpus-and-the-trainer-cost-us-nothing) |
| End-to-end compute vs full FT | ~**67%** | [Thinking Machines](https://thinkingmachines.ai/blog/lora/) |
| Unsloth GRPO, Llama 3.1 8B @ 20K ctx | TRL+FA2 **510.8 GB → 54.3 GB** (**−90%**) | [Unsloth](https://unsloth.ai/blog/grpo) via [summary](https://www.rohan-paul.com/p/unsloth-releases-new-grpo-algorithms) |
| Unsloth GRPO, Qwen2.5 1.5B | 7 GB → **5 GB** | same |

Unsloth's 90% claim is *not* generic LoRA — it is four specific optimizations: a memory-efficient linear cross-entropy for the grouped logits ("slashes memory 8x or more, shaving 68.5 GB … for `num_generations = 8` and 20K context"), FP16/FP8 scaling, offloading gradient checkpointing, and **sharing one memory pool with vLLM** ("shaving 16GB of VRAM"). Vendor-published, single-source, consumer-GPU oriented: **cite as `medium-low` and do not apply it as a general multiplier.** What *is* safely generalizable is that the grouped-logits tensor (`P·G · L_out · vocab`) is a first-order memory term in GRPO that naive implementations materialize — which is the actual reason TRL+FA2 needed 510 GB.

**Recommendation: default RL method to LoRA r=16, all-linear.** It is the only configuration where a 70B RLVR run is single-node-feasible.

---

## 8. DPO / ORPO — firming up the §1.6 multipliers

The fine-tuning doc gives DPO as "~2× memory, ~3–4× compute." Both are wrong in interesting ways.

### 8.1 Exact pass count

Per preference pair, DPO processes two sequences (chosen + rejected):

| Pass | Count | FLOPs/token |
|---|---|---|
| Policy forward | 2 (chosen, rejected) | `2N` each |
| Policy backward | 2 | `4N` each |
| Reference forward | 2 | `2N` each |
| **Total per processed token** | | **`8N`** (vs SFT's `6N`) |

So it is **four forwards + two backwards per pair**, but the right normalization is per *token*:

```
DPO, reference resident:    8N per token × 2 sequences/example  =  2.67× SFT per example
DPO, reference precomputed: 6N per token × 2 sequences/example  =  2.00× SFT per example
ORPO (reference-free):      6N per token × 2 sequences/example  =  2.00× SFT per example
```

**Recommend `2.7×` (resident ref) and `2.0×` (precomputed ref / ORPO)** — tighter and better-grounded than "3–4×". Memory multiplier: **1.125×** model state with a resident reference, **1.0×** without (§6.1), though activations do double because both completions are in flight.

### 8.2 Precomputing reference logprobs is standard practice and should be the default

TRL's `precompute_ref_log_probs=True` "allows you to precompute the reference model log probabilities for the entire training dataset before training begins … the reference model does not need to be kept in memory" ([TRL DPO docs](https://huggingface.co/docs/trl/en/dpo_trainer)). Measured effect: **−31% peak GPU memory**, plus removal of the `2N` reference forward from every step of every epoch.

Its cost is one pass over the dataset: `2N × dataset_tokens`, paid **once** rather than `epochs ×` — so at 3 epochs it converts 3 reference forwards into 1, i.e. the compute saving is `(epochs − 1)/epochs` of the reference term. It is incompatible with `sync_ref_model=True` (moving-reference variants) and with online/augmented data pipelines.

**Default it on, with a note that it blocks reference-model syncing.** This is the single cheapest win available in the DPO path and the existing doc does not mention it.

---

## 9. Published end-to-end RL run costs — the regression targets

| Model / run | Method | Hardware | Wall-clock | GPU-hours | Cost | Source |
|---|---|---|---|---|---|---|
| **Tülu 3 8B** final RL | PPO/RLVR | 8 GPUs | **65 h** | **520** | ~$1,300 @ $2.50 | [Tülu 3](https://arxiv.org/pdf/2411.15124) |
| **Tülu 3 70B** final RL | PPO/RLVR | **48 GPUs** | **60 h** | **2,880** | ~$7,200 | same |
| Tülu 3 8B reward model | RM training | 8× H100 | 9 h | 72 | ~$180 | same |
| **Olmo 3 Think 32B** initial RL | RLVR | 8 train + 20 inference nodes (224 GPUs) | ~**5 days** | ~**26,900** | ~$67k | [Olmo 3](https://arxiv.org/html/2512.13961v2) |
| Olmo 3.1 Think 32B continuation | RLVR | 224 GPUs | **21 days** | **~113,000** | ~$280k | same |
| **Qwen3-32B** RL stage | GRPO-family | — | — | **17,920** | ~$45k | [Qwen3](https://arxiv.org/pdf/2505.09388) |
| Qwen3-235B-A22B | RL | — | 170 steps, AIME'24 70.1→85.1 | — | — | same |
| **DeepSeek-R1** RL stage | GRPO | 512–2048 H800 | 80 h (R1); 198 h (R1-Zero) | — | **~$1M** (Epoch estimate; ±2×) | [Epoch AI](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1), [R1 paper](https://arxiv.org/pdf/2501.12948) |
| DeepSeek-R1 total reported training | — | — | — | — | **$294,000** | [DeepSeek via press](https://www.cnn.com/2025/09/19/business/deepseek-ai-training-cost-china-intl) |
| **SimpleRL 7B/14B** | GRPO | 16× H100-80 | **15 h**, ~100 steps, 8K examples | 240 | ~$600 | [SimpleRL](https://github.com/hkust-nlp/simpleRL-reason) |
| SimpleRL 32B | GRPO | 64× H100-80 | **1.5 days** | 2,304 | ~$5,700 | same |
| **Mini-R1 / Open-R1** 3B | GRPO, G=8 | 4× H100-80 | **6 h**, 450 steps (45–60 s/step) | 24 | ~$60 | [Mini-R1](https://www.philschmid.de/mini-deepseek-r1) |
| verl GRPO, Qwen2.5-VL-7B | GRPO, G=4 | 8× MI355X | 11 h (39,739 s), 50 steps | 88 | — | [ROCm](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html) |
| verl DAPO, Qwen2.5-Math-7B | DAPO, G=16, 8K resp | 8× MI355X | 10 h (35,902 s), 100 steps | 80 | — | same |
| Generic 70B PPO | PPO | multi-cluster | 48–72 h | — | $8,033 (B200 mixed-spot) – $12,096 (on-demand) | [Spheron, aggregator](https://www.spheron.network/blog/rlhf-training-infrastructure-verl-openrlhf-trl-gpu-cloud/) |

**Notes on these anchors.** DeepSeek's $294k headline and Epoch's ~$1M RL estimate are not comparable — Epoch's is an independent reconstruction of the *RL phase alone* from `N·B·G·L ≈ 2 trillion generated tokens` and ~8,000 gradient steps, with the author flagging batch size as "the biggest source of uncertainty … could be too high by a factor of 2." The DeepSeek-R1 RL phase is estimated at ~17% of combined pretrain+RL cost. Cost columns marked "~$" are my conversion at $2.50/H100-hr ([RunPod on-demand, Apr 2026, per Mini-R1](https://www.philschmid.de/mini-deepseek-r1)) and are **mine, not the source's**.

**The four-order-of-magnitude spread — $60 for a 3B GRPO tutorial run to ~$1M for R1's RL stage — is this pane's headline, exactly as the QLoRA-to-full-FT spread is the fine-tuning pane's.**

---

## 10. Proposed engine shape

Mirrors `src/engine/types.ts` conventions and composes with both the fine-tuning memory engine and the inference decode roofline.

```ts
export type RlMethod = 'grpo' | 'dapo' | 'rlvr' | 'ppo' | 'dpo' | 'orpo';
export type RlTopology = 'colocated' | 'colocated_sleep' | 'disaggregated';

export type RlScenario = {
  // --- model (shared with FineTuneScenario; derived from catalog) ---
  modelId: string | 'custom';
  // derived: nTotal, nActive, layers, hidden, kvHeads, headDim, ffnDim, vocab, isMoe, kvBytesPerToken

  // --- method ---
  method: RlMethod;
  tuningMethod: 'full' | 'lora' | 'qlora';   // reuse FineTuneScenario's
  loraRank: number;                           // default 16 for RL (Thinking Machines: 1-32)
  klBeta: number;                             // 0 ⇒ no reference model resident  (default 0)
  rewardSource: 'verifiable' | 'reward_model';
  rewardModelParams?: number;                 // only if rewardSource = 'reward_model'
  criticParams?: number;                      // PPO only; defaults to nTotal

  // --- rollout workload (THE cost drivers; user-supplied) ---
  promptsPerStep: number;        // P       default 64
  groupSize: number;             // G       default 8   (ppo ⇒ forced 1)
  promptTokens: number;          // L_in    default 512
  maxCompletionTokens: number;   // L_out   default 4096
  meanCompletionRatio: number;   //         default 0.35 — tokens trained on vs tokens paid for
  steps: number;                 // S       default 300
  prefixCaching: boolean;        //         default true

  // --- hardware & topology ---
  gpuId: string;
  topology: RlTopology;                       // default 'disaggregated' if >=16 GPUs
  totalGpus?: number;                         // if set, engine derives the split
  inferenceGpus?: number;                     // if set, overrides the derived split
  rolloutPrecision: 'bf16' | 'fp8';           // default bf16
  kvPrecision: 'bf16' | 'fp8';                // default bf16
  trainingBackend: 'megatron' | 'fsdp2';      // drives MFU 0.44 vs 0.13
  asyncRollout: boolean;                      // default false (sync); true ⇒ overlap

  // --- editable assumptions (AssumptionChips) ---
  etaDecode: number;        // 0.60  from inference doc
  etaRollout: number;       // 0.45  §2.2   ← the single most uncertain input
  etaPrefill: number;       // 0.82  from inference doc
  mfuPolicy: number;        // 0.44 megatron / 0.13 fsdp2
  weightSyncBwGbs: number;  // 10    §5.3
  availability: number;
  pricePerGpuHour: number;
};

export type RlResult = {
  // memory (composes with the fine-tuning memory engine)
  trainMemoryGb: { policy; grads; optimizer; critic; reference; activations; overhead };
  rolloutMemoryGb: { engineWeights; kvPool; overhead };
  kvBytesPerSequence: number;
  kvPoolGbAtFullConcurrency: number;      // P·G concurrent — often the binding term
  minGpusTraining: number;
  minGpusInference: number;
  bindingConstraint: 'kv_cache' | 'model_state' | 'decode_bandwidth' | 'policy_compute';

  // the split
  gpuSplit: { inference: number; training: number; inferenceFraction: number };

  // per-step time decomposition (the FormulaTraceDrawer payload)
  stepSeconds: { prefill; decode; logprob; policyUpdate; reward; weightSync; total };
  rolloutToTrainRatio: number;            // r
  generationSharePct: number;             // r/(1+r)
  decodeConcurrency: number;              // B
  decodeWaves: number;
  tpotMs: number;

  // totals
  tokensGenerated: number;                // P·G·L_out·S  — the headline volume
  tokensTrained: number;                  // P·G·(L_in + meanCompletionRatio·L_out)·S
  wallClockHours: number;
  gpuHours: number;
  costUsd: number;

  warnings: string[];
  trace: string[];
  confidence: ConfidenceLevel;            // 'low' for grpo/ppo, 'medium-low' for dpo/orpo
};
```

**User supplies:** model, method, `P`, `G`, `L_in`, `L_out`, `steps`, GPU SKU, total GPUs (optional), topology, LoRA on/off.
**Engine derives:** `kvBytesPerToken` from architecture, `B`/`waves` from the KV budget, `TPOT` from the decode roofline, `r` and hence the GPU split, all memory terms, and both token totals.
**Never derived, always an editable assumption:** `etaRollout`, `mfuPolicy`, `steps`.

A `preset` layer should mirror the inference pane's use-case presets: *Math/code RLVR* (G=8, L_out=8192, verifiable, β=0), *Short-form instruction RL* (G=8, L_out=1024), *Long-CoT reasoning* (G=16, L_out=32768), *Classic RLHF* (PPO, reward model, G=1, L_out=1024), *Preference tuning* (DPO, precomputed ref).

---

## 11. Worked examples (the UI's defaults sanity check)

Colocated GRPO, `P=64` prompts/step, `L_in=512`, `S=500` steps, H100-80 at **$3.00/GPU-hr**, `η_decode=0.60`, `η_rollout=0.45`, `MFU_policy=0.35`, reference logprob forward included, BF16 KV.

| Model | Method | GPUs | G | `L_out` | B_max | B | waves | TPOT | **gen (s)** | policy (s) | logprob (s) | **step (s)** | **gen %** | **r** | **wall-clock (h)** | **cost** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 7B | full | 8 | 8 | 4,096 | 2,152 | 512 | 1 | 5.6 ms | 51 | 39 | 13 | **104** | 49% | 1.0 | **14.4** | **$346** |
| 7B | full | 8 | 8 | 16,384 | 587 | 512 | 1 | 16.8 ms | 613 | 143 | 48 | **804** | 76% | 3.2 | **111.7** | **$2,680** |
| 7B | full | 8 | 16 | 4,096 | 2,152 | 1,024 | 1 | 10.3 ms | 94 | 78 | 26 | **198** | 47% | 0.9 | **27.5** | **$660** |
| 7B | full | 8 | 16 | 16,384 | 587 | 587 | 2 | 19.2 ms | 1,396 | 285 | 95 | **1,777** | 79% | 3.7 | **246.8** | **$5,923** |
| 7B | LoRA | 8 | 8 | 4,096 | 2,152 | 512 | 1 | 5.6 ms | 51 | 26 | 13 | **90** | 57% | 1.3 | **12.5** | **$301** |
| 7B | LoRA | 8 | 8 | 16,384 | 587 | 512 | 1 | 16.8 ms | 613 | 95 | 48 | **756** | 81% | 4.3 | **105.0** | **$2,520** |
| 7B | LoRA | 8 | 16 | 4,096 | 2,152 | 1,024 | 1 | 10.3 ms | 94 | 52 | 26 | **172** | 55% | 1.2 | **23.8** | **$572** |
| 7B | LoRA | 8 | 16 | 16,384 | 587 | 587 | 2 | 19.2 ms | 1,396 | 190 | 95 | **1,681** | 83% | 4.9 | **233.5** | **$5,605** |
| 70B | LoRA | 32 | 8 | 4,096 | 1,453 | 512 | 1 | 8.9 ms | 81 | 60 | 30 | **171** | 47% | 0.9 | **23.8** | **$2,282** |
| 70B | LoRA | 32 | 8 | 16,384 | 396 | 396 | 2 | 19.8 ms | 1,439 | 220 | 110 | **1,769** | 81% | 4.4 | **245.7** | **$23,592** |
| 70B | LoRA | 32 | 16 | 4,096 | 1,453 | 1,024 | 1 | 15.6 ms | 142 | 120 | 60 | **322** | 44% | 0.8 | **44.7** | **$4,295** |
| 70B | LoRA | 32 | 16 | 16,384 | 396 | 396 | 3 | 19.8 ms | 2,158 | 441 | 220 | **2,819** | 77% | 3.3 | **391.6** | **$37,590** |
| 70B | full | 64 | 8 | 4,096 | 3,001 | 512 | 1 | 4.4 ms | 40 | 45 | 15 | **103** | 39% | 0.7 | **14.4** | **$2,758** |
| 70B | full | 64 | 8 | 16,384 | 818 | 512 | 1 | 12.4 ms | 453 | 165 | 55 | **677** | 67% | 2.1 | **94.0** | **$18,043** |
| 70B | full | 64 | 16 | 4,096 | 3,001 | 1,024 | 1 | 7.8 ms | 71 | 90 | 30 | **194** | 37% | 0.6 | **26.9** | **$5,171** |
| 70B | full | 64 | 16 | 16,384 | 818 | 818 | 2 | 19.2 ms | 1,401 | 331 | 110 | **1,844** | 76% | 3.2 | **256.2** | **$49,183** |

**Memory floors** (from §6.3) and the derived **GPU split** for the disaggregated equivalent:

| Model | Method | Model-state floor | KV pool @ full concurrency | Binding term | Min GPUs (state) | Derived inference fraction at this `r` |
|---|---|---|---|---|---|---|
| 7B | full | 162 GiB | 238 GiB (G=8, L_out=8192) | **KV** | 3 | 50% (r≈1) → 76% (r≈3.2) |
| 7B | LoRA | 49 GiB | 238 GiB | **KV** | 1 | 57% → 81% |
| 70B | full | 1,335 GiB | 1,360 GiB | **tie** | 20 | 41% → 68% |
| 70B | LoRA | 286 GiB | 1,360 GiB | **KV (4.8×)** | 5 | 47% → 81% |

**What to read off this table:**

- **`L_out` is a 10–17× cost lever; `G` is a 2× lever.** 4,096 → 16,384 multiplies cost 7.8–11×; G=8 → 16 multiplies it 1.9–2.1×. The UI should make `L_out` the most prominent control.
- **`r` crosses 1.0 between `L_out` 4,096 and 16,384.** Below that, training time still matters and colocation is defensible; above it, the fleet should be mostly generating.
- **70B full-FT on 64 GPUs costs *more* than 70B LoRA on 32** at every row, while being slower to converge per the LoRA-for-RL result (§7). Default to LoRA.
- Cross-check against §9: 70B LoRA / G=8 / L_out=4,096 / 500 steps = **23.8 h on 32 GPUs = 762 GPU-hours**, against Tülu 3's measured 70B RL run of **2,880 GPU-hours**. **3.8× low.** Tülu 3 is PPO with a critic and reward model, used more steps, and ran on a less-optimized 2024-era stack — but a 3.8× residual against the only well-documented 70B anchor is the honest headline uncertainty, and §14 treats it as such.

---

## 12. Confidence ratings

| Quantity | Confidence | Why |
|---|---|---|
| KV bytes/token, KV pool size | **high** | exact arithmetic from architecture; validated in inference doc §2.3 |
| Resident-copy bytes/param (policy, critic, ref, reward) | **high** | exact arithmetic; corroborated by two independent sources |
| Policy-update FLOPs (`6N`/`4N`) and logprob forwards (`2N`) | **high** | established; back-solved against NeMo-RL |
| Prefix-caching saving `(G−1)/G` on prefill | **high** | definitional |
| LoRA viability for RL at low rank | **high** | primary source, replicated into TRL docs |
| DPO/ORPO compute multipliers (2.0× / 2.7×) | **medium** | derived from exact pass counts |
| `MFU_policy` (0.44 Megatron / 0.13 FSDP2) | **medium** | back-solved from one published table; depends on an inferred token count |
| Generation share / `r` as a function of `L_out` | **medium-low** | 5 independent sources agree on the 60–80% band for long-CoT; the `L_out` dependence is interpolated from 3 points |
| Derived GPU split `r/(1+r)` | **medium-low** | brackets Olmo 3 (71%) and ROLL Flash (60%) but no source states the rule |
| Weight sync at 10 GB/s effective | **medium-low** | single back-solve; published collective latencies span 20 ms to 17 s |
| **`η_rollout` = 0.45** | **low** | two back-solves 2× apart (0.28, 0.56); no source states this factor |
| **Total wall-clock and cost for GRPO/PPO** | **low** | compounds `η_rollout` × `MFU` × `steps`; 3.8× residual vs the Tülu 3 anchor |
| Steps to convergence | **low** | a research outcome, not a config value; 100–8,000 observed |
| Unsloth's 90% GRPO memory reduction | **medium-low** | vendor single-source, specific to their kernels |

Per the brief, the GRPO/PPO path ships at **`low`** overall confidence. DPO/ORPO ship at **`medium`** — an upgrade from the fine-tuning doc's `medium-low`, earned by the exact pass-count derivation and the precompute finding.

---

## 13. Warnings worth emitting

- **`L_out > 16384`** → "At this completion length, 75–85% of your GPU time is autoregressive decode at single-digit MFU. Consider FP8 rollout (1.2–1.5×), INT4 KV (1.8× rollout), or async rollout (1.4–2.8×) before buying more GPUs."
- **KV pool exceeds model state** → "Your binding memory constraint is the KV cache for `P×G` concurrent sequences, not the model. Reducing `G` or enabling FP8 KV frees more memory than quantizing the weights."
- **Colocated with ≥16 GPUs** → "Colocation caps the rollout engine at ~30% of VRAM (TRL's default), costing roughly 3× rollout throughput. At this scale, disaggregate."
- **`beta > 0` with verifiable rewards** → "DAPO and current TRL defaults drop the KL term for RLVR. `beta=0` removes the reference model entirely (−2 bytes/param and −33% of the logprob phase)."
- **`beta = 0` with a learned reward model** → "Without KL regularization against a reference policy, a learned reward model will be hacked. Keep `beta > 0` here."
- **PPO selected** → "PPO holds four model copies, two of them trainable (policy and critic, 16 bytes/param each). GRPO removes the critic for ~30–40% less VRAM at comparable quality; it is the 2026 default for verifiable rewards."
- **Full FT selected where LoRA fits** → quantify the delta (see §11: 70B full on 64 GPUs costs more than LoRA on 32 at every configuration) and cite the LoRA-matches-FullFT-for-RL result.
- **FSDP2/DTensor backend on a ≥32B model** → "Megatron-Core cuts policy-training time 3.4× at 70B (97 s → 28 s/step). At short `L_out` this is a 36% end-to-end saving."
- **`P × G < 512`** → "Below ~512 sequences per step, group-relative advantage normalization is unstable."
- **`G > 16`** → "Rollout cost is linear in `G`. DeepSeekMath used G=64 at `L_out`=1,024; at `L_out`=16K that same `G` is ~16× the budget."
- **`steps` is a guess** → "Published RLVR runs converge anywhere from 100 to 8,000 steps. This input dominates the total and cannot be derived."
- **MoE model** → the fine-tuning doc's warning applies twice over: decode bandwidth follows `N_total` while policy FLOPs follow `N_active`, so MoE is *disproportionately* rollout-bound. Inference doc §1.4: FP8 gives +31% on dense but **+69–74% on MoE**.
- **Reasoning-heavy workload + NVFP4 KV** → inference doc §2.4: "should be validated on your specific task … particularly for reasoning-heavy workloads." RLVR *is* the reasoning-heavy workload.

---

## 14. Honest uncertainty — where this can be 5–10× wrong

The brief asked for this plainly, so: **the wall-clock and cost outputs for GRPO/PPO can be wrong by 5× in either direction, and the dominant error is not in any formula — it is in three inputs that no source pins down.**

**1. `η_rollout` (the roofline realization factor) — up to 2× each way.** My two back-solves differ by exactly 2× (0.28 at 7B/1 GPU, 0.56 at 70B/64 GPUs). **No published source states this factor at all**; it is my construction to reconcile the decode roofline with measured RL rollout times. The honest finding here is negative and worth recording: *there is no published measurement of rollout throughput achieved as a fraction of the decode roofline in an RL loop.* Pin it down by having the user supply **one** number: observed tokens/s/GPU during rollout, or observed seconds-per-step, from their own stack. One measured step calibrates everything downstream.

**2. `steps` — unbounded.** Published runs span 100 (SimpleRL) to ~8,000 (R1). This is not a knob; it is how long the reward curve takes to flatten on *their* data. A 3× miss here is a 3× miss in cost, and it is the user's to supply.

**3. The long-tail distribution — up to 10×.** I model the step as gated by `L_out,max`, which is correct for strictly synchronous GRPO. But (a) if the user's completions genuinely cluster well below the cap *and* their stack uses continuous batching with in-flight replacement, the true cost is closer to the mean, i.e. up to 10× cheaper (NeMo-RL's 70B row has mean/max = 398/4096); and (b) RollPacker exists precisely because the tail is expensive, reporting **2.03–2.56× end-to-end speedup over verl** from tail batching alone. **So the same workload has a 10× band depending on scheduling, and the calculator sees none of that.** Expose `meanCompletionRatio` and a `tailGating: 'max' | 'mean' | 'blend'` toggle rather than pretending to know.

**Secondary, 1.3–3× each:**

- **MFU_policy** — 0.13 (DTensor) to 0.44 (Megatron), a 3.4× spread I can only resolve by asking which backend.
- **Async vs sync** — 1.4–2.8× wall-clock ([AReaL 2.77×](https://arxiv.org/pdf/2505.24298), [ROLL Flash 1.8–2.7×](https://arxiv.org/html/2510.11345), "~40–70% speedup from overlapping" per [LLM Stack](https://prakashkagitha.github.io/llm-stack-book/06-rl-infra/02-generation-training-loop.html)). My model is synchronous.
- **Reward/verifier latency** — modeled as ~3%. For code-execution or tool-use rewards with sandboxes, this can be the bottleneck and I have **no sourced figure** for it. Agentic RL with environment steps is out of scope and should be stated as such in the UI.

**The one concrete validation failure, stated plainly.** §11's 70B LoRA / G=8 / L_out=4,096 / 500-step row gives 762 GPU-hours. The only well-documented 70B RL anchor — Tülu 3 — measured **2,880 GPU-hours**. My estimate is **3.8× low**. Part is explicable (PPO vs GRPO, critic + reward model, 2024-era stack, unknown step count) and part is not. **Until a second 70B anchor exists, treat the GRPO wall-clock output as a lower bound and say so in the UI**, e.g. a results-card footnote: *"Compared against the one published 70B RL run we could source, this model runs ~4× optimistic. Treat it as a floor."* That is more useful than a confidence badge.

**What would fix this:** a single "calibrate from my run" input — observed seconds per step, at a stated `(model, GPUs, G, P, L_out, topology)`. One such point collapses `η_rollout`, `MFU_policy`, and the tail factor into one measured constant, and would move this pane from `low` to `medium` confidence. **Recommend building that input in v1, not v2.**

---

## 15. Open design questions

1. **Separate mode, or a method dropdown inside Fine-Tuning?** The memory engine is shared; the time engine is not (decode roofline vs FLOP/MFU). Recommendation: **a method dropdown that switches the pane's hero output** — Fine-Tuning shows "min GPUs + time + cost," RL shows "GPU split + step anatomy + tokens generated."
2. **Hero output: the GPU split, or the step-time waterfall?** The waterfall (generation / logprob / update / sync) teaches the user why the answer is what it is, and the split falls out of it. Lean waterfall.
3. **Should `tokensGenerated` be a headline number?** 500 steps × 64 prompts × 8 completions × 4,096 tokens = **1.05 billion generated tokens** for one modest 7B run. That number is more legible than GPU-hours and connects directly to the Inference pane's per-million-token costs — arguably the best cross-pane link in the app.
4. **Async rollout as a toggle or a phase 2?** It is 1.4–2.8× and the 2026 default in every serious framework, but it introduces staleness as a correctness concern. Suggest a toggle with a fixed 2.0× and a `low` badge, not a derived model.
5. **Agentic RL (multi-turn, tool calls, environment steps)** is the fastest-growing RL workload and this document does not model it: environment latency, not decode, can bind. Scope it out explicitly in the UI rather than silently mis-answering.
6. **Should the pane refuse to answer without a calibration point?** Given §14, an honest alternative to shipping a `low`-confidence number is to require one measured step first. Probably too strict for v1, but worth discussing.

---

## Sources

**Primary — RL algorithms and model reports**
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning](https://arxiv.org/pdf/2402.03300) — GRPO's origin; G=64, lr 1e-6, KL coef 0.04, max len 1024, train batch 1024
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/pdf/2501.12948)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/pdf/2503.14476) — removes the KL term; 50 on AIME'24 at 50% of GRPO's steps
- [Tülu 3: Pushing Frontiers in Open Language Model Post-Training](https://arxiv.org/pdf/2411.15124) — the 8B/70B RL GPU-hour anchors
- [Scaling the Tülu 3 recipes to 405B (AI2 blog)](https://allenai.org/blog/tulu-3-405B)
- [Olmo 3](https://arxiv.org/html/2512.13961v2) — 20 inference / 8 training node split; 4× OlmoRL speedup; 5-day and 21-day runs
- [Qwen3 Technical Report](https://arxiv.org/pdf/2505.09388) — 17,920 GPU-hours RL; 170 steps for AIME'24 70.1→85.1
- [LoRA Without Regret (Thinking Machines Lab)](https://thinkingmachines.ai/blog/lora/) — LoRA matches FullFT for policy gradient at rank 1; ~67% of compute
- [SimpleRL-reason](https://github.com/hkust-nlp/simpleRL-reason) — 16× / 64× H100 run times

**Primary — RL systems and infrastructure**
- [NeMo-RL v0.3: Megatron-Core support (NVIDIA)](https://developer.nvidia.com/blog/reinforcement-learning-with-nvidia-nemo-rl-megatron-core-support-for-optimized-training-throughput/) and [discussion #1161](https://github.com/NVIDIA-NeMo/RL/discussions/1161) — **the per-phase GRPO step-time tables; the most load-bearing source in this document**
- [RollPacker: Mitigating Long-Tail Rollouts](https://arxiv.org/pdf/2509.21009) — 70% rollout / 21–23% training; 2.03–2.56× over verl on ≤128 H800
- [ROLL Flash (Part II)](https://arxiv.org/html/2510.11345) — ">70% rollout"; the `β*` GPU-allocation formula; 24/16 measured optimum
- [AReaL: Large-Scale Asynchronous RL for Language Reasoning](https://arxiv.org/pdf/2505.24298) — 2.77× over synchronous at equal GPUs
- [AReaL-Hex: Asynchronous RL over Heterogeneous GPUs](https://arxiv.org/html/2511.00796) — weight sync 1.9% homogeneous / 13.54% heterogeneous
- [StreamRL: Disaggregated Stream Generation](https://arxiv.org/pdf/2504.15930)
- [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries (Hugging Face)](https://huggingface.co/blog/async-rl-training-landscape) — the 7B/32B rollout timing table; NCCL 100–500 ms vs bucketed ~20 ms; Mooncake 16–17 s for Kimi-K2
- [OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework](https://arxiv.org/pdf/2405.11143) — the 70B GPU-allocation recipes
- [OpenRLHF Hybrid Engine docs](https://openrlhf.readthedocs.io/en/latest/hybrid_engine.html)
- [verl Performance Tuning Guide](https://verl.readthedocs.io/en/latest/perf/perf_tuning.html) and [config reference](https://verl.readthedocs.io/en/latest/examples/config.html)
- [verl LoRA support for RL(HF)](https://verl.readthedocs.io/en/latest/advance/ppo_lora.html)
- [verl FP8 RL](https://verl.readthedocs.io/en/latest/low_precision/fp8.html)
- [Scaling RL with verl on AMD Instinct MI355X (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/verl/README.html) — GRPO/DAPO wall-clock, GPU splits, trainer idle ratios, tok/GPU/s
- [TRL GRPO Trainer docs](https://huggingface.co/docs/trl/main/en/grpo_trainer) — `num_generations=8`, `max_completion_length=512`, `beta=0.0`, `vllm_mode="colocate"`, `vllm_gpu_memory_utilization=0.3`, PEFT adapter-disable reference
- [TRL DPO Trainer docs](https://huggingface.co/docs/trl/en/dpo_trainer) — `precompute_ref_log_probs`
- [TRL: LoRA Without Regret (docs port)](https://huggingface.co/docs/trl/lora_without_regret)
- [vLLM Sleep Mode docs](https://docs.vllm.ai/en/latest/features/sleep_mode/) — level 1 vs level 2, >90% memory release
- [Prefix Grouper: Efficient GRPO Training through Shared-Prefix Forward](https://arxiv.org/html/2506.05433v1) — the `1/G` reduction

**Primary-ish — quantized rollout**
- [Unified FP8 for MoE RL (LMSYS)](https://www.lmsys.org/blog/2025-11-25-fp8-rl/) — FP8 rollout 1.2–1.5× and ~half the rollout memory
- [Jet-RL: On-Policy FP8 RL](https://arxiv.org/html/2601.14243) — 33% rollout / 41% training / 16% end-to-end
- [KV4-RL: INT4 KV Cache for RLVR rollouts](https://arxiv.org/html/2610.14448) — 1.80× rollout, 1.27× full iteration
- [AIS: Adaptive Importance Sampling for Quantized RL](https://arxiv.org/html/2605.13907v1) — retains 1.5–2.76× FP8 rollout speedup at BF16 parity
- [µ-GRPO: How Off-Policy Can GRPO Be?](https://arxiv.org/html/2605.17570v1) — 1.82× rollout / 1.53× wall-clock
- [SparseRL-Sync: Lossless Weight Synchronization with ~100× Less Communication](https://arxiv.org/pdf/2605.07330)

**Secondary / explanatory (flagged as such)**
- [The LLM Stack, ch. 6.2: The Generation–Training Loop & Rollout Engines](https://prakashkagitha.github.io/llm-stack-book/06-rl-infra/02-generation-training-loop.html) — 60–80% generation; the 70/15/10/3/2 breakdown; MFU asymmetry; a worked 7B H100 roofline. **The best single conceptual source found; a book chapter, not peer-reviewed.**
- [Epoch AI: What went into training DeepSeek-R1?](https://epoch.ai/gradient-updates/what-went-into-training-deepseek-r1) — ~2T generated tokens, ~8,000 gradient steps, ~$1M RL phase, with the author's own ±2× caveat
- [Mini-R1: Reproduce DeepSeek R1's "aha moment" (Phil Schmid)](https://www.philschmid.de/mini-deepseek-r1) — 4× H100, 450 steps, 6 h; the $2.49/H100-hr conversion basis
- [We Ran Multi-Node GRPO on 8 GPUs (Union.ai)](https://www.union.ai/blog-post/we-ran-multi-node-grpo-on-8-gpus-and-the-trainer-cost-us-nothing) — 294 s generation vs 140 s training; LoRA adapter sync "a few megabytes"
- [Unsloth: Long-context GRPO](https://unsloth.ai/blog/grpo) (vendor; 403s to direct fetch — figures taken from [this summary](https://www.rohan-paul.com/p/unsloth-releases-new-grpo-algorithms) and [DigiAlps](https://digialps.com/you-can-now-train-reasoning-models-with-5gb-vram-thanks-to-unsloth/)) — 510.8 → 54.3 GB at 20K context; 5 GB for Qwen2.5-1.5B
- [RLHF Training Infrastructure: verl, OpenRLHF, TRL (Spheron)](https://www.spheron.network/blog/rlhf-training-infrastructure-verl-openrlhf-trl-gpu-cloud/) — **aggregator**; the per-copy bytes/param table, the `P×G ≥ 512` stability rule, verl config defaults, GPU hourly rates, the 70B PPO cost band
- [GRPO Fine-Tuning on GPU Cloud (Spheron)](https://www.spheron.network/blog/grpo-fine-tuning-gpu-cloud/) — **aggregator**; the "GRPO removes the critic, −30–40% VRAM" figure
- [DPO vs PPO for Production Alignment (Spheron)](https://www.spheron.network/blog/dpo-vs-ppo-rlhf-algorithm-production-llm-alignment/) — **aggregator**
- [Interconnects: GRPO tweaks, base model RL, data curation](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo) — on when the KL term is and is not needed
- [RollArt: Scaling Agentic RL Training via Disaggregated Infrastructure](https://www.researchgate.net/publication/399175445_RollArt_Scaling_Agentic_RL_Training_via_Disaggregated_Infrastructure) — agentic RL, out of scope here but the obvious next gap
- [DeepSeek's reported $294,000 training cost (CNN)](https://www.cnn.com/2025/09/19/business/deepseek-ai-training-cost-china-intl)

**Composed with (in-repo)**
- `docs/research/2026-10-14-inference-serving-research.md` §1 (prefill/decode roofline, `η_decode=0.60`, `η_prefill=0.82`, hardware constants) and §2 (KV-cache sizing, `B_max`, FP8/NVFP4 KV)
- `docs/research/2026-10-14-finetuning-calculator-research.md` §1 (memory decomposition, bytes/param, activations), §2.1 (`6N`/`4N`), §1.6 (the RL multipliers this document corrects)
