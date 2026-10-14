# Deep Research: Inference Serving Economics and the Buy-vs-Build Model

**Date:** 2026-10-14
**Purpose:** Establish the quantitative foundation for the existing **Inference / buy-vs-build** pane, which was built from two short decisions docs (`docs/decisions/2026-06-23-inference-calculator-decisions.md`, `docs/decisions/2026-06-24-inference-ux-presets.md`) and has never had a research backing.
**Question the pane answers:** *Should I call an API or self-host — and at what volume does that flip?*

---

## 0. The central finding (shapes the whole design)

**Inference is throughput- and bandwidth-bound, and the cost per token is a function of *load*, not of hardware.**

Contrast the three panes:

| Pane | Binding constraint | The question |
|---|---|---|
| Training | **Compute** | `GPUs = total_FLOPs / (deadline × sustained_FLOPs)` |
| Fine-tuning | **Memory capacity** | *Does the model state fit at all?* |
| **Inference** | **Memory bandwidth, then KV-cache capacity** | *How many concurrent sequences can I keep in flight under an SLO?* |

The dominant modeling error in inference sizing is treating throughput as a property of (model, GPU) when it is actually a property of (model, GPU, **concurrency**, **SLO**). The single most important number found in this research:

> On identical H100 hardware serving the same model, measured effective cost ranged from **$0.21 to $15.25 per million output tokens** — a **2.5–24× spread** across enterprise load levels (1–10 rps), and up to **36.3×** between idle and saturation.
> — [Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation](https://arxiv.org/html/2606.11690v1)

That paper's framing should become the pane's framing. It defines effective cost as

```
C_eff = (P_GPU × 1e6) / (3600 × Θ_achieved(λ, L))
```

where `Θ_achieved` depends on the offered arrival rate `λ` and the latency SLO `L`, and explicitly argues:

> "GPU utilization U is not an independent parameter to be assumed … but is itself a dependent variable determined by the interaction of H, M, Q, λ, and L."

The repo currently does the opposite: `src/data/self-host-throughput.json` hardcodes `outputTokensPerSec` per (model, GPU, quantization) as if it were a hardware constant, and `src/engine/inference-calculator.ts:105` derives utilization from it. **Utilization is treated as an output of a constant, when it should be the input that determines the constant.** Everything else in this document follows from inverting that.

Three secondary findings with large cost consequences:

1. **Prefix caching is no longer a rounding error.** OpenRouter reports that **>85% of agentic tokens are cached prompt tokens** ([OpenRouter / Futurum, 2026](https://www.tomshardware.com/tech-industry/artificial-intelligence/futurum-ceo-says-agents-use-ai-5x-more-than-humans-number-will-eventually-hit-10x-but-agents-are-mostly-rereading-what-theyve-already-seen)). Every major API prices cache reads at 2.5–10% of base input. The pane's `ApiModel` type has no cache fields at all (`src/data/types.ts:1-6`), so for the Agentic and RAG presets it overstates API cost by ~34% end-to-end (worked below, §7.3).
2. **TTFT is dominated by input length, which the current model ignores entirely.** `estimatedTtftMs = 50 + paramBillions * 3` (`src/engine/inference-calculator.ts:108-109`) is a function of parameter count only. Prefill FLOPs are `2·N·s` — linear in sequence length. At the Long-context preset (50,000 input tokens) the heuristic is **~8.3× low**; at the Chatbot preset (500 input tokens) it is **~7.5× high**.
3. **Self-hosting at 100% uptime with no redundancy, as currently modeled, understates real cost by 3–5×.** Published figures put average utilization in self-hosted inference clusters at **22%**, the break-even utilization versus API pricing at **~60%**, and all-in TCO at **3–5× the raw GPU rental** ([Cerevisor](https://cerevisor.com/blog/self-hosted-llm-cost-vs-api-utilization)). The breakeven volume the pane reports is therefore 3–5× too low.

---

## 1. First-principles serving performance model

### 1.1 The two-phase decomposition

Every request has two phases with opposite hardware characteristics. This is the standard roofline decomposition ([LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/html/2402.16363v4), [Spheron roofline analysis](https://www.spheron.network/blog/the-roofline-model-for-llm-inference-are-you-compute-bound-o/), [Redis: Prefill vs Decode](https://redis.io/blog/prefill-vs-decode/)):

| Phase | Work | Arithmetic intensity | Bound by | Measured efficiency |
|---|---|---|---|---|
| **Prefill** | `2 · N_active · s` FLOPs | ≈ `s` (batch in tokens), so ~2,048 FLOPs/byte at s=2048 | **Compute** | **80–84% of peak FLOPS** |
| **Decode** | `2 · N_active` FLOPs/token, `N_total · b_w` bytes read/token | ≈ `B` (concurrent sequences) for the linear layers; **~1 FLOP/byte** for the attention/KV read, regardless of B | **HBM bandwidth** | **~60% of peak HBM** |

For a linear layer: FLOPs ≈ `2 · B · d_in · d_out`, bytes moved ≈ `2 · d_in · d_out`, so **arithmetic intensity ≈ B**. That single identity is the whole model: decode's position on the roofline *is* the batch size.

### 1.2 The formulas the engine should implement

```
# Prefill / TTFT
prefill_flops   = 2 · N_active · s_in
TTFT            = T_queue + prefill_flops / (G · F_peak · η_prefill)      η_prefill ≈ 0.82

# Decode / TPOT (per-token, at concurrency B)
bytes_per_step  = N_total · b_w  +  B · s_avg · kv_bytes_per_token
TPOT            = bytes_per_step / (G · BW_peak · η_decode)                η_decode ≈ 0.60
    … until compute-bound, i.e. until B ≥ B_ridge (§1.4), after which
TPOT            = 2 · N_active · B / (G · F_peak · η_prefill) / B

# Aggregate output throughput
Θ_out(B)        = B / TPOT(B)
```

`G` = GPU count in the tensor-parallel group (bandwidth and FLOPS both scale with `G`, to first order, minus comm overhead).

Note the second term of `bytes_per_step`: **the KV cache is re-read every decode step.** At long context it overtakes the weights as the dominant byte traffic, which is why TPOT degrades with context length even though FLOPs/token do not.

### 1.3 Hardware constants (calibration inputs)

Dense (non-sparsity) tensor-core figures. **This table resolves a 2× error that is endemic in the secondary sources.**

| GPU | HBM | BW (TB/s) | BF16 dense (TFLOPS) | FP8 dense (TFLOPS) | Ridge point, BF16 (FLOPs/byte) |
|---|---|---|---|---|---|
| H100 SXM5 | 80 GB HBM3 | **3.35** | **989.5** | **1,979** | **295** |
| H200 SXM | 141 GB HBM3e | **4.8** | 989.5 | 1,979 | ~206 |
| B200 | 180 GB HBM3e | **7.7** | ~1,125 | ~2,250 | ~146 |
| B300 | up to 288 GB HBM3e | ~8.0 | — | — | — |
| MI300X | 192 GB HBM3 | **5.3** | ~653 | ~1,307 | ~123 |
| MI355X | 288 GB HBM3e | **8.0** | ~1,250 | ~2,500 | ~156 |

**Warning for implementers:** aggregator pages routinely quote H100 as "1,979 TFLOPS BF16 / 3,958 TFLOPS FP8" — those are the **with-sparsity** numbers and are 2× the dense figures that apply to LLM inference. [One such page](https://www.spheron.network/blog/nvidia-h100-specs/) states "3,958 TFLOPS" for H100; the roofline source correctly uses 989.5 BF16 / 1,979 FP8 dense. The B200/MI300X/MI355X rows above are derived from aggregator tables by halving the sparse figures and are **medium-low confidence** — verify against NVIDIA/AMD datasheets before shipping. BW figures are consistent across sources and are high confidence.

### 1.4 The roofline knee (critical batch size)

Published H100 BF16 operating points, 70B model ([Spheron roofline](https://www.spheron.network/blog/the-roofline-model-for-llm-inference-are-you-compute-bound-o/)):

| Operating point | Arithmetic intensity | Achieved | % of compute roof |
|---|---|---|---|
| Decode, B=1 | ~1 FLOP/byte | 3.4 TFLOPS | **0.3%** |
| Decode, B=32 | ~32 FLOPs/byte | 107 TFLOPS | 11% |
| Decode, B≈280 | ~280 FLOPs/byte | 938 TFLOPS | 95% |
| Prefill, s=2048 | ~2,048 FLOPs/byte | 989.5 TFLOPS | 100% |

**`B_ridge` ≈ 280 concurrent sequences on H100 at BF16.** Below that, decode is memory-bound and `Θ_out` scales *almost linearly* with concurrency — this is the single most important fact the lookup table hides. Serving at B=1–32 leaves 89–99.7% of the compute roof on the floor.

**Quantization does not move the knee.** FP8 halves the bytes per weight (intensity doubles) *and* doubles the compute roof, so `B_ridge` "barely shifts." FP8 is a **bandwidth-efficiency lever (≈2× decode throughput at fixed B), not a regime change.** Corroborated independently: FP8 gives **+31% peak throughput on dense models but +69–74% on MoE** ([arXiv 2606.11690](https://arxiv.org/html/2606.11690v1)) — MoE benefits 2.2–2.4× more because its decode is even more bandwidth-skewed relative to its active FLOPs.

### 1.5 Back-solved default efficiency factors

```
η_prefill (sustained / peak FLOPS)  = 0.82     # from "prefill runs at 80–84% of peak"
η_decode  (sustained / peak HBM BW) = 0.60     # from "decode achieves 60% of peak HBM throughput"
tp_scaling_efficiency(G)            = 0.95^log2(G)   # low confidence, see §9
```

**Sanity-check the model against a published claim, and watch it catch an error.** One source states "for an 8B model in BF16 on an H100, a single user can achieve 223 tokens per second at batch one" ([The Software Frontier](https://www.thesoftwarefrontier.com/p/llm-inference-benchmarks-seven-stopwatches)). Llama 3.1 8B in BF16 is 16.06 GB of weights. 223 tok/s × 16.06 GB = **3.58 TB/s — above the H100's 3.35 TB/s peak.** The claim is physically impossible at BF16 batch 1; it must be FP8, a smaller model, or speculative decoding. The roofline model predicts 209 tok/s at 100% of peak BW and **125 tok/s at η=0.60**. *This is the argument for replacing the lookup table: a first-principles model can reject bad inputs; a table of magic numbers cannot.*

---

## 2. KV cache sizing — the real concurrency limit

### 2.1 Exact formula

```
kv_bytes_per_token = 2 · L · H_kv · d_head · b_kv          # the leading 2 is K and V
kv_bytes_per_seq   = kv_bytes_per_token · (s_in + s_out)
```

`b_kv` = 2 (BF16/FP16), 1 (FP8/INT8), 0.5 (NVFP4). Under MLA the formula collapses — query-head count disappears entirely:

```
kv_bytes_per_token(MLA) = L · (kv_lora_rank + qk_rope_head_dim) · b_kv
```

### 2.2 Per-token figures for current models (BF16)

Primary source for the architecture-derived column: [Sebastian Raschka, KV Cache Calculations](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/). Cross-checked against [Spheron KV cache guide](https://www.spheron.network/blog/kv-cache-optimization-guide/) and [AWS KV cache tiering](https://aws.amazon.com/blogs/storage/accelerate-inference-with-kv-cache-tiering-on-aws/).

| Model | Attention | KV / token | Derivation | At 8K ctx | At 32K ctx |
|---|---|---|---|---|---|
| Llama 3.1 8B | GQA (32L, 8 kv, 128) | **128 KiB** (0.131 MB) | 32·8·128·4 | 1.07 GB | 4.30 GB |
| Llama 3.1 70B | GQA (80L, 8 kv, 128) | **320 KiB** (0.327 MB) | 80·8·128·4 | 2.68 GB | **10.74 GB** ✓ |
| Llama 3.1 405B | GQA (126L, 8 kv, 128) | **504 KiB** (0.516 MB) | 126·8·128·4 | 4.0 GB | 16.5 GB |
| Qwen3 8B | GQA (36L, 8 kv, 128) | **144 KiB** | 36·8·128·4 | 1.15 GB | 4.6 GB |
| **DeepSeek V3** | **MLA** (61L, d_c=512, rope 64) | **68.6 KiB** | 61·(512+64)·2 | 0.55 GB | 2.2 GB |
| DeepSeek V4.1-Flash | CSA2 (compressed shared) | **3.125 KiB** | — | 0.026 GB | 0.10 GB |
| Qwen3-Next 80B-A3B | sparse full-attn (12L, 2 kv, 256) | **24 KiB** | 12·2·256·4 | 0.19 GB | 0.77 GB |
| Gemma 4 31B | hybrid SWA + global | **840 KiB** | mixed | 6.7 GB | 26.9 GB |
| Gemma 4 26B-A4B | hybrid | **210 KiB** | mixed | 1.7 GB | 6.7 GB |
| xLSTM 7B | none (recurrent) | **0 B** | — | 0 | 0 |

**The spread is 300×** between DeepSeek V3's MLA (68.6 KiB/token) and Gemma 4 31B's hybrid attention (840 KiB/token) — *at similar parameter counts*. MLA is reported to deliver a **~64× reduction versus naive MHA** at DeepSeek's dimensions (1,024 B/layer/token vs 65,536 B) and DeepSeek V3 holds **70 KB/token against Llama 3.1 405B's 516 KB** ([Spheron MLA guide](https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud/), [DeepSeek-V2 paper](https://arxiv.org/pdf/2405.04434), [DeepSeek-V3 hardware reflections](https://arxiv.org/pdf/2505.09343)).

**Attention architecture, not parameter count, determines how many users a GPU can serve.** No field in `SelfHostEntry` (`src/data/types.ts:37-46`) can express this.

### 2.3 KV-cache-limited maximum concurrency

```
vram_usable      = gpu_count · vram_per_gpu · gpu_memory_utilization    # vLLM default 0.90
vram_for_kv      = vram_usable − N_total · b_w − activation_overhead
B_max            = floor(vram_for_kv / kv_bytes_per_seq)
B_effective      = min(B_max, B_slo)                                     # §5
```

`--gpu-memory-utilization 0.90` is the vLLM default; PagedAttention is on by default ([Spheron KV guide](https://www.spheron.network/blog/kv-cache-optimization-guide/)).

Worked examples (BF16 weights, BF16 KV):

| Config | Usable | Weights | KV pool | s=8K → B_max | s=32K → B_max | s=128K → B_max |
|---|---|---|---|---|---|---|
| Llama 3.1 8B, 1× H100 80 GB | 72 GB | 16.1 GB | 55.9 GB | **52** | 13 | 3 |
| Llama 3.1 70B, 4× H100 | 288 GB | 140 GB | 148 GB | **55** | **13** | 3 |
| Llama 3.1 70B, 2× H200 (141 GB) | 253.8 GB | 140 GB | 113.8 GB | 42 | 10 | 2 |
| DeepSeek V3 (MLA, FP8 wts), 8× H200 | 1,015 GB | ~671 GB | 344 GB | **598** | 149 | 37 |

The published anchor confirms the 70B row: "KV cache per request ~10.7 GB (BF16) at 32K; 8 concurrent users ~86 GB" — matching `0.320 MiB × 32768 = 10.5 GB` to within 2%.

**This is the finding that most directly indicts the current data model.** `self-host-throughput.json:7` declares `maxConcurrentRequests: 32` for Llama 3 70B on 4× H100 as a scalar. The true value is **55 at 8K context and 13 at 32K** — wrong by 1.7× in one direction and 2.5× in the other, from the same row, depending on an input the pane already collects (the use-case preset sets input tokens from 500 to 50,000).

### 2.4 KV cache quantization

| Precision | Memory vs BF16 | Production guidance |
|---|---|---|
| FP8 KV | **−50%** | "safe for production across virtually all tasks" |
| INT8 KV | −50% | similar, more calibration-sensitive |
| NVFP4 KV (Blackwell only) | **−75%** | "should be validated on your specific task … particularly for reasoning-heavy workloads" |

Measured compounding on Llama 3.1 70B / H100 ([Spheron KV guide](https://www.spheron.network/blog/kv-cache-optimization-guide/)):

| Config | Output tok/s | Concurrent users |
|---|---|---|
| Naive allocation | 400 | 4 |
| + PagedAttention | 900 | 8 |
| + FP8 KV | **1,800** | 16 |
| + NVFP4 KV (Blackwell) | **2,600** | 32 |

A **6.5× throughput range for the same model on the same GPU**, driven entirely by KV-cache management. The lookup table offers one number (1,200 for 70B on 4× H100) and no way to pick a point on this curve.

---

## 3. Continuous batching, throughput, and goodput

### 3.1 Continuous batching gains

| Lever | Gain | Source |
|---|---|---|
| Continuous vs static batching | **2–3× throughput** | GMI Cloud via [Cerevisor](https://cerevisor.com/blog/self-hosted-llm-cost-vs-api-utilization) |
| Continuous batching vs idle, cost/token | **3–4× cheaper** | same |
| Utilization ceiling with continuous batching | **>70%** (from 22% baseline) | Spheron, Apr 2026, via same |

### 3.2 Published throughput anchors

| Model | Hardware | Framework / config | Output tok/s | Per GPU |
|---|---|---|---|---|
| DeepSeek-V3 | multi-node H200 (CoreWeave, IB + CX-7) | vLLM, Wide-EP, **disaggregated P/D** | — | **2,200** |
| DeepSeek-V3 | same, pre-optimization baseline | vLLM | — | ~1,500 |
| Qwen 3.5 27B | **96× B200** (GKE) | — | **1,103,941** | ~11,500 |
| Llama 3.3 70B | 8× H200 SXM5 | continuous batching, B=256 | 3,600 | 450 |
| 70B class | 8× B200 (spot) | — | 5,200 | 650 |
| Llama 4 70B BF16 | B200, B=8 | vLLM | — | 800–1,200 |
| Qwen2.5-7B | B200 | vLLM | — | 272 tok/s **per user** |
| Qwen2.5-7B | H200 | vLLM | — | 195 tok/s **per user** |

Sources: [vLLM large-scale serving blog](https://vllm.ai/blog/2025-12-17-large-scale-serving) (primary, the 2.2k/1.5k figures), [Google Cloud on GKE + B200](https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592), and [Spheron GPU cost-per-token benchmark](https://www.spheron.network/blog/gpu-cost-per-token-benchmark-llm-inference-2026/) / [RunPod GPU guide](https://www.runpod.io/articles/guides/best-gpu-for-llm-inference) (aggregators, for the 70B rows).

Note the **8× range in per-GPU throughput (272 → 2,200) across these rows**, driven by model, concurrency, parallelism strategy, and disaggregation — not by GPU generation.

### 3.3 Throughput–latency Pareto frontier and goodput

NVIDIA frames this as a Pareto frontier of **datacenter throughput (tok/s/GPU) vs responsiveness (tok/s/user)** ([NVIDIA TensorRT-LLM benchmarking](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/)). **Goodput** counts only work that finished while satisfying the SLO ([Cerebrium](https://cerebrium.ai/resources/llm-serving-goodput-latency-slos)) — distinct from saturation throughput, which has no SLO constraint.

The SLO premium, measured ([arXiv 2606.11690](https://arxiv.org/html/2606.11690v1)), under TTFT p99 ≤ 300 ms and TPOT p99 ≤ 50 ms:

| Model | λ_max under SLO | Cost/MTok at λ_max | Premium over saturation floor |
|---|---|---|---|
| Llama 3.1 8B FP16 | 25 rps | $0.368 | **1.18×** |
| Mixtral 8×7B FP16 | 10 rps | $1.666 | **1.91×** |

**A pane that reports saturation throughput reports a cost 15–48% below what any latency-sensitive product can actually achieve.** The premium is larger for MoE because its decode is more bandwidth-skewed.

---

## 4. Prefix caching / prompt caching

### 4.1 Hit rates by workload

Ranges are wide and source-dependent; present as a range with a conservative default.

| Workload | Reported hit rate | Confidence |
|---|---|---|
| Agents / multi-tenant SaaS with fixed system prompt + tool defs | **60–95%** (75–95% where prefix overlap > 60%) | medium |
| Multi-turn chat | 75–90% (SGLang RadixAttention) vs 10–20% (claimed, vLLM APC) | **low** — see caveat |
| Code analysis | 60–80% vs 5–15% (same comparison) | low |
| RAG with stable instructions | 60–85% "without special effort" | medium |
| Creative generation, personalized recs (every request unique) | **~0%** | high |
| Scaling with prefix length at 80% overlap | ~75% @ 256 tok → ~92% @ 2,048 tok | medium |

**Caveat on the SGLang-vs-vLLM gap:** the 75–90% vs 10–20% comparison comes from a single aggregator family ([Spheron vLLM-vs-SGLang](https://www.spheron.network/blog/vllm-vs-sglang-2026/)) and is not corroborated by primary benchmarks. vLLM's automatic prefix caching is hash-based prefix matching and should not be an order of magnitude worse on the same workload. **Treat the framework gap as unverified; treat the workload-dependence as well-supported.** Eviction policy is the real variable — one source notes prefix caching "saves 80% of prefill cost" but that eviction policies "quietly turn it into 5%" ([dev.to](https://dev.to/tech_nuggets/prefix-caching-at-scale-when-it-saves-you-80-of-prefill-cost-and-the-eviction-policies-that-5e8)).

### 4.2 The hard production datum

> **More than 85% of agent tokens come from cached prompts.** Agentic token usage on OpenRouter rose from 0.51T to 7.3T tokens between February and August 2026 — **14× in six months** — versus 1.4T human tokens.
> — [OpenRouter data](https://openrouter.ai/data), [State of AI: 100 Trillion Token Study](https://arxiv.org/html/2601.10088v1), [Tom's Hardware summary](https://www.tomshardware.com/tech-industry/artificial-intelligence/futurum-ceo-says-agents-use-ai-5x-more-than-humans-number-will-eventually-hit-10x-but-agents-are-mostly-rereading-what-theyve-already-seen)

This is the strongest single argument for adding cache pricing to the pane: the fastest-growing workload class is **85% cache reads by token volume**, and cache reads cost 2.5–10% of base input.

### 4.3 API-side cache pricing (primary sources)

| Provider | Cache write | Cache read | Multiplier vs input |
|---|---|---|---|
| Anthropic (standard) | 1.25× input (5 min) / 2.0× (1 h) | **0.1× input** | 10% |
| Anthropic — Opus 5.5, Sonnet 5.5 | same | **0.05× input** | **5%** |
| Anthropic — Fable 5.1, Mythos 5.1 | same | **0.025× input** | **2.5%** |
| OpenAI | not separately charged | 0.1× input (gpt-5.x/6.x); 0.5× on gpt-4.1/o3; 0.5× on gpt-4o | 10–50% |
| Google Gemini | — | ~0.05–0.1× input **plus hourly storage** ($0.50–$4.50/MTok-hr) | 5–10% + storage |
| DeepSeek | — | **0.02–0.03× input** ($0.022 vs $0.66 cache-miss on v4-pro) | **~3%** |
| Fireworks | — | 0.02–0.2× input (varies by model) | 2–20% |

Sources: [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing), [OpenAI pricing](https://developers.openai.com/api/docs/pricing), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing), [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing), [Fireworks serverless pricing](https://docs.fireworks.ai/serverless/pricing).

**Google's model is structurally different** — explicit context caching with a *storage* charge per MTok-hour, not just a per-read discount. A single `cacheReadPer1M` field cannot represent it; needs an optional `cacheStoragePer1MPerHr`.

---

## 5. Latency SLOs by use case

Primary-ish source: [Spheron LLM Inference SLO Engineering](https://www.spheron.network/blog/llm-inference-slo-ttft-itl-latency-budget-guide-2026/) (aggregator, but the only source found with a complete per-use-case table; corroborated on the chat row by multiple sources).

| Use case | TTFT p99 | ITL p99 | Notes |
|---|---|---|---|
| Voice agent | **150 ms** | **30 ms** | TTS adds 100–200 ms; tightest budget |
| Code completion (inline) | **100 ms** | 25 ms | must feel local |
| Interactive chat | **300 ms** | 50 ms | 500 ms = perceived lag; 800 ms = measurable abandonment |
| Code completion (panel) | 300 ms | 50 ms | |
| RAG-augmented chat | **400 ms** | 80 ms | retrieval precedes prefill |
| Batch / async agent | 3,000 ms | 200 ms | SLO on total job time |

### 5.1 TTFT decomposition and the queueing term

```
TTFT = T_net_in + T_queue + T_prefill        T_net_in ≈ 5–30 ms regional
```

Measured p99/p50 TTFT ratio vs concurrency (same source):

| Concurrent requests | p50 TTFT | p99 TTFT | Ratio |
|---|---|---|---|
| 8 | 45 ms | 90 ms | 2.0× |
| 16 | 52 ms | 160 ms | 3.1× |
| 32 | 68 ms | 280 ms | 4.1× |
| 64 | 95 ms | 480 ms | **5.1×** |

And the resulting utilization ceilings — `p99 ∝ 1/(1−ρ)`:

| TTFT p99 target | Max sustainable utilization |
|---|---|
| 300 ms | **≤63%** |
| 400 ms | ≤70% |
| 500 ms | ≤75% |

**This gives the pane a principled utilization target to replace an arbitrary one.** It also independently corroborates the ~60% break-even utilization figure from §9: you cannot run a latency-sensitive service above ~63–75% utilization, so the economics must close below that.

---

## 6. Acceleration techniques: realistic multipliers

### 6.1 Speculative decoding / EAGLE-3 / MTP

| Claim | Value | Source quality |
|---|---|---|
| EAGLE-3 paper, across models/tasks | **3.0–6.5×**, acceptance length 4.05–7.5 tokens | primary ([EAGLE-3 paper](https://arxiv.org/pdf/2503.01840)) |
| EAGLE-3, H100 SXM5 | 4,400 vs 1,200 tok/s = **3.7×** | aggregator |
| EAGLE-3, H200 SXM5 | 6,200 vs 1,700 tok/s = **3.6×** | aggregator |
| **EAGLE-3 on SGLang, 1× H100, by batch size** | **1.81× at B=2 → 1.38× at B=64** | the number that matters |
| MTP acceptance rate, typical | 50–70% (80–90% best case) | mixed |
| MTP acceptance, pathological | **2.4–4.6%** (Step-3.5-Flash on vLLM — "practically unusable") | primary ([vLLM issue #38339](https://github.com/vllm-project/vllm/issues/38339)) |
| Acceptance < 50% | **slows you down** | consistent across sources |

**The decisive finding: speculative decoding's benefit collapses as concurrency rises** — 1.81× at batch 2 down to 1.38× at batch 64. Speculation spends compute to buy bandwidth; once batching has already converted the workload toward compute-bound, there is nothing left to buy. A throughput-oriented calculator should therefore **default speculative decoding OFF** and present it as a *latency* lever (improves tok/s/user at low concurrency), not a *cost* lever. Recommended default if modeled: **1.4× at high concurrency, 2.0× at B ≤ 8, `medium-low` confidence.**

### 6.2 Disaggregated prefill/decode

Standard in 2026. NVIDIA Dynamo 1.0 shipped as a production release **2026-03-16** ([docs](https://docs.nvidia.com/dynamo/kubernetes/disaggregated-serving/overview), [repo](https://github.com/ai-dynamo/dynamo)); vLLM's large-scale DeepSeek-V3 result uses disaggregated P/D with Wide-EP.

| Claim | Value | Source |
|---|---|---|
| Dynamo disaggregated, DeepSeek R1 on Blackwell | **up to 7× throughput** | NVIDIA, via aggregator |
| Encode-Prefill-Decode disaggregation, multimodal | up to **5× faster TTFT**, 7× faster E2E | [NVIDIA dev blog](https://developer.nvidia.com/blog/when-to-use-encode-prefill-decode-disaggregation-to-accelerate-multimodal-model-serving/) |
| vLLM DeepSeek-V3 H200, with disagg + Wide-EP | 1,500 → **2,200 tok/s/GPU** (+47%) | [vLLM blog](https://vllm.ai/blog/2025-12-17-large-scale-serving) (primary) |
| TTFT effect | drops — "dedicated prefill nodes are never interrupted by decode work" | multiple |

**The sizing consequence is structural, not just a multiplier:** disaggregation means prefill and decode are sized *separately*, with different GPU counts, different parallelism, and potentially different GPU SKUs — prefill wants FLOPS, decode wants bandwidth (one source notes an A100 80 GB at 2 TB/s is competitive with an H100 PCIe for decode, at a lower hourly rate). A single `gpuCount` field cannot express a P/D-disaggregated deployment. **Recommendation: model disaggregation as a phase-2 toggle that splits the GPU count into `prefillGpus` + `decodeGpus`; do not bake a 7× multiplier into the base model** — the 7× is a vendor headline on a specific MoE model, and the independently reproducible vLLM figure is +47%.

---

## 7. Refreshed API pricing (October 2026)

### 7.1 Anthropic — primary source, high confidence

[platform.claude.com/docs/en/about-claude/pricing](https://platform.claude.com/docs/en/about-claude/pricing) and [models overview](https://platform.claude.com/docs/en/models/overview). Batch API = **50% off input and output**. Context window 1M on all current models.

| Model | API model ID | input | 5m cache write | 1h cache write | **cache read** | output | ctx |
|---|---|---|---|---|---|---|---|
| Claude Fable 5.1 | `claude-fable-5-1` | $10 | $12.50 | $20 | **$0.25** (0.025×) | $50 | 1M |
| Claude Opus 5.5 | `claude-opus-5-5` | **$4** | $5 | $8 | **$0.20** (0.05×) | **$20** | 1M |
| Claude Opus 5 | `claude-opus-5` | $5 | $6.25 | $10 | $0.50 | $25 | 1M |
| Claude Sonnet 5.5 | `claude-sonnet-5-5` | **$2** | $2.50 | $4 | **$0.10** (0.05×) | **$10** | 1M |
| Claude Sonnet 5 | `claude-sonnet-5` | $2 | $2.50 | $4 | $0.20 | $10 | 1M |
| Claude Haiku 5.5 (≤100K prompt) | `claude-haiku-5-5` | **$0.10** | $0.125 | $0.20 | $0.01 | **$0.50** | 1M |
| Claude Haiku 5.5 (>100K prompt) | `claude-haiku-5-5` | $0.50 | $0.625 | $1 | $0.05 | $2.50 | 1M |
| Claude Haiku 4.5 | `claude-haiku-4-5` | $1 | $1.25 | $2 | $0.10 | $5 | 200K |

Two pricing *shapes* the repo's `ApiModel` type cannot represent:

1. **Haiku 5.5 is priced by prompt length** — a request over 100,000 input tokens pays 5× on every category, priced per request. At the Long-context preset (50,000 in) it is cheap; at 150,000 it is 5× dearer. There is no `ApiModel` field for a length-tiered rate.
2. **Cache-read multipliers differ per model** (2.5% / 5% / 10%), so a global "cache reads are 10% of input" constant would be wrong for three of the current models.

Also material: **Claude 4.7-and-later models use a newer tokenizer that produces ~30% more tokens for the same text.** A pane that converts "words" or "documents" into tokens, and compares a 4.6-era model against a 5.x model at equal token counts, is comparing unequal amounts of text. Worth a warning at minimum.

### 7.2 OpenAI, Google, DeepSeek, Mistral, Cohere

**OpenAI** — primary: [developers.openai.com/api/docs/pricing](https://developers.openai.com/api/docs/pricing). Batch = **50% off** all categories (input, cached input, output).

| Model | input | cached input | output |
|---|---|---|---|
| gpt-6-astra | $10.00 | $1.00 | $50.00 |
| gpt-6.1-sol | $2.00 | $0.10 | $10.00 |
| gpt-6-sol | $2.00 | $0.20 | $10.00 |
| gpt-6-luna | $0.10 | $0.01 | $0.50 |
| gpt-5.6-sol | $4.00 | $0.40 | $20.00 |
| gpt-5.6-terra | $2.00 | $0.20 | $12.00 |
| gpt-5.6-luna | $0.20 | $0.02 | $1.20 |
| gpt-5.4 | $2.50 | $0.25 | $15.00 |
| gpt-5.4-mini | $0.75 | $0.075 | $4.50 |
| gpt-5.4-nano | $0.20 | $0.02 | $1.25 |
| gpt-5.5-pro | $30.00 | — | $180.00 |
| *(legacy, still priced)* gpt-4o | $2.50 | $1.25 | $10.00 |
| *(legacy)* gpt-4o-mini | $0.15 | $0.075 | $0.60 |
| *(legacy)* o1 | $15.00 | $7.50 | $60.00 |

**Google Gemini** — primary: [ai.google.dev/gemini-api/docs/pricing](https://ai.google.dev/gemini-api/docs/pricing). Note promotional pricing with announced step-ups, and ≤200k / >200k tiers on Pro models.

| Model | input | cached | output | ctx tier | note |
|---|---|---|---|---|---|
| gemini-4-argon | $2.00 → **$4.00 from 2026-12-01** | $0.10/$0.20 | $10.00 / $20.00 | tiered | promo expiring |
| gemini-3.8-flash | $0.75 → **$1.50 from 2027-01-01** | $0.075/$0.15 + $0.50–1.00/MTok-hr storage | $3.75 / $7.50 | tiered | promo expiring |
| gemini-3.6-flash | $0.75 / $1.50 | $0.075 / $0.15 | $3.75 / $7.50 | tiered | |
| gemini-3.1-pro-preview | $2.00 (≤200k) / $4.00 (>200k) | $0.20/$0.40 + $4.50 storage | $12.00 / $18.00 | tiered | |
| gemini-3.5-flash-lite | $0.30 | $0.03 + $1.00 storage | $2.50 | flat | |
| *(legacy)* gemini-2.5-pro | $1.25 / $2.50 | $0.125 / $0.25 | $10.00 / $15.00 | tiered | matches repo |
| *(legacy)* gemini-2.5-flash | $0.30 | $0.03 | $2.50 | flat | repo has $0.15/$0.60 — **wrong** |

Batch = 50% off. **Two shapes the repo cannot represent: announced future price increases, and ≤200k/>200k context tiers.**

**DeepSeek** — primary: [api-docs.deepseek.com/quick_start/pricing](https://api-docs.deepseek.com/quick_start/pricing). Prices are **time-of-day dependent** — off-peak is half of peak; peak is 01:00–04:00 and 06:00–10:00 UTC Mon–Fri.

| Model | cache-hit in (off/peak) | cache-miss in (off/peak) | output (off/peak) |
|---|---|---|---|
| deepseek-flash | $0.003 / $0.006 | $0.15 / $0.30 | $0.60 / $1.20 |
| deepseek-v4-pro | $0.022 / $0.044 | $0.66 / $1.32 | $1.98 / $3.96 |

**Mistral** — official pricing page confirms only Mistral Large at **$0.50 in / $1.50 out** ([mistral.ai/pricing](https://mistral.ai/pricing)). The rest below is **aggregator-sourced, medium-low confidence** ([CloudZero](https://www.cloudzero.com/blog/mistral-api-pricing/), [benchlm](https://benchlm.ai/mistral/api-pricing)); the Large row's agreement with the official page is the only cross-check available. Batch = 50% off.

| Model | input | output | confidence |
|---|---|---|---|
| Mistral Large 3 | $0.50 | $1.50 | **high** (official) |
| Mistral Medium 3.5 | $1.50 | $7.50 | medium-low |
| Mistral Small 4 | $0.15 | $0.60 | medium-low |
| Magistral Medium | $2.00 | $5.00 | medium-low |
| Devstral 2 | $0.40 | $2.00 | medium-low |
| Ministral 3 (3B/8B/14B) | $0.10–0.20 | $0.10–0.20 | medium-low |

**Cohere** — [cohere.com/pricing](https://cohere.com/pricing) **no longer publishes per-token pricing for its current models** (North, Compass, Embed 5, Rerank 4, Parse 5). Only legacy Command models carry public per-token rates (Command R 03-2024 $0.50/$1.50; Command R+ 08-2024 $2.50/$10.00); current models are quoted as Model Vault instances at $3.00–$10.00/hr or $2,000–$6,500/month, or "contact sales."
**Recommendation: drop Cohere from the pane, or list it as "contact sales / instance-priced."** Carrying a stale legacy row implies a per-token offer that no longer exists.

### 7.3 Open-model hosts

| Provider | Model | input | cached in | output | source |
|---|---|---|---|---|---|
| Together | Qwen3.8-2.4T-A95B | $2.00 | — | $6.00 | [together.ai/pricing](https://www.together.ai/pricing) |
| Together | Kimi K3 | $3.00 | — | $15.00 | same |
| Together | GLM-5.3 | $1.40 | — | $4.40 | same |
| Together | GLM-5.3-Flash | $0.15 | — | $0.50 | same |
| Together | DeepSeek V4 Pro 0813 | $1.32 | — | $3.96 | same |
| Together | DeepSeek V4 Flash 0731 | $0.14 | — | $0.28 | same |
| Together | gpt-oss-120B | $0.15 | — | $0.60 | same |
| Together | Llama 3.3 70B | $1.04 | — | $1.04 | same |
| Together | Gemma 4 31B | $0.39 | — | $0.97 | same |
| Fireworks | Kimi K3 | $3.00 | $0.30 | $15.00 | [docs.fireworks.ai](https://docs.fireworks.ai/serverless/pricing) |
| Fireworks | GLM 5.3 | $1.40 | $0.26 | $4.40 | same |
| Fireworks | DeepSeek V4.1 Flash | $0.30 | **$0.006** | $1.20 | same |
| Fireworks | gpt-oss-120B | $0.15 | $0.015 | $0.60 | same |
| Fireworks | Nemotron Lightning 30B | $0.05 | $0.01 | $0.20 | same |
| Fireworks | Qwen 3.8 Max | $2.00 | $0.25 | $6.00 | same |
| Groq | gpt-oss-120b | $0.15 | — | $0.60 | aggregator — **UNVERIFIED** |
| Cerebras | gpt-oss-120b | $0.35 | — | $0.75 | aggregator — **UNVERIFIED** |
| DeepInfra | gpt-oss-120b | $0.037 | — | $0.17 | aggregator — **UNVERIFIED** |
| Baseten | gpt-oss-120b | $0.10 | — | $0.50 | aggregator — **UNVERIFIED** |

**The four UNVERIFIED rows could not be read from vendor pricing pages** (groq.com/pricing served marketing copy; console.groq.com/docs/pricing returned 404; the others were not reachable in this pass). They come from [morphllm](https://www.morphllm.com/llm-api-pricing) and [DigitalApplied's Q2 2026 matrix](https://www.digitalapplied.com/blog/ai-inference-providers-pricing-matrix-q2-2026). **Do not ship them without re-verification.**

Fireworks also publishes **parameter-count-based generic pricing** — a genuinely different pricing model worth noting: <4B $0.10/MTok; 4–16B $0.20; >16B $0.90; MoE ≤56B $0.50; MoE 56–176B $1.20 (same rate in and out).

### 7.4 Worked example: what cache pricing is worth

Agentic preset from `docs/decisions/2026-06-24-inference-ux-presets.md` (16,000 in / 4,000 out) on `claude-sonnet-5` at the repo's default 100,000 req/day (`src/store/inference-store.ts:18`):

```
No cache modeling (what the pane does today):
  input  16,000 × $2.00/1e6  = $0.0320
  output  4,000 × $10.00/1e6 = $0.0400
  per request                = $0.0720   →  $219,150 / month (30.44 d)

With 85% of input served from cache (OpenRouter's agent figure), cache read $0.20/MTok:
  cached   13,600 × $0.20/1e6 = $0.00272
  uncached  2,400 × $2.00/1e6 = $0.00480
  output    4,000 × $10.00/1e6 = $0.04000
  per request                  = $0.0475  →  $144,639 / month
```

**A 34% overstatement of total API cost, and a 76% overstatement of input cost — $74,500/month at the pane's own default volume.** Because this error lands entirely on the API side of the comparison, it biases the pane's central recommendation *toward self-hosting*. On Opus 5.5 (cache read 0.05×) the effect is larger still.

---

## 8. Refreshed cloud GPU pricing (October 2026)

### 8.1 Verified from vendor pages

| Provider | Instance | GPU | n | on-demand $/hr | per GPU | spot/reserved | source |
|---|---|---|---|---|---|---|---|
| CoreWeave | HGX H100 | H100 SXM | 8 | **$49.24** | $6.16 | spot $19.71 | [coreweave.com/pricing](https://www.coreweave.com/pricing) |
| CoreWeave | HGX H200 | H200 | 8 | **$50.44** | $6.31 | spot $20.93 | same |
| CoreWeave | HGX B200 | B200 | 8 | **$68.80** | $8.60 | spot $34.11 | same |
| CoreWeave | HGX B300 | B300 | 8 | contact sales | — | spot $35.84 | same |
| CoreWeave | GB200 NVL72 | GB200 | 4 | **$42.00** | $10.50 | — | same |
| CoreWeave | A100 | A100 | 8 | $21.60 | $2.70 | spot $9.65 | same |
| Lambda | 8x H100 SXM | H100 SXM | 8 | **$31.92** | **$3.99** | — | [lambda.ai](https://lambda.ai/service/gpu-cloud) |
| Lambda | 8x B200 SXM6 | B200 | 8 | **$53.52** | **$6.69** | — | same |
| Lambda | 1x H100 SXM | H100 SXM | 1 | $4.29 | $4.29 | — | same |
| Lambda | 1x GH200 | GH200 | 1 | $2.29 | $2.29 | — | same |
| RunPod | H100 SXM (Secure) | H100 SXM | 1 | **$3.99** | $3.99 | community $2.69 | [runpod.io/pricing](https://www.runpod.io/pricing) |
| RunPod | H200 (Secure) | H200 | 1 | **$5.29** | $5.29 | community $3.59 | same |
| RunPod | B200 (Secure) | B200 | 1 | **$7.99** | $7.99 | community $5.98 | same |
| RunPod | B300 (Secure) | B300 | 1 | $8.99 | $8.99 | — | same |
| Together | H100 dedicated | H100 | 1 | **$5.49** | $5.49 | — | [together.ai/pricing](https://www.together.ai/pricing) |
| Together | B200 dedicated | B200 | 1 | **$8.99** | $8.99 | — | same |
| Fireworks | H100 80 GB | H100 | 1 | **$8.00** | $8.00 | — | [fireworks.ai/pricing](https://fireworks.ai/pricing) |
| Fireworks | H200 141 GB | H200 | 1 | **$8.00** | $8.00 | — | same |
| Fireworks | B200 180 GB | B200 | 1 | **$13.00** | $13.00 | — | same |
| Fireworks | B300 288 GB | B300 | 1 | $15.00 | $15.00 | — | same |

CoreWeave notes reserved capacity at **up to 60% off on-demand** — a useful default for the `reservedPerHr` field where a vendor publishes no reserved rate.

### 8.2 Aggregator-sourced — mark medium-low, re-verify before shipping

Neither AWS, GCP, nor Azure publishes GPU instance prices on the pages that describe those instances, and their calculators were not reachable in this pass.

| Provider | Instance | GPU | n | $/hr | per GPU | source / confidence |
|---|---|---|---|---|---|---|
| AWS | p5.48xlarge | H100 | 8 | **~$55.04** | ~$6.88 | [Spheron AWS H100 2026](https://www.spheron.network/blog/aws-h100-pricing-2026/) — medium-low |
| AWS | p5en.48xlarge | H200 | 8 | ~$54.92 | $6.865 | same — medium-low |
| AWS | p6-b200.48xlarge | B200 | 8 | **~$98.84** | $12.355 | [Spheron, eff. 2026-07-01](https://www.spheron.network/blog/aws-capacity-blocks-pricing-2026-b200-b300-hike/) — medium-low |
| GCP | a3-ultragpu-8g | H200 | 8 | **~$84.81** | ~$10.60 | [ThunderCompute GCP guide](https://www.thundercompute.com/blog/google-cloud-gpu-instances) — medium-low |
| GCP | a4 (B200) | B200 | 8 | ~$128.88 | ~$16.11 | same — **reservation/Spot/Flex-start only, no standard on-demand** |
| Azure | ND GB200 v6 | GB200 | — | — | **~$27** | same — low |
| Nebius | H200 | H200 | 1 | **$5.40** | $5.40 | [usagepricing, eff. 2026-10-01](https://www.usagepricing.com/blueprint/activity/nebius-ai-2026-09-24-gpu-price-increase) — medium |
| Nebius | B200 | B200 | 1 | **$8.50** | $8.50 | same — medium |
| Nebius | H100/H200 spot | — | 1 | from $0.79 | — | same — dynamic spot |
| Nebius | B200/B300 spot | — | 1 | from $0.99 | — | same |
| Crusoe | B200 | B200 | — | **contact sales** | — | no published rate |

**Pricing direction is up, not down.** AWS Capacity Blocks rose ~20% effective 2026-07-01 across P6-B300/P6-B200/P5/P5e/P5en/P4de; Nebius raised Hopper and Blackwell on-demand 17–21% effective 2026-10-01. A `lastUpdated` field alone is insufficient — the pane should show the data's age prominently and warn above some staleness threshold.

**AMD MI300X / MI355X: no published per-hour rate found at any of the surveyed providers.** Marked **UNVERIFIED / absent** rather than guessed. RunPod and CoreWeave do not list them; Fireworks does not.

### 8.3 Suggested type extension

`CloudGpuInstance` (`src/data/types.ts:18-25`) carries no hardware characteristics, so a first-principles engine cannot compute anything from a selected instance. Minimum additions:

```ts
export type CloudGpuInstance = {
  instance: string;
  gpu: string;
  gpuCount: number;
  onDemandPerHr: number;
  reservedPerHr: number | null;
  spotPerHr: number | null;
  // --- new, required by the roofline engine ---
  vramPerGpuGb: number;        // 80 | 141 | 180 | 192 | 288
  hbmBandwidthTBs: number;     // 3.35 | 4.8 | 7.7 | 5.3 | 8.0
  bf16DenseTflops: number;     // DENSE, not with-sparsity
  fp8DenseTflops: number;
  priceConfidence?: 'verified' | 'aggregator' | 'unverified';
  sourceUrl?: string;
};
```

---

## 9. Total cost of self-hosting beyond GPU rental

The current engine bills `pricePerHr × 24 × 30 × instancesNeeded` (`src/engine/inference-calculator.ts:100`): 100% uptime, exactly the GPU count the model needs, no replicas, no non-GPU cost.

| Factor | Published figure | Effect on the pane's self-host number |
|---|---|---|
| Average utilization, self-hosted inference clusters | **22%** (Spheron, Apr 2026, before batching work) | — |
| Utilization achievable with continuous batching | **>70%** | — |
| **Break-even utilization vs API pricing** | **~60%** | the pane has no utilization *input*, only an output |
| Utilization at which an AWS H100 matches a mid-tier API blended price | **~9%** | shows how load-dependent the answer is |
| Idle time on a business-hours workload | **60–70% of the month** | ~2.5–3× effective cost |
| Peak overprovisioning required | **2–4×** | ~2–4× |
| **All-in TCO vs raw GPU rental** | **3–5×** (engineering, redundancy, monitoring) | **3–5×** |
| DevOps cost | ~$145,000+/yr | fixed add |
| Model update cycle | ~$12,000 engineering per cycle | periodic add |
| Redundancy | single-GPU deployment = single failure domain: "fine for batch but not for real-time" | ≥2× for HA |

Sources: [Cerevisor (utilization)](https://cerevisor.com/blog/self-hosted-llm-cost-vs-api-utilization), [Azumo (hidden costs)](https://azumo.com/artificial-intelligence/ai-insights/self-hosting-llms-cost), [DevTk](https://devtk.ai/en/blog/self-hosting-llm-vs-api-cost-2026/), [RenéZander breakeven guide](https://renezander.com/guides/self-hosted-llm-vs-api/). All are aggregators or vendor blogs — **no primary, audited TCO study was found.** Mark the 3–5× multiplier `medium-low`. Note the multipliers partly overlap (the 3–5× figure is described as *including* redundancy and monitoring) so they must not be naively multiplied together; recommend a **single user-editable `tcoMultiplier`, default 3.0**, rather than stacked factors.

### 9.1 Published breakeven points — present as a range

| Claim | Threshold | Source |
|---|---|---|
| Self-host beats API past | **~500M tokens/month** | DevTk |
| Math "strongly favors" self-hosting | **≥10B tokens/month** | Azumo |
| Breakeven on reserved capacity, 12-mo window | **2–5M tokens/day** (= 60–150M/mo) | GPUaaS |
| Median breakeven vs mid-tier APIs, across 900 modeled scenarios | **11–12M tokens/day** (= 330–365M/mo) | GPUaaS |
| Breakeven vs frontier APIs | 2–5M tok/day min; "clearly profitable" above ~100M tok/day | Cerevisor |
| Breakeven vs budget API tiers | **15–25M tokens/day** | Cerevisor |
| Breakeven vs Gemini 3.1 Pro ($12/MTok), Mixtral 8×7B FP16 on H100 | **λ ≈ 1.6 rps** | [arXiv 2606.11690](https://arxiv.org/html/2606.11690v1) (primary) |
| Self-host cost-advantageous across all tested models | **λ ≥ 25 rps** | same (primary) |

**Recommended framing: the breakeven is a *band*, 0.5–10B tokens/month, and its position inside that band is set by (a) which API tier you compare against and (b) your achieved utilization.** The spread across these sources is 20×, which is itself the finding — a single headline breakeven number is not defensible. The pane should show a band and name the two drivers.

### 9.2 Self-hosted cost per 1M tokens — anchors

| Config | $/MTok (output) | Conditions |
|---|---|---|
| Llama 70B, H100, single-stream | **$0.45** | B=1 |
| Llama 70B, H100, B=8 | **$0.12** | |
| Llama 70B, bare vLLM @ $4.06/hr, 6.66M tok/hr | $0.61 | |
| Llama 70B @ 85% GPU utilization | **$0.10–0.13** | optimized |
| 8× H100 pod, Llama 3.1 70B, optimized | ~$1.90 | includes overhead |
| Llama 3.1 8B FP16, H100, saturation | **$0.311** | from arXiv 2606.11690 |
| Llama 3.1 8B FP16, H100, λ=1 rps | **$7.60** (24.4× penalty) | same |
| Mixtral 8×7B FP16, H100, λ=1 rps | **$15.25** | same — *above Claude Sonnet 4.6's $15 list* |

These sources disagree by 15× on the "optimized 70B on H100" figure ($0.10–0.13 vs $0.61 vs $1.90), which is exactly the symptom of quoting cost per token without stating concurrency. **Recommendation: never display a self-host $/MTok without the concurrency and SLO that produced it.**

A back-solve worth noting: the arXiv paper's saturation figure of $0.311/MTok for Llama 3.1 8B on H100 implies **~2,680 output tok/s** at an assumed $3.00/hr GPU price (`Θ = P·1e6/(3600·C)`). The repo's `self-host-throughput.json:5` says **2,800** for Llama 3 8B on H100. **The lookup table's numbers are roughly right at saturation and badly wrong everywhere else** — the table is not inaccurate so much as it answers a question nobody is asking (what if my GPU were perfectly busy?). The paper's own price assumption is not stated, so this back-solve is my own derivation: **medium-low confidence.**

---

## 10. Discrepancies with the current implementation

Unsparing, with `file:line`.

| Quantity | Repo's current value / approach | Researched current value | Impact | Recommended action |
|---|---|---|---|---|
| **TTFT** | `50 + paramBillions × 3` ms — a function of parameter count *only* (`inference-calculator.ts:108-109`) | `T_queue + 2·N_active·s_in / (G·F_peak·0.82)` — **linear in input tokens** | 70B @ 50K in on 4×H100: true ≈ **2,157 ms**, repo says **260 ms** (8.3× low). 8B @ 500 in on 1×H100: true ≈ **9.9 ms**, repo says **74 ms** (7.5× high) | **Replace.** Wrong in both directions; ignores the pane's own most-varied input |
| **Throughput** | Lookup table of 11 `outputTokensPerSec` scalars (`self-host-throughput.json:5-15`) | `Θ_out(B) = B / TPOT(B)`; measured spread **$0.21–$15.25/MTok on identical hardware** (24×) | The entire cost comparison rests on a number that varies 24× with a variable the table does not carry | **Replace with roofline + concurrency.** Keep the table as calibration fixtures only |
| **Max concurrency** | `maxConcurrentRequests` scalar, e.g. 32 for 70B on 4×H100 (`self-host-throughput.json:7`); **never read by the engine** | `B_max = (usable_VRAM − weights) / (kv_bytes_per_token · s)` → **57 @ 8K, 13 @ 32K** | 1.8× high and 2.5× low from the same row; and the field is dead code | **Compute it.** Add KV-cache sizing; the field is currently unused by `calculateSelfHostCost` |
| **KV cache** | Not modeled at all | 68.6 KiB/token (DeepSeek V3, MLA) to 840 KiB/token (Gemma 4 31B) — **300× spread** | Determines concurrency, hence cost per token, hence the whole answer | **Add.** Needs `layers`, `kvHeads`, `headDim`, `attentionType` per model |
| **Batching / concurrency** | Absent. No input, no output, no effect | Continuous batching = 2–3× throughput, 3–4× cheaper per token; `B_ridge ≈ 280` on H100 BF16 | At B=1–32 the GPU runs at 0.3–11% of its compute roof | **Add as the primary driver** |
| **Prefix / prompt caching** | Absent. `ApiModel` has no cache fields (`types.ts:1-6`); cost is `input×rate + output×rate` (`inference-calculator.ts:50-51`) | Cache reads cost **2.5–10%** of input; **>85% of agentic tokens are cached** | **Overstates API cost by 34%** on the Agentic preset ($74.5K/mo at default volume) — and the bias falls entirely on the API side, skewing the recommendation toward self-hosting | **Add `cacheReadPer1M`, `cacheWritePer1M`, and a `cacheHitRate` input** |
| **Anthropic model rows** | Claude Sonnet 4 $3/$15, Claude Opus 4 **$15/$75**, Claude Haiku 3.5 $0.80/$4 (`api-pricing.json:15-17`) | **All three retired.** Current: Opus 5.5 **$4/$20**, Sonnet 5.5 $2/$10, Haiku 5.5 from $0.10/$0.50 | Opus is priced at **3.75× its real rate** — the single worst error in the data file | **Replace with §7.1 table** |
| **Mistral row** | Mistral Large $2.00/$6.00 (`api-pricing.json:30`) | Mistral Large 3 **$0.50/$1.50** (official) | **4× too high** | Replace |
| **DeepSeek row** | DeepSeek-V3 $0.27/$1.10 (`api-pricing.json:37`) | V3 gone; `deepseek-v4-pro` $0.66–1.32 / $1.98–3.96, time-of-day dependent | **~3.6× too low**, and misses off-peak/peak entirely | Replace; add a note on time-of-day pricing |
| **Google rows** | Gemini 2.5 Pro $1.25/$10 ✓, Gemini 2.5 Flash **$0.15/$0.60** ✗ (`api-pricing.json:23-24`) | 2.5 Flash is **$0.30/$2.50**; current flagship is `gemini-4-argon` $2/$10 | Flash output is **4.2× too low**; models 2 generations stale | Replace |
| **OpenAI rows** | GPT-4o $2.50/$10, GPT-4o-mini $0.15/$0.60, o1 $15/$60 (`api-pricing.json:7-9`) | **All three still correctly priced** — but three generations stale | Prices right, model menu obsolete | Replace model list; prices need no correction |
| **AWS p5.48xlarge** | **$98.32/hr** on-demand (`cloud-gpu-pricing.json:7`) | **~$55.04/hr** | **79% too high.** Directly inflates self-host cost and pushes the breakeven up | Replace (medium-low confidence source — re-verify) |
| **GCP a3-highgpu-8g** | $98.35/hr (`cloud-gpu-pricing.json:14`) | a3-ultragpu-8g (H200) ~$84.81/hr; H100 a3 superseded | Stale SKU and stale price | Replace with a3-ultragpu / a4 |
| **Lambda 8× H100** | $27.92/hr (`cloud-gpu-pricing.json:27`) | **$31.92/hr** ($3.99 × 8) | 13% low | Replace |
| **CoreWeave H100** | $4.76/GPU/hr on-demand, $2.21 reserved (`cloud-gpu-pricing.json:33`) | **$6.16/GPU/hr** ($49.24 / 8); reserved "up to 60% off" | 29% low | Replace |
| **RunPod H100 SXM** | $3.99 on-demand, $2.49 spot (`cloud-gpu-pricing.json:39`) | **$3.99 Secure ✓**, $2.69 Community | On-demand exact; spot slightly low | Minor update |
| **Blackwell / H200 coverage** | **Absent.** Newest GPU in the file is H100 | B200, B300, GB200, GB300, H200, MI300X/MI355X all in market | Cannot model any 2026 deployment | Add §8.1 rows |
| **Flat 30-day month** | `monthlyCost = dailyCost × 30` (`:54`) and `monthlyGpuCost = … × 24 × 30` (`:100`), but `annualCost = dailyCost × 365` (`:55`) | 365/12 = **30.4375** | Monthly understated **1.46%**; `annualCost ≠ 12 × monthlyCost`. The repo's own test enshrines the inconsistency (`inference-calculator.test.ts:59` asserts the 365/30 ratio) | Use 30.4375 (or 365/12) everywhere; fix the test |
| **100% uptime, no redundancy, no non-GPU cost** | `pricePerHr × 24 × 30 × instancesNeeded` (`:100`) | All-in TCO **3–5× raw rental**; ~60% utilization needed to break even; 2–4× peak overprovisioning | Self-host cost understated **3–5×**; breakeven volume understated **3–5×** | Add `tcoMultiplier` (default 3.0), `replicas`, `peakToAverageRatio` |
| **Breakeven curve treats self-host cost as flat** | `generateBreakevenCurve` takes a scalar `selfHostMonthlyCost` and emits it unchanged at every volume (`:180-206`) | Self-host cost is a **staircase** — exceeding capacity requires another replica | At the default 100K req/day with the Agentic preset, required output is **4,630 tok/s** vs the best table entry's 1,200 — `canServe` is false, yet the chart still plots a flat self-host line. **Correctness bug, not just imprecision** | Recompute replica count per volume point |
| **`costPerRequest` divides by `requestsPerDay`** | `:56`, with no zero guard; the curve's step list includes `0` (`:189`) | — | `costPerRequest` = **NaN** at the first curve point | Guard the divisor |
| **Utilization warning threshold** | Warns below **10%** (`:117-121`) | Break-even utilization is **~60%**; SLO-sustainable ceiling is **63–75%** | Warns only in the catastrophic case; silent across the entire 10–60% range where self-hosting still loses | Raise to 60%, and add an upper warning above 70% (SLO risk) |
| **Speculative decoding / disaggregation / quantization effects** | Quantization is a free-text label on `SelfHostEntry` with no computed effect; the others absent | FP8 = +31% dense / +69–74% MoE throughput; spec-dec 1.38–1.81×; disagg +47% measured | Cannot model any 2026 serving optimization | Add as explicit, defaulted, editable multipliers |
| **`lastUpdated` dates** | **`2026-06-01`** in all three data files (`api-pricing.json:2`, `cloud-gpu-pricing.json:2`, `self-host-throughput.json:2`) | 4.5 months stale; AWS +20% (Jul), Nebius +17–21% (Oct), full Anthropic/OpenAI/Gemini lineup turnover | Every price in the pane predates two rounds of increases and a model generation | Refresh; surface the age in the UI and warn past ~60 days |
| **Tokenizer differences** | Not modeled | Claude 4.7+ produces **~30% more tokens for the same text** than 4.6-era models | Comparing a 4.6-era model to a 5.x model at equal token counts compares unequal text | Add a warning; consider a per-model `tokenizerFactor` |
| **No length-tiered or time-of-day pricing** | `ApiModel` is a flat pair of rates | Haiku 5.5: 5× above 100K prompt. Gemini Pro: 2× above 200K. DeepSeek: 2× at peak hours. Gemini promos step up Dec 2026 / Jan 2027 | Three current providers use pricing shapes the type cannot express | Make rates optionally tiered: `inputPer1M: number \| Tier[]` |

---

## 11. Proposed first-principles engine shape

Replaces the lookup table; keeps the existing adapter/trace/confidence conventions (`src/engine/adapters/*`, `FormulaTraceDrawer`, `AssumptionChips`, `ConfidenceBadge`).

```ts
// --- new: per-model architecture, required for KV-cache sizing ---
export type ServedModel = {
  id: string;
  name: string;
  paramsTotalB: number;        // memory follows total
  paramsActiveB: number;       // compute follows active (MoE)
  layers: number;
  kvHeads: number;             // GQA/MQA group count
  headDim: number;
  attentionType: 'mha' | 'gqa' | 'mqa' | 'mla' | 'hybrid-swa';
  kvLoraRank?: number;         // MLA only
  qkRopeHeadDim?: number;      // MLA only
  kvBytesPerTokenOverride?: number;  // for hybrid/SWA, where the formula doesn't close
  contextWindow: number;
  license: string;
};

export type ServingConfig = {
  weightPrecision: 'bf16' | 'fp8' | 'int4';
  kvPrecision: 'bf16' | 'fp8' | 'nvfp4';
  gpuMemoryUtilization: number;      // default 0.90 (vLLM)
  framework: 'vllm' | 'sglang' | 'trtllm';
  continuousBatching: boolean;       // default true
  prefixCaching: boolean;            // default true
  cacheHitRate: number;              // default 0.0; 0.85 for agentic
  speculativeDecoding: boolean;      // default FALSE — see §6.1
  disaggregatedPd: boolean;          // default false (phase 2)
};

export type SelfHostScenario = {
  model: ServedModel;
  config: ServingConfig;
  instance: CloudGpuInstance;        // now carries BW + dense TFLOPS
  gpuCount: number;                  // tensor-parallel group
  replicas: number;                  // default 2 for HA
  // workload
  requestsPerDay: number;
  avgInputTokens: number;
  avgOutputTokens: number;
  peakToAverageRatio: number;        // default 2.5
  // SLO
  ttftP99TargetMs: number;           // default 300 (interactive chat)
  itlP99TargetMs: number;            // default 50
  // economics
  pricingTier: 'onDemand' | 'reserved' | 'spot';
  tcoMultiplier: number;             // default 3.0 — non-GPU TCO, §9
};

export type SelfHostResult = {
  // --- capacity, from first principles ---
  kvBytesPerToken: number;
  kvBytesPerSequence: number;
  weightsGb: number;
  kvPoolGb: number;
  concurrencyKvLimit: number;        // B_max  (§2.3)
  concurrencySloLimit: number;       // B_slo  (§5)
  concurrencyEffective: number;      // min of the two — the binding constraint
  bindingConstraint: 'kv-cache' | 'slo' | 'demand';
  ridgeBatchSize: number;            // B_ridge (§1.4)
  regime: 'memory-bound' | 'compute-bound';

  // --- latency, from the roofline ---
  prefillFlops: number;
  ttftMs: number;                    // T_queue + prefill (§1.2)
  tpotMs: number;
  outputTokensPerSecPerReplica: number;
  meetsSlo: boolean;

  // --- demand vs capacity ---
  requiredOutputTokensPerSec: number;
  requiredPeakOutputTokensPerSec: number;
  replicasRequired: number;          // ceil(peak demand / per-replica capacity)
  utilizationPercent: number;

  // --- cost ---
  gpuCostPerMonth: number;           // 30.4375-day month
  allInCostPerMonth: number;         // × tcoMultiplier
  costPer1MOutputTokens: number;
  costPer1MBlendedTokens: number;

  trace: string[];
  warnings: string[];
  confidence: ConfidenceLevel;
};
```

API side, extending `ApiModel` (`src/data/types.ts:1-6`):

```ts
export type PriceTier = { maxPromptTokens: number | null; rate: number };

export type ApiModel = {
  model: string;
  modelId?: string;                      // exact API id, e.g. 'claude-opus-5-5'
  inputPer1M: number | PriceTier[];      // tiered: Haiku 5.5, Gemini Pro
  outputPer1M: number | PriceTier[];
  cacheReadPer1M?: number;               // §4.3
  cacheWritePer1M?: number;
  cacheStoragePer1MPerHr?: number;       // Google only
  batchDiscount?: number;                // 0.5 almost universally
  contextWindow: number;
  tokenizerFactor?: number;              // 1.3 for Claude 4.7+, §7.1
  effectiveFrom?: string;                // announced step-ups (Gemini)
  sourceUrl: string;                     // one per row — non-negotiable
};
```

### 11.1 Warnings worth emitting

- Utilization **< 60%** → "self-hosting loses to API pricing below ~60% utilization" (replaces the 10% threshold)
- Utilization **> 70%** → "p99 TTFT grows as 1/(1−ρ); this utilization will miss a 300 ms SLO"
- `bindingConstraint === 'kv-cache'` → "concurrency is capped by KV-cache capacity at N sequences; FP8 KV would roughly double it"
- `concurrencyEffective < 32` and `regime === 'memory-bound'` → "running at X% of the compute roof; `B_ridge` is ~280 on this GPU"
- MoE model → "memory follows total params, compute and FLOPs follow active params"
- MoE + FP8 → "FP8 gains 69–74% on MoE vs 31% on dense — worth enabling"
- `avgInputTokens > 100_000` and provider is Anthropic Haiku 5.5 → "prompts over 100K tokens are billed at 5× on this model"
- `cacheHitRate === 0` and preset is Agentic or RAG → "production agentic workloads report >85% cached input tokens; modeling 0% overstates API cost by ~34%"
- `speculativeDecoding && concurrencyEffective > 32` → "speculative decoding's gain falls to ~1.38× at batch 64; it is a latency lever, not a cost lever"
- `replicas === 1` → "single replica is a single failure domain — fine for batch, not for real-time"
- Data age > 60 days → "GPU and API pricing moved 17–21% in single revisions during 2026; re-verify before deciding"
- Comparing a Claude 4.6-era model against 4.7+ → "newer tokenizer produces ~30% more tokens for the same text"

### 11.2 Confidence ratings (reusing `ConfidenceLevel`, `src/engine/types.ts:54`)

| Quantity | Confidence | Why |
|---|---|---|
| KV cache bytes/token (catalog model, GQA/MHA/MLA) | **high** | exact arithmetic; validated against the 10.7 GB @32K anchor |
| KV cache bytes/token (hybrid SWA, e.g. Gemma 4) | **medium-low** | formula doesn't close; needs per-model override |
| Prefill FLOPs / TTFT compute term | **high** | `2·N·s` is exact; η=0.82 from measurement |
| Decode TPOT at batch 1 | **high** | bandwidth arithmetic; caught a physically impossible published claim |
| `B_max` (KV-cache-limited concurrency) | **high** | exact arithmetic given VRAM and `gpu_memory_utilization` |
| `B_ridge` on H100 | **high** | 280 @ BF16, published and derivable |
| `B_ridge` on B200 / MI300X / MI355X | **medium-low** | derived from aggregator TFLOPS that may be sparse figures |
| Aggregate `Θ_out(B)` in the memory-bound region | **medium** | linear-in-B is a good approximation, not exact |
| `Θ_out(B)` near and above the ridge | **medium-low** | comm overhead and kernel efficiency dominate |
| TTFT queueing term | **medium-low** | p99/p50 table from one aggregator |
| Anthropic / OpenAI / Gemini / DeepSeek pricing | **high** | vendor pricing pages, read this session |
| Mistral pricing (beyond Large 3) | **medium-low** | aggregators; one row cross-checked |
| Groq / Cerebras / DeepInfra / Baseten pricing | **low — UNVERIFIED** | vendor pages not reachable |
| Cohere per-token pricing | **n/a** | no longer published for current models |
| CoreWeave / Lambda / RunPod / Together / Fireworks GPU pricing | **high** | vendor pricing pages |
| AWS / GCP / Azure GPU pricing | **medium-low** | aggregators; no vendor page publishes it |
| MI300X / MI355X cloud pricing | **low — absent** | no published rate found |
| FP8 throughput gain (+31% dense / +69–74% MoE) | **medium** | one primary paper |
| Prefix-cache hit rates | **medium** for the workload dependence, **low** for the framework gap | see §4.1 caveat |
| Speculative decoding speedup | **medium-low** | 1.38–6.5× range; batch-size dependence is the reliable part |
| Disaggregated P/D gain | **medium-low** | +47% reproducible; 7× is a vendor headline |
| 3–5× non-GPU TCO multiplier | **medium-low** | no primary audited study found |
| Breakeven volume | **low** | published thresholds span 20× |

---

## 12. Regression-target table

Published tuples usable directly as test fixtures. Each row names its source and whether it is primary.

| # | Model | Hardware | Quant | Framework / config | Concurrency | Output tok/s | TTFT | Source | Primary? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | DeepSeek-V3 | multi-node H200 | — | vLLM, Wide-EP + disagg P/D | production | **2,200 /GPU** | — | [vLLM blog](https://vllm.ai/blog/2025-12-17-large-scale-serving) | **yes** |
| 2 | DeepSeek-V3 | multi-node H200 | — | vLLM, pre-optimization | production | ~1,500 /GPU | — | same | **yes** |
| 3 | Llama 3.1 8B | 1× H100 | FP16 | vLLM, saturation | saturated | ~2,680 /GPU (back-solved from $0.311/MTok @ $3.00/hr) | — | [arXiv 2606.11690](https://arxiv.org/html/2606.11690v1) | **yes** (throughput is my derivation) |
| 4 | Llama 3.1 8B | 1× H100 | FP16 | vLLM, λ=10 rps | — | — | **84.5 ms p99** | same | **yes** |
| 5 | Llama 3.1 8B | 1× H100 | FP16 | vLLM, λ=1 rps | — | — | 54.3 ms p99 | same | **yes** |
| 6 | Llama 3.1 8B | 1× H100 | FP16 | vLLM, λ=50 rps | — | — | 994.8 ms p99 | same | **yes** |
| 7 | Llama 3.1 8B | 1× H100 | FP16 | vLLM, TTFT p99≤300/TPOT p99≤50 | λ_max=25 rps | — | ≤300 ms | same | **yes** |
| 8 | Mixtral 8×7B | 1× H100 | FP16 | vLLM, same SLO | λ_max=10 rps | — | ≤300 ms | same | **yes** |
| 9 | 70B class | 1× H100 | BF16 | roofline, decode B=1 | 1 | 3.4 TFLOPS (0.3% of roof) | — | [Spheron roofline](https://www.spheron.network/blog/the-roofline-model-for-llm-inference-are-you-compute-bound-o/) | no |
| 10 | 70B class | 1× H100 | BF16 | roofline, decode B=32 | 32 | 107 TFLOPS (11%) | — | same | no |
| 11 | 70B class | 1× H100 | BF16 | roofline, decode B=280 | 280 | 938 TFLOPS (95%) | — | same | no |
| 12 | 70B class | 1× H100 | BF16 | roofline, prefill s=2048 | — | 989.5 TFLOPS (100%) | — | same | no |
| 13 | Llama 3.1 70B | 1× H100 | FP16 | naive allocation | 4 | 400 | — | [Spheron KV guide](https://www.spheron.network/blog/kv-cache-optimization-guide/) | no |
| 14 | Llama 3.1 70B | 1× H100 | FP16 | + PagedAttention | 8 | 900 | — | same | no |
| 15 | Llama 3.1 70B | 1× H100 | FP16 + FP8 KV | + FP8 KV | 16 | 1,800 | — | same | no |
| 16 | Llama 3.1 70B | 1× H100 | FP16 + NVFP4 KV | + NVFP4 KV | 32 | 2,600 | — | same | no |
| 17 | Llama 3.3 70B | 8× H200 SXM5 | — | continuous batching | 256 | 3,600 total / 450 per GPU | — | aggregator | no |
| 18 | Qwen 3.5 27B | 96× B200 (GKE) | — | — | — | **1,103,941 total** / ~11,500 per GPU | — | [Google Cloud](https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592) | semi |
| 19 | Qwen2.5-7B | 1× B200 | — | vLLM | per-user | 272 tok/s/user | — | aggregator | no |
| 20 | Qwen2.5-7B | 1× H200 | — | vLLM | per-user | 195 tok/s/user | — | aggregator | no |
| 21 | — (EAGLE-3) | 1× H100 SXM5 | — | spec-dec vs baseline | — | 4,400 vs 1,200 (3.7×) | — | aggregator | no |
| 22 | — (EAGLE-3) | 1× H100 | — | SGLang, B=2 | 2 | 1.81× baseline | — | aggregator | no |
| 23 | — (EAGLE-3) | 1× H100 | — | SGLang, B=64 | 64 | **1.38× baseline** | — | aggregator | no |

**KV-cache capacity fixtures** (exact arithmetic, suitable as `toBeCloseTo` assertions):

| Model | GPU config | ctx | Expected `kvBytesPerSequence` | Expected `B_max` |
|---|---|---|---|---|
| Llama 3.1 70B | 4× H100, BF16 wts + KV, util 0.90 | 32,768 | **10.74 GB** | **13** |
| Llama 3.1 70B | 4× H100, BF16 wts + KV, util 0.90 | 8,192 | 2.68 GB | **55** |
| Llama 3.1 8B | 1× H100, BF16 wts + KV, util 0.90 | 8,192 | 1.07 GB | **52** |
| DeepSeek V3 | 8× H200, FP8 wts, BF16 KV, util 0.90 | 8,192 | 0.575 GB | **598** |

**TTFT fixtures** (from `2·N_active·s / (G · 989.5e12 · 0.82)`):

| Model | GPU config | s_in | Expected TTFT (compute only) | Repo's current output |
|---|---|---|---|---|
| Llama 3 70B | 4× H100 | 50,000 | **2,157 ms** | 260 ms |
| Llama 3 70B | 4× H100 | 4,000 | **173 ms** | 260 ms |
| Llama 3 8B | 1× H100 | 500 | **9.9 ms** | 74 ms |
| Llama 3 8B | 1× H100 | 8,000 | 158 ms | 74 ms |

---

## 13. Open design questions

1. **Does the lookup table survive at all?** Recommendation: keep `self-host-throughput.json` but reclassify it — rename to `self-host-benchmarks.json`, add `concurrency`, `inputTokens`, `outputTokens`, `sloMet`, `sourceUrl` to every row, and use it as **calibration fixtures for the roofline engine**, never as the engine's input. The current rows have no concurrency annotation and so cannot be used as fixtures as-is.
2. **Is concurrency a user input or a derived output?** The arXiv paper argues the user's real inputs are arrival rate `λ` and SLO `L`, with concurrency and utilization falling out. That is the more honest model and matches how people actually procure. But the pane already collects `requestsPerDay`, which gives `λ` for free. **Recommendation: derive concurrency from `λ` and the SLO; show it prominently as the headline derived quantity, and let power users override.**
3. **Hero output: breakeven volume, or a cost-per-MTok-vs-concurrency curve?** The 24× cost spread argues the curve is the more truthful hero, with the breakeven as a band annotated on it. A single breakeven number is the thing most likely to be quoted out of context — and the published thresholds span 20×.
4. **Do we model disaggregated P/D in phase 1?** It is standard in 2026 and is where the best published numbers come from, but it splits `gpuCount` into two independently-sized pools and changes the type shape. **Recommendation: phase 2**, with a visible "this model assumes colocated prefill+decode" assumption chip in phase 1.
5. **How do we handle pricing shapes the flat type can't hold** (Haiku 5.5's 100K threshold, Gemini's 200K tier and announced step-ups, DeepSeek's peak/off-peak)? A tiered-rate union type covers two of the three; time-of-day needs a separate concept. **Recommendation: implement tiers, and handle DeepSeek with a `pricingNote` string plus a peak/off-peak toggle.**
6. **Should the pane show a "both" option?** Every breakeven source notes that production deployments route by request: cheap self-hosted model for bulk, API for hard requests. A blended-routing mode may be more useful than a binary.
7. **What is the staleness policy?** Prices moved 17–21% in single revisions during 2026. Options: a `scripts/` fetcher (already proposed in the 2026-06-23 decisions doc, never built), a CI check that fails past N days, or a prominent in-UI age badge. **Recommendation: all three, and make `sourceUrl` a required field on every priced row so verification is a click.**
8. **Does the pane need the `ServedModel` architecture catalog, or can it borrow the fine-tuning tab's?** The fine-tuning research (`docs/research/2026-10-14-finetuning-calculator-research.md` §5) asks for almost exactly the same fields (`layers`, `hidden`, `kvHeads`, `ffnDim`, `vocab`, `isMoe`). **Recommendation: one shared catalog, built once, consumed by both panes.** This is the strongest argument for doing the two panes' data work together.

---

## Sources

**Roofline, prefill/decode, and arithmetic intensity**
- [LLM Inference Unveiled: Survey and Roofline Model Insights](https://arxiv.org/html/2402.16363v4) — the canonical roofline treatment
- [The Roofline Model for LLM Inference (Spheron)](https://www.spheron.network/blog/the-roofline-model-for-llm-inference-are-you-compute-bound-o/) — H100 ridge point 295 FLOPs/byte, B_ridge≈280, the four operating points, correct **dense** TFLOPS
- [Prefill vs Decode: LLM Inference Phases Explained (Redis)](https://redis.io/blog/prefill-vs-decode/) — 80–84% of peak prefill, 60% of peak HBM decode
- [GPU Memory Bandwidth vs Compute for LLM Inference (NeevCloud)](https://neevcloud.com/blogs/gpu-memory-bandwidth-vs-compute-what-actually-limits-llm-inf)
- [AI Hardware Accelerators for LLMs: Architectures and the Memory Wall](https://arxiv.org/pdf/2608.28048)
- [SPAD: Specialized Prefill and Decode Hardware for Disaggregated LLM Inference](https://arxiv.org/pdf/2510.08544)
- [vllm-inference-profiling: roofline analysis of vLLM on A100](https://github.com/YuvrajSinghBhadoria2/vllm-inference-profiling)

**Cost-per-token as a function of load (the central source)**
- [Beyond Per-Token Pricing: A Concurrency-Aware Methodology for LLM Infrastructure Cost Estimation](https://arxiv.org/html/2606.11690v1) — the $0.21–$15.25/MTok spread, `C_eff` formula, utilization-as-output argument, SLO premium, FP8 dense-vs-MoE gains, API crossover points. **The most decision-relevant source in this document.**

**KV cache**
- [KV Cache Calculations (Sebastian Raschka)](https://sebastianraschka.com/llm-architecture-gallery/kv-cache-calculations/) — per-token figures for current models incl. MLA and hybrid attention
- [KV Cache Optimization Guide (Spheron)](https://www.spheron.network/blog/kv-cache-optimization-guide/) — concurrency formula, FP8/NVFP4 savings, the 400→2,600 tok/s staged table, vLLM defaults
- [Multi-Head Latent Attention on GPU Cloud (Spheron)](https://www.spheron.network/blog/multi-head-latent-attention-mla-gpu-cloud/) — ~98% KV reduction claim
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE Language Model](https://arxiv.org/pdf/2405.04434) — MLA formulation
- [Insights into DeepSeek-V3: Hardware for AI Architectures](https://arxiv.org/pdf/2505.09343) — 70 KB/token vs Llama 405B's 516 KB
- [Accelerate inference with KV cache tiering on AWS](https://aws.amazon.com/blogs/storage/accelerate-inference-with-kv-cache-tiering-on-aws/) — Llama 3.1 405B at ~516 KB/token
- [Predictive Multi-Tier Memory Management for KV Cache](https://arxiv.org/pdf/2604.26968)

**Throughput, batching, goodput**
- [vLLM Large Scale Serving: DeepSeek @ 2.2k tok/s/H200 with Wide-EP](https://vllm.ai/blog/2025-12-17-large-scale-serving) — primary; 1,500→2,200 tok/s/GPU, disaggregated P/D
- [1 Million Tokens Per Second: Qwen 3.5 27B on GKE with B200 (Google Cloud)](https://medium.com/google-cloud/1-million-tokens-per-second-qwen-3-5-27b-on-gke-with-b200-gpus-161da5c1b592)
- [LLM Inference Benchmarking: Performance Tuning with TensorRT-LLM (NVIDIA)](https://developer.nvidia.com/blog/llm-inference-benchmarking-performance-tuning-with-tensorrt-llm/) — throughput/responsiveness Pareto frontier
- [LLM Serving Goodput Under Latency SLOs (Cerebrium)](https://cerebrium.ai/resources/llm-serving-goodput-latency-slos)
- [GPU Cost Per Token: 7 Major LLMs Across GPU Types (Spheron)](https://www.spheron.network/blog/gpu-cost-per-token-benchmark-llm-inference-2026/) — aggregator
- [Best GPU for LLM inference: what the benchmarks show in 2026 (RunPod)](https://www.runpod.io/articles/guides/best-gpu-for-llm-inference) — aggregator
- [vLLM vs TensorRT-LLM vs SGLang H100 benchmarks (Spheron)](https://www.spheron.network/blog/vllm-vs-tensorrt-llm-vs-sglang-benchmarks/) — aggregator
- [LLM Inference Benchmarks: Seven Stopwatches, One Server](https://www.thesoftwarefrontier.com/p/llm-inference-benchmarks-seven-stopwatches) — source of the 223 tok/s claim the roofline rejects (§1.5)

**Prefix / prompt caching**
- [OpenRouter Data](https://openrouter.ai/data) and [State of AI: An Empirical 100 Trillion Token Study with OpenRouter](https://arxiv.org/html/2601.10088v1)
- [Agents use 5x more tokens than humans as cached prompts explode (Tom's Hardware / Futurum)](https://www.tomshardware.com/tech-industry/artificial-intelligence/futurum-ceo-says-agents-use-ai-5x-more-than-humans-number-will-eventually-hit-10x-but-agents-are-mostly-rereading-what-theyve-already-seen) — the >85% cached-agent-tokens figure
- [Agentic Tokens Jumped 14x: What OpenRouter's Data Shows](https://stack-archive.com/blog/agentic-ai-self-consumption-tokens-14x-openrouter-2026/)
- [Prefix caching at scale: 80% savings, and the eviction policies that turn it into 5%](https://dev.to/tech_nuggets/prefix-caching-at-scale-when-it-saves-you-80-of-prefill-cost-and-the-eviction-policies-that-5e8)
- [SGLang Production Deployment Guide: RadixAttention (Spheron)](https://www.spheron.network/blog/sglang-production-deployment-guide/) — aggregator
- [vLLM vs SGLang 2026: RadixAttention vs PagedAttention (Spheron)](https://www.spheron.network/blog/vllm-vs-sglang-2026/) — aggregator; source of the uncorroborated framework gap
- [Context Engineering for Production AI Agents (Spheron)](https://www.spheron.network/blog/context-engineering-production-ai-agents-kv-cache-long-context/)

**Speculative decoding**
- [EAGLE-3: Scaling up Inference Acceleration via Training-Time Test](https://arxiv.org/pdf/2503.01840) — primary; 3.0–6.5×, acceptance length 4.05–7.5
- [vLLM issue #38339: Step-3.5-Flash MTP acceptance rate 2.4–4.6%](https://github.com/vllm-project/vllm/issues/38339) — primary; the failure mode
- [From research to production: EAGLE-3 on Vertex (LMSYS)](https://www.lmsys.org/blog/2025-12-01-eagle3-vertex/)
- [2x Faster on a 229B MoE: EAGLE3 for MiniMax-M2.5 (Hugging Face)](https://huggingface.co/blog/lujangusface/tw-eagle3-minimax)
- [The Limits of Speculation: Bounding Speculative Decoding in MoE](https://arxiv.org/pdf/2609.22156)
- [Speculative Decoding Production Guide (Spheron)](https://www.spheron.network/blog/speculative-decoding-production-guide/) — aggregator; the batch-size dependence
- [Performance improvements with speculative decoding in vLLM for gpt-oss (Red Hat)](https://developers.redhat.com/articles/2026/04/16/performance-improvements-speculative-decoding-vllm-gpt-oss)

**Disaggregated prefill/decode**
- [Disaggregated Serving (NVIDIA Dynamo docs)](https://docs.nvidia.com/dynamo/kubernetes/disaggregated-serving/overview)
- [ai-dynamo/dynamo](https://github.com/ai-dynamo/dynamo)
- [When to Use Encode-Prefill-Decode Disaggregation (NVIDIA)](https://developer.nvidia.com/blog/when-to-use-encode-prefill-decode-disaggregation-to-accelerate-multimodal-model-serving/)
- [Prefill/decode disaggregation reaches production: state of play, May 2026](https://lecompute.fr/en/runtimes/disaggregation-prefill-decode-production/)
- [Disaggregated Prefill/Decode LLM Serving Guide (Cerebrium)](https://cerebrium.ai/resources/disaggregated-prefill-decode-llm-serving)
- [Scaling multi-node LLM inference with NVIDIA Dynamo on AKS (Microsoft)](https://blog.aks.azure.com/2025/10/24/dynamo-on-aks)
- [NVIDIA Dynamo 1.0 Deployment Guide (Spheron)](https://www.spheron.network/blog/nvidia-dynamo-disaggregated-inference-guide/) — aggregator

**Latency SLOs**
- [LLM Inference SLO Engineering: TTFT, ITL, P99 Latency Budgets (Spheron)](https://www.spheron.network/blog/llm-inference-slo-ttft-itl-latency-budget-guide-2026/) — the per-use-case SLO table, p99/p50 ratios, utilization ceilings
- [Key metrics for LLM inference (BentoML)](https://bentoml.com/llm/llm-inference-basics/llm-inference-metrics)
- [TTFT vs Tokens Per Second (GMI Cloud)](https://www.gmicloud.ai/en/blog/ttft-llm-speed-metrics)
- [AI Model Latency Benchmarks 2026 (DigitalApplied)](https://www.digitalapplied.com/blog/ai-model-latency-benchmarks-2026-ttft-throughput)
- [Memory Offloading for LLM Inference with Latency SLO Guarantees](https://arxiv.org/pdf/2502.08182)
- [FairBatching: Fairness-Aware Batch Formation for LLM Inference](https://arxiv.org/pdf/2510.14392)

**API pricing — primary (vendor pages, read 2026-10-14)**
- [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) — full model table, cache write/read, batch, long-context and data-residency multipliers
- [Anthropic models overview](https://platform.claude.com/docs/en/models/overview) — exact model IDs, context windows, tokenizer note
- [OpenAI API pricing](https://developers.openai.com/api/docs/pricing)
- [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing)
- [DeepSeek pricing](https://api-docs.deepseek.com/quick_start/pricing)
- [Mistral pricing](https://mistral.ai/pricing) (Large 3 only) and [Mistral models overview](https://docs.mistral.ai/getting-started/models/models_overview/)
- [Cohere pricing](https://cohere.com/pricing) — per-token rates no longer published for current models
- [Together AI pricing](https://www.together.ai/pricing)
- [Fireworks serverless pricing](https://docs.fireworks.ai/serverless/pricing) and [Fireworks pricing](https://fireworks.ai/pricing)

**API pricing — aggregators (used only where no vendor page was reachable)**
- [Mistral API Pricing in 2026 (CloudZero)](https://www.cloudzero.com/blog/mistral-api-pricing/)
- [Mistral API Pricing October 2026 (benchlm)](https://benchlm.ai/mistral/api-pricing)
- [LLM API Pricing 2026 (morphllm)](https://www.morphllm.com/llm-api-pricing) — Groq/Cerebras/DeepInfra/Baseten rows, **UNVERIFIED**
- [AI Inference Providers Pricing Matrix Q2 2026 (DigitalApplied)](https://www.digitalapplied.com/blog/ai-inference-providers-pricing-matrix-q2-2026) — **UNVERIFIED**
- [Best LLM inference providers 2026 (CloudZero)](https://www.cloudzero.com/blog/llm-inference-providers/)

**Cloud GPU pricing — primary (vendor pages, read 2026-10-14)**
- [CoreWeave pricing](https://www.coreweave.com/pricing)
- [Lambda GPU Cloud](https://lambda.ai/service/gpu-cloud)
- [RunPod pricing](https://www.runpod.io/pricing)
- [Together AI pricing](https://www.together.ai/pricing) (dedicated GPU section)
- [Fireworks pricing](https://fireworks.ai/pricing) (on-demand GPU section)
- [Amazon EC2 P5 instance types](https://aws.amazon.com/ec2/instance-types/p5/) — specs only; **AWS does not publish prices here**

**Cloud GPU pricing — aggregators (AWS/GCP/Azure/Nebius; no vendor page available)**
- [AWS H100 Pricing 2026 (Spheron)](https://www.spheron.network/blog/aws-h100-pricing-2026/)
- [AWS Capacity Blocks Pricing 2026: B200/B300 Price Hike (Spheron)](https://www.spheron.network/blog/aws-capacity-blocks-pricing-2026-b200-b300-hike/)
- [Google Cloud GPU Instances: Every Machine Type, Spec, and Price (ThunderCompute)](https://www.thundercompute.com/blog/google-cloud-gpu-instances)
- [Nebius GPU price increase, eff. 2026-10-01 (UsagePricing)](https://www.usagepricing.com/blueprint/activity/nebius-ai-2026-09-24-gpu-price-increase)
- [GPU Cloud Pricing Comparison 2026 (Spheron)](https://www.spheron.network/blog/gpu-cloud-pricing-comparison-2026/)
- [Best GPU Neoclouds 2026 (MarkTechPost)](https://www.marktechpost.com/2026/08/23/best-gpu-neoclouds-2026/)
- [Data Center GPU Pricing 2026 (IntuitionLabs)](https://intuitionlabs.ai/articles/data-center-gpu-pricing-2026)

**GPU hardware specs**
- [NVIDIA H100 Specs (Spheron)](https://www.spheron.network/blog/nvidia-h100-specs/) — **quotes with-sparsity TFLOPS; see §1.3 warning**
- [H100 vs H200 vs B200 vs B300 buying guide (OpenMetal)](https://openmetal.io/resources/blog/h100-vs-h200-vs-b200-vs-b300-ai-gpu-server/)
- [AMD MI300X vs NVIDIA H200 (Spheron)](https://www.spheron.network/blog/amd-mi300x-vs-nvidia-h200/)
- [H200 vs MI355X specs and live rental prices](https://gpuperhour.com/compare/h200-vs-mi355x)
- [Comparing NVIDIA's B200 and H100 (Civo)](https://www.civo.com/blog/comparing-nvidia-b200-and-h100)

**Self-host TCO and breakeven**
- [Self-hosted LLM cost vs the API: the utilization number your board never sees (Cerevisor)](https://cerevisor.com/blog/self-hosted-llm-cost-vs-api-utilization) — 22% baseline utilization, ~60% break-even utilization, 3–5× TCO, continuous-batching multipliers, $1.90/MTok 8×H100 anchor
- [Self-Hosting LLMs: Hidden Costs You're Missing (Azumo)](https://azumo.com/artificial-intelligence/ai-insights/self-hosting-llms-cost) — 2–4× overprovisioning, 60–70% idle, DevOps $145K/yr
- [Self-Host LLM vs API: Real Cost Breakdown 2026 (DevTk)](https://devtk.ai/en/blog/self-hosting-llm-vs-api-cost-2026/) — ~500M tok/mo threshold
- [Self-Hosted LLM vs API: Break-Even Analysis (RenéZander)](https://renezander.com/guides/self-hosted-llm-vs-api/)
- [LLM Inference Cost on H100 in 2026 (GPUaaS)](https://gpuaas.com/blog/h100-llm-inference-cost-2026) — 2–5M tok/day and 11–12M tok/day medians
- [LLM Inference Cost 2026: Cost per Million Tokens (packet.ai)](https://packet.ai/blog/llm-inference-cost)
- [Self-Hosting LLMs in 2026: When Open-Weight Models Pay (DigyAI)](https://digyai.com/insights/self-hosting-llms/)
- [Is Self-Hosting LLMs Cheaper Than APIs in 2026? (GigaGPU)](https://gigagpu.com/is-self-hosting-llms-cheaper-than-apis/)

**Workload characterization (carried forward from the UX presets decision doc)**
- [A Systematic Characterization of LLM Inference on GPUs](https://arxiv.org/pdf/2512.01644) — Microsoft Azure production traces
- [OpenRouter State of AI](https://openrouter.ai/state-of-ai)
- [DeepSeek V4 Is Earning Agentic Token Share (OpenRouter)](https://openrouter.ai/blog/insights/deepseek-v4-adoption/)
