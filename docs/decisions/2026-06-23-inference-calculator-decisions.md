# Inference Calculator — Design Decisions

**Date:** 2026-06-23
**Context:** Adding an inference build-vs-buy calculator as a top-level mode alongside the existing training GPU calculator.

---

## UI Placement
**Decision:** Top-level tab — "Training Calculator" and "Inference Calculator" — above the model family tabs. The model family tabs (LLM/TS/Tabular/Classical) only appear when Training is selected.

## API Providers (initial)
- OpenAI: GPT-4o, GPT-4o-mini, o1
- Anthropic: Claude Sonnet, Claude Opus, Claude Haiku
- Google: Gemini Pro, Gemini Flash
- Mistral: Mistral Large, Mistral Small
- DeepSeek: DeepSeek-V3

## Self-Host Models (initial)
- Llama 3: 8B, 70B, 405B
- Mistral 7B, Mixtral 8x7B
- Qwen 2.5: 7B, 72B
- DeepSeek-V3

## Cloud GPU Providers (initial)
- AWS (p5, p4d)
- GCP (a3, a2)
- Azure (ND H100)
- Lambda Labs
- CoreWeave
- RunPod

## Pricing Data Approach
- Curated JSON tables in `src/data/` (hand-editable)
- Node script in `scripts/` to fetch/scrape current pricing and output draft JSON
- User reviews and commits

## Throughput Modeling
- Both cost-per-token comparison AND latency/throughput modeling
- Throughput data from vLLM/TGI benchmarks, curated per model × GPU SKU
- Latency estimate: TTFT and tokens/sec at given request rate
- Breakeven analysis: API cheaper below X req/day, self-host cheaper above
