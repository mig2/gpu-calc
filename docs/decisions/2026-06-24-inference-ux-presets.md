# Inference Calculator UX — Use-Case Presets and Scale Methods

**Date:** 2026-06-24
**Issue:** #60
**Context:** Users don't think in requests/day and avg tokens. They think in use cases and business scale.

---

## Decision

Replace raw infrastructure inputs with a two-step guided flow:

1. **Use-case preset** selects typical input/output token sizes
2. **Scale method** determines request volume via a natural metric

### Use-Case Presets

Based on production workload data from Azure inference traces, OpenRouter's 100T-token study, and enterprise token optimization patterns.

| Use Case | Input Tokens | Output Tokens | Rationale |
|----------|-------------|--------------|-----------|
| Chatbot / Customer Support | 500 | 300 | System prompt + short turns. Azure data: balanced workloads. |
| RAG / Q&A over Documents | 4,000 | 500 | Retrieved context chunks + query. Context-heavy pattern. |
| Document Summarization | 10,000 | 500 | Long input docs, concise output. Legal/enterprise typical. |
| Coding Assistant | 8,000 | 2,000 | Code context + generation. OpenRouter: coding >50% of volume. |
| Agentic Workflow | 16,000 | 4,000 | Multi-step tool use. SWE-bench: 1-3.5M tokens/task. |
| Long-context Analysis | 50,000 | 1,000 | Full documents. Azure 2024: 91.6% context-heavy. |
| Custom | manual | manual | Power users. |

### Scale Methods

| Method | Input | Derived |
|--------|-------|---------|
| By users | users × req/user/day | requests/day |
| By monthly budget | total tokens/month | requests/day (using avg token sizes) |
| Direct | requests/day | (raw input) |

### Design principle

Raw technical fields (requests/day, input tokens, output tokens) remain visible and editable at all times. The presets and scale methods are convenience layers on top, not replacements.

## References

- [A Systematic Characterization of LLM Inference on GPUs](https://arxiv.org/pdf/2512.01644) — Microsoft Azure production traces 2023-2024
- [OpenRouter State of AI 2025](https://openrouter.ai/state-of-ai) — 100T token usage study
- [Enterprise LLM Pricing Trends (YipitData)](https://www.yipitdata.com/resources/blog/cloud-llm-pricing-trends)
- [LLM Token Optimization (Redis)](https://redis.io/blog/llm-token-optimization-speed-up-apps/)
