# Inference Economics — Replica Scaling and the Breakeven Redefinition

**Date:** 2026-10-14
**Issues:** #141 (replica scaling), #101 (breakeven staircase), #107 (cloud pricing), #106 (API pricing)
**Research:** [`docs/research/2026-10-14-inference-serving-research.md`](../research/2026-10-14-inference-serving-research.md)
**Context:** Reported from use — the pane said *"Required throughput (23148.1 tok/s) exceeds capacity (997 tok/s). Add more GPUs or reduce request volume"* while offering no control to add GPUs.

---

## Decision 1 — Self-host capacity and cost scale with replicas

The reported symptom was a warning advising an impossible action. The underlying defect was larger: **demand never entered the cost at all.**

```ts
const instancesNeeded = Math.ceil(throughput.gpuCount / instance.gpuCount)
```

That sizes the instance to **fit the model**, not to **serve the load**. It is a function of model size and instance shape only. So self-host monthly cost was identical at 1% and at 2,300% of capacity, and the pane reported `canServe: false` beside the price of a configuration it already knew could not serve the workload.

Replicas are now sized to the offered load by default and editable:

```
replicasNeeded = ceil(required_tok_s / per_replica_tok_s)
capacity       = per_replica_tok_s × replicas
monthlyCost    = pricePerHr × 24 × DAYS_PER_MONTH × instancesPerReplica × replicas
utilization    = required / total capacity
```

**Auto-sizing is the default rather than an opt-in.** Reporting the cost of an unservable configuration is not a useful answer to "what would this cost", and the previous behaviour amounted to that. An explicit override remains, for modelling deliberate under- or over-provisioning.

At 10M req/day the pane now auto-sizes to 59 replicas, 98.4% utilization, $4.23M/month. It previously reported "cannot serve" and $71,774.

## Decision 2 — The breakeven volume is no longer the headline number

`calculateBreakeven` carried the comment `// Self-host is fixed.` That was its premise, and Decision 1 falsified it.

With both sides scaling linearly in volume, a single crossover volume stops being generally meaningful: if self-host is cheaper per token it wins at every volume that keeps the replicas busy, and if it is dearer it never wins. The variable that actually decides it is **utilization**, which is also what the research found independently — published break-even utilization against API pricing is **~60%**, and measured cost on identical H100 hardware spans **$0.21–$15.25 per million output tokens** purely with concurrency.

So the result now also carries:

- `apiCostPerMillionOutputTokens` and `selfHostCostPerMillionOutputTokens`
- `selfHostCheaperPerToken`
- `breakevenNote`, stating in words when the crossover figure does and does not apply

The crossover volume is retained because it is still meaningful while a single replica suffices. It is labelled rather than presented as the answer.

**This changed the meaning of a number users may have recorded.** A remembered "breakeven requests/day" from an earlier version is not comparable to today's. That is the strongest argument for the explicit `breakevenNote` over a silent improvement.

## Decision 3 — Two tests were replaced rather than relaxed

Both encoded the old premise:

| Test | Asserted | Why it had to go |
|---|---|---|
| `warns when throughput exceeded` | `canServe === false` | Asserted that reporting an unservable configuration is the correct output |
| `self-host is cheaper at high volume` | self-host always wins at 500K req/day | True only while self-host cost was wrongly fixed |

The second is the more interesting failure: it passed for years *because* of the bug. A test that encodes a defect will defend it, so the rule applied was to replace the assertion with the corrected behaviour and record why in a comment — never to widen a tolerance until it passes.

## Decision 4 — Pricing is vendor-verified or explicitly marked

Both pricing files were four months stale, and the earlier research marked several replacement figures medium-low on aggregator evidence. The rule adopted: **every rate comes from the provider's own page or public pricing API, or it is marked `aggregator`, or it is omitted.**

Outcome: API pricing 38/38 vendor-sourced; cloud GPU pricing 29/32, with the three GCP rows marked `aggregator` because Google renders its GPU tables client-side and its Billing Catalog API requires a key.

The caution proved warranted in size if not in direction. AWS `p5.48xlarge` was stored at `$98.32/hr`; AWS's own price feed gives **$55.04** — 79% too high. That was verified directly against `b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2` rather than taken on trust, and all four P-family rows matched to the cent.

Two supporting rules:

- **Rows carry `region`.** They previously carried none, which made every rate ambiguous given how much GPU pricing varies by region.
- **No derived discounts.** Where a vendor advertises "up to 60% off" rather than a rate, the field stays `null` with the reason recorded. A number computed from a marketing percentage and displayed as a quote is false precision.

## What this does *not* fix

Replica scaling is linear in a quantity that is itself wrong. `outputTokensPerSec` remains a constant per hardware configuration when it actually varies ~24× with concurrency and SLO — see #95, and `src/engine/rl-calculator.ts` for a working roofline implementation that reproduces the published KV anchors and could be ported.

Also outstanding: prompt-cache fields exist on `ApiModel` but the cost engine does not read them (#99); tiered pricing is unmodelled (#105); data age is not surfaced in the UI (#108); and TCO beyond raw rental — redundancy, peak-to-average, non-GPU cost — is absent (#100), which published figures put at 3–5× raw rental.

## References

See the research document linked above; every figure in this file is sourced there or in the commit that implemented it.
