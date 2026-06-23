import { useState, useMemo, useEffect } from 'react'
import { calculateApiCost, calculateSelfHostCost, calculateBreakeven, generateBreakevenCurve } from '../engine/inference-calculator'
import { useInferenceStore } from '../store/inference-store'
import apiPricingData from '../data/api-pricing.json'
import cloudGpuData from '../data/cloud-gpu-pricing.json'
import throughputData from '../data/self-host-throughput.json'
import type { ApiModel, CloudGpuInstance, SelfHostEntry } from '../data/types'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Legend, ReferenceLine } from 'recharts'

type PricingTier = 'onDemand' | 'reserved' | 'spot'

export function InferenceCalculator() {
  // Usage pattern
  const [requestsPerDay, setRequestsPerDay] = useState(100000)
  const [avgInputTokens, setAvgInputTokens] = useState(2000)
  const [avgOutputTokens, setAvgOutputTokens] = useState(1000)

  // API selection
  const [apiProviderIdx, setApiProviderIdx] = useState(0)
  const [apiModelIdx, setApiModelIdx] = useState(0)

  // Self-host selection
  const [selfHostIdx, setSelfHostIdx] = useState(2) // Llama 3 70B on H100
  const [cloudProviderIdx, setCloudProviderIdx] = useState(0)
  const [cloudInstanceIdx, setCloudInstanceIdx] = useState(0)
  const [pricingTier, setPricingTier] = useState<PricingTier>('onDemand')

  const usage = { requestsPerDay, avgInputTokens, avgOutputTokens }

  // Lookups
  const apiProvider = apiPricingData.providers[apiProviderIdx]
  const apiModel: ApiModel = apiProvider?.models[apiModelIdx] ?? apiProvider?.models[0]
  const selfHostEntry: SelfHostEntry = throughputData.entries[selfHostIdx]
  const cloudProvider = cloudGpuData.providers[cloudProviderIdx]
  const cloudInstance: CloudGpuInstance = cloudProvider?.instances[cloudInstanceIdx] ?? cloudProvider?.instances[0]

  // Calculations
  const apiResult = useMemo(
    () => apiModel ? calculateApiCost(apiModel, apiProvider.provider, usage) : null,
    [apiModel, apiProvider, usage.requestsPerDay, usage.avgInputTokens, usage.avgOutputTokens],
  )

  const selfHostResult = useMemo(
    () => selfHostEntry && cloudInstance
      ? calculateSelfHostCost(selfHostEntry, cloudInstance, cloudProvider.provider, usage, pricingTier)
      : null,
    [selfHostEntry, cloudInstance, cloudProvider, usage.requestsPerDay, usage.avgInputTokens, usage.avgOutputTokens, pricingTier],
  )

  const breakeven = useMemo(
    () => apiResult && selfHostResult ? calculateBreakeven(apiResult, selfHostResult, usage) : null,
    [apiResult, selfHostResult, usage.requestsPerDay],
  )

  const chartData = useMemo(
    () => apiModel && selfHostResult
      ? generateBreakevenCurve(apiModel, apiProvider.provider, selfHostResult.monthlyGpuCost, avgInputTokens, avgOutputTokens, Math.max(100000, requestsPerDay * 3))
      : [],
    [apiModel, apiProvider, selfHostResult?.monthlyGpuCost, avgInputTokens, avgOutputTokens, requestsPerDay],
  )

  // Sync to store for header export controls
  const setResults = useInferenceStore((s) => s.setResults)
  useEffect(() => {
    setResults({ usage, apiResult, selfHostResult, breakeven })
  }, [apiResult, selfHostResult, breakeven, usage, setResults])

  return (
    <div className="inference-calculator">
      {/* Usage Pattern */}
      <div className="inference-section">
        <h2>Usage Pattern</h2>
        <div className="inference-usage-grid">
          <fieldset>
            <legend>Requests / Day</legend>
            <input type="number" min={1} value={requestsPerDay} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setRequestsPerDay(v) }} />
            <div className="presets">
              {[10000, 50000, 100000, 500000, 1000000, 5000000].map((n) => (
                <button key={n} className={requestsPerDay === n ? 'active' : ''} onClick={() => setRequestsPerDay(n)}>
                  {n >= 1e6 ? `${n / 1e6}M` : `${n / 1000}K`}
                </button>
              ))}
            </div>
          </fieldset>
          <fieldset>
            <legend>Avg Input Tokens</legend>
            <input type="number" min={1} value={avgInputTokens} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setAvgInputTokens(v) }} />
            <div className="presets">
              {[500, 1000, 2000, 4000, 8000, 16000].map((n) => (
                <button key={n} className={avgInputTokens === n ? 'active' : ''} onClick={() => setAvgInputTokens(n)}>
                  {n >= 1000 ? `${n / 1000}K` : n}
                </button>
              ))}
            </div>
          </fieldset>
          <fieldset>
            <legend>Avg Output Tokens</legend>
            <input type="number" min={1} value={avgOutputTokens} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setAvgOutputTokens(v) }} />
            <div className="presets">
              {[200, 500, 1000, 2000, 4000, 8000].map((n) => (
                <button key={n} className={avgOutputTokens === n ? 'active' : ''} onClick={() => setAvgOutputTokens(n)}>
                  {n >= 1000 ? `${n / 1000}K` : n}
                </button>
              ))}
            </div>
          </fieldset>
          <p className="inference-data-note" style={{ marginTop: '0.5rem' }}>
            Daily volume: <strong>{((requestsPerDay * avgInputTokens) / 1e6).toFixed(1)}M input tokens</strong> + <strong>{((requestsPerDay * avgOutputTokens) / 1e6).toFixed(1)}M output tokens</strong> = <strong>{((requestsPerDay * (avgInputTokens + avgOutputTokens)) / 1e6).toFixed(1)}M total tokens/day</strong>
          </p>
        </div>
      </div>

      <div className="inference-comparison-grid">
        {/* BUY (API) */}
        <div className="inference-panel">
          <h3 className="panel-title buy-title">Buy (API)</h3>
          <fieldset>
            <legend>Provider</legend>
            <select value={apiProviderIdx} onChange={(e) => { setApiProviderIdx(Number(e.target.value)); setApiModelIdx(0) }}>
              {apiPricingData.providers.map((p, i) => (<option key={i} value={i}>{p.provider}</option>))}
            </select>
          </fieldset>
          <fieldset>
            <legend>Model</legend>
            <select value={apiModelIdx} onChange={(e) => setApiModelIdx(Number(e.target.value))}>
              {apiProvider?.models.map((m, i) => (
                <option key={i} value={i}>{m.model} (${m.inputPer1M}/${m.outputPer1M} per M)</option>
              ))}
            </select>
          </fieldset>

          {apiResult && (
            <div className="inference-result-card">
              <div className="result-gpu-count">
                <span className="big-number">${apiResult.monthlyCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                <span className="label">/ month</span>
              </div>
              <dl className="result-details">
                <div><dt>Cost per request</dt><dd>${apiResult.costPerRequest.toFixed(4)}</dd></div>
                <div><dt>Annual cost</dt><dd>${apiResult.annualCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</dd></div>
                <div><dt>Input cost share</dt><dd>{(apiResult.inputCostShare * 100).toFixed(0)}%</dd></div>
                <div><dt>Output cost share</dt><dd>{(apiResult.outputCostShare * 100).toFixed(0)}%</dd></div>
              </dl>
              <details className="trace-details">
                <summary>Calculation trace</summary>
                <pre className="trace-pre">{apiResult.trace.join('\n')}</pre>
              </details>
            </div>
          )}
        </div>

        {/* BUILD (Self-host) */}
        <div className="inference-panel">
          <h3 className="panel-title build-title">Build (Self-host)</h3>
          <fieldset>
            <legend>Model</legend>
            <select value={selfHostIdx} onChange={(e) => setSelfHostIdx(Number(e.target.value))}>
              {throughputData.entries.map((e, i) => (
                <option key={i} value={i}>{e.model} — {e.gpuCount}x {e.gpu} ({e.quantization})</option>
              ))}
            </select>
          </fieldset>
          <fieldset>
            <legend>Cloud Provider</legend>
            <select value={cloudProviderIdx} onChange={(e) => { setCloudProviderIdx(Number(e.target.value)); setCloudInstanceIdx(0) }}>
              {cloudGpuData.providers.map((p, i) => (<option key={i} value={i}>{p.provider}</option>))}
            </select>
          </fieldset>
          <fieldset>
            <legend>Instance</legend>
            <select value={cloudInstanceIdx} onChange={(e) => setCloudInstanceIdx(Number(e.target.value))}>
              {cloudProvider?.instances.map((inst, i) => (
                <option key={i} value={i}>{inst.instance} — {inst.gpuCount}x {inst.gpu} (${inst.onDemandPerHr}/hr)</option>
              ))}
            </select>
          </fieldset>
          <fieldset>
            <legend>Pricing Tier</legend>
            <div className="presets">
              {(['onDemand', 'reserved', 'spot'] as PricingTier[]).map((tier) => (
                <button key={tier} className={pricingTier === tier ? 'active' : ''} onClick={() => setPricingTier(tier)}>
                  {tier === 'onDemand' ? 'On-Demand' : tier === 'reserved' ? 'Reserved' : 'Spot'}
                </button>
              ))}
            </div>
          </fieldset>

          {selfHostResult && (
            <div className="inference-result-card">
              <div className="result-gpu-count">
                <span className="big-number">${selfHostResult.monthlyGpuCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}</span>
                <span className="label">/ month</span>
              </div>
              <dl className="result-details">
                <div><dt>Max throughput</dt><dd>{selfHostResult.maxOutputTokensPerSec.toLocaleString()} tok/s</dd></div>
                <div><dt>Required throughput</dt><dd>{selfHostResult.requiredTokensPerSec.toFixed(1)} tok/s</dd></div>
                <div><dt>Utilization</dt><dd>{selfHostResult.utilizationPercent.toFixed(1)}%</dd></div>
                <div><dt>Est. TTFT</dt><dd>~{selfHostResult.estimatedTtftMs}ms</dd></div>
              </dl>
              {!selfHostResult.canServe && (
                <p className="warning">Cannot serve required throughput with this configuration.</p>
              )}
              {selfHostResult.warnings.map((w, i) => (
                <p key={i} className="warning">{w}</p>
              ))}
              <details className="trace-details">
                <summary>Calculation trace</summary>
                <pre className="trace-pre">{selfHostResult.trace.join('\n')}</pre>
              </details>
            </div>
          )}
        </div>
      </div>

      {/* Breakeven Analysis */}
      {breakeven && apiResult && selfHostResult && (
        <div className="inference-section breakeven-section">
          <h3>Build vs Buy Analysis</h3>
          <div className="breakeven-summary">
            {breakeven.apiCheaperBelow ? (
              <p>
                At <strong>{requestsPerDay.toLocaleString()} req/day</strong>, <strong className="buy-accent">API is cheaper</strong> by ${Math.abs(breakeven.monthlySavingsAtCurrentVolume).toLocaleString(undefined, { maximumFractionDigits: 0 })}/month.
                Self-hosting becomes cheaper above <strong>{breakeven.breakevenRequestsPerDay.toLocaleString()} req/day</strong>.
              </p>
            ) : (
              <p>
                At <strong>{requestsPerDay.toLocaleString()} req/day</strong>, <strong className="build-accent">self-hosting is cheaper</strong> by ${Math.abs(breakeven.monthlySavingsAtCurrentVolume).toLocaleString(undefined, { maximumFractionDigits: 0 })}/month
                (${Math.abs(breakeven.annualSavingsAtCurrentVolume).toLocaleString(undefined, { maximumFractionDigits: 0 })}/year).
              </p>
            )}
          </div>

          {chartData.length > 0 && (
            <div className="breakeven-chart">
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData} margin={{ top: 10, right: 30, bottom: 20, left: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#2e3346" />
                  <XAxis
                    dataKey="requestsPerDay"
                    tickFormatter={(v) => v >= 1000 ? `${v / 1000}K` : v}
                    label={{ value: 'Requests / Day', position: 'bottom', offset: 0, fill: '#8b90a0', fontSize: 11 }}
                    tick={{ fill: '#8b90a0', fontSize: 11 }}
                  />
                  <YAxis
                    tickFormatter={(v) => `$${v >= 1000 ? `${(v / 1000).toFixed(0)}K` : v}`}
                    label={{ value: 'Monthly Cost', angle: -90, position: 'insideLeft', fill: '#8b90a0', fontSize: 11 }}
                    tick={{ fill: '#8b90a0', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#1a1d27', border: '1px solid #2e3346', borderRadius: 8, color: '#e1e4ed' }}
                    formatter={(value: number, name: string) => [`$${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}`, name]}
                    labelFormatter={(label) => `${Number(label).toLocaleString()} req/day`}
                  />
                  <Legend wrapperStyle={{ color: '#8b90a0', fontSize: 11 }} />
                  <Line type="monotone" dataKey="apiMonthlyCost" name="API Cost" stroke="#f59e0b" strokeWidth={2} dot={false} />
                  <Line type="monotone" dataKey="selfHostMonthlyCost" name="Self-Host Cost" stroke="#6366f1" strokeWidth={2} dot={false} />
                  {breakeven.breakevenRequestsPerDay < Infinity && (
                    <ReferenceLine x={breakeven.breakevenRequestsPerDay} stroke="#22c55e" strokeDasharray="5 5" label={{ value: 'Breakeven', fill: '#22c55e', fontSize: 10 }} />
                  )}
                  <ReferenceLine x={requestsPerDay} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Current', fill: '#ef4444', fontSize: 10 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}

      <p className="inference-data-note">
        Pricing data last updated: {apiPricingData.lastUpdated}. Throughput benchmarks are approximate.
        <br />
        Self-host costs do not include engineering time, ops overhead, or networking.
      </p>
    </div>
  )
}
