import { useState } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { GPU_SKUS } from '../engine/gpu-data'
import { getPeakFlops } from '../engine/hardware-estimator'
import { REFERENCE_MODELS } from '../engine/reference-models'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceDot, CartesianGrid } from 'recharts'

type BudgetMode = 'scenario' | 'gpu'

const TPP_RANGE = [5, 10, 15, 20, 30, 40, 50, 75, 100, 150, 200, 300]

function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(0)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return n.toFixed(0)
}

export function IsoFlopExplorer() {
  const [open, setOpen] = useState(false)
  const [budgetMode, setBudgetMode] = useState<BudgetMode>('scenario')
  const [gpuCount, setGpuCount] = useState(256)
  const [gpuSkuId, setGpuSkuId] = useState('h100-sxm')
  const [windowDays, setWindowDays] = useState(30)
  const [mfu, setMfu] = useState(0.40)

  const results = useScenarioStore((s) => s.results)
  const scenario = useScenarioStore((s) => s.scenario)
  const modelFamily = useScenarioStore((s) => s.modelFamily)

  // Only relevant for transformer-based modes
  if (modelFamily === 'classical_tabular') return null

  const overhead = scenario.overheadFactor
  const architectureFactor = 6

  // Compute FLOP budget based on mode
  let availableFlops: number
  if (budgetMode === 'gpu') {
    const gpu = GPU_SKUS.find((g) => g.id === gpuSkuId) ?? GPU_SKUS[0]
    const peak = getPeakFlops(gpu, scenario.precision)
    const sustained = peak * mfu * scenario.availability
    availableFlops = gpuCount * sustained * windowDays * 86_400
  } else {
    availableFlops = results.length > 0 ? results[0].totalFlops : 0
  }

  if (availableFlops <= 0) return null

  // Compute curve: for each TPP, what's the max model size?
  const curveData = TPP_RANGE.map((tpp) => {
    const maxN = Math.sqrt(availableFlops / (architectureFactor * tpp * overhead))
    return { tpp, maxParams: maxN, maxParamsLabel: formatParams(maxN) }
  })

  // Current scenario dot (for LLM mode)
  const currentTpp = modelFamily === 'llm' ? scenario.tokensPerParameter : null
  const currentN = modelFamily === 'llm' ? scenario.modelParameters : null

  // Reference model dots that are within ~0.1x to 10x of the budget
  const refDots = REFERENCE_MODELS
    .filter((ref) => ref.estimatedFlops > availableFlops * 0.05 && ref.estimatedFlops < availableFlops * 20)
    .map((ref) => {
      const tpp = ref.trainingTokens / ref.parameters
      return { name: ref.name, tpp, params: ref.parameters }
    })
    .filter((d) => d.tpp >= 5 && d.tpp <= 300)

  return (
    <div className="isoflop-section">
      <button
        className="advanced-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? '\u25BE' : '\u25B8'} IsoFLOP Explorer
      </button>
      {open && (
        <div className="isoflop-content">
          <p className="sensitivity-subtitle">
            For a fixed compute budget, explore the tradeoff between model size and tokens-per-parameter.
          </p>

          <div className="isoflop-controls">
            <fieldset>
              <legend>Budget Source</legend>
              <div className="presets">
                <button className={budgetMode === 'scenario' ? 'active' : ''} onClick={() => setBudgetMode('scenario')}>
                  Current Scenario
                </button>
                <button className={budgetMode === 'gpu' ? 'active' : ''} onClick={() => setBudgetMode('gpu')}>
                  GPU Budget
                </button>
              </div>
            </fieldset>

            {budgetMode === 'gpu' && (
              <>
                <fieldset>
                  <legend>GPUs</legend>
                  <input type="number" min={1} value={gpuCount} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setGpuCount(v) }} />
                  <div className="presets">
                    {[64, 128, 256, 512, 1024].map((n) => (
                      <button key={n} className={gpuCount === n ? 'active' : ''} onClick={() => setGpuCount(n)}>{n}</button>
                    ))}
                  </div>
                </fieldset>
                <fieldset>
                  <legend>GPU SKU</legend>
                  <select value={gpuSkuId} onChange={(e) => setGpuSkuId(e.target.value)}>
                    {GPU_SKUS.map((g) => (<option key={g.id} value={g.id}>{g.label}</option>))}
                  </select>
                </fieldset>
                <fieldset>
                  <legend>Window (days)</legend>
                  <input type="number" min={1} value={windowDays} onChange={(e) => { const v = parseInt(e.target.value); if (v > 0) setWindowDays(v) }} />
                </fieldset>
                <fieldset>
                  <legend>MFU</legend>
                  <input type="number" min={0.1} max={0.7} step={0.05} value={mfu} onChange={(e) => { const v = parseFloat(e.target.value); if (v > 0) setMfu(v) }} />
                </fieldset>
              </>
            )}

            <p className="isoflop-budget">
              Budget: <strong>{availableFlops.toExponential(2)} FLOPs</strong>
              {budgetMode === 'gpu' && <> ({gpuCount} × {GPU_SKUS.find(g => g.id === gpuSkuId)?.label ?? gpuSkuId} × {windowDays}d)</>}
            </p>
          </div>

          <div className="isoflop-chart">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={curveData} margin={{ top: 10, right: 30, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#2e3346" />
                <XAxis
                  dataKey="tpp"
                  label={{ value: 'Tokens per Parameter', position: 'bottom', offset: 0, fill: '#8b90a0', fontSize: 11 }}
                  tick={{ fill: '#8b90a0', fontSize: 11 }}
                />
                <YAxis
                  dataKey="maxParams"
                  scale="log"
                  domain={['auto', 'auto']}
                  tickFormatter={formatParams}
                  label={{ value: 'Max Model Size', angle: -90, position: 'insideLeft', fill: '#8b90a0', fontSize: 11 }}
                  tick={{ fill: '#8b90a0', fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{ background: '#1a1d27', border: '1px solid #2e3346', borderRadius: 8, color: '#e1e4ed' }}
                  formatter={(value: number) => [formatParams(value), 'Max Model Size']}
                  labelFormatter={(label) => `TPP: ${label}`}
                />
                <Line type="monotone" dataKey="maxParams" stroke="#6366f1" strokeWidth={2} dot={false} />
                {currentTpp && currentN && (
                  <ReferenceDot x={currentTpp} y={currentN} r={6} fill="#22c55e" stroke="#fff" strokeWidth={2} />
                )}
                {refDots.map((ref) => (
                  <ReferenceDot key={ref.name} x={ref.tpp} y={ref.params} r={4} fill="#f59e0b" stroke="none">
                  </ReferenceDot>
                ))}
              </LineChart>
            </ResponsiveContainer>
            <div className="isoflop-legend">
              <span className="isoflop-legend-item"><span className="legend-dot" style={{ background: '#6366f1' }}></span> Budget frontier</span>
              {currentTpp && <span className="isoflop-legend-item"><span className="legend-dot" style={{ background: '#22c55e' }}></span> Your scenario</span>}
              {refDots.length > 0 && <span className="isoflop-legend-item"><span className="legend-dot" style={{ background: '#f59e0b' }}></span> Known models</span>}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
