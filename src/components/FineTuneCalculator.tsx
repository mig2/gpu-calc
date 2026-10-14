import { Tooltip } from './Tooltip'
import { ConfidenceBadge } from './ConfidenceBadge'
import { isPeftMethod, PER_GPU_OVERHEAD_BYTES, USABLE_MEMORY_FRACTION } from '../engine/finetune-calculator'
import type {
  FineTuneEstimate,
  FineTuneMethod,
  FitState,
  LoraTargets,
  MemoryBreakdown,
  OptimizerKind,
  ShardingStrategy,
  WeightPrecision,
} from '../engine/finetune-types'
import type { GpuSku } from '../engine/types'
import {
  CATALOG_LAST_UPDATED,
  CUSTOM_MODEL_ID,
  FINE_TUNE_GPUS,
  FINE_TUNE_METHODS,
  FINE_TUNE_MODELS,
  GOAL_PRESETS,
  METHOD_LABELS,
  OPTIMIZER_LABELS,
  PRECISION_LABELS,
  PRICING_LAST_UPDATED,
  PRICING_TIER_LABELS,
  SHARDING_LABELS,
  TARGET_LABELS,
  getModelEntry,
  selectCellResult,
  useFineTuneStore,
  type PricingTier,
} from '../store/finetune-store'

/**
 * Fine-tuning pane.
 *
 * Decisions: docs/decisions/2026-10-14-finetuning-tab-decisions.md
 *
 * The hero output is the method x GPU matrix. Every cell leads with the GPU
 * count and fit state because that is what actually varies between methods
 * (20-30x); wall-clock on its own inverts the ranking and cost is close to
 * method-invariant, so neither can carry the cell alone (Decision 2).
 */

// ---------------------------------------------------------------------------
// Formatting
// ---------------------------------------------------------------------------

function formatGb(bytes: number): string {
  const gb = bytes / 1e9
  if (!Number.isFinite(gb)) return '—'
  if (gb >= 1000) return `${Math.round(gb).toLocaleString()} GB`
  if (gb >= 10) return `${gb.toFixed(0)} GB`
  return `${gb.toFixed(1)} GB`
}

function formatHours(hours: number): string {
  if (!Number.isFinite(hours)) return '—'
  if (hours >= 48) return `${(hours / 24).toFixed(1)} d`
  if (hours >= 1) return `${hours.toFixed(1)} h`
  if (hours >= 0.05) return `${hours.toFixed(2)} h`
  return `${Math.max(1, Math.round(hours * 60))} min`
}

function formatUsd(usd: number): string {
  if (usd >= 1000) return `$${Math.round(usd).toLocaleString()}`
  if (usd >= 10) return `$${usd.toFixed(0)}`
  return `$${usd.toFixed(2)}`
}

function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(n >= 100e9 ? 0 : 1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return n.toFixed(0)
}

function formatTokens(n: number): string {
  if (!Number.isFinite(n)) return '—'
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  return Math.round(n).toLocaleString()
}

const FIT_LABELS: Record<FitState, string> = {
  fits: 'FITS',
  sharded: 'SHARDED',
  oom: 'OOM',
}

const FIT_DESCRIPTIONS: Record<FitState, string> = {
  fits: 'fits on a single GPU without sharding',
  sharded: 'needs sharded model state across several GPUs',
  oom: 'does not fit at any practical GPU count',
}

/** Usable HBM per GPU, matching the engine's own capacity accounting. */
function usableBytesPerGpu(gpu: GpuSku): number {
  return gpu.memoryGb * 1e9 * USABLE_MEMORY_FRACTION - PER_GPU_OVERHEAD_BYTES
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

type CellResult = { estimate: FineTuneEstimate; costUsd: number | null }

function MatrixCell({
  method,
  gpu,
  cell,
  selected,
  priced,
  onSelect,
}: {
  method: FineTuneMethod
  gpu: GpuSku
  cell: CellResult | undefined
  selected: boolean
  priced: boolean
  onSelect: () => void
}) {
  if (!cell) return <td className="ft-cell-empty">—</td>

  const { estimate, costUsd } = cell
  const fit = estimate.fitState
  const feasible = fit !== 'oom'
  const gpuCount = feasible ? estimate.minGpusMemory : null

  const costTitle = !priced
    ? `No instance price for ${gpu.label} in cloud-gpu-pricing.json. Cost is left blank rather than borrowed from a different accelerator.`
    : 'Infeasible at any practical GPU count, so there is no run to price.'

  const label = [
    `${METHOD_LABELS[method]} on ${gpu.label}:`,
    FIT_DESCRIPTIONS[fit],
    feasible ? `${gpuCount} GPU${gpuCount === 1 ? '' : 's'},` : '',
    `${formatGb(estimate.memory.totalBytes)} of model state,`,
    feasible ? `${formatHours(estimate.wallClockHours)} wall-clock,` : '',
    costUsd != null ? formatUsd(costUsd) : 'no cost estimate',
  ]
    .filter(Boolean)
    .join(' ')

  return (
    <td className="ft-matrix-td">
      <button
        type="button"
        className={`ft-cell ft-cell-${fit}${selected ? ' selected' : ''}`}
        data-fit-state={fit}
        aria-pressed={selected}
        aria-label={label}
        onClick={onSelect}
      >
        <span className="ft-cell-head">
          <span className="ft-cell-gpus">{feasible ? `${gpuCount}×` : '—'}</span>
          <span className={`ft-fit-badge ft-fit-${fit}`}>{FIT_LABELS[fit]}</span>
        </span>
        <span className="ft-cell-mem">{formatGb(estimate.memory.totalBytes)} state</span>
        <span className="ft-cell-foot">
          <span className="ft-cell-time">{feasible ? formatHours(estimate.wallClockHours) : '—'}</span>
          {costUsd != null ? (
            <span className="ft-cell-cost">{formatUsd(costUsd)}</span>
          ) : (
            <span className="ft-cell-cost ft-cell-cost-none" title={costTitle}>
              {'—'}
            </span>
          )}
        </span>
      </button>
    </td>
  )
}

function MethodMatrix() {
  const matrix = useFineTuneStore((s) => s.matrix)
  const rates = useFineTuneStore((s) => s.rates)
  const method = useFineTuneStore((s) => s.config.method)
  const selectedGpuId = useFineTuneStore((s) => s.selectedGpuId)
  const selectCell = useFineTuneStore((s) => s.selectCell)

  const unpriced = matrix.gpus.filter((gpu) => rates[gpu.id] == null)

  return (
    <div className="ft-matrix-wrapper">
      <table className="ft-matrix">
        <thead>
          <tr>
            <th scope="col" className="ft-matrix-corner">
              Method
            </th>
            {matrix.gpus.map((gpu) => (
              <th scope="col" key={gpu.id}>
                {gpu.label}
                <span className="ft-matrix-colnote">
                  {gpu.memoryGb} GB · {formatGb(usableBytesPerGpu(gpu))} usable
                  {rates[gpu.id]
                    ? ` · ${formatUsd(rates[gpu.id].usdPerGpuHour)}/GPU-hr`
                    : ' · no price'}
                </span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.methods.map((m) => (
            <tr key={m}>
              <th scope="row" className="ft-matrix-rowhead">
                {METHOD_LABELS[m]}
              </th>
              {matrix.gpus.map((gpu) => (
                <MatrixCell
                  key={gpu.id}
                  method={m}
                  gpu={gpu}
                  cell={matrix.cells[m]?.[gpu.id]}
                  priced={rates[gpu.id] != null}
                  selected={m === method && gpu.id === selectedGpuId}
                  onSelect={() => selectCell(m, gpu.id)}
                />
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="ft-matrix-legend">
        <span className="ft-fit-badge ft-fit-fits">FITS</span>
        <span className="ft-legend-text">one GPU holds all model state</span>
        <span className="ft-fit-badge ft-fit-sharded">SHARDED</span>
        <span className="ft-legend-text">state split across the listed GPU count</span>
        <span className="ft-fit-badge ft-fit-oom">OOM</span>
        <span className="ft-legend-text">infeasible at any practical count</span>
      </div>
      {unpriced.length > 0 && (
        <p className="ft-note">
          No instance price in <code>cloud-gpu-pricing.json</code> for{' '}
          {unpriced.map((g) => g.label).join(', ')}, so those columns show no cost. Rates
          elsewhere are the cheapest listed instance price divided by its GPU count.
        </p>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Memory breakdown
// ---------------------------------------------------------------------------

const MEMORY_SEGMENTS: { key: keyof MemoryBreakdown; label: string; cls: string }[] = [
  { key: 'weightsBytes', label: 'Weights', cls: 'ft-seg-weights' },
  { key: 'gradientsBytes', label: 'Gradients', cls: 'ft-seg-gradients' },
  { key: 'optimizerBytes', label: 'Optimizer', cls: 'ft-seg-optimizer' },
  { key: 'activationsBytes', label: 'Activations', cls: 'ft-seg-activations' },
  { key: 'auxiliaryBytes', label: 'Auxiliary', cls: 'ft-seg-auxiliary' },
  { key: 'overheadBytes', label: 'Overhead', cls: 'ft-seg-overhead' },
]

function MemoryBreakdownBar({
  memory,
  gpu,
  gpuCount,
}: {
  memory: MemoryBreakdown
  gpu: GpuSku
  gpuCount: number
}) {
  const perGpu = usableBytesPerGpu(gpu)
  const capacity = perGpu * (Number.isFinite(gpuCount) && gpuCount > 0 ? gpuCount : 1)
  const scale = Math.max(memory.totalBytes, capacity) * 1.04

  return (
    <div className="ft-memory">
      <div className="ft-memory-track">
        {MEMORY_SEGMENTS.map(({ key, label, cls }) => {
          const bytes = memory[key]
          if (bytes <= 0) return null
          return (
            <div
              key={key}
              className={`ft-memory-seg ${cls}`}
              style={{ width: `${(bytes / scale) * 100}%` }}
              title={`${label}: ${formatGb(bytes)}`}
            />
          )
        })}
        <div
          className="ft-memory-capacity"
          style={{ left: `${Math.min(100, (capacity / scale) * 100)}%` }}
          aria-hidden="true"
        />
      </div>
      <p className="ft-memory-caption">
        {formatGb(memory.totalBytes)} of model state against{' '}
        {formatGb(capacity)} usable on{' '}
        {Number.isFinite(gpuCount) ? `${gpuCount}× ${gpu.label}` : gpu.label}{' '}
        ({formatGb(perGpu)} per GPU, {(USABLE_MEMORY_FRACTION * 100).toFixed(0)}% of
        advertised HBM minus {formatGb(PER_GPU_OVERHEAD_BYTES)} reserved)
      </p>
      <ul className="ft-memory-legend">
        {MEMORY_SEGMENTS.map(({ key, label, cls }) => (
          <li key={key}>
            <span className={`ft-memory-swatch ${cls}`} aria-hidden="true" />
            {label}
            <strong>{formatGb(memory[key])}</strong>
          </li>
        ))}
      </ul>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Selected-cell detail
// ---------------------------------------------------------------------------

function SelectedCellDetail() {
  const cell = useFineTuneStore(selectCellResult)
  const selectedGpuId = useFineTuneStore((s) => s.selectedGpuId)
  const method = useFineTuneStore((s) => s.config.method)
  const matrix = useFineTuneStore((s) => s.matrix)

  const gpu = matrix.gpus.find((g) => g.id === selectedGpuId)
  if (!cell || !gpu) {
    return (
      <div className="ft-section">
        <h2>Selected configuration</h2>
        <p className="ft-note">Pick a cell in the matrix to see its breakdown.</p>
      </div>
    )
  }

  const { estimate, costUsd } = cell
  const gpuHours =
    Number.isFinite(estimate.wallClockHours) && Number.isFinite(estimate.minGpusMemory)
      ? estimate.wallClockHours * estimate.minGpusMemory
      : null

  return (
    <div className="ft-section">
      <div className="ft-detail-head">
        <h2>
          {METHOD_LABELS[method]} on {gpu.label}
        </h2>
        <ConfidenceBadge level={estimate.confidence} />
      </div>

      <div className={`ft-fit-banner ft-cell-${estimate.fitState}`}>
        <span className={`ft-fit-badge ft-fit-${estimate.fitState}`}>
          {FIT_LABELS[estimate.fitState]}
        </span>
        <span>
          {estimate.fitState === 'oom'
            ? `${formatGb(estimate.memory.totalBytes)} of model state exceeds what any practical number of ${gpu.label} can hold under this sharding strategy.`
            : `${estimate.minGpusMemory}× ${gpu.label} — ${FIT_DESCRIPTIONS[estimate.fitState]}.`}
        </span>
      </div>

      <MemoryBreakdownBar
        memory={estimate.memory}
        gpu={gpu}
        gpuCount={estimate.minGpusMemory}
      />

      <dl className="result-details">
        <div>
          <dt>
            <Tooltip text="Minimum GPUs set by the memory floor, rounded to a deployable count (1, 2, 4, 8, then whole nodes of 8).">
              Minimum GPUs
            </Tooltip>
          </dt>
          <dd>{Number.isFinite(estimate.minGpusMemory) ? estimate.minGpusMemory : '—'}</dd>
        </div>
        <div>
          <dt>Wall-clock</dt>
          <dd>{formatHours(estimate.wallClockHours)}</dd>
        </div>
        <div>
          <dt>GPU-hours</dt>
          <dd>{gpuHours != null ? gpuHours.toFixed(2) : '—'}</dd>
        </div>
        <div>
          <dt>
            <Tooltip text="GPU-hours x the cheapest listed per-GPU hourly rate for this SKU. Blank when the pricing file has no row for it.">
              Cost
            </Tooltip>
          </dt>
          <dd>{costUsd != null ? formatUsd(costUsd) : '—'}</dd>
        </div>
        <div>
          <dt>Trainable parameters</dt>
          <dd>
            {formatParams(estimate.trainableParameters)} (
            {(estimate.trainableFraction * 100).toFixed(3)}%)
          </dd>
        </div>
        <div>
          <dt>Tokens processed</dt>
          <dd>{formatTokens(estimate.tokensProcessed)}</dd>
        </div>
        <div>
          <dt>FLOPs / token</dt>
          <dd>{estimate.flopsPerToken.toExponential(2)}</dd>
        </div>
        <div>
          <dt>Total FLOPs</dt>
          <dd>{estimate.totalFlops.toExponential(2)}</dd>
        </div>
        <div>
          <dt>
            <Tooltip text="Fine-tuning MFU is lower than pretraining's: 0.40 single GPU, 0.30 sharded single node, 0.27 multi-node, times the method and scaling factors.">
              Effective MFU
            </Tooltip>
          </dt>
          <dd>{(estimate.effectiveMfu * 100).toFixed(1)}%</dd>
        </div>
      </dl>

      {estimate.warnings.length > 0 && (
        <div className="warnings-panel">
          <h3>Warnings &amp; Notes</h3>
          {estimate.warnings.map((w) => (
            <p key={w} className="warning">
              {w}
            </p>
          ))}
        </div>
      )}

      <details className="trace-details">
        <summary>
          Calculation trace — {METHOD_LABELS[method]} on {gpu.label}
        </summary>
        <pre className="trace-pre">{estimate.trace.join('\n')}</pre>
      </details>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

function ModelSection() {
  const modelId = useFineTuneStore((s) => s.modelId)
  const setModelId = useFineTuneStore((s) => s.setModelId)
  const custom = useFineTuneStore((s) => s.custom)
  const setCustomField = useFineTuneStore((s) => s.setCustomField)

  const entry = getModelEntry(modelId)
  const families = [...new Set(FINE_TUNE_MODELS.map((m) => m.family))]

  return (
    <div className="ft-section">
      <h2>1. Which model?</h2>
      <div className="scale-inputs">
        <fieldset>
          <legend>Base model</legend>
          <select
            aria-label="Base model"
            value={modelId}
            onChange={(e) => setModelId(e.target.value)}
          >
            {families.map((family) => (
              <optgroup key={family} label={family}>
                {FINE_TUNE_MODELS.filter((m) => m.family === family).map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                    {m.isMoe && m.totalParameters != null && m.activeParameters != null
                      ? ` — ${formatParams(m.totalParameters)} total / ${formatParams(m.activeParameters)} active`
                      : ''}
                  </option>
                ))}
              </optgroup>
            ))}
          </select>
        </fieldset>
      </div>

      {entry.isMoe && entry.totalParameters != null && entry.activeParameters != null && (
        <p className="ft-moe-note">
          <strong>MoE:</strong> memory follows {formatParams(entry.totalParameters)} total
          parameters — every expert stays resident — while compute follows{' '}
          {formatParams(entry.activeParameters)} active per token. Conflating the two
          overstates FLOPs by{' '}
          {(entry.totalParameters / entry.activeParameters).toFixed(0)}
          {'×'}. Do not 4-bit QLoRA an MoE: bitsandbytes lacks native MoE support and
          quantization interferes with expert routing.
        </p>
      )}

      {!entry.isMoe && entry.totalParameters != null && (
        <p className="ft-note">
          {formatParams(entry.totalParameters)} parameters
          {entry.layers != null && entry.hidden != null
            ? `, L=${entry.layers}, d=${entry.hidden}`
            : ''}
          {entry.dimsVerified
            ? ' — dimensions read from the model config, so adapter counts are exact.'
            : ' — dimensions unverified for the exact adapter formula; the engine falls back and lowers confidence.'}
        </p>
      )}

      {modelId === CUSTOM_MODEL_ID && (
        <div className="scale-inputs">
          <fieldset>
            <legend>Total parameters (B)</legend>
            <input
              type="number"
              min={0.01}
              step={0.1}
              aria-label="Total parameters in billions"
              value={custom.totalParameters / 1e9}
              onChange={(e) => {
                const v = parseFloat(e.target.value)
                if (v > 0) setCustomField('totalParameters', v * 1e9)
              }}
            />
          </fieldset>
          <fieldset>
            <legend>Active parameters (B)</legend>
            <input
              type="number"
              min={0.01}
              step={0.1}
              aria-label="Active parameters in billions"
              value={custom.activeParameters / 1e9}
              onChange={(e) => {
                const v = parseFloat(e.target.value)
                if (v > 0) setCustomField('activeParameters', v * 1e9)
              }}
            />
          </fieldset>
          {(
            [
              ['layers', 'Layers'],
              ['hidden', 'Hidden size'],
              ['ffnDim', 'FFN dim'],
              ['kvHeads', 'KV heads'],
              ['headDim', 'Head dim'],
            ] as const
          ).map(([key, label]) => (
            <fieldset key={key}>
              <legend>{label} (optional)</legend>
              <input
                type="number"
                min={1}
                aria-label={label}
                placeholder="estimated"
                value={custom[key] ?? ''}
                onChange={(e) => {
                  const raw = e.target.value
                  if (raw === '') {
                    setCustomField(key, null)
                    return
                  }
                  const v = parseInt(raw, 10)
                  if (v > 0) setCustomField(key, v)
                }}
              />
            </fieldset>
          ))}
        </div>
      )}
    </div>
  )
}

function GoalSection() {
  const presetId = useFineTuneStore((s) => s.presetId)
  const applyPreset = useFineTuneStore((s) => s.applyPreset)

  return (
    <div className="ft-section">
      <h2>2. What are you teaching it?</h2>
      <div className="use-case-grid">
        {GOAL_PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            className={`use-case-card ${presetId === preset.id ? 'active' : ''}`}
            onClick={() => applyPreset(preset.id)}
          >
            <span className="use-case-label">{preset.label}</span>
            <span className="use-case-desc">{preset.description}</span>
            <span className="use-case-tokens">
              {preset.examples.toLocaleString()} examples · {preset.epochs} epoch
              {preset.epochs === 1 ? '' : 's'} · {METHOD_LABELS[preset.method]}
              {isPeftMethod(preset.method) ? ` r=${preset.loraRank}` : ''}
            </span>
            <span className="ft-preset-basis">{preset.basis}</span>
          </button>
        ))}
      </div>
      <p className="ft-note">
        Presets only fill the fields below — every raw number stays editable, and editing
        one clears the preset.
      </p>
    </div>
  )
}

function WorkloadSection() {
  const workload = useFineTuneStore((s) => s.workload)
  const setWorkloadField = useFineTuneStore((s) => s.setWorkloadField)

  const datasetTokens = workload.examples * workload.avgTokensPerExample

  return (
    <div className="ft-section">
      <h2>3. Workload</h2>
      <div className="scale-inputs">
        <fieldset>
          <legend>Training examples</legend>
          <input
            type="number"
            min={1}
            aria-label="Training examples"
            value={workload.examples}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10)
              if (v > 0) setWorkloadField('examples', v)
            }}
          />
          <div className="presets">
            {[500, 1000, 2000, 10000, 20000, 200000].map((n) => (
              <button
                key={n}
                type="button"
                className={workload.examples === n ? 'active' : ''}
                onClick={() => setWorkloadField('examples', n)}
              >
                {n >= 1000 ? `${n / 1000}K` : n}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>Avg tokens / example</legend>
          <input
            type="number"
            min={1}
            aria-label="Average tokens per example"
            value={workload.avgTokensPerExample}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10)
              if (v > 0) setWorkloadField('avgTokensPerExample', v)
            }}
          />
        </fieldset>
        <fieldset>
          <legend>Epochs</legend>
          <input
            type="number"
            min={1}
            aria-label="Epochs"
            value={workload.epochs}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10)
              if (v > 0) setWorkloadField('epochs', v)
            }}
          />
          <div className="presets">
            {[1, 2, 3, 5].map((n) => (
              <button
                key={n}
                type="button"
                className={workload.epochs === n ? 'active' : ''}
                onClick={() => setWorkloadField('epochs', n)}
              >
                {n}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>
            <Tooltip text="Max sequence length. Activation memory scales linearly with it, and it is the only batch-dependent memory term.">
              Sequence length
            </Tooltip>
          </legend>
          <input
            type="number"
            min={1}
            aria-label="Sequence length"
            value={workload.sequenceLength}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10)
              if (v > 0) setWorkloadField('sequenceLength', v)
            }}
          />
          <div className="presets">
            {[512, 1024, 2048, 4096, 8192, 16384].map((n) => (
              <button
                key={n}
                type="button"
                className={workload.sequenceLength === n ? 'active' : ''}
                onClick={() => setWorkloadField('sequenceLength', n)}
              >
                {n >= 1024 ? `${n / 1024}K` : n}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>Micro-batch size (per GPU)</legend>
          <input
            type="number"
            min={1}
            aria-label="Micro-batch size"
            value={workload.microBatchSize}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10)
              if (v > 0) setWorkloadField('microBatchSize', v)
            }}
          />
        </fieldset>
        <fieldset>
          <legend>
            <Tooltip text="Packing concatenates variable-length samples to eliminate padding: 0.95 efficiency packed against 0.55 unpacked.">
              Sequence packing
            </Tooltip>
          </legend>
          <label className="ft-checkbox">
            <input
              type="checkbox"
              checked={workload.packing}
              onChange={(e) => setWorkloadField('packing', e.target.checked)}
            />
            <span>Pack sequences (0.95 efficiency)</span>
          </label>
        </fieldset>
      </div>
      <p className="ft-note">
        Dataset is {formatTokens(datasetTokens)} tokens before epochs, objective and
        packing are applied.
      </p>
    </div>
  )
}

function ConfigSection() {
  const config = useFineTuneStore((s) => s.config)
  const setConfigField = useFineTuneStore((s) => s.setConfigField)
  const availability = useFineTuneStore((s) => s.availability)
  const setAvailability = useFineTuneStore((s) => s.setAvailability)
  const overheadFactor = useFineTuneStore((s) => s.overheadFactor)
  const setOverheadFactor = useFineTuneStore((s) => s.setOverheadFactor)
  const pricingTier = useFineTuneStore((s) => s.pricingTier)
  const setPricingTier = useFineTuneStore((s) => s.setPricingTier)

  const peft = isPeftMethod(config.method)

  return (
    <div className="ft-section">
      <h2>4. Method &amp; configuration</h2>
      <div className="scale-inputs">
        <fieldset>
          <legend>Method (selects the matrix row)</legend>
          <select
            aria-label="Method"
            value={config.method}
            onChange={(e) => setConfigField('method', e.target.value as FineTuneMethod)}
          >
            {FINE_TUNE_METHODS.map((m) => (
              <option key={m} value={m}>
                {METHOD_LABELS[m]}
              </option>
            ))}
          </select>
        </fieldset>
        <fieldset>
          <legend>Optimizer</legend>
          <select
            aria-label="Optimizer"
            value={config.optimizer}
            onChange={(e) => setConfigField('optimizer', e.target.value as OptimizerKind)}
          >
            {(Object.keys(OPTIMIZER_LABELS) as OptimizerKind[]).map((o) => (
              <option key={o} value={o}>
                {OPTIMIZER_LABELS[o]}
              </option>
            ))}
          </select>
        </fieldset>
        <fieldset>
          <legend>Base weight precision</legend>
          <select
            aria-label="Base weight precision"
            value={config.weightPrecision}
            onChange={(e) =>
              setConfigField('weightPrecision', e.target.value as WeightPrecision)
            }
          >
            {(Object.keys(PRECISION_LABELS) as WeightPrecision[]).map((p) => (
              <option key={p} value={p}>
                {PRECISION_LABELS[p]}
              </option>
            ))}
          </select>
        </fieldset>
        <fieldset>
          <legend>
            <Tooltip text="LoRA rank. Adapter parameters scale linearly with it. Only meaningful for LoRA, QLoRA and DoRA.">
              LoRA rank
            </Tooltip>
          </legend>
          <input
            type="number"
            min={1}
            aria-label="LoRA rank"
            disabled={!peft}
            value={config.loraRank}
            onChange={(e) => {
              const v = parseInt(e.target.value, 10)
              if (v > 0) setConfigField('loraRank', v)
            }}
          />
          <div className="presets">
            {[8, 16, 32, 64, 128].map((n) => (
              <button
                key={n}
                type="button"
                disabled={!peft}
                className={config.loraRank === n ? 'active' : ''}
                onClick={() => setConfigField('loraRank', n)}
              >
                {n}
              </button>
            ))}
          </div>
        </fieldset>
        <fieldset>
          <legend>LoRA targets</legend>
          <select
            aria-label="LoRA targets"
            disabled={!peft}
            value={config.loraTargets}
            onChange={(e) => setConfigField('loraTargets', e.target.value as LoraTargets)}
          >
            {(Object.keys(TARGET_LABELS) as LoraTargets[]).map((t) => (
              <option key={t} value={t}>
                {TARGET_LABELS[t]}
              </option>
            ))}
          </select>
        </fieldset>
        <fieldset>
          <legend>Sharding strategy</legend>
          <select
            aria-label="Sharding strategy"
            value={config.sharding}
            onChange={(e) => setConfigField('sharding', e.target.value as ShardingStrategy)}
          >
            {(Object.keys(SHARDING_LABELS) as ShardingStrategy[]).map((s) => (
              <option key={s} value={s}>
                {SHARDING_LABELS[s]}
              </option>
            ))}
          </select>
        </fieldset>
        <fieldset>
          <legend>
            <Tooltip text="Gradient checkpointing trades +2N FLOPs/token for a large activation-memory saving. On is the standard fine-tuning configuration.">
              Gradient checkpointing
            </Tooltip>
          </legend>
          <label className="ft-checkbox">
            <input
              type="checkbox"
              checked={config.gradientCheckpointing}
              onChange={(e) => setConfigField('gradientCheckpointing', e.target.checked)}
            />
            <span>Recompute activations (+2N FLOPs/token)</span>
          </label>
        </fieldset>
        <fieldset>
          <legend>
            <Tooltip text="DPO and ORPO only. Precomputing reference logprobs offline removes the online reference forward from every step.">
              Precompute reference logprobs
            </Tooltip>
          </legend>
          <label className="ft-checkbox">
            <input
              type="checkbox"
              checked={config.precomputeReferenceLogprobs ?? false}
              onChange={(e) =>
                setConfigField('precomputeReferenceLogprobs', e.target.checked)
              }
            />
            <span>Offline reference pass (DPO / ORPO)</span>
          </label>
        </fieldset>
        <fieldset>
          <legend>Availability</legend>
          <input
            type="number"
            min={0.1}
            max={1}
            step={0.05}
            aria-label="Availability"
            value={availability}
            onChange={(e) => {
              const v = parseFloat(e.target.value)
              if (v > 0 && v <= 1) setAvailability(v)
            }}
          />
        </fieldset>
        <fieldset>
          <legend>Overhead factor</legend>
          <input
            type="number"
            min={1}
            step={0.05}
            aria-label="Overhead factor"
            value={overheadFactor}
            onChange={(e) => {
              const v = parseFloat(e.target.value)
              if (v >= 1) setOverheadFactor(v)
            }}
          />
        </fieldset>
        <fieldset>
          <legend>Pricing tier</legend>
          <div className="presets">
            {(Object.keys(PRICING_TIER_LABELS) as PricingTier[]).map((tier) => (
              <button
                key={tier}
                type="button"
                className={pricingTier === tier ? 'active' : ''}
                onClick={() => setPricingTier(tier)}
              >
                {PRICING_TIER_LABELS[tier]}
              </button>
            ))}
          </div>
        </fieldset>
      </div>
      {!peft && (
        <p className="ft-note">
          LoRA rank and targets are inert for {METHOD_LABELS[config.method]} — it trains
          the full parameter set.
        </p>
      )}
    </div>
  )
}

function GpuColumnSection() {
  const selectedGpuIds = useFineTuneStore((s) => s.selectedGpuIds)
  const toggleGpu = useFineTuneStore((s) => s.toggleGpu)

  return (
    <div className="ft-section">
      <h2>5. GPU columns</h2>
      <div className="gpu-checkboxes">
        {FINE_TUNE_GPUS.map((gpu) => (
          <label key={gpu.id}>
            <input
              type="checkbox"
              aria-label={gpu.label}
              checked={selectedGpuIds.includes(gpu.id)}
              onChange={() => toggleGpu(gpu.id)}
            />
            <span>{gpu.label}</span>
            <span className="chip" style={{ marginLeft: 'auto' }}>
              {gpu.memoryGb}GB
            </span>
          </label>
        ))}
      </div>
    </div>
  )
}

function AssumptionRow() {
  const config = useFineTuneStore((s) => s.config)
  const workload = useFineTuneStore((s) => s.workload)
  const availability = useFineTuneStore((s) => s.availability)
  const overheadFactor = useFineTuneStore((s) => s.overheadFactor)

  const chips = [
    `Method: ${METHOD_LABELS[config.method]}`,
    `Optimizer: ${config.optimizer}`,
    `Weights: ${config.weightPrecision}`,
    `Checkpointing: ${config.gradientCheckpointing ? 'on' : 'off'}`,
    `Sharding: ${config.sharding}`,
    `Packing: ${workload.packing ? '0.95' : '0.55'}`,
    `Availability: ${(availability * 100).toFixed(0)}%`,
    `Overhead: ${overheadFactor}x`,
    'MFU: 0.40 / 0.30 / 0.27 by scale',
    'Cost is near method-invariant',
  ]

  return (
    <div className="assumption-chips" aria-label="Active assumptions">
      {chips.map((chip) => (
        <span key={chip} className="chip">
          {chip}
        </span>
      ))}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Pane
// ---------------------------------------------------------------------------

export function FineTuneCalculator() {
  return (
    <div className="ft-calculator">
      <ModelSection />
      <GoalSection />
      <WorkloadSection />
      <ConfigSection />
      <GpuColumnSection />

      <div className="ft-section">
        <h2>Method {'×'} GPU feasibility</h2>
        <AssumptionRow />
        <p className="ft-finding">
          Read this matrix for <strong>feasibility</strong>, not for the bill. On a fixed
          workload, memory footprint and minimum GPU count vary 20-30{'×'} across
          methods, while cost varies well under 2{'×'}: billed by the GPU-hour, a
          method needing 24{'×'} the GPUs finishes in roughly 1/24 the wall-clock and
          the two cancel. Wall-clock alone therefore <em>inverts</em> the ranking — full
          fine-tuning looks fastest precisely because it is spread over three nodes. The
          method choice is about what hardware you can obtain and how long you will wait.
        </p>
        <MethodMatrix />
      </div>

      <SelectedCellDetail />

      <p className="inference-data-note">
        Model catalog last updated {CATALOG_LAST_UPDATED}; GPU pricing{' '}
        {PRICING_LAST_UPDATED}. Per-GPU rates are an instance price divided by its GPU
        count, cheapest listed provider. Costs exclude data preparation, failed runs,
        evaluation and engineering time.
      </p>
    </div>
  )
}
