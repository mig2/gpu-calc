import { useState } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { GPU_SKUS } from '../engine/gpu-data'
import { calibrateMfu } from '../engine/calibration'
import type { GpuSku, Precision } from '../engine/types'

function ThroughputCalibration() {
  const setClassicalField = useScenarioStore((s) => s.setClassicalField)

  const [calRows, setCalRows] = useState('1000000')
  const [calCols, setCalCols] = useState('100')
  const [calRounds, setCalRounds] = useState('1000')
  const [calSeconds, setCalSeconds] = useState('')
  const [throughputResult, setThroughputResult] = useState<number | null>(null)

  function calibrate() {
    const r = parseFloat(calRows)
    const c = parseFloat(calCols)
    const rounds = parseFloat(calRounds)
    const sec = parseFloat(calSeconds)
    if ([r, c, rounds, sec].some((v) => !(v > 0))) {
      setThroughputResult(null)
      return
    }
    const workUnits = r * c * rounds
    const throughput = workUnits / sec
    setThroughputResult(throughput)
  }

  function applyThroughput() {
    if (throughputResult && throughputResult > 0) {
      setClassicalField('throughputCoefficient', throughputResult)
    }
  }

  return (
    <div className="calibration-content">
      <p className="sensitivity-subtitle">
        Enter a known GBDT training run to back-solve for throughput coefficient.
      </p>

      <fieldset>
        <legend>Rows</legend>
        <input type="text" value={calRows} onChange={(e) => setCalRows(e.target.value)} placeholder="e.g. 1000000" />
      </fieldset>

      <fieldset>
        <legend>Columns</legend>
        <input type="text" value={calCols} onChange={(e) => setCalCols(e.target.value)} placeholder="e.g. 100" />
      </fieldset>

      <fieldset>
        <legend>Boosting Rounds</legend>
        <input type="text" value={calRounds} onChange={(e) => setCalRounds(e.target.value)} placeholder="e.g. 1000" />
      </fieldset>

      <fieldset>
        <legend>Wall-Clock Seconds</legend>
        <input type="text" value={calSeconds} onChange={(e) => setCalSeconds(e.target.value)} placeholder="e.g. 120" />
      </fieldset>

      <button className="apply-mfu-btn" onClick={calibrate} style={{ marginTop: '0.5rem' }}>
        Calculate throughput
      </button>

      {throughputResult !== null && (
        <div className="calibration-result">
          <div className="result-gpu-count">
            <span className="big-number">
              {throughputResult.toExponential(2)}
            </span>
            <span className="label">work units/sec</span>
          </div>
          <button className="apply-mfu-btn" onClick={applyThroughput}>
            Apply throughput coefficient
          </button>
        </div>
      )}
    </div>
  )
}

function MfuCalibration() {
  const setMfuForGpu = useScenarioStore((s) => s.setMfuForGpu)
  const customGpus = useScenarioStore((s) => s.customGpus)
  const allGpus: GpuSku[] = [...GPU_SKUS, ...customGpus]

  const [modelParams, setModelParams] = useState('70e9')
  const [tokens, setTokens] = useState('1.4e12')
  const [days, setDays] = useState('30')
  const [gpuCount, setGpuCount] = useState('701')
  const [gpuId, setGpuId] = useState('h100-sxm')
  const [precision, setPrecision] = useState<Precision>('BF16_DENSE')
  const [overhead, setOverhead] = useState('1.10')
  const [availability, setAvailability] = useState('0.90')

  const gpu = allGpus.find((g) => g.id === gpuId)

  function getResult() {
    if (!gpu) return null
    const params = parseFloat(modelParams)
    const tok = parseFloat(tokens)
    const d = parseFloat(days)
    const gc = parseInt(gpuCount)
    const oh = parseFloat(overhead)
    const av = parseFloat(availability)
    if ([params, tok, d, gc, oh, av].some((v) => !(v > 0))) return null

    return calibrateMfu({
      modelParameters: params,
      tokensTrainedOn: tok,
      wallClockDays: d,
      gpuCount: gc,
      gpu,
      precision,
      overheadFactor: oh,
      availability: av,
    })
  }

  const result = getResult()

  function applyMfu() {
    if (result && gpu) {
      setMfuForGpu(gpu.id, result.achievedMfu)
    }
  }

  return (
    <div className="calibration-content">
      <p className="sensitivity-subtitle">
        Enter a known training run to back-solve for achieved MFU.
      </p>

      <fieldset>
        <legend>Model Parameters</legend>
        <input type="text" value={modelParams} onChange={(e) => setModelParams(e.target.value)} placeholder="e.g. 70e9" />
      </fieldset>

      <fieldset>
        <legend>Tokens Trained On</legend>
        <input type="text" value={tokens} onChange={(e) => setTokens(e.target.value)} placeholder="e.g. 1.4e12" />
      </fieldset>

      <fieldset>
        <legend>Wall-Clock Days</legend>
        <input type="number" value={days} onChange={(e) => setDays(e.target.value)} min="0.1" step="0.1" />
      </fieldset>

      <fieldset>
        <legend>GPU Count</legend>
        <input type="number" value={gpuCount} onChange={(e) => setGpuCount(e.target.value)} min="1" />
      </fieldset>

      <fieldset>
        <legend>GPU SKU</legend>
        <select value={gpuId} onChange={(e) => setGpuId(e.target.value)}>
          {allGpus.map((g) => (
            <option key={g.id} value={g.id}>{g.label}</option>
          ))}
        </select>
      </fieldset>

      <fieldset>
        <legend>Precision</legend>
        <select value={precision} onChange={(e) => setPrecision(e.target.value as Precision)}>
          <option value="BF16_DENSE">BF16 Dense</option>
          <option value="FP8_DENSE">FP8 Dense</option>
        </select>
      </fieldset>

      <fieldset>
        <legend>Overhead Factor</legend>
        <input type="number" value={overhead} onChange={(e) => setOverhead(e.target.value)} min="1" step="0.01" />
      </fieldset>

      <fieldset>
        <legend>Availability</legend>
        <input type="number" value={availability} onChange={(e) => setAvailability(e.target.value)} min="0.1" max="1" step="0.01" />
      </fieldset>

      {result && (
        <div className="calibration-result">
          <div className="result-gpu-count">
            <span className={`big-number ${!result.isReasonable ? 'unreasonable' : ''}`}>
              {result.achievedMfuPercent}
            </span>
            <span className="label">achieved MFU</span>
          </div>
          {!result.isReasonable && (
            <p className="warning">
              This MFU value is outside the typical range (10-70%). Check your inputs.
            </p>
          )}
          <button className="apply-mfu-btn" onClick={applyMfu}>
            Apply {result.achievedMfuPercent} MFU to {gpu?.label}
          </button>
          <details className="trace-details">
            <summary>Calculation trace</summary>
            <pre className="trace-pre">{result.trace.join('\n')}</pre>
          </details>
        </div>
      )}
    </div>
  )
}

export function CalibrationMode() {
  const [open, setOpen] = useState(false)
  const modelFamily = useScenarioStore((s) => s.modelFamily)

  const isClassical = modelFamily === 'classical_tabular'

  return (
    <div className="calibration-section">
      <button
        className="advanced-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? '\u25BE' : '\u25B8'} Calibration Mode
      </button>
      {open && (
        isClassical ? <ThroughputCalibration /> : <MfuCalibration />
      )}
    </div>
  )
}
