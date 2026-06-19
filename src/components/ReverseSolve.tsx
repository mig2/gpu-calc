import { useState } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { getGpuById } from '../engine/gpu-data'
import { solveForTrainingTime, solveForMaxModelSize } from '../engine/reverse-solve'

type SolveMode = 'time' | 'size'

export function ReverseSolve() {
  const [open, setOpen] = useState(false)
  const [mode, setMode] = useState<SolveMode>('time')
  const [gpuCount, setGpuCount] = useState(256)
  const scenario = useScenarioStore((s) => s.scenario)
  const primaryGpuId = scenario.selectedGpuIds[0]
  const gpu = getGpuById(primaryGpuId)

  if (!gpu) return null

  const timeResult = mode === 'time' ? solveForTrainingTime(scenario, gpu, gpuCount) : null
  const sizeResult = mode === 'size' ? solveForMaxModelSize(scenario, gpu, gpuCount) : null

  return (
    <div className="reverse-solve-section">
      <button
        className="advanced-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? '\u25BE' : '\u25B8'} Reverse Solve
      </button>
      {open && (
        <div className="reverse-solve-content">
          <p className="sensitivity-subtitle">Given a fixed GPU budget, find training time or max model size.</p>

          <fieldset>
            <legend>Solve For</legend>
            <div className="presets">
              <button className={mode === 'time' ? 'active' : ''} onClick={() => setMode('time')}>
                Training Time
              </button>
              <button className={mode === 'size' ? 'active' : ''} onClick={() => setMode('size')}>
                Max Model Size
              </button>
            </div>
          </fieldset>

          <fieldset>
            <legend>Available GPUs ({gpu.label})</legend>
            <input
              type="number"
              min={1}
              value={gpuCount}
              onChange={(e) => {
                const val = parseInt(e.target.value)
                if (val > 0) setGpuCount(val)
              }}
              aria-label="GPU count for reverse solve"
            />
            <div className="presets">
              {[64, 128, 256, 512, 1024].map((n) => (
                <button key={n} className={gpuCount === n ? 'active' : ''} onClick={() => setGpuCount(n)}>
                  {n}
                </button>
              ))}
            </div>
          </fieldset>

          {timeResult && (
            <div className="reverse-result">
              <div className="result-gpu-count">
                <span className="big-number">{timeResult.trainingDays.toFixed(1)}</span>
                <span className="label">days to train</span>
              </div>
              <details className="trace-details">
                <summary>Calculation trace</summary>
                <pre className="trace-pre">{timeResult.trace.join('\n')}</pre>
              </details>
            </div>
          )}

          {sizeResult && (
            <div className="reverse-result">
              <div className="result-gpu-count">
                <span className="big-number">{sizeResult.maxParametersLabel}</span>
                <span className="label">max model size</span>
              </div>
              <details className="trace-details">
                <summary>Calculation trace</summary>
                <pre className="trace-pre">{sizeResult.trace.join('\n')}</pre>
              </details>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
