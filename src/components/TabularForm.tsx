import { useState, useEffect } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { Tooltip } from './Tooltip'
import type { Precision, TabularTokenizationMode } from '../engine/types'

const MODEL_PRESETS = [
  { label: '50M', value: 50e6 },
  { label: '100M', value: 100e6 },
  { label: '500M', value: 500e6 },
  { label: '1B', value: 1e9 },
]

const TASK_PRESETS = [
  { label: '100K', value: 100e3 },
  { label: '1M', value: 1e6 },
  { label: '10M', value: 10e6 },
]

function parseWindow(input: string): number | null {
  const match = input.trim().match(/^(\d+(?:\.\d+)?)\s*(h|hours?|d|days?|w|weeks?)?$/i)
  if (!match) return null
  const num = parseFloat(match[1])
  if (num <= 0) return null
  const unit = (match[2] || 'd')[0].toLowerCase()
  switch (unit) {
    case 'h': return num * 3600
    case 'd': return num * 86_400
    case 'w': return num * 7 * 86_400
    default: return num * 86_400
  }
}

function formatWindow(seconds: number): string {
  const days = seconds / 86_400
  if (days >= 7 && days % 7 === 0) return `${days / 7}w`
  if (Number.isInteger(days)) return `${days}d`
  const hours = seconds / 3600
  return `${hours}h`
}

export function TabularForm() {
  const tabConfig = useScenarioStore((s) => s.tabConfig)
  const scenario = useScenarioStore((s) => s.scenario)
  const setTabField = useScenarioStore((s) => s.setTabField)
  const setTrainingWindowSeconds = useScenarioStore((s) => s.setTrainingWindowSeconds)
  const setPrecision = useScenarioStore((s) => s.setPrecision)

  const [windowInput, setWindowInput] = useState(formatWindow(scenario.trainingWindowSeconds))
  const [windowError, setWindowError] = useState('')

  useEffect(() => {
    const timer = setTimeout(() => {
      const parsed = parseWindow(windowInput)
      if (parsed) {
        setWindowError('')
        setTrainingWindowSeconds(parsed)
      } else if (windowInput.trim()) {
        setWindowError('Enter a number with h, d, or w (e.g. 30d)')
      }
    }, 400)
    return () => clearTimeout(timer)
  }, [windowInput, setTrainingWindowSeconds])

  return (
    <>
      <fieldset>
        <legend>
          <Tooltip text="Total number of parameters in the tabular foundation model.">
            Model Parameters
          </Tooltip>
        </legend>
        <div className="input-with-presets">
          <input
            type="number"
            value={tabConfig.modelParameters}
            min={1}
            onChange={(e) => {
              const val = parseFloat(e.target.value)
              if (val > 0) setTabField('modelParameters', val)
            }}
            aria-label="Model parameters"
          />
          <div className="presets">
            {MODEL_PRESETS.map((p) => (
              <button
                key={p.label}
                className={tabConfig.modelParameters === p.value ? 'active' : ''}
                onClick={() => setTabField('modelParameters', p.value)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Total number of synthetic or real tabular tasks used for pretraining.">
            Pretraining Tasks
          </Tooltip>
        </legend>
        <div className="input-with-presets">
          <input
            type="number"
            value={tabConfig.numberOfPretrainingTasks}
            min={1}
            onChange={(e) => {
              const val = parseFloat(e.target.value)
              if (val > 0) setTabField('numberOfPretrainingTasks', val)
            }}
            aria-label="Pretraining tasks"
          />
          <div className="presets">
            {TASK_PRESETS.map((p) => (
              <button
                key={p.label}
                className={tabConfig.numberOfPretrainingTasks === p.value ? 'active' : ''}
                onClick={() => setTabField('numberOfPretrainingTasks', p.value)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of rows (observations) per tabular task.">
            Rows per Task
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tabConfig.rowsPerTask}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTabField('rowsPerTask', val)
          }}
          aria-label="Rows per task"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of columns (features) per tabular task.">
            Columns per Task
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tabConfig.columnsPerTask}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTabField('columnsPerTask', val)
          }}
          aria-label="Columns per task"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="How tabular data maps to tokens. Row: one token per row. Cell: one token per cell. Axial: rows + columns (additive). Custom: specify directly.">
            Tokenization Mode
          </Tooltip>
        </legend>
        <select
          value={tabConfig.tokenizationMode}
          onChange={(e) => setTabField('tokenizationMode', e.target.value as TabularTokenizationMode)}
          aria-label="Tokenization mode"
        >
          <option value="row">Row (1 token per row)</option>
          <option value="cell">Cell (1 token per cell)</option>
          <option value="axial">Axial (rows + columns)</option>
          <option value="custom">Custom</option>
        </select>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of complete passes over the training data.">
            Epochs
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tabConfig.epochs}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTabField('epochs', val)
          }}
          aria-label="Epochs"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Multiplier for the FLOPs-per-token formula. 6 is standard for decoder transformers.">
            Architecture Factor
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tabConfig.architectureFactor}
          min={1}
          onChange={(e) => {
            const val = parseFloat(e.target.value)
            if (val > 0) setTabField('architectureFactor', val)
          }}
          aria-label="Architecture factor"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Multiplier for test-time compute (e.g. in-context learning iterations). 1 = no extra compute.">
            Test-Time Compute Multiplier
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tabConfig.testTimeComputeMultiplier}
          min={1}
          onChange={(e) => {
            const val = parseFloat(e.target.value)
            if (val >= 1) setTabField('testTimeComputeMultiplier', val)
          }}
          aria-label="Test-time compute multiplier"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Wall-clock time available for training. Shorter windows require more GPUs.">
            Training Window
          </Tooltip>
        </legend>
        <input
          type="text"
          value={windowInput}
          onChange={(e) => setWindowInput(e.target.value)}
          aria-label="Training window"
          placeholder="e.g. 30d"
        />
        {windowError && <span className="input-error">{windowError}</span>}
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Numerical precision for training. BF16 is standard. FP8 is experimental.">
            Precision
          </Tooltip>
        </legend>
        <select
          value={scenario.precision}
          onChange={(e) => setPrecision(e.target.value as Precision)}
          aria-label="Precision"
        >
          <option value="BF16_DENSE">BF16 Dense</option>
          <option value="FP8_DENSE">FP8 Dense (experimental)</option>
        </select>
      </fieldset>
    </>
  )
}
