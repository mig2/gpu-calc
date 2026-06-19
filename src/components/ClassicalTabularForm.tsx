import { useState, useEffect } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { Tooltip } from './Tooltip'
import type { GbdtAlgorithm } from '../engine/types'

const ROW_PRESETS = [
  { label: '100K', value: 100e3 },
  { label: '1M', value: 1e6 },
  { label: '10M', value: 10e6 },
  { label: '100M', value: 100e6 },
]

const ALGORITHMS: { label: string; value: GbdtAlgorithm }[] = [
  { label: 'LightGBM', value: 'lightgbm' },
  { label: 'XGBoost', value: 'xgboost' },
  { label: 'CatBoost', value: 'catboost' },
  { label: 'Random Forest', value: 'random_forest' },
  { label: 'Custom', value: 'custom' },
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

export function ClassicalTabularForm() {
  const classicalConfig = useScenarioStore((s) => s.classicalConfig)
  const scenario = useScenarioStore((s) => s.scenario)
  const setClassicalField = useScenarioStore((s) => s.setClassicalField)
  const setTrainingWindowSeconds = useScenarioStore((s) => s.setTrainingWindowSeconds)

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
          <Tooltip text="Gradient boosted decision tree algorithm. Different implementations have different GPU throughput characteristics.">
            Algorithm
          </Tooltip>
        </legend>
        <select
          value={classicalConfig.algorithm}
          onChange={(e) => setClassicalField('algorithm', e.target.value as GbdtAlgorithm)}
          aria-label="Algorithm"
        >
          {ALGORITHMS.map((a) => (
            <option key={a.value} value={a.value}>{a.label}</option>
          ))}
        </select>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of rows (observations) in the training dataset.">
            Rows
          </Tooltip>
        </legend>
        <div className="input-with-presets">
          <input
            type="number"
            value={classicalConfig.rows}
            min={1}
            onChange={(e) => {
              const val = parseFloat(e.target.value)
              if (val > 0) setClassicalField('rows', val)
            }}
            aria-label="Rows"
          />
          <div className="presets">
            {ROW_PRESETS.map((p) => (
              <button
                key={p.label}
                className={classicalConfig.rows === p.value ? 'active' : ''}
                onClick={() => setClassicalField('rows', p.value)}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of columns (features) in the training dataset.">
            Columns
          </Tooltip>
        </legend>
        <input
          type="number"
          value={classicalConfig.columns}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setClassicalField('columns', val)
          }}
          aria-label="Columns"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of boosting iterations (trees). More rounds = better fit but longer training.">
            Boosting Rounds
          </Tooltip>
        </legend>
        <input
          type="number"
          value={classicalConfig.boostingRounds}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setClassicalField('boostingRounds', val)
          }}
          aria-label="Boosting rounds"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Maximum depth of each tree. Deeper trees capture more complex patterns but risk overfitting.">
            Max Depth
          </Tooltip>
        </legend>
        <input
          type="number"
          value={classicalConfig.maxDepth}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setClassicalField('maxDepth', val)
          }}
          aria-label="Max depth"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of cross-validation folds. Each fold retrains the model, multiplying total work.">
            CV Folds
          </Tooltip>
        </legend>
        <input
          type="number"
          value={classicalConfig.cvFolds}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setClassicalField('cvFolds', val)
          }}
          aria-label="CV folds"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of hyperparameter search trials. Each trial runs a full CV loop, multiplying total work.">
            HP Trials
          </Tooltip>
        </legend>
        <input
          type="number"
          value={classicalConfig.hyperparameterTrials}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setClassicalField('hyperparameterTrials', val)
          }}
          aria-label="HP trials"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Whether training runs on CPU or GPU. GPU is typically 5-10x faster for histogram-based GBDTs.">
            Implementation
          </Tooltip>
        </legend>
        <select
          value={classicalConfig.cpuOrGpu}
          onChange={(e) => setClassicalField('cpuOrGpu', e.target.value as 'cpu' | 'gpu')}
          aria-label="CPU or GPU"
        >
          <option value="gpu">GPU</option>
          <option value="cpu">CPU</option>
        </select>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Empirical throughput in work units per second. Set to 0 to use benchmark defaults. Calibrate from a known run for better accuracy.">
            Throughput Coefficient
          </Tooltip>
        </legend>
        <input
          type="number"
          value={classicalConfig.throughputCoefficient}
          min={0}
          onChange={(e) => {
            const val = parseFloat(e.target.value)
            if (val >= 0) setClassicalField('throughputCoefficient', val)
          }}
          aria-label="Throughput coefficient"
          placeholder="0 = use defaults"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Wall-clock time budget. For classical models, this determines how many parallel workers are needed.">
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
    </>
  )
}
