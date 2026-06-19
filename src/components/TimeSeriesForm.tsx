import { useState, useEffect } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { Tooltip } from './Tooltip'
import type { Precision } from '../engine/types'

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

export function TimeSeriesForm() {
  const tsConfig = useScenarioStore((s) => s.tsConfig)
  const scenario = useScenarioStore((s) => s.scenario)
  const setTsField = useScenarioStore((s) => s.setTsField)
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
          <Tooltip text="Total number of parameters in the time-series foundation model.">
            Model Parameters
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.modelParameters}
          min={1}
          onChange={(e) => {
            const val = parseFloat(e.target.value)
            if (val > 0) setTsField('modelParameters', val)
          }}
          aria-label="Model parameters"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Total number of distinct time series in the training dataset.">
            Number of Series
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.numberOfSeries}
          min={1}
          onChange={(e) => {
            const val = parseFloat(e.target.value)
            if (val > 0) setTsField('numberOfSeries', val)
          }}
          aria-label="Number of series"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Average number of timesteps (observations) per series.">
            Avg Timesteps per Series
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.averageTimestepsPerSeries}
          min={1}
          onChange={(e) => {
            const val = parseFloat(e.target.value)
            if (val > 0) setTsField('averageTimestepsPerSeries', val)
          }}
          aria-label="Average timesteps per series"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of variables (channels) measured at each timestep.">
            Variables per Series
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.variablesPerSeries}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTsField('variablesPerSeries', val)
          }}
          aria-label="Variables per series"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of past timesteps the model sees as context for each prediction.">
            Lookback Window
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.lookbackWindow}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTsField('lookbackWindow', val)
          }}
          aria-label="Lookback window"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of future timesteps the model predicts.">
            Forecast Horizon
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.forecastHorizon}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTsField('forecastHorizon', val)
          }}
          aria-label="Forecast horizon"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Step size for sliding the training window across each series. Smaller stride = more windows = more data.">
            Stride
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.stride}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTsField('stride', val)
          }}
          aria-label="Stride"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Number of timesteps grouped into a single patch (token input unit).">
            Patch Size
          </Tooltip>
        </legend>
        <input
          type="number"
          value={tsConfig.patchSize}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTsField('patchSize', val)
          }}
          aria-label="Patch size"
        />
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="How variables map to tokens. Compressed: one token per patch. Expanded: one token per variable per patch.">
            Tokenization Mode
          </Tooltip>
        </legend>
        <select
          value={tsConfig.tokenizationMode}
          onChange={(e) => setTsField('tokenizationMode', e.target.value as 'channel_compressed' | 'channel_expanded' | 'custom')}
          aria-label="Tokenization mode"
        >
          <option value="channel_compressed">Channel compressed</option>
          <option value="channel_expanded">Channel expanded</option>
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
          value={tsConfig.epochs}
          min={1}
          onChange={(e) => {
            const val = parseInt(e.target.value)
            if (val > 0) setTsField('epochs', val)
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
          value={tsConfig.architectureFactor}
          min={1}
          onChange={(e) => {
            const val = parseFloat(e.target.value)
            if (val > 0) setTsField('architectureFactor', val)
          }}
          aria-label="Architecture factor"
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
