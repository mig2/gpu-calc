import { useState, useEffect } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import type { TrainingMode } from '../engine/types'
import { Tooltip } from './Tooltip'

const MODEL_PRESETS = [
  { label: '7B', value: 7e9 },
  { label: '13B', value: 13e9 },
  { label: '34B', value: 34e9 },
  { label: '70B', value: 70e9 },
  { label: '130B', value: 130e9 },
  { label: '405B', value: 405e9 },
]

const WINDOW_PRESETS = [
  { label: '7 days', value: 7 * 86_400 },
  { label: '14 days', value: 14 * 86_400 },
  { label: '30 days', value: 30 * 86_400 },
  { label: '60 days', value: 60 * 86_400 },
]

const TPP_PRESETS = [
  { label: '20 (Chinchilla)', value: 20 },
  { label: '50', value: 50 },
  { label: '100', value: 100 },
]

const TRAINING_MODES: { label: string; value: TrainingMode }[] = [
  { label: 'Full pretraining', value: 'FULL_PRETRAINING' },
  { label: 'Continued pretraining', value: 'CONTINUED_PRETRAINING' },
  { label: 'SFT', value: 'SFT' },
  { label: 'LoRA', value: 'LORA' },
  { label: 'RLHF', value: 'RLHF' },
  { label: 'Distillation', value: 'DISTILLATION' },
]

function parseModelSize(input: string): number | null {
  const match = input.trim().match(/^(\d+(?:\.\d+)?)\s*([MBTmbt])?$/i)
  if (!match) return null
  const num = parseFloat(match[1])
  if (num <= 0) return null
  const unit = (match[2] || 'B').toUpperCase()
  switch (unit) {
    case 'M': return num * 1e6
    case 'B': return num * 1e9
    case 'T': return num * 1e12
    default: return null
  }
}

function formatModelSize(params: number): string {
  if (params >= 1e12) return `${params / 1e12}T`
  if (params >= 1e9) return `${params / 1e9}B`
  return `${params / 1e6}M`
}

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

export function LlmForm() {
  const scenario = useScenarioStore((s) => s.scenario)
  const {
    setModelParameters,
    setTokensPerParameter,
    setTrainingWindowSeconds,
    setTrainingMode,
    setPrecision,
  } = useScenarioStore()

  const [modelInput, setModelInput] = useState(formatModelSize(scenario.modelParameters))
  const [windowInput, setWindowInput] = useState(formatWindow(scenario.trainingWindowSeconds))
  const [tppInput, setTppInput] = useState(String(scenario.tokensPerParameter))
  const [modelError, setModelError] = useState('')
  const [windowError, setWindowError] = useState('')

  useEffect(() => {
    const timer = setTimeout(() => {
      const parsed = parseModelSize(modelInput)
      if (parsed) {
        setModelError('')
        setModelParameters(parsed)
      } else if (modelInput.trim()) {
        setModelError('Enter a number with M, B, or T (e.g. 70B)')
      }
    }, 400)
    return () => clearTimeout(timer)
  }, [modelInput, setModelParameters])

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
          <Tooltip text="Total number of model parameters. The calculator uses this with tokens-per-parameter to determine training compute.">
            Model Size
          </Tooltip>
        </legend>
        <div className="input-with-presets">
          <input
            type="text"
            value={modelInput}
            onChange={(e) => setModelInput(e.target.value)}
            aria-label="Model parameters"
            placeholder="e.g. 70B"
          />
          {modelError && <span className="input-error">{modelError}</span>}
          <div className="presets">
            {MODEL_PRESETS.map((p) => (
              <button
                key={p.label}
                className={scenario.modelParameters === p.value ? 'active' : ''}
                onClick={() => {
                  setModelInput(p.label)
                  setModelParameters(p.value)
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Wall-clock time available for training. Shorter windows require more GPUs.">
            Training Window
          </Tooltip>
        </legend>
        <div className="input-with-presets">
          <input
            type="text"
            value={windowInput}
            onChange={(e) => setWindowInput(e.target.value)}
            aria-label="Training window"
            placeholder="e.g. 30d"
          />
          {windowError && <span className="input-error">{windowError}</span>}
          <div className="presets">
            {WINDOW_PRESETS.map((p) => (
              <button
                key={p.label}
                className={scenario.trainingWindowSeconds === p.value ? 'active' : ''}
                onClick={() => {
                  setWindowInput(formatWindow(p.value))
                  setTrainingWindowSeconds(p.value)
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="How many tokens to train on per parameter. Chinchilla-optimal is ~20. Modern recipes often use 50-100+ for inference efficiency.">
            Tokens per Parameter
          </Tooltip>
        </legend>
        <div className="input-with-presets">
          <input
            type="number"
            value={tppInput}
            min={1}
            onChange={(e) => {
              setTppInput(e.target.value)
              const val = parseFloat(e.target.value)
              if (val > 0) setTokensPerParameter(val)
            }}
            aria-label="Tokens per parameter"
          />
          <div className="presets">
            {TPP_PRESETS.map((p) => (
              <button
                key={p.label}
                className={scenario.tokensPerParameter === p.value ? 'active' : ''}
                onClick={() => {
                  setTppInput(String(p.value))
                  setTokensPerParameter(p.value)
                }}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Type of training run. The 6ND formula is calibrated for full pretraining; other modes may need less compute.">
            Training Mode
          </Tooltip>
        </legend>
        <select
          value={scenario.trainingMode}
          onChange={(e) => setTrainingMode(e.target.value as TrainingMode)}
          aria-label="Training mode"
        >
          {TRAINING_MODES.map((m) => (
            <option key={m.value} value={m.value}>{m.label}</option>
          ))}
        </select>
      </fieldset>

      <fieldset>
        <legend>
          <Tooltip text="Numerical precision for training. BF16 is standard. FP8 is experimental and may not deliver theoretical peak.">
            Precision
          </Tooltip>
        </legend>
        <select
          value={scenario.precision}
          onChange={(e) => setPrecision(e.target.value as 'BF16_DENSE' | 'FP8_DENSE')}
          aria-label="Precision"
        >
          <option value="BF16_DENSE">BF16 Dense</option>
          <option value="FP8_DENSE">FP8 Dense (experimental)</option>
        </select>
      </fieldset>
    </>
  )
}
