import { useState, useEffect } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import type { Precision, TrainingMode } from '../engine/types'
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

/**
 * Tokens per parameter, keyed to published practice rather than round numbers.
 * Sourced in docs/research/2026-10-14-llm-training-research.md section 1.5;
 * the TPP column of section 1.2 is the evidence behind each.
 */
const TPP_PRESETS = [
  { label: '20 (Chinchilla)', value: 20, title: 'Compute-optimal: minimises loss for a fixed TRAINING budget (Hoffmann et al. 2022). Still the right answer to that question, but no 2025-26 production run trains here.' },
  { label: '26 (replicated)', value: 26, title: "Epoch AI's re-fit of Chinchilla puts the optimum at 25.6. DeepMind's own parametric form reads as high as ~70; the three methods in the original paper disagree." },
  { label: '150 (frontier dense)', value: 150, title: '2026 practice. Median of Llama 3.1 70B (223), OLMo 3 32B (172), Marin 32B (201) and Qwen3-235B on total params (153).' },
  { label: '400 (MoE active)', value: 400, title: 'Sparse models on ACTIVE parameters: DeepSeek-V3 is 400, GLM-5 is 712 — while sitting at 22-38 on total params. Use with the active count.' },
  { label: '1000 (small, served)', value: 1000, title: 'Inference-optimal for a small model serving high volume. Llama 3.1 8B is 1,950; Marin 8B 1,588. Loss keeps falling to ~10,000 TPP with sharply diminishing returns.' },
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
          <Tooltip text="Tokens of training data per model parameter. Chinchilla's 20 minimises loss for a fixed TRAINING budget; once a model will be served, lifetime compute (train + serve) optimises differently and favours a smaller model trained longer. Published 2026 practice: frontier dense models 150-250, frontier MoE 20-40 on total params (but 400-700 on active), small served models 1,000+. FLOPs scale linearly with this, so it moves the answer directly.">
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
          <Tooltip text="Numerical precision for training. BF16 is standard. FP8 and FP4 swap in a lower-precision peak but are penalised by an MFU factor (FP8 0.70x, FP4 0.50x) because real recipes quantize only transformer-block linears.">
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
          <option value="FP4_DENSE">FP4 Dense (Blackwell/MI355X only)</option>
        </select>
      </fieldset>
    </>
  )
}
