import { useState } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { GPU_SKUS } from '../engine/gpu-data'
import { Tooltip } from './Tooltip'

export function AdvancedAssumptions() {
  const [open, setOpen] = useState(false)
  const scenario = useScenarioStore((s) => s.scenario)
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const {
    setMfuForGpu,
    setAvailability,
    setOverheadFactor,
    setMemoryBytesPerParameter,
    setArchitectureFactor,
    setTrainingTokensOverride,
  } = useScenarioStore()

  return (
    <div className="advanced-assumptions">
      <button
        className="advanced-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? '\u25BE' : '\u25B8'} Advanced Assumptions
      </button>
      {open && (
        <div className="advanced-content">
          {scenario.selectedGpuIds.map((gpuId) => {
            const gpu = GPU_SKUS.find((g) => g.id === gpuId)
            const mfu = scenario.mfuByGpuId[gpuId] ?? gpu?.defaultMfu ?? 0.4
            return (
              <div key={gpuId} className="slider-group">
                <label>
                  <span>
                    <Tooltip text="Model FLOP Utilization — fraction of theoretical GPU peak actually achieved. Depends on architecture, parallelism, and software stack.">
                      MFU — {gpu?.label ?? gpuId}
                    </Tooltip>
                  </span>
                  <span>{(mfu * 100).toFixed(0)}%</span>
                </label>
                <input
                  type="range"
                  min={0.10}
                  max={0.70}
                  step={0.01}
                  value={mfu}
                  onChange={(e) => setMfuForGpu(gpuId, parseFloat(e.target.value))}
                  aria-label={`MFU for ${gpu?.label ?? gpuId}`}
                />
                <div className="presets">
                  <button onClick={() => setMfuForGpu(gpuId, 0.30)}>30%</button>
                  <button onClick={() => setMfuForGpu(gpuId, 0.40)}>40%</button>
                  <button onClick={() => setMfuForGpu(gpuId, 0.50)}>50%</button>
                </div>
              </div>
            )
          })}

          <div className="slider-group">
            <label>
              <span>
                <Tooltip text="Fraction of wall-clock time the cluster is actually training (vs failures, restarts, maintenance, queueing).">
                  Availability
                </Tooltip>
              </span>
              <span>{(scenario.availability * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min={0.50}
              max={1.00}
              step={0.01}
              value={scenario.availability}
              onChange={(e) => setAvailability(parseFloat(e.target.value))}
              aria-label="Availability"
            />
          </div>

          <div className="slider-group">
            <label>
              <span>
                <Tooltip text="Extra compute for checkpointing, evaluation, data stalls, and restarts. 1.10 means 10% overhead.">
                  Overhead Factor
                </Tooltip>
              </span>
              <span>{scenario.overheadFactor.toFixed(2)}x</span>
            </label>
            <input
              type="range"
              min={1.00}
              max={1.50}
              step={0.01}
              value={scenario.overheadFactor}
              onChange={(e) => setOverheadFactor(parseFloat(e.target.value))}
              aria-label="Overhead factor"
            />
          </div>

          <fieldset>
            <legend>
              <Tooltip text="GPU memory per parameter for optimizer state (Adam ~16 bytes). Does not include activations.">
                Memory Bytes / Parameter
              </Tooltip>
            </legend>
            <input
              type="number"
              min={1}
              value={scenario.memoryBytesPerParameter}
              onChange={(e) => {
                const val = parseInt(e.target.value)
                if (val > 0) setMemoryBytesPerParameter(val)
              }}
              aria-label="Memory bytes per parameter"
            />
          </fieldset>

          {modelFamily === 'llm' && (
            <>
              <fieldset>
                <legend>
                  <Tooltip text="Multiplier in the compute formula (default 6 for standard dense transformers). Adjust for non-standard architectures.">
                    Architecture Factor
                  </Tooltip>
                </legend>
                <input
                  type="number"
                  min={1}
                  step={1}
                  value={scenario.architectureFactor ?? 6}
                  onChange={(e) => {
                    const val = parseFloat(e.target.value)
                    if (val > 0) setArchitectureFactor(val)
                  }}
                  aria-label="Architecture factor"
                />
              </fieldset>

              <fieldset>
                <legend>
                  <Tooltip text="Override the total training tokens (instead of TPP × N). Accepts scientific notation like 1.4e12. Leave blank to use TPP × N.">
                    Training Tokens Override
                  </Tooltip>
                </legend>
                <input
                  type="text"
                  placeholder="e.g. 1.4e12 (blank = TPP × N)"
                  value={scenario.trainingTokensOverride != null ? String(scenario.trainingTokensOverride) : ''}
                  onChange={(e) => {
                    const raw = e.target.value.trim()
                    if (raw === '') {
                      setTrainingTokensOverride(undefined)
                    } else {
                      const val = Number(raw)
                      if (!isNaN(val) && val > 0) setTrainingTokensOverride(val)
                    }
                  }}
                  aria-label="Training tokens override"
                />
              </fieldset>
            </>
          )}
        </div>
      )}
    </div>
  )
}
