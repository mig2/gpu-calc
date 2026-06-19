import { useState } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import { GPU_SKUS } from '../engine/gpu-data'

export function AdvancedAssumptions() {
  const [open, setOpen] = useState(false)
  const scenario = useScenarioStore((s) => s.scenario)
  const {
    setMfuForGpu,
    setAvailability,
    setOverheadFactor,
    setMemoryBytesPerParameter,
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
                  <span>MFU — {gpu?.label ?? gpuId}</span>
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
              <span>Availability</span>
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
              <span>Overhead Factor</span>
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
            <legend>Memory Bytes / Parameter</legend>
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
        </div>
      )}
    </div>
  )
}
