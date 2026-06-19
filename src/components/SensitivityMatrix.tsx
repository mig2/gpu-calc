import { useScenarioStore } from '../store/scenario-store'
import { getGpuById, getH100Reference } from '../engine/gpu-data'
import { estimateTrainingRun } from '../engine/calculator'

const MFU_VALUES = [0.30, 0.35, 0.40, 0.45, 0.50]
const WINDOW_DAYS = [7, 14, 30, 60]

export function SensitivityMatrix() {
  const scenario = useScenarioStore((s) => s.scenario)
  const primaryGpuId = scenario.selectedGpuIds[0]
  const gpu = getGpuById(primaryGpuId)
  const h100 = getH100Reference()

  if (!gpu) return null

  // Compute grid
  const grid = MFU_VALUES.map((mfu) =>
    WINDOW_DAYS.map((days) => {
      const tweaked = {
        ...scenario,
        mfuByGpuId: { ...scenario.mfuByGpuId, [primaryGpuId]: mfu },
        trainingWindowSeconds: days * 86_400,
      }
      return estimateTrainingRun(tweaked, gpu, h100).requiredGpus
    }),
  )

  // Find min/max for heatmap coloring
  const allValues = grid.flat()
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)

  function heatColor(val: number): string {
    if (maxVal === minVal) return 'var(--surface)'
    const ratio = (val - minVal) / (maxVal - minVal)
    // Green (low) to red (high)
    const r = Math.round(34 + ratio * 205)
    const g = Math.round(197 - ratio * 150)
    const b = Math.round(94 - ratio * 26)
    return `rgb(${r}, ${g}, ${b})`
  }

  return (
    <div className="sensitivity-section">
      <h3>Sensitivity Matrix — {gpu.label}</h3>
      <p className="sensitivity-subtitle">Required GPUs by MFU and training window</p>
      <div className="sensitivity-table-wrapper">
        <table className="sensitivity-table">
          <thead>
            <tr>
              <th>MFU</th>
              {WINDOW_DAYS.map((d) => (
                <th key={d}>{d}d</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MFU_VALUES.map((mfu, ri) => (
              <tr key={mfu}>
                <td className="mfu-label">{(mfu * 100).toFixed(0)}%</td>
                {WINDOW_DAYS.map((_, ci) => {
                  const val = grid[ri][ci]
                  return (
                    <td
                      key={ci}
                      className="heat-cell"
                      style={{ backgroundColor: heatColor(val), color: '#fff' }}
                    >
                      {val.toLocaleString()}
                    </td>
                  )
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
