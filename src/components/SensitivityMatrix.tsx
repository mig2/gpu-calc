import { useScenarioStore } from '../store/scenario-store'
import { getGpuById, getH100Reference } from '../engine/gpu-data'
import { estimateTrainingRun } from '../engine/calculator'
import { computeTimeSeriesFlops } from '../engine/adapters/time-series-adapter'
import { computeTabularFlops } from '../engine/adapters/tabular-adapter'
import { estimateClassicalTabular } from '../engine/adapters/classical-tabular-adapter'
import { estimateHardware } from '../engine/hardware-estimator'

const MFU_VALUES = [0.30, 0.35, 0.40, 0.45, 0.50]
const WINDOW_DAYS = [7, 14, 30, 60]

const LOOKBACK_VALUES = [64, 128, 256, 512]
const PATCH_SIZES = [8, 16, 32, 64]

const TAB_ROWS = [256, 512, 1024, 2048]
const TAB_COLUMNS = [20, 50, 100, 200]

const CLASSICAL_ROWS = [100e3, 1e6, 10e6, 100e6]
const CLASSICAL_ROUNDS = [100, 500, 1000, 5000]

function heatColor(val: number, minVal: number, maxVal: number): string {
  if (maxVal === minVal) return 'var(--surface)'
  const ratio = (val - minVal) / (maxVal - minVal)
  const r = Math.round(34 + ratio * 205)
  const g = Math.round(197 - ratio * 150)
  const b = Math.round(94 - ratio * 26)
  return `rgb(${r}, ${g}, ${b})`
}

export function SensitivityMatrix() {
  const scenario = useScenarioStore((s) => s.scenario)
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const tsConfig = useScenarioStore((s) => s.tsConfig)
  const tabConfig = useScenarioStore((s) => s.tabConfig)
  const classicalConfig = useScenarioStore((s) => s.classicalConfig)
  const primaryGpuId = scenario.selectedGpuIds[0]
  const gpu = getGpuById(primaryGpuId)
  const h100 = getH100Reference()

  if (!gpu) return null

  if (modelFamily === 'classical_tabular') {
    const grid = CLASSICAL_ROWS.map((rows) =>
      CLASSICAL_ROUNDS.map((rounds) => {
        const classicalScenario = {
          modelFamily: 'classical_tabular' as const,
          ...classicalConfig,
          rows,
          boostingRounds: rounds,
          trainingWindowSeconds: scenario.trainingWindowSeconds,
          precision: scenario.precision,
          selectedGpuIds: scenario.selectedGpuIds,
          mfuByGpuId: scenario.mfuByGpuId,
          availability: scenario.availability,
          overheadFactor: scenario.overheadFactor,
        }
        const result = estimateClassicalTabular(classicalScenario)
        return Math.max(1, Math.ceil(result.estimatedSeconds / scenario.trainingWindowSeconds))
      }),
    )

    const allValues = grid.flat()
    const minVal = Math.min(...allValues)
    const maxVal = Math.max(...allValues)

    return (
      <div className="sensitivity-section">
        <h3>Sensitivity Matrix — {gpu.label}</h3>
        <p className="sensitivity-subtitle">Required parallel workers by rows and boosting rounds</p>
        <div className="sensitivity-table-wrapper">
          <table className="sensitivity-table">
            <thead>
              <tr>
                <th>Rows</th>
                {CLASSICAL_ROUNDS.map((r) => (
                  <th key={r}>{r.toLocaleString()} rounds</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {CLASSICAL_ROWS.map((rows, ri) => (
                <tr key={rows}>
                  <td className="mfu-label">{rows >= 1e6 ? `${rows / 1e6}M` : `${rows / 1e3}K`}</td>
                  {CLASSICAL_ROUNDS.map((_, ci) => {
                    const val = grid[ri][ci]
                    return (
                      <td
                        key={ci}
                        className="heat-cell"
                        style={{ backgroundColor: heatColor(val, minVal, maxVal), color: '#fff' }}
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

  if (modelFamily === 'tabular_foundation') {
    const grid = TAB_ROWS.map((rows) =>
      TAB_COLUMNS.map((cols) => {
        const tabScenario = {
          modelFamily: 'tabular_foundation' as const,
          ...tabConfig,
          rowsPerTask: rows,
          columnsPerTask: cols,
          trainingWindowSeconds: scenario.trainingWindowSeconds,
          precision: scenario.precision,
          selectedGpuIds: scenario.selectedGpuIds,
          mfuByGpuId: scenario.mfuByGpuId,
          availability: scenario.availability,
          overheadFactor: scenario.overheadFactor,
        }

        const adapterResult = computeTabularFlops(tabScenario)
        const hwResult = estimateHardware(
          {
            totalFlops: adapterResult.totalFlops,
            modelParameters: tabConfig.modelParameters,
            memoryBytesPerParameter: tabConfig.memoryBytesPerParameter,
            hardware: {
              trainingWindowSeconds: scenario.trainingWindowSeconds,
              precision: scenario.precision,
              selectedGpuIds: scenario.selectedGpuIds,
              mfuByGpuId: scenario.mfuByGpuId,
              availability: scenario.availability,
              overheadFactor: scenario.overheadFactor,
            },
          },
          gpu,
          h100,
        )
        return hwResult.requiredGpus
      }),
    )

    const allValues = grid.flat()
    const minVal = Math.min(...allValues)
    const maxVal = Math.max(...allValues)

    return (
      <div className="sensitivity-section">
        <h3>Sensitivity Matrix — {gpu.label}</h3>
        <p className="sensitivity-subtitle">Required GPUs by rows per task and columns per task</p>
        <div className="sensitivity-table-wrapper">
          <table className="sensitivity-table">
            <thead>
              <tr>
                <th>Rows</th>
                {TAB_COLUMNS.map((c) => (
                  <th key={c}>{c} cols</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {TAB_ROWS.map((rows, ri) => (
                <tr key={rows}>
                  <td className="mfu-label">{rows.toLocaleString()}</td>
                  {TAB_COLUMNS.map((_, ci) => {
                    const val = grid[ri][ci]
                    return (
                      <td
                        key={ci}
                        className="heat-cell"
                        style={{ backgroundColor: heatColor(val, minVal, maxVal), color: '#fff' }}
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

  if (modelFamily === 'time_series_foundation') {
    const grid = LOOKBACK_VALUES.map((lookback) =>
      PATCH_SIZES.map((patch) => {
        const tsScenario = {
          modelFamily: 'time_series_foundation' as const,
          ...tsConfig,
          lookbackWindow: lookback,
          patchSize: patch,
          trainingWindowSeconds: scenario.trainingWindowSeconds,
          precision: scenario.precision,
          selectedGpuIds: scenario.selectedGpuIds,
          mfuByGpuId: scenario.mfuByGpuId,
          availability: scenario.availability,
          overheadFactor: scenario.overheadFactor,
        }

        const adapterResult = computeTimeSeriesFlops(tsScenario)
        const hwResult = estimateHardware(
          {
            totalFlops: adapterResult.totalFlops,
            modelParameters: tsConfig.modelParameters,
            memoryBytesPerParameter: tsConfig.memoryBytesPerParameter,
            hardware: {
              trainingWindowSeconds: scenario.trainingWindowSeconds,
              precision: scenario.precision,
              selectedGpuIds: scenario.selectedGpuIds,
              mfuByGpuId: scenario.mfuByGpuId,
              availability: scenario.availability,
              overheadFactor: scenario.overheadFactor,
            },
          },
          gpu,
          h100,
        )
        return hwResult.requiredGpus
      }),
    )

    const allValues = grid.flat()
    const minVal = Math.min(...allValues)
    const maxVal = Math.max(...allValues)

    return (
      <div className="sensitivity-section">
        <h3>Sensitivity Matrix — {gpu.label}</h3>
        <p className="sensitivity-subtitle">Required GPUs by lookback window and patch size</p>
        <div className="sensitivity-table-wrapper">
          <table className="sensitivity-table">
            <thead>
              <tr>
                <th>Lookback</th>
                {PATCH_SIZES.map((p) => (
                  <th key={p}>patch {p}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {LOOKBACK_VALUES.map((lookback, ri) => (
                <tr key={lookback}>
                  <td className="mfu-label">{lookback}</td>
                  {PATCH_SIZES.map((_, ci) => {
                    const val = grid[ri][ci]
                    return (
                      <td
                        key={ci}
                        className="heat-cell"
                        style={{ backgroundColor: heatColor(val, minVal, maxVal), color: '#fff' }}
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

  // LLM mode — original sensitivity matrix
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

  const allValues = grid.flat()
  const minVal = Math.min(...allValues)
  const maxVal = Math.max(...allValues)

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
                      style={{ backgroundColor: heatColor(val, minVal, maxVal), color: '#fff' }}
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
