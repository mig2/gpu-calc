import { useScenarioStore } from '../store/scenario-store'
import { computeTimeSeriesFlops } from '../engine/adapters/time-series-adapter'

export function TimeSeriesBreakdown() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const tsConfig = useScenarioStore((s) => s.tsConfig)
  const scenario = useScenarioStore((s) => s.scenario)

  if (modelFamily !== 'time_series_foundation') return null

  const tsScenario = {
    modelFamily: 'time_series_foundation' as const,
    ...tsConfig,
    trainingWindowSeconds: scenario.trainingWindowSeconds,
    precision: scenario.precision,
    selectedGpuIds: scenario.selectedGpuIds,
    mfuByGpuId: scenario.mfuByGpuId,
    availability: scenario.availability,
    overheadFactor: scenario.overheadFactor,
  }

  const result = computeTimeSeriesFlops(tsScenario)
  if (!result.dataBreakdown) return null

  return (
    <div className="ts-breakdown">
      <h3>Data Pipeline</h3>
      <dl className="result-details">
        {Object.entries(result.dataBreakdown).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  )
}
