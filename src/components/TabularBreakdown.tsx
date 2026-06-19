import { useScenarioStore } from '../store/scenario-store'
import { computeTabularFlops } from '../engine/adapters/tabular-adapter'

export function TabularBreakdown() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const tabConfig = useScenarioStore((s) => s.tabConfig)
  const scenario = useScenarioStore((s) => s.scenario)

  if (modelFamily !== 'tabular_foundation') return null

  const tabScenario = {
    modelFamily: 'tabular_foundation' as const,
    ...tabConfig,
    trainingWindowSeconds: scenario.trainingWindowSeconds,
    precision: scenario.precision,
    selectedGpuIds: scenario.selectedGpuIds,
    mfuByGpuId: scenario.mfuByGpuId,
    availability: scenario.availability,
    overheadFactor: scenario.overheadFactor,
  }

  const result = computeTabularFlops(tabScenario)
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
