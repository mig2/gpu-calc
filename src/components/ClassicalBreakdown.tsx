import { useScenarioStore } from '../store/scenario-store'
import { estimateClassicalTabular } from '../engine/adapters/classical-tabular-adapter'

export function ClassicalBreakdown() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const classicalConfig = useScenarioStore((s) => s.classicalConfig)
  const scenario = useScenarioStore((s) => s.scenario)

  if (modelFamily !== 'classical_tabular') return null

  const classicalScenario = {
    modelFamily: 'classical_tabular' as const,
    ...classicalConfig,
    trainingWindowSeconds: scenario.trainingWindowSeconds,
    precision: scenario.precision,
    selectedGpuIds: scenario.selectedGpuIds,
    mfuByGpuId: scenario.mfuByGpuId,
    availability: scenario.availability,
    overheadFactor: scenario.overheadFactor,
  }

  const result = estimateClassicalTabular(classicalScenario)

  return (
    <div className="ts-breakdown">
      <h3>Classical Tabular Estimate</h3>
      <dl className="result-details">
        {Object.entries(result.dataBreakdown).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </div>
        ))}
        <div>
          <dt>Estimated time</dt>
          <dd>
            {result.estimatedDays >= 1
              ? `${result.estimatedDays.toFixed(1)} days`
              : result.estimatedHours >= 1
                ? `${result.estimatedHours.toFixed(1)} hours`
                : `${result.estimatedSeconds.toFixed(0)} seconds`}
          </dd>
        </div>
      </dl>
    </div>
  )
}
