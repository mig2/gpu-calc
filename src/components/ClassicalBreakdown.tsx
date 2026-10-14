import { useScenarioStore } from '../store/scenario-store'
import { estimateClassicalTabular } from '../engine/adapters/classical-tabular-adapter'
import type { ClassicalGpuVerdict } from '../engine/adapters/classical-tabular-adapter'

const VERDICT_LABEL: Record<ClassicalGpuVerdict, string> = {
  gpu_recommended: 'GPU: yes — 1 GPU',
  cpu_sufficient: 'GPU: not needed',
  gpu_likely_slower: 'GPU: no — likely slower',
}

function formatSeconds(seconds: number): string {
  if (seconds < 90) return `${seconds.toFixed(0)} seconds`
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)} minutes`
  if (seconds < 172_800) return `${(seconds / 3600).toFixed(1)} hours`
  return `${(seconds / 86_400).toFixed(1)} days`
}

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
      <div className={`gpu-verdict gpu-verdict-${result.gpuVerdict.replace(/_/g, '-')}`}>
        <strong>{VERDICT_LABEL[result.gpuVerdict]}</strong>
        <span>{result.gpuVerdictReason}</span>
      </div>
      <dl className="result-details">
        {Object.entries(result.dataBreakdown).map(([key, value]) => (
          <div key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </div>
        ))}
        <div>
          <dt>On CPU</dt>
          <dd>{formatSeconds(result.cpuAlternativeSeconds)}</dd>
        </div>
        <div>
          <dt>On 1 GPU</dt>
          <dd>{formatSeconds(result.gpuAlternativeSeconds)}</dd>
        </div>
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
