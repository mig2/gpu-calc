import { useScenarioStore } from '../store/scenario-store'
import type { Precision } from '../engine/types'

const PRECISION_LABELS: Record<Precision, string> = {
  BF16_DENSE: 'BF16',
  FP8_DENSE: 'FP8',
  FP4_DENSE: 'FP4',
}

export function AssumptionChips() {
  const scenario = useScenarioStore((s) => s.scenario)

  const chips = [
    `TPP: ${scenario.tokensPerParameter}`,
    `Availability: ${(scenario.availability * 100).toFixed(0)}%`,
    `Overhead: ${scenario.overheadFactor}x`,
    `Precision: ${PRECISION_LABELS[scenario.precision]}`,
    ...scenario.selectedGpuIds.map(
      (id) => `MFU ${id}: ${((scenario.mfuByGpuId[id] ?? 0.4) * 100).toFixed(0)}%`,
    ),
  ]

  return (
    <div className="assumption-chips" aria-label="Active assumptions">
      {chips.map((chip) => (
        <span key={chip} className="chip">{chip}</span>
      ))}
    </div>
  )
}
