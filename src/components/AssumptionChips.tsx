import { useScenarioStore } from '../store/scenario-store'

export function AssumptionChips() {
  const scenario = useScenarioStore((s) => s.scenario)

  const chips = [
    `TPP: ${scenario.tokensPerParameter}`,
    `Availability: ${(scenario.availability * 100).toFixed(0)}%`,
    `Overhead: ${scenario.overheadFactor}x`,
    `Precision: ${scenario.precision === 'BF16_DENSE' ? 'BF16' : 'FP8'}`,
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
