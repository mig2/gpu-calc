import { useScenarioStore } from '../store/scenario-store'
import type { ModelFamily } from '../engine/types'

const TABS: { family: ModelFamily; label: string; enabled: boolean }[] = [
  { family: 'llm', label: 'LLM', enabled: true },
  { family: 'time_series_foundation', label: 'Time Series', enabled: true },
  { family: 'tabular_foundation', label: 'Tabular', enabled: true },
  { family: 'classical_tabular', label: 'Classical', enabled: true },
]

export function ModelFamilyTabs() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const setModelFamily = useScenarioStore((s) => s.setModelFamily)

  return (
    <div className="family-tabs" role="tablist" aria-label="Model family">
      {TABS.map((tab) => (
        <button
          key={tab.family}
          role="tab"
          aria-selected={modelFamily === tab.family}
          className={`family-tab ${modelFamily === tab.family ? 'active' : ''} ${!tab.enabled ? 'disabled' : ''}`}
          onClick={() => tab.enabled && setModelFamily(tab.family)}
          disabled={!tab.enabled}
        >
          {tab.label}
          {!tab.enabled && <span className="coming-soon">Soon</span>}
        </button>
      ))}
    </div>
  )
}
