import { useScenarioStore } from '../store/scenario-store'
import { LlmForm } from './LlmForm'
import { TimeSeriesForm } from './TimeSeriesForm'

export function ScenarioForm() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  return (
    <div className="scenario-form">
      <h2>Scenario</h2>
      {modelFamily === 'llm' && <LlmForm />}
      {modelFamily === 'time_series_foundation' && <TimeSeriesForm />}
      {(modelFamily === 'tabular_foundation' || modelFamily === 'classical_tabular') && (
        <p className="coming-soon-msg">This model family is coming soon.</p>
      )}
    </div>
  )
}
