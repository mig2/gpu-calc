import { useScenarioStore } from '../store/scenario-store'
import { LlmForm } from './LlmForm'
import { TimeSeriesForm } from './TimeSeriesForm'
import { TabularForm } from './TabularForm'
import { ClassicalTabularForm } from './ClassicalTabularForm'

export function ScenarioForm() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  return (
    <div className="scenario-form">
      <h2>Scenario</h2>
      {modelFamily === 'llm' && <LlmForm />}
      {modelFamily === 'time_series_foundation' && <TimeSeriesForm />}
      {modelFamily === 'tabular_foundation' && <TabularForm />}
      {modelFamily === 'classical_tabular' && <ClassicalTabularForm />}
    </div>
  )
}
