import { ScenarioForm } from './ScenarioForm'
import { GpuSelector } from './GpuSelector'
import { AdvancedAssumptions } from './AdvancedAssumptions'
import { ResultCards } from './ResultCards'
import { AssumptionChips } from './AssumptionChips'
import { GpuComparisonTable } from './GpuComparisonTable'
import { SensitivityMatrix } from './SensitivityMatrix'
import { ReverseSolve } from './ReverseSolve'

export default function App() {
  return (
    <div className="app">
      <header className="app-header">
        <h1>LLM Training GPU Calculator</h1>
        <p className="subtitle">
          Estimate accelerator requirements for training dense language models
        </p>
      </header>
      <main className="app-main">
        <aside className="input-rail">
          <ScenarioForm />
          <GpuSelector />
          <AdvancedAssumptions />
        </aside>
        <section className="results-area">
          <AssumptionChips />
          <ResultCards />
          <GpuComparisonTable />
          <SensitivityMatrix />
          <ReverseSolve />
        </section>
      </main>
    </div>
  )
}
