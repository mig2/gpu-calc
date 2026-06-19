import { ScenarioForm } from './ScenarioForm'
import { GpuSelector } from './GpuSelector'
import { AdvancedAssumptions } from './AdvancedAssumptions'
import { ResultCards } from './ResultCards'
import { FormulaTraceDrawer } from './FormulaTraceDrawer'
import { AssumptionChips } from './AssumptionChips'
import { GpuComparisonTable } from './GpuComparisonTable'
import { SensitivityMatrix } from './SensitivityMatrix'
import { ReverseSolve } from './ReverseSolve'
import { WarningsPanel } from './WarningsPanel'
import { CustomGpuEditor } from './CustomGpuEditor'

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
          <CustomGpuEditor />
        </aside>
        <section className="results-area">
          <AssumptionChips />
          <ResultCards />
          <FormulaTraceDrawer />
          <GpuComparisonTable />
          <SensitivityMatrix />
          <ReverseSolve />
          <WarningsPanel />
        </section>
      </main>
    </div>
  )
}
