import { useEffect } from 'react'
import { ScenarioForm } from './ScenarioForm'
import { GpuSelector } from './GpuSelector'
import { AdvancedAssumptions } from './AdvancedAssumptions'
import { ResultCards } from './ResultCards'
import { FormulaTraceDrawer } from './FormulaTraceDrawer'
import { AssumptionChips } from './AssumptionChips'
import { GpuComparisonTable } from './GpuComparisonTable'
import { SensitivityMatrix } from './SensitivityMatrix'
import { ReverseSolve } from './ReverseSolve'
import { CalibrationMode } from './CalibrationMode'
import { WarningsPanel } from './WarningsPanel'
import { CustomGpuEditor } from './CustomGpuEditor'
import { ExportControls } from './ExportControls'
import { decodeScenarioFromHash } from '../engine/export'
import { useScenarioStore } from '../store/scenario-store'

export default function App() {
  useEffect(() => {
    const partial = decodeScenarioFromHash(window.location.hash)
    if (partial) {
      const store = useScenarioStore.getState()
      const merged = { ...store.scenario, ...partial }
      store.setScenario(merged)
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1>LLM Training GPU Calculator</h1>
        <p className="subtitle">
          Estimate accelerator requirements for training dense language models
        </p>
        <ExportControls />
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
          <CalibrationMode />
          <WarningsPanel />
        </section>
      </main>
    </div>
  )
}
