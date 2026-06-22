import { useEffect } from 'react'
import { ScenarioForm } from './ScenarioForm'
import { GpuSelector } from './GpuSelector'
import { AdvancedAssumptions } from './AdvancedAssumptions'
import { ResultCards } from './ResultCards'
import { FormulaTraceDrawer } from './FormulaTraceDrawer'
import { AssumptionChips } from './AssumptionChips'
import { GpuComparisonTable } from './GpuComparisonTable'
import { SensitivityMatrix } from './SensitivityMatrix'
import { ReferenceComparison } from './ReferenceComparison'
import { IsoFlopExplorer } from './IsoFlopExplorer'
import { ReverseSolve } from './ReverseSolve'
import { CalibrationMode } from './CalibrationMode'
import { WarningsPanel } from './WarningsPanel'
import { CustomGpuEditor } from './CustomGpuEditor'
import { ExportControls } from './ExportControls'
import { ModelFamilyTabs } from './ModelFamilyTabs'
import { TimeSeriesBreakdown } from './TimeSeriesBreakdown'
import { TabularBreakdown } from './TabularBreakdown'
import { ClassicalBreakdown } from './ClassicalBreakdown'
import { decodeScenarioFromHash } from '../engine/export'
import { useScenarioStore } from '../store/scenario-store'
import type { ModelFamily } from '../engine/types'

const FAMILY_TITLES: Record<string, { title: string; subtitle: string }> = {
  llm: {
    title: 'LLM Training GPU Calculator',
    subtitle: 'Estimate accelerator requirements for training dense language models',
  },
  time_series_foundation: {
    title: 'Time-Series Foundation Model Calculator',
    subtitle: 'Estimate accelerator requirements for training time-series foundation models',
  },
  tabular_foundation: {
    title: 'Tabular Foundation Model Calculator',
    subtitle: 'Estimate accelerator requirements for training tabular foundation models',
  },
  classical_tabular: {
    title: 'Classical Tabular Calculator',
    subtitle: 'Estimate compute requirements for classical tabular models',
  },
}

export default function App() {
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const { title, subtitle } = FAMILY_TITLES[modelFamily] ?? FAMILY_TITLES.llm

  useEffect(() => {
    const decoded = decodeScenarioFromHash(window.location.hash)
    if (decoded) {
      const store = useScenarioStore.getState()
      const merged = { ...store.scenario, ...decoded.scenario }
      if (decoded.modelFamily) {
        store.setModelFamily(decoded.modelFamily as ModelFamily)
      }
      store.setScenario(merged)
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1>{title}</h1>
        <p className="subtitle">
          {subtitle}
        </p>
        <div className="header-actions">
          <ExportControls />
          <a href="/help.html" target="_blank" rel="noopener noreferrer" className="help-link">
            Help
          </a>
        </div>
      </header>
      <main className="app-main">
        <aside className="input-rail">
          <ModelFamilyTabs />
          <ScenarioForm />
          <GpuSelector />
          <AdvancedAssumptions />
          <CustomGpuEditor />
        </aside>
        <section className="results-area">
          <AssumptionChips />
          <TimeSeriesBreakdown />
          <TabularBreakdown />
          <ClassicalBreakdown />
          <ResultCards />
          <FormulaTraceDrawer />
          <GpuComparisonTable />
          <SensitivityMatrix />
          <ReferenceComparison />
          <IsoFlopExplorer />
          <ReverseSolve />
          <CalibrationMode />
          <WarningsPanel />
        </section>
      </main>
    </div>
  )
}
