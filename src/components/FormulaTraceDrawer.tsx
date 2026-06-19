import { useScenarioStore } from '../store/scenario-store'
import { getGpuById } from '../engine/gpu-data'

export function FormulaTraceDrawer() {
  const results = useScenarioStore((s) => s.results)

  if (results.length === 0) return null

  return (
    <div className="trace-drawers">
      {results.map((result) => {
        const gpu = getGpuById(result.gpuId)
        return (
          <details key={result.gpuId} className="trace-details">
            <summary>
              Calculation trace — {gpu?.label ?? result.gpuId}
            </summary>
            <pre className="trace-pre">{result.trace.join('\n')}</pre>
          </details>
        )
      })}
    </div>
  )
}
