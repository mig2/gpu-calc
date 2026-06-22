import { useScenarioStore } from '../store/scenario-store'
import { REFERENCE_MODELS, findNearestReferences, formatFlopsShort } from '../engine/reference-models'

export function ReferenceComparison() {
  const results = useScenarioStore((s) => s.results)
  const modelFamily = useScenarioStore((s) => s.modelFamily)

  // Only show for transformer-based modes that produce FLOPs
  if (modelFamily === 'classical_tabular') return null
  if (results.length === 0) return null

  const primaryResult = results[0]
  const totalFlops = primaryResult.totalFlops
  if (totalFlops <= 0) return null

  const { smaller, larger, closest } = findNearestReferences(totalFlops)
  const ratio = totalFlops / closest.estimatedFlops

  return (
    <div className="reference-comparison">
      <h3>Compute Context</h3>
      <p className="reference-summary">
        Your scenario uses <strong>{formatFlopsShort(totalFlops)} FLOPs</strong>
        {ratio >= 0.8 && ratio <= 1.2
          ? <>, comparable to <strong>{closest.name}</strong> ({closest.source})</>
          : ratio < 0.8
          ? <>, about <strong>{(ratio).toFixed(1)}×</strong> the compute of {closest.name}</>
          : <>, about <strong>{(ratio).toFixed(1)}×</strong> the compute of {closest.name}</>
        }
      </p>
      <div className="reference-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Parameters</th>
              <th>Tokens</th>
              <th>~FLOPs</th>
              <th>vs Yours</th>
            </tr>
          </thead>
          <tbody>
            {REFERENCE_MODELS
              .sort((a, b) => a.estimatedFlops - b.estimatedFlops)
              .map((ref) => {
                const refRatio = totalFlops / ref.estimatedFlops
                const isClosest = ref.name === closest.name
                return (
                  <tr key={ref.name} className={isClosest ? 'highlight-row' : ''}>
                    <td>{ref.name}</td>
                    <td className="mono">{ref.parameters >= 1e9 ? `${ref.parameters / 1e9}B` : `${ref.parameters / 1e6}M`}</td>
                    <td className="mono">{ref.trainingTokens >= 1e12 ? `${(ref.trainingTokens / 1e12).toFixed(1)}T` : `${(ref.trainingTokens / 1e9).toFixed(0)}B`}</td>
                    <td className="mono">{formatFlopsShort(ref.estimatedFlops)}</td>
                    <td className="mono">{refRatio >= 100 ? `${refRatio.toFixed(0)}×` : refRatio >= 1 ? `${refRatio.toFixed(1)}×` : `${refRatio.toFixed(2)}×`}</td>
                  </tr>
                )
              })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
