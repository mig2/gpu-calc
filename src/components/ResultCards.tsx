import { useScenarioStore } from '../store/scenario-store'
import { getGpuById } from '../engine/gpu-data'

function formatNumber(n: number): string {
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PFLOP/s`
  if (n >= 1e12) return `${(n / 1e12).toFixed(2)}T`
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)}M`
  return n.toLocaleString()
}

function formatFlops(n: number): string {
  if (n >= 1e24) return `${(n / 1e24).toFixed(2)} YottaFLOPs`
  if (n >= 1e21) return `${(n / 1e21).toFixed(2)} ZettaFLOPs`
  if (n >= 1e18) return `${(n / 1e18).toFixed(2)} ExaFLOPs`
  if (n >= 1e15) return `${(n / 1e15).toFixed(2)} PetaFLOPs`
  return n.toExponential(2)
}

export function ResultCards() {
  const results = useScenarioStore((s) => s.results)

  if (results.length === 0) {
    return <div className="result-cards"><p>Select at least one GPU to see results.</p></div>
  }

  return (
    <div className="result-cards">
      {results.map((result) => (
        <div key={result.gpuId} className="result-card">
          <div className="result-card-header">
            <h3>{getGpuById(result.gpuId)?.label ?? result.gpuId}</h3>
          </div>
          <div className="result-gpu-count">
            <span className="big-number">{result.requiredGpus.toLocaleString()}</span>
            <span className="label">GPUs required</span>
          </div>
          <dl className="result-details">
            <div>
              <dt>H100 equivalents</dt>
              <dd>{result.h100Equivalents.toFixed(1)}</dd>
            </div>
            <div>
              <dt>Training tokens</dt>
              <dd>{formatNumber(result.tokens)}</dd>
            </div>
            <div>
              <dt>Base FLOPs</dt>
              <dd>{formatFlops(result.baseFlops)}</dd>
            </div>
            <div>
              <dt>Total FLOPs (with overhead)</dt>
              <dd>{formatFlops(result.totalFlops)}</dd>
            </div>
            <div>
              <dt>Sustained throughput/GPU</dt>
              <dd>{formatNumber(result.sustainedFlopsPerGpu)}</dd>
            </div>
            <div>
              <dt>Memory lower-bound GPUs</dt>
              <dd>{result.memoryLowerBoundGpus}</dd>
            </div>
          </dl>
          {result.warnings.length > 0 && (
            <div className="result-warnings">
              {result.warnings.map((w, i) => (
                <p key={i} className="warning">{w}</p>
              ))}
            </div>
          )}
        </div>
      ))}
    </div>
  )
}
