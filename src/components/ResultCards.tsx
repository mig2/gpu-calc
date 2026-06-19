import { useScenarioStore } from '../store/scenario-store'
import { getGpuById } from '../engine/gpu-data'
import { Tooltip } from './Tooltip'
import { ConfidenceBadge } from './ConfidenceBadge'
import type { ConfidenceLevel } from '../engine/types'

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

function getConfidence(modelFamily: string, trainingMode: string): ConfidenceLevel {
  if (modelFamily === 'time_series_foundation') return 'medium'
  switch (trainingMode) {
    case 'FULL_PRETRAINING': return 'high'
    case 'CONTINUED_PRETRAINING': return 'medium'
    default: return 'medium-low'
  }
}

export function ResultCards() {
  const results = useScenarioStore((s) => s.results)
  const modelFamily = useScenarioStore((s) => s.modelFamily)
  const scenario = useScenarioStore((s) => s.scenario)
  const confidence = getConfidence(modelFamily, scenario.trainingMode)

  if (results.length === 0) {
    return <div className="result-cards"><p>Select at least one GPU to see results.</p></div>
  }

  return (
    <div className="result-cards">
      {results.map((result) => {
        const effectiveGpus = Math.max(result.requiredGpus, result.memoryLowerBoundGpus)
        return (
        <div key={result.gpuId} className="result-card">
          <div className="result-card-header">
            <h3>{getGpuById(result.gpuId)?.label ?? result.gpuId}</h3>
            <ConfidenceBadge level={confidence} />
          </div>
          <div className="result-gpu-count">
            <span className="big-number">{effectiveGpus.toLocaleString()}</span>
            <span className="label">
              <Tooltip text="Minimum GPUs needed to complete training in the specified window at the given MFU and availability.">
                GPUs required
              </Tooltip>
            </span>
          </div>
          {result.memoryLowerBoundGpus > result.requiredGpus ? (
            <div className="binding-indicator memory-bound">
              <span className="binding-badge">Memory-bound</span>
              <span className="binding-detail">
                {result.memoryLowerBoundGpus.toLocaleString()} GPUs needed for memory
                (vs {result.requiredGpus.toLocaleString()} for compute)
              </span>
            </div>
          ) : (
            <div className="binding-indicator compute-bound">
              <span className="binding-badge">Compute-bound</span>
              {result.memoryLowerBoundGpus > 1 && (
                <span className="binding-detail">
                  Memory needs {result.memoryLowerBoundGpus.toLocaleString()} GPUs
                  (compute needs {result.requiredGpus.toLocaleString()})
                </span>
              )}
            </div>
          )}
          <dl className="result-details">
            <div>
              <dt>
                <Tooltip text="How many H100 GPUs at default MFU would provide equivalent throughput.">
                  H100 equivalents
                </Tooltip>
              </dt>
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
              <dt>
                <Tooltip text="Actual useful compute per GPU: peak FLOP/s x MFU x availability.">
                  Sustained throughput/GPU
                </Tooltip>
              </dt>
              <dd>{formatNumber(result.sustainedFlopsPerGpu)}</dd>
            </div>
            <div>
              <dt>
                <Tooltip text="Minimum GPUs needed just to hold model state in memory, before considering compute.">
                  Memory lower-bound GPUs
                </Tooltip>
              </dt>
              <dd>{result.memoryLowerBoundGpus}</dd>
            </div>
          </dl>
        </div>
        )
      })}
    </div>
  )
}
