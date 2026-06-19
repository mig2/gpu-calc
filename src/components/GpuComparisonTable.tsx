import { useScenarioStore } from '../store/scenario-store'
import { getGpuById } from '../engine/gpu-data'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const BAR_COLORS = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444']

export function GpuComparisonTable() {
  const results = useScenarioStore((s) => s.results)

  if (results.length < 2) return null

  const chartData = results.map((r) => {
    const gpu = getGpuById(r.gpuId)
    return {
      name: gpu?.label ?? r.gpuId,
      gpus: r.requiredGpus,
    }
  })

  return (
    <div className="comparison-section">
      <h3>GPU Comparison</h3>
      <div className="comparison-table-wrapper">
        <table className="comparison-table">
          <thead>
            <tr>
              <th>GPU</th>
              <th>Required GPUs</th>
              <th>H100 Equiv.</th>
              <th>Sustained FLOP/s</th>
              <th>Memory</th>
              <th>Bandwidth</th>
              <th>Min Memory GPUs</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => {
              const gpu = getGpuById(r.gpuId)
              return (
                <tr key={r.gpuId}>
                  <td>{gpu?.label ?? r.gpuId}</td>
                  <td className="mono">{r.requiredGpus.toLocaleString()}</td>
                  <td className="mono">{r.h100Equivalents.toFixed(1)}</td>
                  <td className="mono">{(r.sustainedFlopsPerGpu / 1e12).toFixed(1)} TFLOP/s</td>
                  <td className="mono">{gpu?.memoryGb ?? '—'} GB</td>
                  <td className="mono">{gpu?.bandwidthTbps ? `${gpu.bandwidthTbps} TB/s` : '—'}</td>
                  <td className="mono">{r.memoryLowerBoundGpus}</td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
      <div className="comparison-chart">
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
            <XAxis dataKey="name" tick={{ fill: '#8b90a0', fontSize: 11 }} />
            <YAxis tick={{ fill: '#8b90a0', fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: '#1a1d27', border: '1px solid #2e3346', borderRadius: 8, color: '#e1e4ed' }}
            />
            <Bar dataKey="gpus" name="Required GPUs" radius={[4, 4, 0, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}
