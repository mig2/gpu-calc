import { useInferenceStore } from '../store/inference-store'

export function InferenceExportControls() {
  const { usage, apiResult, selfHostResult, breakeven } = useInferenceStore()

  function exportJson() {
    const data = { schema_version: '2.0', mode: 'inference', usage, apiResult, selfHostResult, breakeven }
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'inference-comparison.json'; a.click()
    URL.revokeObjectURL(url)
  }

  function exportCsv() {
    const lines = [
      'Side,Metric,Value',
      `API,Provider,${apiResult?.provider ?? ''}`,
      `API,Model,${apiResult?.model ?? ''}`,
      `API,Monthly Cost,$${apiResult?.monthlyCost.toFixed(2) ?? ''}`,
      `API,Annual Cost,$${apiResult?.annualCost.toFixed(2) ?? ''}`,
      `API,Cost/Request,$${apiResult?.costPerRequest.toFixed(4) ?? ''}`,
      `Self-Host,Model,${selfHostResult?.model ?? ''}`,
      `Self-Host,Cloud,${selfHostResult?.cloudProvider ?? ''} ${selfHostResult?.instance ?? ''}`,
      `Self-Host,Monthly Cost,$${selfHostResult?.monthlyGpuCost.toFixed(2) ?? ''}`,
      `Self-Host,Max Throughput,${selfHostResult?.maxOutputTokensPerSec ?? ''} tok/s`,
      `Self-Host,Utilization,${selfHostResult?.utilizationPercent.toFixed(1) ?? ''}%`,
      `Breakeven,Requests/Day,${breakeven?.breakevenRequestsPerDay ?? ''}`,
      `Breakeven,Monthly Savings,$${breakeven?.monthlySavingsAtCurrentVolume.toFixed(2) ?? ''}`,
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url; a.download = 'inference-comparison.csv'; a.click()
    URL.revokeObjectURL(url)
  }

  function copyMarkdown() {
    const lines = [
      '# Inference Build vs Buy Comparison',
      '',
      `**Usage:** ${usage.requestsPerDay.toLocaleString()} req/day × ${usage.avgInputTokens} input + ${usage.avgOutputTokens} output tokens`,
      `**Daily volume:** ${((usage.requestsPerDay * (usage.avgInputTokens + usage.avgOutputTokens)) / 1e6).toFixed(1)}M tokens/day`,
      '',
      '## API (Buy)',
      apiResult ? `- **${apiResult.provider} ${apiResult.model}**: $${apiResult.monthlyCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}/month ($${apiResult.costPerRequest.toFixed(4)}/req)` : '- No selection',
      '',
      '## Self-Host (Build)',
      selfHostResult ? `- **${selfHostResult.model}** on ${selfHostResult.gpuCount}× GPU (${selfHostResult.cloudProvider}): $${selfHostResult.monthlyGpuCost.toLocaleString(undefined, { maximumFractionDigits: 0 })}/month` : '- No selection',
      selfHostResult ? `- Throughput: ${selfHostResult.maxOutputTokensPerSec} tok/s (${selfHostResult.utilizationPercent.toFixed(1)}% utilized)` : '',
      '',
      '## Breakeven',
      breakeven
        ? breakeven.apiCheaperBelow
          ? `API is cheaper at current volume. Self-hosting breaks even at ${breakeven.breakevenRequestsPerDay.toLocaleString()} req/day.`
          : `Self-hosting saves $${Math.abs(breakeven.monthlySavingsAtCurrentVolume).toLocaleString(undefined, { maximumFractionDigits: 0 })}/month at current volume.`
        : 'No comparison available.',
    ]
    navigator.clipboard.writeText(lines.join('\n')).then(() => alert('Markdown copied to clipboard'))
  }

  return (
    <div className="export-controls">
      <button onClick={copyMarkdown}>Copy Markdown</button>
      <button onClick={exportJson}>Export JSON</button>
      <button onClick={exportCsv}>Export CSV</button>
    </div>
  )
}
