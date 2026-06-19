import { useScenarioStore } from '../store/scenario-store'
import { exportToJson, exportToCsv, exportToMarkdown, encodeScenarioToHash } from '../engine/export'

export function ExportControls() {
  const scenario = useScenarioStore((s) => s.scenario)
  const results = useScenarioStore((s) => s.results)

  function copyToClipboard(text: string, label: string) {
    navigator.clipboard.writeText(text).then(() => {
      alert(`${label} copied to clipboard`)
    })
  }

  function downloadFile(content: string, filename: string, mime: string) {
    const blob = new Blob([content], { type: mime })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }

  function shareUrl() {
    const hash = encodeScenarioToHash(scenario)
    const url = window.location.origin + window.location.pathname + hash
    navigator.clipboard.writeText(url).then(() => {
      alert('Shareable URL copied to clipboard')
    })
  }

  return (
    <div className="export-controls">
      <button onClick={() => copyToClipboard(exportToMarkdown(scenario, results), 'Markdown')}>
        Copy Markdown
      </button>
      <button onClick={() => downloadFile(exportToJson(scenario, results), 'gpu-calc.json', 'application/json')}>
        Export JSON
      </button>
      <button onClick={() => downloadFile(exportToCsv(results), 'gpu-calc.csv', 'text/csv')}>
        Export CSV
      </button>
      <button onClick={shareUrl}>
        Copy Share URL
      </button>
    </div>
  )
}
