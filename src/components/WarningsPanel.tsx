import { useScenarioStore } from '../store/scenario-store'

export function WarningsPanel() {
  const results = useScenarioStore((s) => s.results)

  // Collect and deduplicate warnings across all GPU results
  const seen = new Set<string>()
  const warnings: string[] = []
  for (const result of results) {
    for (const w of result.warnings) {
      if (!seen.has(w)) {
        seen.add(w)
        warnings.push(w)
      }
    }
  }

  if (warnings.length === 0) return null

  return (
    <div className="warnings-panel">
      <h3>Warnings &amp; Notes</h3>
      {warnings.map((w, i) => (
        <p key={i} className="warning">{w}</p>
      ))}
    </div>
  )
}
