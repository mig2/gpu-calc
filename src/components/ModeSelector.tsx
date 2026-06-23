import type { AppMode } from './App'

export function ModeSelector({ mode, setMode }: { mode: AppMode; setMode: (m: AppMode) => void }) {
  return (
    <div className="mode-selector" role="tablist" aria-label="Calculator mode">
      <button
        role="tab"
        aria-selected={mode === 'training'}
        className={`mode-tab ${mode === 'training' ? 'active' : ''}`}
        onClick={() => setMode('training')}
      >
        Training Calculator
      </button>
      <button
        role="tab"
        aria-selected={mode === 'inference'}
        className={`mode-tab ${mode === 'inference' ? 'active' : ''}`}
        onClick={() => setMode('inference')}
      >
        Inference Calculator
      </button>
    </div>
  )
}
