import { useState } from 'react'
import { useScenarioStore } from '../store/scenario-store'
import type { GpuSku } from '../engine/types'

export function CustomGpuEditor() {
  const [open, setOpen] = useState(false)
  const customGpus = useScenarioStore((s) => s.customGpus)
  const addCustomGpu = useScenarioStore((s) => s.addCustomGpu)
  const removeCustomGpu = useScenarioStore((s) => s.removeCustomGpu)

  const [label, setLabel] = useState('')
  const [bf16, setBf16] = useState('')
  const [fp8, setFp8] = useState('')
  const [memory, setMemory] = useState('')
  const [bandwidth, setBandwidth] = useState('')
  const [mfu, setMfu] = useState('0.40')
  const [error, setError] = useState('')

  function handleAdd() {
    if (!label.trim()) { setError('Label is required'); return }
    const bf16Val = parseFloat(bf16)
    if (!(bf16Val > 0)) { setError('BF16 peak FLOP/s must be positive'); return }
    const memVal = parseFloat(memory)
    if (!(memVal > 0)) { setError('Memory (GB) must be positive'); return }
    const mfuVal = parseFloat(mfu)
    if (!(mfuVal > 0 && mfuVal <= 1)) { setError('MFU must be between 0 and 1'); return }

    const id = `custom-${label.toLowerCase().replace(/[^a-z0-9]+/g, '-')}-${Date.now()}`
    const gpu: GpuSku = {
      id,
      label: label.trim(),
      bf16DenseFlops: bf16Val,
      memoryGb: memVal,
      defaultMfu: mfuVal,
    }
    const fp8Val = parseFloat(fp8)
    if (fp8Val > 0) gpu.fp8DenseFlops = fp8Val
    const bwVal = parseFloat(bandwidth)
    if (bwVal > 0) gpu.bandwidthTbps = bwVal

    addCustomGpu(gpu)
    setLabel('')
    setBf16('')
    setFp8('')
    setMemory('')
    setBandwidth('')
    setMfu('0.40')
    setError('')
  }

  return (
    <div className="custom-gpu-section">
      <button
        className="advanced-toggle"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? '\u25BE' : '\u25B8'} Custom GPU
      </button>
      {open && (
        <div className="custom-gpu-content">
          {customGpus.length > 0 && (
            <div className="custom-gpu-list">
              {customGpus.map((g) => (
                <div key={g.id} className="custom-gpu-item">
                  <span>{g.label}</span>
                  <button className="remove-btn" onClick={() => removeCustomGpu(g.id)}>Remove</button>
                </div>
              ))}
            </div>
          )}

          <div className="custom-gpu-form">
            <fieldset>
              <legend>Label</legend>
              <input type="text" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. MI300X" />
            </fieldset>
            <fieldset>
              <legend>Dense BF16 Peak (FLOP/s)</legend>
              <input type="number" value={bf16} onChange={(e) => setBf16(e.target.value)} placeholder="e.g. 1.3e15" />
            </fieldset>
            <fieldset>
              <legend>Dense FP8 Peak (FLOP/s, optional)</legend>
              <input type="number" value={fp8} onChange={(e) => setFp8(e.target.value)} placeholder="e.g. 2.6e15" />
            </fieldset>
            <fieldset>
              <legend>Memory (GB)</legend>
              <input type="number" value={memory} onChange={(e) => setMemory(e.target.value)} placeholder="e.g. 192" />
            </fieldset>
            <fieldset>
              <legend>Bandwidth (TB/s, optional)</legend>
              <input type="number" value={bandwidth} onChange={(e) => setBandwidth(e.target.value)} placeholder="e.g. 5.3" />
            </fieldset>
            <fieldset>
              <legend>Default MFU (0-1)</legend>
              <input type="number" value={mfu} onChange={(e) => setMfu(e.target.value)} step="0.01" min="0.01" max="1" />
            </fieldset>
            {error && <p className="input-error">{error}</p>}
            <button className="add-gpu-btn" onClick={handleAdd}>Add Custom GPU</button>
          </div>
        </div>
      )}
    </div>
  )
}
