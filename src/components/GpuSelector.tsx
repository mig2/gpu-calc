import { useScenarioStore } from '../store/scenario-store'
import { GPU_SKUS } from '../engine/gpu-data'

export function GpuSelector() {
  const selectedGpuIds = useScenarioStore((s) => s.scenario.selectedGpuIds)
  const setSelectedGpuIds = useScenarioStore((s) => s.setSelectedGpuIds)

  function toggleGpu(id: string) {
    if (selectedGpuIds.includes(id)) {
      if (selectedGpuIds.length > 1) {
        setSelectedGpuIds(selectedGpuIds.filter((g) => g !== id))
      }
    } else {
      setSelectedGpuIds([...selectedGpuIds, id])
    }
  }

  return (
    <div className="gpu-selector">
      <h3>GPU SKUs</h3>
      <div className="gpu-checkboxes">
        {GPU_SKUS.map((gpu) => (
          <label key={gpu.id}>
            <input
              type="checkbox"
              checked={selectedGpuIds.includes(gpu.id)}
              onChange={() => toggleGpu(gpu.id)}
            />
            <span>{gpu.label}</span>
            <span className="chip" style={{ marginLeft: 'auto' }}>
              {gpu.memoryGb}GB
            </span>
          </label>
        ))}
      </div>
    </div>
  )
}
