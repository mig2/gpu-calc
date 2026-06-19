import type { ConfidenceLevel } from '../engine/types'
import { Tooltip } from './Tooltip'

const LABELS: Record<ConfidenceLevel, { text: string; className: string; tip: string }> = {
  high: { text: 'High confidence', className: 'confidence-high', tip: 'Formula is widely used and inputs are well-defined.' },
  medium: { text: 'Medium confidence', className: 'confidence-medium', tip: 'Formula is plausible but architecture/data-unit choices matter.' },
  'medium-low': { text: 'Medium-low confidence', className: 'confidence-medium-low', tip: 'Estimate is useful for comparison but not procurement.' },
  low: { text: 'Low confidence', className: 'confidence-low', tip: 'Estimate depends heavily on custom implementation or empirical calibration.' },
}

export function ConfidenceBadge({ level }: { level: ConfidenceLevel }) {
  const info = LABELS[level]
  return (
    <Tooltip text={info.tip}>
      <span className={`confidence-badge ${info.className}`}>{info.text}</span>
    </Tooltip>
  )
}
