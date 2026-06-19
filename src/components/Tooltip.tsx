import { useId } from 'react'

type TooltipProps = {
  text: string;
  children?: React.ReactNode;
}

export function Tooltip({ text, children }: TooltipProps) {
  const id = useId()
  return (
    <span className="tooltip-wrapper">
      {children}
      <span
        className="tooltip-trigger"
        tabIndex={0}
        role="button"
        aria-describedby={id}
      >
        i
      </span>
      <span className="tooltip-content" id={id} role="tooltip">
        {text}
      </span>
    </span>
  )
}
