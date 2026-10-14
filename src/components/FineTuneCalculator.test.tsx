import { describe, it, expect, beforeEach } from 'vitest'
import { render, screen, fireEvent } from '@testing-library/react'
import { FineTuneCalculator } from './FineTuneCalculator'
import { useFineTuneStore, FINE_TUNE_METHODS } from '../store/finetune-store'

/**
 * Behavioural tests for the fine-tuning pane.
 *
 * The pane's subject is feasibility, so the assertions are about what a reader
 * can actually tell from a cell: GPU count, fit state, and whether an absent
 * price is presented honestly rather than as NaN.
 */

function selectValue(label: string, value: string) {
  const el = screen.getByLabelText(label) as HTMLSelectElement
  fireEvent.change(el, { target: { value } })
  return el
}

function numberInput(label: string) {
  return screen.getByLabelText(label) as HTMLInputElement
}

beforeEach(() => {
  useFineTuneStore.getState().reset()
})

describe('FineTuneCalculator', () => {
  it('renders the pane with its input sections and the matrix', () => {
    render(<FineTuneCalculator />)

    expect(screen.getByText('1. Which model?')).toBeTruthy()
    expect(screen.getByText('3. Workload')).toBeTruthy()
    expect(screen.getByText(/Method . GPU feasibility/)).toBeTruthy()
    expect(screen.getByLabelText('Base model')).toBeTruthy()
    // The near-constant-cost finding is stated, not left to be inferred.
    expect(screen.getByText(/billed by the GPU-hour/)).toBeTruthy()
  })

  it('renders one cell per method per selected GPU', () => {
    const { container } = render(<FineTuneCalculator />)
    const gpuCount = useFineTuneStore.getState().selectedGpuIds.length

    expect(gpuCount).toBeGreaterThan(1)
    expect(container.querySelectorAll('[data-fit-state]').length).toBe(
      FINE_TUNE_METHODS.length * gpuCount,
    )
  })

  it('shrinks and grows the matrix as GPU columns are toggled', () => {
    const { container } = render(<FineTuneCalculator />)
    const before = container.querySelectorAll('[data-fit-state]').length

    fireEvent.click(screen.getByLabelText('NVIDIA H200 SXM'))
    expect(container.querySelectorAll('[data-fit-state]').length).toBe(
      before - FINE_TUNE_METHODS.length,
    )
  })

  it('fills the raw workload fields from a goal preset and leaves them editable', () => {
    render(<FineTuneCalculator />)

    fireEvent.click(screen.getByText('Instruction following'))

    expect(numberInput('Training examples').value).toBe('10000')
    expect(numberInput('Epochs').value).toBe('2')
    expect(numberInput('Sequence length').value).toBe('2048')
    expect(useFineTuneStore.getState().config.method).toBe('lora')
    expect(useFineTuneStore.getState().config.loraRank).toBe(32)

    // Still editable, and editing clears the preset rather than fighting it.
    const examples = numberInput('Training examples')
    expect(examples.disabled).toBe(false)
    fireEvent.change(examples, { target: { value: '7500' } })

    expect(numberInput('Training examples').value).toBe('7500')
    expect(useFineTuneStore.getState().workload.examples).toBe(7500)
    expect(useFineTuneStore.getState().presetId).toBeNull()
  })

  it('surfaces the total-vs-active split when an MoE model is selected', () => {
    const { container } = render(<FineTuneCalculator />)

    expect(container.querySelector('.ft-moe-note')).toBeNull()

    selectValue('Base model', 'qwen3-235b-a22b')

    const note = container.querySelector('.ft-moe-note')
    expect(note).toBeTruthy()
    expect(note!.textContent).toMatch(/memory follows 235B total parameters/)
    expect(note!.textContent).toMatch(/compute follows 22\.2B active/)
    // The MoE + 4-bit hard warning from the decisions doc.
    expect(note!.textContent).toMatch(/Do not 4-bit QLoRA an MoE/)
  })

  it('shows an em dash, never NaN, for a GPU with no pricing row', () => {
    const { container } = render(<FineTuneCalculator />)

    // H200 is in gpu-data.ts but has no instance row in cloud-gpu-pricing.json.
    expect(useFineTuneStore.getState().rates['h200-sxm']).toBeUndefined()
    expect(useFineTuneStore.getState().rates['h100-sxm']).toBeTruthy()

    const unpriced = screen.getAllByLabelText(/NVIDIA H200 SXM: .*no cost estimate/)
    expect(unpriced.length).toBeGreaterThan(0)
    expect(unpriced[0].querySelector('.ft-cell-cost-none')!.textContent).toBe('—')

    expect(container.textContent).not.toMatch(/NaN/)
    expect(screen.getByText(/No instance price in/)).toBeTruthy()
  })

  it('marks an OOM cell distinguishably from a feasible one', () => {
    const { container } = render(<FineTuneCalculator />)

    // DDP replicates model state, so anything over one GPU's capacity is OOM.
    selectValue('Sharding strategy', 'ddp')

    const oom = container.querySelectorAll('[data-fit-state="oom"]')
    const feasible = container.querySelectorAll(
      '[data-fit-state="fits"], [data-fit-state="sharded"]',
    )
    expect(oom.length).toBeGreaterThan(0)
    expect(feasible.length).toBeGreaterThan(0)

    expect(oom[0].className).toContain('ft-cell-oom')
    expect(oom[0].className).not.toContain('ft-cell-fits')
    expect(oom[0].textContent).toContain('OOM')
    expect(feasible[0].textContent).not.toContain('OOM')
    // No misleading GPU count on an infeasible cell.
    expect(oom[0].querySelector('.ft-cell-gpus')!.textContent).toBe('—')
  })

  it('opens the clicked cell in the detail panel with its trace and warnings', () => {
    const { container } = render(<FineTuneCalculator />)

    const qloraCell = screen.getByLabelText(/^QLoRA on NVIDIA H100 SXM:/)
    fireEvent.click(qloraCell)

    expect(useFineTuneStore.getState().config.method).toBe('qlora')
    expect(useFineTuneStore.getState().selectedGpuId).toBe('h100-sxm')
    expect(screen.getByText('QLoRA on NVIDIA H100 SXM')).toBeTruthy()
    expect(container.querySelector('.ft-memory-track')).toBeTruthy()
    expect(container.querySelector('.trace-pre')!.textContent).toMatch(/Total memory =/)
    // QLoRA carries an explicit throughput-factor caveat.
    expect(container.querySelector('.warnings-panel')!.textContent).toMatch(
      /QLoRA throughput factor/,
    )
  })

  it('disables the LoRA controls for methods that train the full parameter set', () => {
    render(<FineTuneCalculator />)

    expect(numberInput('LoRA rank').disabled).toBe(false)

    selectValue('Method', 'full')

    expect(numberInput('LoRA rank').disabled).toBe(true)
    expect((screen.getByLabelText('LoRA targets') as HTMLSelectElement).disabled).toBe(true)
    expect(screen.getByText(/are inert for Full fine-tune/)).toBeTruthy()
  })
})
