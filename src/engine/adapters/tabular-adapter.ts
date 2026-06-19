import type { AdapterResult, BaseHardwareConfig, TabularFoundationConfig, ConfidenceLevel } from '../types';

type TabularScenario = BaseHardwareConfig & TabularFoundationConfig;

function formatSci(n: number, digits = 2): string {
  return n.toExponential(digits);
}

function getConfidence(tokenizationMode: string): ConfidenceLevel {
  switch (tokenizationMode) {
    case 'row': return 'medium-low';
    case 'axial': return 'medium-low';
    case 'cell': return 'low';
    default: return 'low';
  }
}

export function computeTabularFlops(scenario: TabularScenario): AdapterResult {
  const {
    modelParameters: N,
    numberOfPretrainingTasks,
    rowsPerTask,
    columnsPerTask,
    tokenizationMode,
    customTokensPerTask,
    epochs,
    architectureFactor,
    testTimeComputeMultiplier,
    overheadFactor,
  } = scenario;

  const warnings: string[] = [];
  const trace: string[] = [];

  // Compute effective tokens per task based on tokenization mode
  let tokensPerTask: number;
  let seqLengthPerTask: number;

  switch (tokenizationMode) {
    case 'row':
      tokensPerTask = rowsPerTask;
      seqLengthPerTask = rowsPerTask;
      break;
    case 'cell':
      tokensPerTask = rowsPerTask * columnsPerTask;
      seqLengthPerTask = rowsPerTask * columnsPerTask;
      break;
    case 'axial':
      // Conservative: additive rather than multiplicative
      tokensPerTask = rowsPerTask + columnsPerTask;
      seqLengthPerTask = Math.max(rowsPerTask, columnsPerTask);
      break;
    case 'custom':
      tokensPerTask = customTokensPerTask ?? rowsPerTask;
      seqLengthPerTask = tokensPerTask;
      break;
    default:
      tokensPerTask = rowsPerTask;
      seqLengthPerTask = rowsPerTask;
  }

  const effectiveTokens = numberOfPretrainingTasks * tokensPerTask * epochs;
  const baseFlops = architectureFactor * N * effectiveTokens;
  const totalFlops = baseFlops * overheadFactor * testTimeComputeMultiplier;

  // Trace
  trace.push(`N = ${formatSci(N)}`);
  trace.push(`Tasks = ${numberOfPretrainingTasks.toLocaleString()}`);
  trace.push(`Rows/task = ${rowsPerTask.toLocaleString()}, Columns/task = ${columnsPerTask}`);
  trace.push(`Tokenization = ${tokenizationMode} → ${tokensPerTask.toLocaleString()} tokens/task`);
  trace.push(`Effective tokens = ${numberOfPretrainingTasks.toLocaleString()} × ${tokensPerTask.toLocaleString()}${epochs > 1 ? ` × ${epochs}` : ''} = ${formatSci(effectiveTokens)}`);
  trace.push(`Base FLOPs = ${architectureFactor} × ${formatSci(N)} × ${formatSci(effectiveTokens)} = ${formatSci(baseFlops)}`);
  if (testTimeComputeMultiplier > 1) {
    trace.push(`Test-time multiplier = ${testTimeComputeMultiplier}×`);
  }
  trace.push(`Total FLOPs = ${formatSci(baseFlops)} × ${overheadFactor}${testTimeComputeMultiplier > 1 ? ` × ${testTimeComputeMultiplier}` : ''} = ${formatSci(totalFlops)}`);

  // Dense attention feasibility warnings
  if (seqLengthPerTask > 100000) {
    warnings.push(
      `Severe attention warning: sequence length of ${seqLengthPerTask.toLocaleString()} per task makes dense attention infeasible. Consider sparse, factorized, or row-level attention.`,
    );
  } else if (seqLengthPerTask > 65536) {
    warnings.push(
      `High attention warning: sequence length of ${seqLengthPerTask.toLocaleString()} per task is very large for dense attention. Training may be extremely slow.`,
    );
  } else if (seqLengthPerTask > 16384) {
    warnings.push(
      `Attention warning: sequence length of ${seqLengthPerTask.toLocaleString()} per task may strain dense attention. Consider attention-efficient architectures.`,
    );
  }

  if (tokenizationMode === 'cell' && columnsPerTask > 50) {
    warnings.push(
      `Cell-tokenized mode with ${columnsPerTask} columns creates very long sequences (${seqLengthPerTask.toLocaleString()} per task). Row-tokenized or axial attention may be more practical.`,
    );
  }

  if (testTimeComputeMultiplier > 5) {
    warnings.push(
      `High test-time compute multiplier (${testTimeComputeMultiplier}×) significantly increases total FLOPs. This may dominate serving cost.`,
    );
  }

  warnings.push(
    'Tabular foundation model estimates are less mature than LLM estimates. Architecture and tokenization choices can change compute by orders of magnitude.',
  );

  return {
    effectiveTokens,
    baseFlops,
    totalFlops,
    trace,
    warnings,
    confidence: getConfidence(tokenizationMode),
    dataBreakdown: {
      'Pretraining tasks': numberOfPretrainingTasks.toLocaleString(),
      'Rows per task': rowsPerTask.toLocaleString(),
      'Columns per task': columnsPerTask,
      'Tokenization mode': tokenizationMode,
      'Tokens per task': tokensPerTask.toLocaleString(),
      'Sequence length': seqLengthPerTask.toLocaleString(),
      'Effective training tokens': formatSci(effectiveTokens),
    },
  };
}
