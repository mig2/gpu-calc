import type { AdapterResult, BaseHardwareConfig, LlmConfig, ConfidenceLevel, TrainingMode } from '../types';

type LlmScenario = BaseHardwareConfig & LlmConfig;

function formatSci(n: number, digits = 2): string {
  return n.toExponential(digits);
}

function getConfidence(trainingMode: TrainingMode): ConfidenceLevel {
  switch (trainingMode) {
    case 'FULL_PRETRAINING': return 'high';
    case 'CONTINUED_PRETRAINING': return 'medium';
    default: return 'medium-low';
  }
}

export function computeLlmFlops(scenario: LlmScenario): AdapterResult {
  const N = scenario.modelParameters;
  const D = scenario.trainingTokensOverride ?? (scenario.tokensPerParameter * N);
  const baseFlops = scenario.architectureFactor * N * D;
  const totalFlops = baseFlops * scenario.overheadFactor;

  const tokenSource = scenario.trainingTokensOverride
    ? `D = ${formatSci(scenario.trainingTokensOverride)} (override)`
    : `D = ${scenario.tokensPerParameter} × N = ${formatSci(D)}`;

  const warnings = buildLlmWarnings(scenario, N);

  return {
    effectiveTokens: D,
    baseFlops,
    totalFlops,
    confidence: getConfidence(scenario.trainingMode),
    trace: [
      `N = ${formatSci(N)}`,
      tokenSource,
      `Base FLOPs = ${scenario.architectureFactor} × N × D = ${formatSci(baseFlops)}`,
      `Total FLOPs = ${formatSci(baseFlops)} × ${scenario.overheadFactor} = ${formatSci(totalFlops)}`,
    ],
    warnings,
    dataBreakdown: {
      'Model parameters': formatSci(N),
      'Training tokens': formatSci(D),
      'Token source': scenario.trainingTokensOverride ? 'Manual override' : `${scenario.tokensPerParameter} × N`,
      'Architecture factor': scenario.architectureFactor,
    },
  };
}

function buildLlmWarnings(scenario: LlmScenario, N: number): string[] {
  const warnings: string[] = [];
  if (N > 70e9 && scenario.trainingWindowSeconds < 14 * 86_400) {
    warnings.push('Large-cluster warning: training a model >70B in under 14 days requires a very large cluster with significant operational complexity.');
  }
  if (scenario.trainingMode !== 'FULL_PRETRAINING') {
    warnings.push('The full-pretraining formula (6ND) may overestimate compute for this training mode. Task-specific formulas are recommended.');
  }
  if (scenario.precision === 'FP8_DENSE') {
    warnings.push('FP8 training is experimental. End-to-end FP8 recipes may not deliver the simple peak-based estimate shown here.');
  }
  return warnings;
}
