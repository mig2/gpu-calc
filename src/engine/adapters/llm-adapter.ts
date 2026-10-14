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
  /*
   * Data-constrained scaling (research sect. 1.4). TPP is not freely dialable:
   * D = TPP x N has to come from a real corpus. Muennighoff et al. found ~4
   * epochs of repetition nearly free with returns decaying sharply after, and
   * a 2026 refinement found even that holds only for ~3B models near 1x
   * Chinchilla, with the free-epoch count falling as N and TPP grow.
   *
   * Deliberately warns only in the regime where the corpus requirement becomes
   * implausible, not whenever a user picks a high preset -- a warning that
   * fires on a legitimate choice is just fatigue.
   */
  const tpp = scenario.trainingTokensOverride
    ? scenario.trainingTokensOverride / N
    : scenario.tokensPerParameter;
  const requiredTokens = tpp * N;
  if (requiredTokens > 30e12) {
    warnings.push(
      `Data-availability warning: ${tpp.toFixed(0)} tokens/parameter at this model size implies ${(requiredTokens / 1e12).toFixed(1)}T training tokens. The largest published corpora are around 30-36T, so this likely requires heavy repetition -- roughly 4 epochs is near-free, after which added compute buys little. The compute estimate stays correct; the quality implication does not.`,
    );
  }

  if (scenario.precision === 'FP8_DENSE' || scenario.precision === 'FP4_DENSE') {
    const label = scenario.precision === 'FP4_DENSE' ? 'FP4' : 'FP8';
    warnings.push(`${label} training is experimental. End-to-end ${label} recipes may not deliver the simple peak-based estimate shown here.`);
  }
  return warnings;
}
