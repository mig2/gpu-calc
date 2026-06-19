import type { AdapterResult, BaseHardwareConfig, TimeSeriesConfig } from '../types';

type TimeSeriesScenario = BaseHardwareConfig & TimeSeriesConfig;

function formatSci(n: number, digits = 2): string {
  return n.toExponential(digits);
}

export function computeTimeSeriesFlops(scenario: TimeSeriesScenario): AdapterResult {
  const {
    modelParameters: N,
    numberOfSeries,
    averageTimestepsPerSeries,
    variablesPerSeries,
    lookbackWindow,
    forecastHorizon,
    stride,
    patchSize,
    tokenizationMode,
    customTokensPerWindow,
    epochs,
    architectureFactor,
    overheadFactor,
  } = scenario;

  const warnings: string[] = [];
  const trace: string[] = [];

  const usableTimesteps = averageTimestepsPerSeries - lookbackWindow - forecastHorizon;

  if (usableTimesteps < 0) {
    warnings.push(
      'Invalid window geometry: lookback + horizon exceeds average timesteps per series. No training windows can be generated.',
    );
    return {
      effectiveTokens: 0,
      baseFlops: 0,
      totalFlops: 0,
      trace: [`N = ${formatSci(N)}`, 'Invalid window geometry — no tokens generated.'],
      warnings,
      confidence: 'medium',
      dataBreakdown: {
        'Windows per series': 0,
        'Tokens per window': 0,
        'Effective training tokens': 0,
      },
    };
  }

  const windowsPerSeries = Math.floor(usableTimesteps / stride) + 1;
  const patchesPerWindow = Math.ceil(lookbackWindow / patchSize);

  let tokensPerWindow: number;
  switch (tokenizationMode) {
    case 'channel_compressed':
      tokensPerWindow = patchesPerWindow;
      break;
    case 'channel_expanded':
      tokensPerWindow = variablesPerSeries * patchesPerWindow;
      break;
    case 'custom':
      tokensPerWindow = customTokensPerWindow ?? patchesPerWindow;
      break;
    default:
      tokensPerWindow = patchesPerWindow;
  }

  const effectiveTokens = numberOfSeries * windowsPerSeries * tokensPerWindow * epochs;
  const baseFlops = architectureFactor * N * effectiveTokens;
  const totalFlops = baseFlops * overheadFactor;

  trace.push(`N = ${formatSci(N)}`);
  trace.push(`Series = ${numberOfSeries.toLocaleString()}`);
  trace.push(`Windows/series = floor((${averageTimestepsPerSeries} - ${lookbackWindow} - ${forecastHorizon}) / ${stride}) + 1 = ${windowsPerSeries}`);
  trace.push(`Patches/window = ceil(${lookbackWindow} / ${patchSize}) = ${patchesPerWindow}`);
  trace.push(`Tokens/window = ${tokenizationMode === 'channel_expanded' ? `${variablesPerSeries} × ${patchesPerWindow} = ` : ''}${tokensPerWindow}`);
  trace.push(`Effective tokens = ${numberOfSeries.toLocaleString()} × ${windowsPerSeries} × ${tokensPerWindow}${epochs > 1 ? ` × ${epochs}` : ''} = ${formatSci(effectiveTokens)}`);
  trace.push(`Base FLOPs = ${architectureFactor} × ${formatSci(N)} × ${formatSci(effectiveTokens)} = ${formatSci(baseFlops)}`);
  trace.push(`Total FLOPs = ${formatSci(baseFlops)} × ${overheadFactor} = ${formatSci(totalFlops)}`);

  if (stride <= lookbackWindow * 0.1) {
    warnings.push(
      `Small stride (${stride}) relative to lookback (${lookbackWindow}) creates many overlapping windows and may inflate data volume significantly.`,
    );
  }

  if (tokenizationMode === 'channel_expanded' && variablesPerSeries > 20) {
    warnings.push(
      `Channel-expanded tokenization with ${variablesPerSeries} variables creates ${tokensPerWindow} tokens per window. This may cause significant compute inflation.`,
    );
  }

  if (patchSize > lookbackWindow) {
    warnings.push(
      'Patch size is larger than lookback window. Each window produces less than one patch.',
    );
  }

  warnings.push(
    'Time-series compute estimate uses a transformer-style approximation (factor × N × tokens). This is not an empirically calibrated time-series scaling law.',
  );

  return {
    effectiveTokens,
    baseFlops,
    totalFlops,
    trace,
    warnings,
    confidence: 'medium',
    dataBreakdown: {
      'Series': numberOfSeries.toLocaleString(),
      'Timesteps per series': averageTimestepsPerSeries.toLocaleString(),
      'Windows per series': windowsPerSeries,
      'Patches per window': patchesPerWindow,
      'Tokens per window': tokensPerWindow,
      'Effective training tokens': formatSci(effectiveTokens),
    },
  };
}
