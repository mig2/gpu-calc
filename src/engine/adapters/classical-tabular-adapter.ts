import type { BaseHardwareConfig, ConfidenceLevel } from '../types';
import type { GbdtAlgorithm, ClassicalTabularConfig } from '../types';

type ClassicalScenario = BaseHardwareConfig & ClassicalTabularConfig;

// Benchmark-derived defaults: work_units per second on a single GPU
// These are rough estimates for typical workloads
const DEFAULT_THROUGHPUT: Record<string, number> = {
  lightgbm: 5e8,    // ~500M work units/sec on GPU
  xgboost: 3e8,     // ~300M work units/sec on GPU
  catboost: 4e8,    // ~400M work units/sec on GPU
  random_forest: 2e8, // ~200M work units/sec on GPU
  custom: 3e8,
};

const CPU_THROUGHPUT_FACTOR = 0.1; // CPU is roughly 10x slower

export type ClassicalEstimateResult = {
  workUnits: number;
  estimatedSeconds: number;
  estimatedHours: number;
  estimatedDays: number;
  throughputUsed: number;
  confidence: ConfidenceLevel;
  trace: string[];
  warnings: string[];
  dataBreakdown: Record<string, string | number>;
};

export function estimateClassicalTabular(scenario: ClassicalScenario): ClassicalEstimateResult {
  const {
    algorithm,
    rows,
    columns,
    boostingRounds,
    cvFolds,
    hyperparameterTrials,
    cpuOrGpu,
    throughputCoefficient,
  } = scenario;

  const warnings: string[] = [];

  const baseWorkUnits = rows * columns * boostingRounds;
  const totalWorkUnits = baseWorkUnits * cvFolds * hyperparameterTrials;

  const defaultThroughput = DEFAULT_THROUGHPUT[algorithm] ?? DEFAULT_THROUGHPUT['custom'];
  const throughput = throughputCoefficient > 0
    ? throughputCoefficient
    : defaultThroughput * (cpuOrGpu === 'cpu' ? CPU_THROUGHPUT_FACTOR : 1);

  const estimatedSeconds = totalWorkUnits / throughput;
  const estimatedHours = estimatedSeconds / 3600;
  const estimatedDays = estimatedSeconds / 86_400;

  const confidence: ConfidenceLevel = throughputCoefficient > 0 ? 'medium' : 'low';

  const trace: string[] = [
    `Algorithm = ${algorithm}`,
    `Base work = ${rows.toLocaleString()} rows x ${columns} cols x ${boostingRounds.toLocaleString()} rounds = ${baseWorkUnits.toExponential(2)}`,
    `Total work = ${baseWorkUnits.toExponential(2)} x ${cvFolds} folds x ${hyperparameterTrials} trials = ${totalWorkUnits.toExponential(2)}`,
    `Throughput = ${throughput.toExponential(2)} work units/sec (${cpuOrGpu.toUpperCase()}${throughputCoefficient > 0 ? ', calibrated' : ', estimated'})`,
    `Estimated time = ${totalWorkUnits.toExponential(2)} / ${throughput.toExponential(2)} = ${estimatedSeconds.toFixed(0)} seconds`,
    `= ${estimatedHours.toFixed(1)} hours = ${estimatedDays.toFixed(2)} days`,
  ];

  if (throughputCoefficient <= 0) {
    warnings.push(
      'No empirical throughput coefficient provided. Estimate uses rough benchmark defaults and may be significantly off. Calibrate with a known run for better accuracy.',
    );
  }

  if (cvFolds * hyperparameterTrials > 50) {
    warnings.push(
      `Cross-validation (${cvFolds} folds) x hyperparameter search (${hyperparameterTrials} trials) = ${cvFolds * hyperparameterTrials}x multiplier. This likely dominates total training time.`,
    );
  }

  if (cpuOrGpu === 'cpu') {
    warnings.push(
      'CPU implementation selected. GPU implementations of histogram-based GBDTs are typically 5-10x faster.',
    );
  }

  warnings.push(
    'Classical tabular estimates use empirical throughput models, not theoretical FLOPs. Results depend heavily on implementation, data characteristics, and hardware.',
  );

  return {
    workUnits: totalWorkUnits,
    estimatedSeconds,
    estimatedHours,
    estimatedDays,
    throughputUsed: throughput,
    confidence,
    trace,
    warnings,
    dataBreakdown: {
      'Algorithm': algorithm,
      'Rows': rows.toLocaleString(),
      'Columns': columns,
      'Boosting rounds': boostingRounds.toLocaleString(),
      'CV folds': cvFolds,
      'HP trials': hyperparameterTrials,
      'Work multiplier': `${cvFolds * hyperparameterTrials}x`,
      'Total work units': totalWorkUnits.toExponential(2),
      'Throughput': `${throughput.toExponential(2)}/sec`,
      'Implementation': cpuOrGpu.toUpperCase(),
    },
  };
}
