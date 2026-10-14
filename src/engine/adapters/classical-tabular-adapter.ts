import type { BaseHardwareConfig, ConfidenceLevel } from '../types';
import type { GbdtAlgorithm, ClassicalTabularConfig } from '../types';

type ClassicalScenario = BaseHardwareConfig & ClassicalTabularConfig;

/**
 * Throughput defaults, in work units/second, where one work unit is
 * `1 row x 1 column x 1 boosting round` — exactly the quantity this adapter counts.
 *
 * PROVENANCE. These are not invented. Each pair is a geometric mean over the
 * >=1M-row datasets of a published benchmark, recomputed as
 * `rows * columns * rounds / seconds` so it maps onto this adapter's work unit
 * one-for-one. See docs/research/2026-10-14-alt-model-families-research.md §3.2-3.3.
 *
 *   - XGBoost / LightGBM / CatBoost: GBM-Benchmarks (XGBoost maintainer's suite),
 *     AWS p3.16xlarge = 8x V100-16GB GPU / 64 vCPU Xeon E5-2686 v4 CPU,
 *     single GPU used. Datasets: YearPredictionMSD, CoverType, Bosch, Synthetic,
 *     Higgs, Airline. https://github.com/RAMitchell/GBM-Benchmarks
 *   - LightGBM CPU is cross-checked against LightGBM's own single-machine
 *     experiments (2x E5-2690 v4, 28 cores): 1.13e9 on Higgs, 2.21e9 on MS LTR,
 *     2.17e9 on Yahoo LTR; dense geometric mean 1.76e9.
 *     https://lightgbm.readthedocs.io/en/latest/Experiments.html
 *   - Random forest: cuML accelerator benchmarks, RTX PRO 6000 Blackwell vs
 *     Threadripper PRO 7975WX (32 cores), 100 estimators. Measured GPU range is
 *     3e9-3.2e10; 5e9 is the deliberately conservative low end of that range
 *     because the CPU baseline there is scikit-learn, a much weaker
 *     implementation than LightGBM.
 *     https://docs.nvidia.com/cuml/latest/cuml-accel/benchmarks/
 *
 * VALID REGIME. All anchors were measured at:
 *   - max_depth = 6 (LightGBM's num_leaves=31 ~ 5 effective levels)
 *   - 500 boosting rounds
 *   - >=1M rows
 *   - dense numeric features (see WIDE_DATA_COLUMN_THRESHOLD and KNOWN LIMITS below)
 *   - default bin counts (~128-256)
 * `maxDepth` and `bins` scale off these anchor points; see DEPTH_CALIBRATION_POINT
 * and BINS_CALIBRATION_POINT.
 *
 * KNOWN LIMITS.
 *   - The GPU anchors are V100-era (2018). H100/Blackwell histogram building is
 *     bandwidth-bound and ~2-3x faster, so these defaults are conservative on
 *     modern hardware. That adjustment is extrapolation and is deliberately NOT
 *     applied here.
 *   - LightGBM's GPU figure is for the CUDA backend. The older OpenCL backend
 *     measured 7.7e8 and was *slower than its own CPU path on 5 of 6* datasets
 *     in GBM-Benchmarks, which is why its GPU and CPU defaults are equal rather
 *     than separated by a speedup factor.
 *   - The `rows x columns` work model collapses on sparse / one-hot data:
 *     LightGBM's Expo (11M x 700 one-hot) implies 6.15e10 and Allstate
 *     (13.2M x 4,228) implies 1.88e11, i.e. 35x and 100x the dense rate, because
 *     Exclusive Feature Bundling collapses the one-hot columns. A warning fires
 *     above WIDE_DATA_COLUMN_THRESHOLD; the numbers are not corrected.
 */
const THROUGHPUT_DEFAULTS: Record<GbdtAlgorithm, { gpu: number; cpu: number }> = {
  // GBM-Benchmarks geomean >=1M rows: GPU 8.57e9, CPU 5.40e8 (speedup 13.4x, range 4.5-23.4x)
  xgboost: { gpu: 8e9, cpu: 5e8 },
  // GBM-Benchmarks OpenCL geomean: GPU 7.66e8 vs CPU 1.03e9 => 0.69x, GPU *slower*.
  // 1.5e9 for both reflects the newer CUDA backend at rough parity with CPU.
  lightgbm: { gpu: 1.5e9, cpu: 1.5e9 },
  // GBM-Benchmarks geomean >=1M rows: GPU 5.62e9, CPU 7.19e8 (speedup 9.7x, range 3.9-13.2x)
  catboost: { gpu: 5e9, cpu: 7e8 },
  // cuML: GPU 3e9-3.2e10 measured, conservative low end taken; CPU from the same suite
  random_forest: { gpu: 5e9, cpu: 1e9 },
  // No measurement exists for "custom". Mid-range placeholder, flagged low confidence.
  custom: { gpu: 2e9, cpu: 5e8 },
};

/**
 * Depth at which THROUGHPUT_DEFAULTS were measured. Histogram-based GBDT cost is
 * linear in depth: at each tree level the union of nodes covers all rows, so a
 * level costs `rows x features` bin increments, over `depth` levels.
 * Research §3.1. GBM-Benchmarks fixed `max_depth = 6`.
 */
const DEPTH_CALIBRATION_POINT = 6;

/**
 * Bin count the GPU coefficients are anchored at. Lower `max_bin` makes the GPU
 * disproportionately faster while leaving CPU nearly flat.
 *
 * Slope source: CatBoost's own GPU-vs-CPU benchmark on Epsilon (400K x 2,000,
 * dense numeric, V100-PCI): 9.26x speedup at 128 bins, 13.33x at 32 bins.
 * That is 1.44x over two halvings of the bin count, i.e. 1.20x per halving, so
 * the exponent is log2(1.20) = 0.263.
 * https://github.com/catboost/catboost/tree/master/catboost/benchmarks/gpu_vs_cpu_training_speed
 *
 * Clamped because the measured span is only 32-128 bins; anything outside that
 * is extrapolation.
 */
const BINS_CALIBRATION_POINT = 256;
const BINS_EXPONENT = 0.263;
const BINS_FACTOR_MIN = 0.7;
const BINS_FACTOR_MAX = 2.0;

/**
 * Row count above which the GPU coefficients hold, and below which GPU loses its
 * advantage to transfer and kernel-launch overhead. Research §3.4: a 100K-row
 * dataset "can train 2-3x slower on GPU".
 * https://lightgbm.readthedocs.io/en/latest/GPU-Performance.html
 */
const GPU_FULL_EFFICIENCY_ROWS = 1e6;
const GPU_SMALL_DATA_ROWS = 1e5;
/** Midpoint of the cited 2-3x small-data GPU penalty, expressed against CPU. */
const GPU_SMALL_DATA_SLOWDOWN_VS_CPU = 2.5;

/** Column count above which the `rows x columns` work model is unreliable (§3.2, §4 item 5). */
const WIDE_DATA_COLUMN_THRESHOLD = 500;

/**
 * CV x HPO multiplier thresholds. TabArena — the current reference tabular
 * benchmark — uses 200 configs x 8 inner folds = 1,600 fits per model per outer
 * fold, so the previous `> 50` threshold fired on every realistic configuration.
 * Research §3.6, arXiv:2506.16791.
 */
const HPO_NOTE_THRESHOLD = 100;
const HPO_DOMINATES_THRESHOLD = 500;

/**
 * Wall-clock above which a GPU starts to be worth its operational cost, and the
 * minimum speedup that justifies one. These are product judgments, not measured
 * values — research §3.6 only observes that a 250x CV x HPO multiplier "turns the
 * repo's 12-60 s single fit into a 50-minute to 4-hour job, which is the only
 * place a GPU starts to matter for mid-size data".
 */
const CPU_SUFFICIENT_SECONDS = 900;
const GPU_WORTH_IT_SPEEDUP = 2;
/**
 * Below this speedup a GPU is called slower outright. The 5% band keeps exact
 * CPU/GPU parity — which is what LightGBM's coefficients encode — out of the
 * "slower" bucket, where it would overstate the evidence.
 */
const GPU_SLOWER_THRESHOLD = 0.95;

/**
 * The GBDT pane's real question is not "how many GPUs" but "do I need a GPU at
 * all" (research §0.3, §4 item 18). 1 GPU is the answer whenever a GPU is the
 * answer: XGBoost 3.0 external memory trains a 1 TB dataset on a single GH200.
 */
export type ClassicalGpuVerdict = 'gpu_recommended' | 'cpu_sufficient' | 'gpu_likely_slower';

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
  /** "Do I need a GPU at all" verdict — the headline answer for this pane. */
  gpuVerdict: ClassicalGpuVerdict;
  /** One-line justification for `gpuVerdict`, safe to render verbatim. */
  gpuVerdictReason: string;
  /** Wall-clock on the CPU path, regardless of which path is selected. */
  cpuAlternativeSeconds: number;
  /** Wall-clock on a single GPU, regardless of which path is selected. */
  gpuAlternativeSeconds: number;
  /** Always 1 when a GPU is worth using. Multi-GPU GBDT is a niche. */
  recommendedGpuCount: number;
};

/**
 * GPU efficiency multiplier for small row counts, derived (not invented) from the
 * two cited anchors: full efficiency at >=1e6 rows, and at 1e5 rows a GPU runs
 * ~2.5x slower than the same library's CPU path. Log-interpolated between them;
 * held flat below 1e5, where overhead dominates.
 */
function gpuSmallDataEfficiency(rows: number, coeffs: { gpu: number; cpu: number }): number {
  if (rows >= GPU_FULL_EFFICIENCY_ROWS) return 1;
  const efficiencyAtSmallData = coeffs.cpu / GPU_SMALL_DATA_SLOWDOWN_VS_CPU / coeffs.gpu;
  if (rows <= GPU_SMALL_DATA_ROWS) return Math.min(1, efficiencyAtSmallData);
  const t =
    (Math.log10(rows) - Math.log10(GPU_SMALL_DATA_ROWS)) /
    (Math.log10(GPU_FULL_EFFICIENCY_ROWS) - Math.log10(GPU_SMALL_DATA_ROWS));
  return Math.min(1, Math.pow(efficiencyAtSmallData, 1 - t));
}

/** `max_bin` scaling, GPU only — CPU throughput is nearly flat in bin count. */
function binsFactor(bins: number): number {
  if (!bins || bins <= 0) return 1;
  const raw = Math.pow(BINS_CALIBRATION_POINT / bins, BINS_EXPONENT);
  return Math.min(BINS_FACTOR_MAX, Math.max(BINS_FACTOR_MIN, raw));
}

export function estimateClassicalTabular(scenario: ClassicalScenario): ClassicalEstimateResult {
  const {
    algorithm,
    rows,
    columns,
    boostingRounds,
    maxDepth,
    bins,
    cvFolds,
    hyperparameterTrials,
    cpuOrGpu,
    throughputCoefficient,
  } = scenario;

  const warnings: string[] = [];

  const effectiveDepth = maxDepth > 0 ? maxDepth : DEPTH_CALIBRATION_POINT;
  const depthFactor = effectiveDepth / DEPTH_CALIBRATION_POINT;

  const baseWorkUnits = rows * columns * boostingRounds;
  const depthAdjustedWork = baseWorkUnits * depthFactor;
  const searchMultiplier = cvFolds * hyperparameterTrials;
  const totalWorkUnits = depthAdjustedWork * searchMultiplier;

  const coeffs = THROUGHPUT_DEFAULTS[algorithm] ?? THROUGHPUT_DEFAULTS.custom;
  const gpuEfficiency = gpuSmallDataEfficiency(rows, coeffs);
  const binFactor = binsFactor(bins);
  const defaultGpuThroughput = coeffs.gpu * gpuEfficiency * binFactor;
  const defaultCpuThroughput = coeffs.cpu;

  const calibrated = throughputCoefficient > 0;
  const defaultThroughput = cpuOrGpu === 'cpu' ? defaultCpuThroughput : defaultGpuThroughput;
  const throughput = calibrated ? throughputCoefficient : defaultThroughput;

  const estimatedSeconds = totalWorkUnits / throughput;
  const estimatedHours = estimatedSeconds / 3600;
  const estimatedDays = estimatedSeconds / 86_400;

  // Both sides of the verdict always use the default coefficients for the path the
  // user did NOT select, so the comparison stays apples-to-apples.
  const cpuAlternativeSeconds =
    totalWorkUnits / (calibrated && cpuOrGpu === 'cpu' ? throughputCoefficient : defaultCpuThroughput);
  const gpuAlternativeSeconds =
    totalWorkUnits / (calibrated && cpuOrGpu === 'gpu' ? throughputCoefficient : defaultGpuThroughput);

  const gpuSpeedup = cpuAlternativeSeconds / gpuAlternativeSeconds;

  let gpuVerdict: ClassicalGpuVerdict;
  let gpuVerdictReason: string;
  if (gpuSpeedup < GPU_SLOWER_THRESHOLD) {
    gpuVerdict = 'gpu_likely_slower';
    gpuVerdictReason =
      `A GPU is predicted to be ${(1 / gpuSpeedup).toFixed(1)}x slower here` +
      (algorithm === 'lightgbm'
        ? ' — LightGBM GPU was slower than its own CPU path on 5 of 6 reference datasets.'
        : ` — at ${rows.toLocaleString()} rows, transfer and kernel-launch overhead dominate.`);
  } else if (gpuSpeedup < GPU_WORTH_IT_SPEEDUP) {
    gpuVerdict = 'cpu_sufficient';
    gpuVerdictReason =
      (gpuSpeedup < 1 / GPU_SLOWER_THRESHOLD
        ? `A GPU buys you nothing here — it is no faster than CPU`
        : `A GPU is only ${gpuSpeedup.toFixed(1)}x faster here, which does not justify the added complexity`) +
      (algorithm === 'lightgbm'
        ? ` (LightGBM GPU measures 0.5-1.5x its own CPU path). CPU finishes in ${formatSeconds(cpuAlternativeSeconds)}.`
        : `. CPU finishes in ${formatSeconds(cpuAlternativeSeconds)}.`);
  } else if (cpuAlternativeSeconds <= CPU_SUFFICIENT_SECONDS) {
    gpuVerdict = 'cpu_sufficient';
    gpuVerdictReason =
      `CPU finishes in ${formatSeconds(cpuAlternativeSeconds)}; a GPU would cut that to ` +
      `${formatSeconds(gpuAlternativeSeconds)} but is not needed at this scale.`;
  } else {
    gpuVerdict = 'gpu_recommended';
    gpuVerdictReason =
      `1 GPU cuts ${formatSeconds(cpuAlternativeSeconds)} on CPU to ${formatSeconds(gpuAlternativeSeconds)} ` +
      `(${gpuSpeedup.toFixed(1)}x). One GPU is enough — multi-GPU GBDT is a niche.`;
  }

  const confidence: ConfidenceLevel = calibrated
    ? 'medium'
    : algorithm === 'custom' || gpuEfficiency < 1
      ? 'low'
      : 'medium-low';

  const trace: string[] = [
    `Algorithm = ${algorithm}`,
    `Base work = ${rows.toLocaleString()} rows x ${columns} cols x ${boostingRounds.toLocaleString()} rounds = ${baseWorkUnits.toExponential(2)}`,
    `Depth scaling = x ${effectiveDepth} / ${DEPTH_CALIBRATION_POINT} (coefficients calibrated at max_depth ${DEPTH_CALIBRATION_POINT}) = ${depthAdjustedWork.toExponential(2)}`,
    `Total work = ${depthAdjustedWork.toExponential(2)} x ${cvFolds} folds x ${hyperparameterTrials} trials = ${totalWorkUnits.toExponential(2)}`,
    calibrated
      ? `Throughput = ${throughput.toExponential(2)} work units/sec (${cpuOrGpu.toUpperCase()}, user-calibrated)`
      : `Throughput = ${throughput.toExponential(2)} work units/sec (${cpuOrGpu.toUpperCase()} default: ${coeffs[cpuOrGpu].toExponential(2)}${cpuOrGpu === 'gpu' ? ` x ${gpuEfficiency.toFixed(3)} small-data efficiency x ${binFactor.toFixed(2)} bins(${bins})` : ''})`,
    `Estimated time = ${totalWorkUnits.toExponential(2)} / ${throughput.toExponential(2)} = ${estimatedSeconds.toFixed(0)} seconds`,
    `= ${estimatedHours.toFixed(1)} hours = ${estimatedDays.toFixed(2)} days`,
    `GPU verdict = ${gpuVerdict} (CPU ${formatSeconds(cpuAlternativeSeconds)} vs 1 GPU ${formatSeconds(gpuAlternativeSeconds)})`,
  ];

  if (!calibrated) {
    trace.push(
      `Coefficient provenance: ${algorithm === 'random_forest' ? 'cuML accelerator benchmarks' : algorithm === 'custom' ? 'none (mid-range placeholder)' : 'GBM-Benchmarks (1x V100 / 64 vCPU Xeon E5-2686 v4)'}, 500 rounds, max_depth 6, >=1M dense numeric rows`,
    );
    trace.push(
      `Published GPU speedup ranges at >=1M rows: XGBoost 4.5-23.4x, CatBoost 3.9-13.2x, LightGBM-OpenCL 0.5-1.5x (often <1x)`,
    );
  }

  if (!calibrated) {
    warnings.push(
      'No empirical throughput coefficient provided. Estimate uses benchmark-derived defaults (GBM-Benchmarks / LightGBM Experiments / CatBoost GPU benchmarks, 500 rounds at max_depth 6 on >=1M dense rows) and is accurate to roughly 2-3x at best. Calibrate with a known run for better accuracy.',
    );
  }

  if (algorithm === 'custom') {
    warnings.push(
      'No published benchmark exists for a custom implementation. The default coefficient is a mid-range placeholder — calibrate it against one of your own runs.',
    );
  }

  if (searchMultiplier > HPO_DOMINATES_THRESHOLD) {
    warnings.push(
      `Cross-validation (${cvFolds} folds) x hyperparameter search (${hyperparameterTrials} trials) = ${searchMultiplier}x multiplier. At this scale the search dominates total training time — prune it (Hyperband / early stopping) rather than buying hardware.`,
    );
  } else if (searchMultiplier > HPO_NOTE_THRESHOLD) {
    warnings.push(
      `Cross-validation x hyperparameter search = ${searchMultiplier}x multiplier, so the search is most of the wall-clock. This is normal: TabArena uses 200 configs x 8 inner folds = 1,600 fits per model per outer fold. The practical literature puts diminishing returns around 50-100 trials.`,
    );
  }

  if (cpuOrGpu === 'gpu' && gpuVerdict === 'gpu_likely_slower') {
    warnings.push(
      `GPU will likely be SLOWER than CPU for this configuration. ${gpuVerdictReason} GPU histogram building only pays off above roughly 1M rows of dense numeric features; a 100K-row dataset can train 2-3x slower on GPU.`,
    );
  } else if (cpuOrGpu === 'gpu' && gpuSpeedup < GPU_WORTH_IT_SPEEDUP) {
    warnings.push(
      `GPU gains little here (${gpuSpeedup.toFixed(1)}x vs CPU). ${gpuVerdictReason} The published 5-10x GPU speedup applies to XGBoost and CatBoost on >=1M rows of dense numeric features, not to every library or every dataset size.`,
    );
  }

  if (cpuOrGpu === 'cpu' && gpuVerdict !== 'gpu_likely_slower') {
    const speedupText =
      algorithm === 'xgboost'
        ? 'XGBoost GPU measured 13.4x faster (range 4.5-23.4x)'
        : algorithm === 'catboost'
          ? 'CatBoost GPU measured 9.7x faster (range 3.9-13.2x)'
          : algorithm === 'random_forest'
            ? 'cuML random forest measured 3-26x faster, rising with dataset size'
            : algorithm === 'lightgbm'
              ? 'LightGBM GPU measures only 0.5-1.5x its own CPU path, so there is little to gain'
              : 'the measured speedup for this algorithm is modest';
    warnings.push(
      `CPU implementation selected. On >=1M rows of dense numeric features, ${speedupText} — here a single GPU would take about ${formatSeconds(gpuAlternativeSeconds)} versus ${formatSeconds(cpuAlternativeSeconds)} on CPU.`,
    );
  }

  if (columns > WIDE_DATA_COLUMN_THRESHOLD) {
    warnings.push(
      `${columns} columns: if any of these are one-hot encoded or otherwise sparse, the rows x columns work model breaks down by up to ~100x. LightGBM's Exclusive Feature Bundling collapses sparse columns, so its measured rate on Expo (11M x 700 one-hot) is 35x and on Allstate (13.2M x 4,228) is 100x the dense rate. This estimate assumes dense numeric features and will be a large over-estimate otherwise.`,
    );
  }

  if (rows < GPU_FULL_EFFICIENCY_ROWS) {
    warnings.push(
      `At ${rows.toLocaleString()} rows you are below the ~1M-row regime the throughput coefficients were measured in. GPU efficiency is scaled down by ${gpuEfficiency.toFixed(2)}x by interpolation, which is derived rather than directly measured — treat the GPU number as indicative only.`,
    );
  }

  warnings.push(
    'VRAM is almost never the binding constraint for GBDTs: a binned matrix is roughly rows x columns bytes (a 1M x 100 table is ~100 MB), and XGBoost 3.0 external memory trains a 1 TB dataset on a single GH200. If memory binds, the answer is external memory on one device, not more GPUs.',
  );

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
    gpuVerdict,
    gpuVerdictReason,
    cpuAlternativeSeconds,
    gpuAlternativeSeconds,
    recommendedGpuCount: 1,
    dataBreakdown: {
      'Algorithm': algorithm,
      'Rows': rows.toLocaleString(),
      'Columns': columns,
      'Boosting rounds': boostingRounds.toLocaleString(),
      'Max depth': `${effectiveDepth} (x${depthFactor.toFixed(2)} vs depth-${DEPTH_CALIBRATION_POINT} calibration)`,
      'Max bins': cpuOrGpu === 'gpu' ? `${bins} (x${binFactor.toFixed(2)} GPU throughput)` : `${bins} (CPU is flat in bins)`,
      'CV folds': cvFolds,
      'HP trials': hyperparameterTrials,
      'Work multiplier': `${searchMultiplier}x`,
      'Total work units': totalWorkUnits.toExponential(2),
      'Throughput': `${throughput.toExponential(2)}/sec`,
      'Implementation': cpuOrGpu.toUpperCase(),
    },
  };
}

function formatSeconds(seconds: number): string {
  if (seconds < 90) return `${seconds.toFixed(0)} s`;
  if (seconds < 5400) return `${(seconds / 60).toFixed(1)} min`;
  if (seconds < 172_800) return `${(seconds / 3600).toFixed(1)} h`;
  return `${(seconds / 86_400).toFixed(1)} days`;
}
