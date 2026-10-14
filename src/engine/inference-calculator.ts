import type { ApiModel, CloudGpuInstance, SelfHostEntry } from '../data/types';

/**
 * Average days in a month (365 / 12 ≈ 30.4167). Used everywhere a monthly
 * figure is derived so that monthly and annual costs stay consistent:
 * twelve such months are exactly one 365-day year.
 */
export const DAYS_PER_MONTH = 365 / 12;

export type UsagePattern = {
  requestsPerDay: number;
  avgInputTokens: number;
  avgOutputTokens: number;
};

export type ApiCostResult = {
  provider: string;
  model: string;
  monthlyCost: number;
  costPerRequest: number;
  annualCost: number;
  inputCostShare: number;   // fraction of cost from input tokens
  outputCostShare: number;
  trace: string[];
};

export type SelfHostCostResult = {
  model: string;
  cloudProvider: string;
  instance: string;
  /** GPUs per replica, from the throughput table row. */
  gpuCount: number;
  /** Replicas actually provisioned (user-chosen, or the computed requirement). */
  replicas: number;
  /** Replicas the offered load requires, before any user override. */
  replicasNeeded: number;
  /** GPUs across all replicas. */
  totalGpus: number;
  monthlyGpuCost: number;
  /** Capacity of ONE replica. */
  maxOutputTokensPerSec: number;
  /** Capacity across all provisioned replicas. */
  totalOutputTokensPerSec: number;
  requiredTokensPerSec: number;
  canServe: boolean;
  utilizationPercent: number;
  estimatedTtftMs: number;
  trace: string[];
  warnings: string[];
};

export type BreakevenResult = {
  breakevenRequestsPerDay: number;
  apiCheaperBelow: boolean;
  monthlySavingsAtCurrentVolume: number;
  annualSavingsAtCurrentVolume: number;
  /** Effective $ per million output tokens, API side. */
  apiCostPerMillionOutputTokens: number;
  /** Effective $ per million output tokens, self-host side, at this provisioning. */
  selfHostCostPerMillionOutputTokens: number;
  /**
   * True when self-host is cheaper per token, so it wins at every volume that
   * keeps the replicas busy -- i.e. there is no crossover volume to find.
   */
  selfHostCheaperPerToken: boolean;
  /** Why the breakeven volume is or is not a meaningful number here. */
  breakevenNote: string;
};

export function calculateApiCost(
  apiModel: ApiModel,
  providerName: string,
  usage: UsagePattern,
): ApiCostResult {
  const dailyInputTokens = usage.requestsPerDay * usage.avgInputTokens;
  const dailyOutputTokens = usage.requestsPerDay * usage.avgOutputTokens;

  const dailyInputCost = (dailyInputTokens / 1e6) * apiModel.inputPer1M;
  const dailyOutputCost = (dailyOutputTokens / 1e6) * apiModel.outputPer1M;
  const dailyCost = dailyInputCost + dailyOutputCost;

  const monthlyCost = dailyCost * DAYS_PER_MONTH;
  // Derived from monthlyCost so annualCost === 12 * monthlyCost holds exactly
  // (equivalent to dailyCost * 365).
  const annualCost = monthlyCost * 12;

  const requestsPerDayNonZero = usage.requestsPerDay || 1;
  const costPerRequest = dailyCost / requestsPerDayNonZero;

  const totalDailyCostNonZero = dailyCost || 1;
  const inputCostShare = dailyInputCost / totalDailyCostNonZero;
  const outputCostShare = dailyOutputCost / totalDailyCostNonZero;

  return {
    provider: providerName,
    model: apiModel.model,
    monthlyCost,
    costPerRequest,
    annualCost,
    inputCostShare,
    outputCostShare,
    trace: [
      `Requests/day = ${usage.requestsPerDay.toLocaleString()}`,
      `Avg input tokens = ${usage.avgInputTokens.toLocaleString()}, Avg output tokens = ${usage.avgOutputTokens.toLocaleString()}`,
      `Daily input tokens = ${dailyInputTokens.toLocaleString()} → $${dailyInputCost.toFixed(2)} (@ $${apiModel.inputPer1M}/M)`,
      `Daily output tokens = ${dailyOutputTokens.toLocaleString()} → $${dailyOutputCost.toFixed(2)} (@ $${apiModel.outputPer1M}/M)`,
      `Daily cost = $${dailyCost.toFixed(2)}`,
      `Monthly cost = $${monthlyCost.toFixed(2)}`,
      `Annual cost = $${annualCost.toFixed(2)}`,
      `Cost per request = $${costPerRequest.toFixed(4)}`,
    ],
  };
}

export function calculateSelfHostCost(
  throughput: SelfHostEntry,
  instance: CloudGpuInstance,
  cloudProviderName: string,
  usage: UsagePattern,
  pricingTier: 'onDemand' | 'reserved' | 'spot' = 'onDemand',
  /**
   * Replicas to provision. Omit to auto-size to the offered load, which is
   * almost always what the user means: reporting the cost of a configuration
   * that provably cannot serve the workload is not a useful answer.
   * Pass an explicit number to model deliberate under- or over-provisioning.
   */
  replicasOverride?: number,
): SelfHostCostResult {
  const warnings: string[] = [];

  // Cost
  const pricePerHr =
    pricingTier === 'reserved' ? (instance.reservedPerHr ?? instance.onDemandPerHr) :
    pricingTier === 'spot' ? (instance.spotPerHr ?? instance.onDemandPerHr) :
    instance.onDemandPerHr;

  // Instances to hold ONE replica of the model. This is a function of model
  // size and instance shape only -- it says nothing about serving capacity.
  const instancesPerReplica = Math.ceil(throughput.gpuCount / instance.gpuCount);

  // Offered load, and how many replicas it takes to serve it.
  const requiredOutputTokensPerSec = (usage.requestsPerDay * usage.avgOutputTokens) / 86_400;
  const perReplicaCapacity = throughput.outputTokensPerSec;
  const replicasNeeded =
    perReplicaCapacity > 0
      ? Math.max(1, Math.ceil(requiredOutputTokensPerSec / perReplicaCapacity))
      : 1;
  const replicas = Math.max(1, Math.floor(replicasOverride ?? replicasNeeded));

  const totalOutputTokensPerSec = perReplicaCapacity * replicas;
  const totalGpus = throughput.gpuCount * replicas;
  const monthlyGpuCost =
    pricePerHr * 24 * DAYS_PER_MONTH * instancesPerReplica * replicas;

  const canServe = requiredOutputTokensPerSec <= totalOutputTokensPerSec;
  const utilizationPercent =
    totalOutputTokensPerSec > 0
      ? (requiredOutputTokensPerSec / totalOutputTokensPerSec) * 100
      : 0;

  // Rough TTFT estimate (larger models = higher latency)
  const paramBillions = parseFloat(throughput.parameters) || 7;
  const estimatedTtftMs = Math.round(50 + paramBillions * 3);

  if (!canServe) {
    warnings.push(
      `Under-provisioned: ${requiredOutputTokensPerSec.toFixed(1)} tok/s required against ${totalOutputTokensPerSec.toFixed(1)} tok/s from ${replicas} replica${replicas === 1 ? '' : 's'}. Raise replicas to ${replicasNeeded} (${throughput.gpuCount * replicasNeeded} GPUs total) or reduce request volume.`,
    );
  }

  if (replicasOverride == null && replicasNeeded > 1) {
    warnings.push(
      `Auto-sized to ${replicasNeeded} replicas (${totalGpus} GPUs) to serve ${requiredOutputTokensPerSec.toFixed(1)} tok/s. One replica of ${throughput.model} on ${throughput.gpuCount}x ${throughput.gpu} serves ${perReplicaCapacity} tok/s.`,
    );
  }

  if (utilizationPercent < 10) {
    warnings.push(
      `GPU utilization is only ${utilizationPercent.toFixed(1)}%. The hardware is significantly over-provisioned for this usage level.`,
    );
  }

  if (pricingTier === 'spot') {
    warnings.push('Spot/preemptible instances can be interrupted. Not suitable for latency-sensitive production workloads.');
  }

  return {
    model: throughput.model,
    cloudProvider: cloudProviderName,
    instance: instance.instance,
    gpuCount: throughput.gpuCount,
    replicas,
    replicasNeeded,
    totalGpus,
    monthlyGpuCost,
    maxOutputTokensPerSec: perReplicaCapacity,
    totalOutputTokensPerSec,
    requiredTokensPerSec: requiredOutputTokensPerSec,
    canServe,
    utilizationPercent,
    estimatedTtftMs,
    trace: [
      `Model: ${throughput.model} on ${throughput.gpuCount}× ${throughput.gpu} (${throughput.framework}, ${throughput.quantization})`,
      `Cloud: ${cloudProviderName} ${instance.instance} @ $${pricePerHr.toFixed(2)}/hr (${pricingTier})`,
      `Instances per replica: ${instancesPerReplica} (sized to hold the model, not to serve load)`,
      `Required throughput = ${usage.requestsPerDay.toLocaleString()} req/day × ${usage.avgOutputTokens} tokens / 86400s = ${requiredOutputTokensPerSec.toFixed(1)} tok/s`,
      `Capacity per replica = ${perReplicaCapacity} tok/s`,
      `Replicas needed = ceil(${requiredOutputTokensPerSec.toFixed(1)} / ${perReplicaCapacity}) = ${replicasNeeded}${replicasOverride != null ? `; overridden to ${replicas}` : ''}`,
      `Total capacity = ${perReplicaCapacity} × ${replicas} = ${totalOutputTokensPerSec.toFixed(1)} tok/s across ${totalGpus} GPUs`,
      `Monthly cost = $${pricePerHr.toFixed(2)} × 24h × ${DAYS_PER_MONTH.toFixed(4)}d × ${instancesPerReplica} inst × ${replicas} replicas = $${monthlyGpuCost.toFixed(2)}`,
      `Utilization = ${utilizationPercent.toFixed(1)}%`,
      `Estimated TTFT ≈ ${estimatedTtftMs}ms`,
    ],
    warnings,
  };
}

export function calculateBreakeven(
  apiCost: ApiCostResult,
  selfHostCost: SelfHostCostResult,
  usage: UsagePattern,
): BreakevenResult {
  /*
   * The original model assumed self-host cost was FIXED and API cost variable,
   * so a crossover volume always existed. Once replicas are sized to load
   * (#141) both sides scale with volume, and that framing largely collapses:
   * if self-host is cheaper per token it wins at every volume that keeps the
   * replicas busy, and if it is dearer it never wins at all.
   *
   * What actually decides it is UTILIZATION, not volume -- which matches the
   * published finding that break-even utilization against API pricing is ~60%.
   * The crossover volume is still reported, because it is meaningful in the
   * one-replica case, but it is labelled rather than presented as the answer.
   */
  const monthlyOutputTokens =
    usage.requestsPerDay * usage.avgOutputTokens * DAYS_PER_MONTH;

  const apiCostPerMillionOutputTokens =
    monthlyOutputTokens > 0 ? (apiCost.monthlyCost / monthlyOutputTokens) * 1e6 : 0;

  // Self-host is billed for provisioned capacity whether or not it is used,
  // so divide by what is actually served, not by what could be.
  const selfHostCostPerMillionOutputTokens =
    monthlyOutputTokens > 0
      ? (selfHostCost.monthlyGpuCost / monthlyOutputTokens) * 1e6
      : Infinity;

  const selfHostCheaperPerToken =
    selfHostCostPerMillionOutputTokens < apiCostPerMillionOutputTokens;

  // Volume at which API cost overtakes the CURRENT self-host bill.
  const dailySelfHostCost = selfHostCost.monthlyGpuCost / DAYS_PER_MONTH;
  const breakevenRequestsPerDay =
    apiCost.costPerRequest > 0
      ? Math.ceil(dailySelfHostCost / apiCost.costPerRequest)
      : Infinity;

  const apiCheaperBelow = usage.requestsPerDay < breakevenRequestsPerDay;

  let breakevenNote: string;
  if (selfHostCost.replicas > 1) {
    breakevenNote =
      `Self-host is provisioned as ${selfHostCost.replicas} replicas, so its cost scales with volume too. ` +
      `A single crossover volume is not meaningful here -- compare cost per token instead ` +
      `($${selfHostCostPerMillionOutputTokens.toFixed(2)} self-host vs $${apiCostPerMillionOutputTokens.toFixed(2)} API per million output tokens) ` +
      `and treat utilization, not volume, as the deciding variable.`;
  } else if (selfHostCost.utilizationPercent < 60) {
    breakevenNote =
      `One replica at ${selfHostCost.utilizationPercent.toFixed(1)}% utilization. Published break-even utilization against API pricing is around 60%, ` +
      `so this configuration pays for idle capacity.`;
  } else {
    breakevenNote =
      `One replica at ${selfHostCost.utilizationPercent.toFixed(1)}% utilization; the crossover volume below is meaningful while a single replica suffices.`;
  }

  const monthlySavings = apiCost.monthlyCost - selfHostCost.monthlyGpuCost;
  const annualSavings = apiCost.annualCost - selfHostCost.monthlyGpuCost * 12;

  return {
    breakevenRequestsPerDay,
    apiCheaperBelow,
    monthlySavingsAtCurrentVolume: monthlySavings,
    annualSavingsAtCurrentVolume: annualSavings,
    apiCostPerMillionOutputTokens,
    selfHostCostPerMillionOutputTokens,
    selfHostCheaperPerToken,
    breakevenNote,
  };
}

/**
 * Generate data points for a breakeven chart: cost vs request volume
 */
export function generateBreakevenCurve(
  apiModel: ApiModel,
  providerName: string,
  /**
   * Cost of ONE replica per month. Self-host cost is a staircase, not a flat
   * line: serving more than one replica's capacity requires another replica.
   * Passing a single total here was the bug behind #141/#101.
   */
  selfHostMonthlyCostPerReplica: number,
  avgInputTokens: number,
  avgOutputTokens: number,
  maxRequestsPerDay: number = 100000,
  /** Output tokens/sec one replica serves. Omit to keep the legacy flat line. */
  perReplicaTokensPerSec?: number,
): {
  requestsPerDay: number;
  apiMonthlyCost: number;
  selfHostMonthlyCost: number;
  replicas: number;
}[] {
  const points: {
    requestsPerDay: number;
    apiMonthlyCost: number;
    selfHostMonthlyCost: number;
    replicas: number;
  }[] = [];

  const steps = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    .filter((v) => v <= maxRequestsPerDay);

  for (const rpd of steps) {
    const result = calculateApiCost(apiModel, providerName, {
      requestsPerDay: rpd,
      avgInputTokens,
      avgOutputTokens,
    });

    // Replicas required to serve this volume, so the self-host line steps.
    let replicas = 1;
    if (perReplicaTokensPerSec && perReplicaTokensPerSec > 0) {
      const requiredTokensPerSec = (rpd * avgOutputTokens) / 86_400;
      replicas = Math.max(1, Math.ceil(requiredTokensPerSec / perReplicaTokensPerSec));
    }

    points.push({
      requestsPerDay: rpd,
      apiMonthlyCost: result.monthlyCost,
      selfHostMonthlyCost: selfHostMonthlyCostPerReplica * replicas,
      replicas,
    });
  }

  return points;
}
