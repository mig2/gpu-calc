import type { ApiModel, CloudGpuInstance, SelfHostEntry } from '../data/types';

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
  gpuCount: number;
  monthlyGpuCost: number;
  maxOutputTokensPerSec: number;
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

  const monthlyCost = dailyCost * 30;
  const annualCost = dailyCost * 365;
  const costPerRequest = dailyCost / usage.requestsPerDay;

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
): SelfHostCostResult {
  const warnings: string[] = [];

  // Cost
  const pricePerHr =
    pricingTier === 'reserved' ? (instance.reservedPerHr ?? instance.onDemandPerHr) :
    pricingTier === 'spot' ? (instance.spotPerHr ?? instance.onDemandPerHr) :
    instance.onDemandPerHr;

  // Scale cost to match GPU count needed for the model
  const instancesNeeded = Math.ceil(throughput.gpuCount / instance.gpuCount);
  const monthlyGpuCost = pricePerHr * 24 * 30 * instancesNeeded;

  // Throughput check
  const requiredOutputTokensPerSec = (usage.requestsPerDay * usage.avgOutputTokens) / 86_400;
  const canServe = requiredOutputTokensPerSec <= throughput.outputTokensPerSec;
  const utilizationPercent = (requiredOutputTokensPerSec / throughput.outputTokensPerSec) * 100;

  // Rough TTFT estimate (larger models = higher latency)
  const paramBillions = parseFloat(throughput.parameters) || 7;
  const estimatedTtftMs = Math.round(50 + paramBillions * 3);

  if (!canServe) {
    warnings.push(
      `Required throughput (${requiredOutputTokensPerSec.toFixed(1)} tok/s) exceeds capacity (${throughput.outputTokensPerSec} tok/s). Add more GPUs or reduce request volume.`,
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
    monthlyGpuCost,
    maxOutputTokensPerSec: throughput.outputTokensPerSec,
    requiredTokensPerSec: requiredOutputTokensPerSec,
    canServe,
    utilizationPercent,
    estimatedTtftMs,
    trace: [
      `Model: ${throughput.model} on ${throughput.gpuCount}× ${throughput.gpu} (${throughput.framework}, ${throughput.quantization})`,
      `Cloud: ${cloudProviderName} ${instance.instance} @ $${pricePerHr.toFixed(2)}/hr (${pricingTier})`,
      `Instances needed: ${instancesNeeded} (${instancesNeeded * instance.gpuCount} GPUs total)`,
      `Monthly cost = $${pricePerHr.toFixed(2)} × 24h × 30d × ${instancesNeeded} = $${monthlyGpuCost.toFixed(2)}`,
      `Required throughput = ${usage.requestsPerDay.toLocaleString()} req/day × ${usage.avgOutputTokens} tokens / 86400s = ${requiredOutputTokensPerSec.toFixed(1)} tok/s`,
      `Max throughput = ${throughput.outputTokensPerSec} tok/s`,
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
  // Find breakeven: at what request volume does self-host become cheaper?
  // API cost scales linearly with requests. Self-host is fixed.
  // breakeven = selfHostMonthlyCost / costPerRequest / 30
  const dailySelfHostCost = selfHostCost.monthlyGpuCost / 30;
  const breakevenRequestsPerDay = apiCost.costPerRequest > 0
    ? Math.ceil(dailySelfHostCost / apiCost.costPerRequest)
    : Infinity;

  const apiCheaperBelow = usage.requestsPerDay < breakevenRequestsPerDay;
  const monthlySavings = apiCost.monthlyCost - selfHostCost.monthlyGpuCost;
  const annualSavings = apiCost.annualCost - (selfHostCost.monthlyGpuCost * 12);

  return {
    breakevenRequestsPerDay,
    apiCheaperBelow,
    monthlySavingsAtCurrentVolume: monthlySavings,
    annualSavingsAtCurrentVolume: annualSavings,
  };
}

/**
 * Generate data points for a breakeven chart: cost vs request volume
 */
export function generateBreakevenCurve(
  apiModel: ApiModel,
  providerName: string,
  selfHostMonthlyCost: number,
  avgInputTokens: number,
  avgOutputTokens: number,
  maxRequestsPerDay: number = 100000,
): { requestsPerDay: number; apiMonthlyCost: number; selfHostMonthlyCost: number }[] {
  const points: { requestsPerDay: number; apiMonthlyCost: number; selfHostMonthlyCost: number }[] = [];
  const steps = [0, 100, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    .filter((v) => v <= maxRequestsPerDay);

  for (const rpd of steps) {
    const result = calculateApiCost(apiModel, providerName, {
      requestsPerDay: rpd,
      avgInputTokens,
      avgOutputTokens,
    });
    points.push({
      requestsPerDay: rpd,
      apiMonthlyCost: result.monthlyCost,
      selfHostMonthlyCost,
    });
  }

  return points;
}
