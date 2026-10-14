import { describe, it, expect } from 'vitest'
import { calculateApiCost, calculateSelfHostCost, calculateBreakeven, generateBreakevenCurve, DAYS_PER_MONTH } from './inference-calculator'
import type { ApiModel, CloudGpuInstance, SelfHostEntry } from '../data/types'

const mockApiModel: ApiModel = {
  model: 'GPT-4o',
  inputPer1M: 2.50,
  outputPer1M: 10.00,
  contextWindow: 128000,
}

const mockInstance: CloudGpuInstance = {
  instance: 'p5.48xlarge',
  gpu: 'H100 SXM',
  gpuCount: 8,
  onDemandPerHr: 98.32,
  reservedPerHr: 63.91,
  spotPerHr: null,
}

const mockThroughput: SelfHostEntry = {
  model: 'Llama 3 70B',
  parameters: '70B',
  gpu: 'H100 SXM',
  gpuCount: 4,
  quantization: 'FP16',
  framework: 'vLLM',
  outputTokensPerSec: 1200,
  maxConcurrentRequests: 32,
}

const defaultUsage = {
  requestsPerDay: 10000,
  avgInputTokens: 1000,
  avgOutputTokens: 500,
}

describe('calculateApiCost', () => {
  it('computes correct monthly cost', () => {
    const result = calculateApiCost(mockApiModel, 'OpenAI', defaultUsage)
    // Daily: 10K * 1000 = 10M input tokens → $25; 10K * 500 = 5M output → $50
    // Daily total = $75, monthly = $75 * (365/12) = $2281.25
    expect(result.monthlyCost).toBeCloseTo(75 * DAYS_PER_MONTH, 2)
  })

  it('computes correct cost per request', () => {
    const result = calculateApiCost(mockApiModel, 'OpenAI', defaultUsage)
    // $75/day / 10000 = $0.0075
    expect(result.costPerRequest).toBeCloseTo(0.0075, 4)
  })

  it('output tokens dominate cost for expensive output pricing', () => {
    const result = calculateApiCost(mockApiModel, 'OpenAI', defaultUsage)
    expect(result.outputCostShare).toBeGreaterThan(result.inputCostShare)
  })

  it('annual cost is exactly 12x monthly', () => {
    const result = calculateApiCost(mockApiModel, 'OpenAI', defaultUsage)
    // A month is 365/12 days, so twelve months is exactly one 365-day year.
    expect(result.annualCost).toBeCloseTo(result.monthlyCost * 12, 6)
    expect(result.annualCost).toBeCloseTo(75 * 365, 2)
  })

  it('cost per request is 0 (not NaN) at zero requests per day', () => {
    const result = calculateApiCost(mockApiModel, 'OpenAI', {
      requestsPerDay: 0,
      avgInputTokens: 1000,
      avgOutputTokens: 500,
    })
    expect(result.costPerRequest).toBe(0)
    expect(result.monthlyCost).toBe(0)
    expect(result.annualCost).toBe(0)
  })

  it('produces trace', () => {
    const result = calculateApiCost(mockApiModel, 'OpenAI', defaultUsage)
    expect(result.trace.length).toBeGreaterThanOrEqual(5)
  })
})

describe('calculateSelfHostCost', () => {
  it('computes monthly GPU cost', () => {
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', defaultUsage)
    // 4 GPUs needed, instance has 8 → 1 instance
    // $98.32 * 24 * (365/12) = ~$71,774
    expect(result.monthlyGpuCost).toBeCloseTo(98.32 * 24 * DAYS_PER_MONTH, 2)
  })

  it('checks throughput feasibility', () => {
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', defaultUsage)
    // Required: 10000 * 500 / 86400 ≈ 57.9 tok/s. Capacity: 1200. Can serve.
    expect(result.canServe).toBe(true)
    expect(result.requiredTokensPerSec).toBeCloseTo(57.87, 0)
  })

  /*
   * Previously this asserted canServe === false, i.e. that the pane reports an
   * impossible configuration and tells the user to "add more GPUs" -- advice
   * the UI offered no way to follow (#141). Capacity is now auto-sized to the
   * load, so the default answer is a servable configuration with its real cost.
   */
  it('auto-sizes replicas to serve the offered load', () => {
    const heavyUsage = { requestsPerDay: 500000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', heavyUsage)
    // Required: 500K * 500 / 86400 ≈ 2894 tok/s against 1200 per replica -> 3
    expect(result.replicasNeeded).toBe(3)
    expect(result.replicas).toBe(3)
    expect(result.canServe).toBe(true)
    expect(result.totalOutputTokensPerSec).toBe(mockThroughput.outputTokensPerSec * 3)
    expect(result.totalGpus).toBe(mockThroughput.gpuCount * 3)
    expect(result.warnings).toContainEqual(expect.stringContaining('Auto-sized to 3 replicas'))
  })

  it('reports a shortfall only when replicas are explicitly held down', () => {
    const heavyUsage = { requestsPerDay: 500000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', heavyUsage, 'onDemand', 1)
    expect(result.canServe).toBe(false)
    expect(result.replicas).toBe(1)
    expect(result.replicasNeeded).toBe(3)
    // The remedy named must be one the UI can actually perform.
    expect(result.warnings).toContainEqual(expect.stringContaining('Raise replicas to 3'))
  })

  it('scales cost linearly with replicas', () => {
    const light = { requestsPerDay: 10000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const one = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', light, 'onDemand', 1)
    const three = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', light, 'onDemand', 3)
    expect(three.monthlyGpuCost).toBeCloseTo(one.monthlyGpuCost * 3, 6)
  })

  it('computes utilization against total capacity, not one replica', () => {
    const heavyUsage = { requestsPerDay: 500000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const r = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', heavyUsage)
    expect(r.utilizationPercent).toBeLessThanOrEqual(100)
    expect(r.utilizationPercent).toBeCloseTo(
      (r.requiredTokensPerSec / r.totalOutputTokensPerSec) * 100, 6,
    )
  })

  it('warns about low utilization', () => {
    const lightUsage = { requestsPerDay: 10, avgInputTokens: 100, avgOutputTokens: 50 }
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', lightUsage)
    expect(result.warnings).toContainEqual(expect.stringContaining('over-provisioned'))
  })

  it('uses reserved pricing when selected', () => {
    const onDemand = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', defaultUsage, 'onDemand')
    const reserved = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', defaultUsage, 'reserved')
    expect(reserved.monthlyGpuCost).toBeLessThan(onDemand.monthlyGpuCost)
  })

  it('produces trace', () => {
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', defaultUsage)
    expect(result.trace.length).toBeGreaterThanOrEqual(5)
  })
})

describe('calculateBreakeven', () => {
  it('finds breakeven point', () => {
    const apiResult = calculateApiCost(mockApiModel, 'OpenAI', defaultUsage)
    const selfHostResult = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', defaultUsage)
    const breakeven = calculateBreakeven(apiResult, selfHostResult, defaultUsage)
    expect(breakeven.breakevenRequestsPerDay).toBeGreaterThan(0)
    expect(breakeven.breakevenRequestsPerDay).toBeLessThan(1e6)
  })

  it('API is cheaper at low volume', () => {
    const lowUsage = { requestsPerDay: 100, avgInputTokens: 1000, avgOutputTokens: 500 }
    const apiResult = calculateApiCost(mockApiModel, 'OpenAI', lowUsage)
    const selfHostResult = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', lowUsage)
    const breakeven = calculateBreakeven(apiResult, selfHostResult, lowUsage)
    expect(breakeven.apiCheaperBelow).toBe(true)
  })

  it('returns Infinity breakeven at zero volume', () => {
    const zeroUsage = { requestsPerDay: 0, avgInputTokens: 1000, avgOutputTokens: 500 }
    const apiResult = calculateApiCost(mockApiModel, 'OpenAI', zeroUsage)
    const selfHostResult = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', zeroUsage)
    const breakeven = calculateBreakeven(apiResult, selfHostResult, zeroUsage)
    // costPerRequest is 0, so the > 0 guard yields Infinity (never worth self-hosting)
    expect(breakeven.breakevenRequestsPerDay).toBe(Infinity)
    expect(breakeven.apiCheaperBelow).toBe(true)
  })

  /*
   * This previously asserted that self-host is unconditionally cheaper at high
   * volume. That only held because self-host cost was FIXED -- the bug in #141.
   * With replicas sized to load, both sides scale with volume, so the question
   * stops being "what volume?" and becomes "cheaper per token, at what
   * utilization?". Asserting the structure rather than a direction.
   */
  it('compares cost per token once both sides scale with volume', () => {
    const highUsage = { requestsPerDay: 500000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const apiResult = calculateApiCost(mockApiModel, 'OpenAI', highUsage)
    const selfHostResult = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', highUsage)
    const breakeven = calculateBreakeven(apiResult, selfHostResult, highUsage)

    expect(breakeven.apiCostPerMillionOutputTokens).toBeGreaterThan(0)
    expect(breakeven.selfHostCostPerMillionOutputTokens).toBeGreaterThan(0)
    expect(breakeven.selfHostCheaperPerToken).toBe(
      breakeven.selfHostCostPerMillionOutputTokens < breakeven.apiCostPerMillionOutputTokens,
    )
    // Multi-replica: the single crossover volume is flagged as not meaningful.
    expect(selfHostResult.replicas).toBeGreaterThan(1)
    expect(breakeven.breakevenNote).toContain('not meaningful')
  })

  it('still reports a meaningful crossover while one replica suffices', () => {
    const light = { requestsPerDay: 5000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const apiResult = calculateApiCost(mockApiModel, 'OpenAI', light)
    const selfHostResult = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', light)
    const breakeven = calculateBreakeven(apiResult, selfHostResult, light)
    expect(selfHostResult.replicas).toBe(1)
    expect(breakeven.breakevenNote).not.toContain('not meaningful')
    expect(breakeven.breakevenRequestsPerDay).toBeGreaterThan(0)
  })
})

describe('DAYS_PER_MONTH', () => {
  it('is 365/12 so monthly and annual figures agree', () => {
    expect(DAYS_PER_MONTH).toBeCloseTo(30.416667, 6)
    expect(DAYS_PER_MONTH * 12).toBeCloseTo(365, 10)
  })
})

describe('generateBreakevenCurve', () => {
  it('returns data points for chart', () => {
    const points = generateBreakevenCurve(mockApiModel, 'OpenAI', 70000, 1000, 500)
    expect(points.length).toBeGreaterThanOrEqual(5)
    expect(points[0].requestsPerDay).toBe(0)
    expect(points[0].apiMonthlyCost).toBe(0)
    expect(points[0].selfHostMonthlyCost).toBe(70000)
  })

  it('API cost increases linearly', () => {
    const points = generateBreakevenCurve(mockApiModel, 'OpenAI', 70000, 1000, 500)
    const costs = points.map((p) => p.apiMonthlyCost)
    for (let i = 1; i < costs.length; i++) {
      expect(costs[i]).toBeGreaterThanOrEqual(costs[i - 1])
    }
  })

  it('self-host cost is constant', () => {
    const points = generateBreakevenCurve(mockApiModel, 'OpenAI', 70000, 1000, 500)
    for (const p of points) {
      expect(p.selfHostMonthlyCost).toBe(70000)
    }
  })
})
