import { describe, it, expect } from 'vitest'
import { calculateApiCost, calculateSelfHostCost, calculateBreakeven, generateBreakevenCurve } from './inference-calculator'
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
    // Daily total = $75, monthly = $2250
    expect(result.monthlyCost).toBeCloseTo(2250, 0)
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

  it('annual cost is ~12x monthly', () => {
    const result = calculateApiCost(mockApiModel, 'OpenAI', defaultUsage)
    expect(result.annualCost).toBeCloseTo(result.monthlyCost * (365 / 30), -1)
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
    // $98.32 * 24 * 30 = ~$70,790
    expect(result.monthlyGpuCost).toBeCloseTo(98.32 * 24 * 30, 0)
  })

  it('checks throughput feasibility', () => {
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', defaultUsage)
    // Required: 10000 * 500 / 86400 ≈ 57.9 tok/s. Capacity: 1200. Can serve.
    expect(result.canServe).toBe(true)
    expect(result.requiredTokensPerSec).toBeCloseTo(57.87, 0)
  })

  it('warns when throughput exceeded', () => {
    const heavyUsage = { requestsPerDay: 500000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const result = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', heavyUsage)
    // Required: 500K * 500 / 86400 ≈ 2894 tok/s > 1200
    expect(result.canServe).toBe(false)
    expect(result.warnings).toContainEqual(expect.stringContaining('exceeds capacity'))
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

  it('self-host is cheaper at high volume', () => {
    const highUsage = { requestsPerDay: 500000, avgInputTokens: 1000, avgOutputTokens: 500 }
    const apiResult = calculateApiCost(mockApiModel, 'OpenAI', highUsage)
    const selfHostResult = calculateSelfHostCost(mockThroughput, mockInstance, 'AWS', highUsage)
    const breakeven = calculateBreakeven(apiResult, selfHostResult, highUsage)
    expect(breakeven.apiCheaperBelow).toBe(false)
    expect(breakeven.monthlySavingsAtCurrentVolume).toBeGreaterThan(0)
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
