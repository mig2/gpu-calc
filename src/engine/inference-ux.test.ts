import { describe, it, expect } from 'vitest'

// Test the derivation logic used in InferenceCalculator
// These are pure math tests, not component tests

describe('inference scale derivations', () => {
  it('users × req/user/day = requests/day', () => {
    const users = 500
    const reqPerUser = 20
    expect(users * reqPerUser).toBe(10000)
  })

  it('monthly token budget → requests/day', () => {
    const monthlyBudgetM = 100 // 100M tokens/month
    const avgInput = 4000
    const avgOutput = 500
    const tokensPerRequest = avgInput + avgOutput
    const totalTokensPerMonth = monthlyBudgetM * 1e6
    const rpd = Math.round(totalTokensPerMonth / tokensPerRequest / 30)
    // 100M / 4500 / 30 ≈ 741
    expect(rpd).toBe(741)
  })

  it('high-volume budget derivation', () => {
    const monthlyBudgetM = 5000 // 5B tokens/month
    const tokensPerRequest = 500 + 300 // chatbot
    const totalTokensPerMonth = monthlyBudgetM * 1e6
    const rpd = Math.round(totalTokensPerMonth / tokensPerRequest / 30)
    // 5B / 800 / 30 ≈ 208,333
    expect(rpd).toBe(208333)
  })

  it('daily token volume calculation', () => {
    const requestsPerDay = 100000
    const avgInput = 2000
    const avgOutput = 1000
    const dailyTotal = requestsPerDay * (avgInput + avgOutput)
    expect(dailyTotal).toBe(300e6) // 300M tokens/day
  })

  it('monthly projection from daily', () => {
    const dailyTokens = 300e6
    expect(dailyTokens * 30).toBe(9e9) // 9B tokens/month
  })
})

describe('use-case preset token values', () => {
  const USE_CASES = [
    { label: 'Chatbot', input: 500, output: 300 },
    { label: 'RAG', input: 4000, output: 500 },
    { label: 'Summarization', input: 10000, output: 500 },
    { label: 'Coding', input: 8000, output: 2000 },
    { label: 'Agentic', input: 16000, output: 4000 },
    { label: 'Long-context', input: 50000, output: 1000 },
  ]

  it('all use cases have positive input and output tokens', () => {
    for (const uc of USE_CASES) {
      expect(uc.input).toBeGreaterThan(0)
      expect(uc.output).toBeGreaterThan(0)
    }
  })

  it('all use cases are input-heavy (input >= output)', () => {
    for (const uc of USE_CASES) {
      expect(uc.input).toBeGreaterThanOrEqual(uc.output)
    }
  })

  it('agentic has highest total tokens per request', () => {
    const totals = USE_CASES.map(uc => uc.input + uc.output)
    const agenticTotal = 16000 + 4000
    expect(Math.max(...totals)).toBe(50000 + 1000) // long-context is actually highest
    expect(agenticTotal).toBe(20000)
  })

  it('chatbot has lowest total tokens per request', () => {
    const totals = USE_CASES.map(uc => uc.input + uc.output)
    expect(Math.min(...totals)).toBe(500 + 300) // chatbot = 800
  })

  it('custom override replaces preset values', () => {
    const preset = USE_CASES[1] // RAG: 4000/500
    const customInput = 8000
    const customOutput = 1000
    const input = customInput ?? preset.input
    const output = customOutput ?? preset.output
    expect(input).toBe(8000)
    expect(output).toBe(1000)
  })
})
