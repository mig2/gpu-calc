import { create } from 'zustand';
import type { ApiCostResult, SelfHostCostResult, BreakevenResult, UsagePattern } from '../engine/inference-calculator';

type InferenceStore = {
  usage: UsagePattern;
  apiResult: ApiCostResult | null;
  selfHostResult: SelfHostCostResult | null;
  breakeven: BreakevenResult | null;
  setResults: (data: {
    usage: UsagePattern;
    apiResult: ApiCostResult | null;
    selfHostResult: SelfHostCostResult | null;
    breakeven: BreakevenResult | null;
  }) => void;
};

export const useInferenceStore = create<InferenceStore>((set) => ({
  usage: { requestsPerDay: 100000, avgInputTokens: 2000, avgOutputTokens: 1000 },
  apiResult: null,
  selfHostResult: null,
  breakeven: null,
  setResults: (data) => set(data),
}));
