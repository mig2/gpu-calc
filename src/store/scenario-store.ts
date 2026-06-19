import { create } from 'zustand';
import type { TrainingScenario, EstimateResult, Precision, TrainingMode } from '../engine/types';
import { estimateTrainingRun } from '../engine/calculator';
import { GPU_SKUS, getGpuById, getH100Reference } from '../engine/gpu-data';

type ScenarioStore = {
  scenario: TrainingScenario;
  results: EstimateResult[];

  setModelParameters: (params: number) => void;
  setTokensPerParameter: (tpp: number) => void;
  setTrainingWindowSeconds: (seconds: number) => void;
  setPrecision: (precision: Precision) => void;
  setTrainingMode: (mode: TrainingMode) => void;
  setSelectedGpuIds: (ids: string[]) => void;
  setMfuForGpu: (gpuId: string, mfu: number) => void;
  setAvailability: (availability: number) => void;
  setOverheadFactor: (overhead: number) => void;
  setMemoryBytesPerParameter: (bytes: number) => void;
};

const defaultScenario: TrainingScenario = {
  modelParameters: 70e9,
  tokensPerParameter: 20,
  trainingWindowSeconds: 30 * 86_400,
  precision: 'BF16_DENSE',
  selectedGpuIds: ['h100-sxm'],
  mfuByGpuId: Object.fromEntries(GPU_SKUS.map((g) => [g.id, g.defaultMfu])),
  availability: 0.90,
  overheadFactor: 1.10,
  trainingMode: 'FULL_PRETRAINING',
  memoryBytesPerParameter: 16,
};

function computeResults(scenario: TrainingScenario): EstimateResult[] {
  const h100 = getH100Reference();
  return scenario.selectedGpuIds
    .map((id) => getGpuById(id))
    .filter((gpu): gpu is NonNullable<typeof gpu> => gpu != null)
    .map((gpu) => estimateTrainingRun(scenario, gpu, h100));
}

export const useScenarioStore = create<ScenarioStore>((set) => ({
  scenario: defaultScenario,
  results: computeResults(defaultScenario),

  setModelParameters: (params) =>
    set((state) => {
      const scenario = { ...state.scenario, modelParameters: params };
      return { scenario, results: computeResults(scenario) };
    }),

  setTokensPerParameter: (tpp) =>
    set((state) => {
      const scenario = { ...state.scenario, tokensPerParameter: tpp };
      return { scenario, results: computeResults(scenario) };
    }),

  setTrainingWindowSeconds: (seconds) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingWindowSeconds: seconds };
      return { scenario, results: computeResults(scenario) };
    }),

  setPrecision: (precision) =>
    set((state) => {
      const scenario = { ...state.scenario, precision };
      return { scenario, results: computeResults(scenario) };
    }),

  setTrainingMode: (mode) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingMode: mode };
      return { scenario, results: computeResults(scenario) };
    }),

  setSelectedGpuIds: (ids) =>
    set((state) => {
      const scenario = { ...state.scenario, selectedGpuIds: ids };
      return { scenario, results: computeResults(scenario) };
    }),

  setMfuForGpu: (gpuId, mfu) =>
    set((state) => {
      const scenario = {
        ...state.scenario,
        mfuByGpuId: { ...state.scenario.mfuByGpuId, [gpuId]: mfu },
      };
      return { scenario, results: computeResults(scenario) };
    }),

  setAvailability: (availability) =>
    set((state) => {
      const scenario = { ...state.scenario, availability };
      return { scenario, results: computeResults(scenario) };
    }),

  setOverheadFactor: (overhead) =>
    set((state) => {
      const scenario = { ...state.scenario, overheadFactor: overhead };
      return { scenario, results: computeResults(scenario) };
    }),

  setMemoryBytesPerParameter: (bytes) =>
    set((state) => {
      const scenario = { ...state.scenario, memoryBytesPerParameter: bytes };
      return { scenario, results: computeResults(scenario) };
    }),
}));
