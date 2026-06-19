import { create } from 'zustand';
import type { TrainingScenario, EstimateResult, Precision, TrainingMode, GpuSku } from '../engine/types';
import { estimateTrainingRun } from '../engine/calculator';
import { GPU_SKUS, getGpuById, getH100Reference, setExtraGpus } from '../engine/gpu-data';

function loadCustomGpus(): GpuSku[] {
  try {
    const saved = localStorage.getItem('gpu-calc-custom-gpus');
    return saved ? JSON.parse(saved) : [];
  } catch {
    return [];
  }
}

function saveCustomGpus(gpus: GpuSku[]) {
  localStorage.setItem('gpu-calc-custom-gpus', JSON.stringify(gpus));
}

export function getAllGpus(customGpus: GpuSku[]): GpuSku[] {
  return [...GPU_SKUS, ...customGpus];
}

type ScenarioStore = {
  scenario: TrainingScenario;
  results: EstimateResult[];
  customGpus: GpuSku[];

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
  addCustomGpu: (gpu: GpuSku) => void;
  removeCustomGpu: (id: string) => void;
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

function computeResults(scenario: TrainingScenario, customGpus: GpuSku[]): EstimateResult[] {
  const h100 = getH100Reference();
  const allGpus = getAllGpus(customGpus);
  return scenario.selectedGpuIds
    .map((id) => allGpus.find((g) => g.id === id) ?? getGpuById(id))
    .filter((gpu): gpu is NonNullable<typeof gpu> => gpu != null)
    .map((gpu) => estimateTrainingRun(scenario, gpu, h100));
}

const initialCustomGpus = loadCustomGpus();
setExtraGpus(initialCustomGpus);

export const useScenarioStore = create<ScenarioStore>((set, get) => ({
  scenario: defaultScenario,
  results: computeResults(defaultScenario, initialCustomGpus),
  customGpus: initialCustomGpus,

  setModelParameters: (params) =>
    set((state) => {
      const scenario = { ...state.scenario, modelParameters: params };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setTokensPerParameter: (tpp) =>
    set((state) => {
      const scenario = { ...state.scenario, tokensPerParameter: tpp };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setTrainingWindowSeconds: (seconds) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingWindowSeconds: seconds };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setPrecision: (precision) =>
    set((state) => {
      const scenario = { ...state.scenario, precision };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setTrainingMode: (mode) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingMode: mode };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setSelectedGpuIds: (ids) =>
    set((state) => {
      const scenario = { ...state.scenario, selectedGpuIds: ids };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setMfuForGpu: (gpuId, mfu) =>
    set((state) => {
      const scenario = {
        ...state.scenario,
        mfuByGpuId: { ...state.scenario.mfuByGpuId, [gpuId]: mfu },
      };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setAvailability: (availability) =>
    set((state) => {
      const scenario = { ...state.scenario, availability };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setOverheadFactor: (overhead) =>
    set((state) => {
      const scenario = { ...state.scenario, overheadFactor: overhead };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  setMemoryBytesPerParameter: (bytes) =>
    set((state) => {
      const scenario = { ...state.scenario, memoryBytesPerParameter: bytes };
      return { scenario, results: computeResults(scenario, state.customGpus) };
    }),

  addCustomGpu: (gpu) =>
    set((state) => {
      const customGpus = [...state.customGpus, gpu];
      saveCustomGpus(customGpus);
      setExtraGpus(customGpus);
      const scenario = {
        ...state.scenario,
        mfuByGpuId: { ...state.scenario.mfuByGpuId, [gpu.id]: gpu.defaultMfu },
      };
      return { customGpus, scenario, results: computeResults(scenario, customGpus) };
    }),

  removeCustomGpu: (id) =>
    set((state) => {
      const customGpus = state.customGpus.filter((g) => g.id !== id);
      saveCustomGpus(customGpus);
      setExtraGpus(customGpus);
      const selectedGpuIds = state.scenario.selectedGpuIds.filter((gid) => gid !== id);
      const scenario = {
        ...state.scenario,
        selectedGpuIds: selectedGpuIds.length > 0 ? selectedGpuIds : ['h100-sxm'],
      };
      return { customGpus, scenario, results: computeResults(scenario, customGpus) };
    }),
}));
