import { create } from 'zustand';
import type { TrainingScenario, EstimateResult, Precision, TrainingMode, GpuSku, ModelFamily, TimeSeriesConfig, BaseHardwareConfig } from '../engine/types';
import { estimateTrainingRun } from '../engine/calculator';
import { computeTimeSeriesFlops } from '../engine/adapters/time-series-adapter';
import { estimateHardware } from '../engine/hardware-estimator';
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

const defaultTsConfig = {
  modelParameters: 1e9,
  numberOfSeries: 10e6,
  averageTimestepsPerSeries: 1000,
  variablesPerSeries: 1,
  lookbackWindow: 256,
  forecastHorizon: 64,
  stride: 64,
  patchSize: 16,
  tokenizationMode: 'channel_compressed' as const,
  epochs: 1,
  architectureType: 'decoder_transformer' as const,
  architectureFactor: 6,
  memoryBytesPerParameter: 16,
};

type ScenarioStore = {
  scenario: TrainingScenario;
  results: EstimateResult[];
  customGpus: GpuSku[];
  modelFamily: ModelFamily;
  tsConfig: {
    modelParameters: number;
    numberOfSeries: number;
    averageTimestepsPerSeries: number;
    variablesPerSeries: number;
    lookbackWindow: number;
    forecastHorizon: number;
    stride: number;
    patchSize: number;
    tokenizationMode: 'channel_compressed' | 'channel_expanded' | 'custom';
    customTokensPerWindow?: number;
    epochs: number;
    architectureType: 'decoder_transformer' | 'encoder_transformer' | 'encoder_decoder' | 'patch_transformer' | 'custom';
    architectureFactor: number;
    memoryBytesPerParameter: number;
  };

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
  setScenario: (scenario: TrainingScenario) => void;
  setModelFamily: (family: ModelFamily) => void;
  setTsField: <K extends keyof ScenarioStore['tsConfig']>(key: K, value: ScenarioStore['tsConfig'][K]) => void;
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

function computeResults(
  scenario: TrainingScenario,
  customGpus: GpuSku[],
  modelFamily: ModelFamily = 'llm',
  tsConfig: ScenarioStore['tsConfig'] = defaultTsConfig,
): EstimateResult[] {
  if (modelFamily === 'time_series_foundation') {
    const h100 = getH100Reference();
    const allGpus = getAllGpus(customGpus);

    const tsScenario = {
      modelFamily: 'time_series_foundation' as const,
      ...tsConfig,
      trainingWindowSeconds: scenario.trainingWindowSeconds,
      precision: scenario.precision,
      selectedGpuIds: scenario.selectedGpuIds,
      mfuByGpuId: scenario.mfuByGpuId,
      availability: scenario.availability,
      overheadFactor: scenario.overheadFactor,
    };

    const adapterResult = computeTimeSeriesFlops(tsScenario);

    return scenario.selectedGpuIds
      .map((id) => allGpus.find((g) => g.id === id) ?? getGpuById(id))
      .filter((gpu): gpu is NonNullable<typeof gpu> => gpu != null)
      .map((gpu) => {
        const hwResult = estimateHardware(
          {
            totalFlops: adapterResult.totalFlops,
            modelParameters: tsConfig.modelParameters,
            memoryBytesPerParameter: tsConfig.memoryBytesPerParameter,
            hardware: {
              trainingWindowSeconds: scenario.trainingWindowSeconds,
              precision: scenario.precision,
              selectedGpuIds: scenario.selectedGpuIds,
              mfuByGpuId: scenario.mfuByGpuId,
              availability: scenario.availability,
              overheadFactor: scenario.overheadFactor,
            },
          },
          gpu,
          h100,
        );

        const warnings = [...adapterResult.warnings];
        if (hwResult.requiredGpus > 1024) {
          warnings.push(
            `Distributed-systems warning: ${hwResult.requiredGpus.toLocaleString()} GPUs implies challenges with networking, checkpointing, stragglers, and cluster fragmentation.`,
          );
        }
        if (hwResult.requiredGpus > 10000) {
          warnings.push(
            `Hyperscale warning: ${hwResult.requiredGpus.toLocaleString()} GPUs implies a multi-rack, multi-datacenter-class deployment with extreme operational complexity.`,
          );
        }
        const mfu = scenario.mfuByGpuId[gpu.id] ?? gpu.defaultMfu;
        if (mfu > 0.60) {
          warnings.push(
            `Optimistic MFU warning: ${(mfu * 100).toFixed(0)}% MFU exceeds typical achieved utilization. Most production runs achieve 30-50%.`,
          );
        }
        if (hwResult.memoryLowerBoundGpus > hwResult.requiredGpus) {
          warnings.push(
            `Memory bound exceeds compute bound: at least ${hwResult.memoryLowerBoundGpus} GPUs needed for model state memory alone (vs ${hwResult.requiredGpus} for compute).`,
          );
        }

        return {
          gpuId: gpu.id,
          tokens: adapterResult.effectiveTokens,
          baseFlops: adapterResult.baseFlops,
          totalFlops: adapterResult.totalFlops,
          sustainedFlopsPerGpu: hwResult.sustainedFlopsPerGpu,
          requiredGpus: hwResult.requiredGpus,
          h100Equivalents: hwResult.h100Equivalents,
          memoryLowerBoundGpus: hwResult.memoryLowerBoundGpus,
          warnings,
          trace: [...adapterResult.trace, ...hwResult.hardwareTrace],
        };
      });
  }

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
  modelFamily: 'llm',
  tsConfig: { ...defaultTsConfig },

  setModelParameters: (params) =>
    set((state) => {
      const scenario = { ...state.scenario, modelParameters: params };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setTokensPerParameter: (tpp) =>
    set((state) => {
      const scenario = { ...state.scenario, tokensPerParameter: tpp };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setTrainingWindowSeconds: (seconds) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingWindowSeconds: seconds };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setPrecision: (precision) =>
    set((state) => {
      const scenario = { ...state.scenario, precision };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setTrainingMode: (mode) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingMode: mode };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setSelectedGpuIds: (ids) =>
    set((state) => {
      const scenario = { ...state.scenario, selectedGpuIds: ids };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setMfuForGpu: (gpuId, mfu) =>
    set((state) => {
      const scenario = {
        ...state.scenario,
        mfuByGpuId: { ...state.scenario.mfuByGpuId, [gpuId]: mfu },
      };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setAvailability: (availability) =>
    set((state) => {
      const scenario = { ...state.scenario, availability };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setOverheadFactor: (overhead) =>
    set((state) => {
      const scenario = { ...state.scenario, overheadFactor: overhead };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
    }),

  setMemoryBytesPerParameter: (bytes) =>
    set((state) => {
      const scenario = { ...state.scenario, memoryBytesPerParameter: bytes };
      return { scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) };
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
      return { customGpus, scenario, results: computeResults(scenario, customGpus, state.modelFamily, state.tsConfig) };
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
      return { customGpus, scenario, results: computeResults(scenario, customGpus, state.modelFamily, state.tsConfig) };
    }),

  setScenario: (scenario) =>
    set((state) => ({ scenario, results: computeResults(scenario, state.customGpus, state.modelFamily, state.tsConfig) })),

  setModelFamily: (family) =>
    set((state) => ({
      modelFamily: family,
      results: computeResults(state.scenario, state.customGpus, family, state.tsConfig),
    })),

  setTsField: (key, value) =>
    set((state) => {
      const tsConfig = { ...state.tsConfig, [key]: value };
      return { tsConfig, results: computeResults(state.scenario, state.customGpus, state.modelFamily, tsConfig) };
    }),
}));
