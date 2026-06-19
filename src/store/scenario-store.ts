import { create } from 'zustand';
import type { TrainingScenario, EstimateResult, Precision, TrainingMode, GpuSku, ModelFamily, TimeSeriesConfig, BaseHardwareConfig, ClassicalTabularConfig } from '../engine/types';
import { estimateTrainingRun } from '../engine/calculator';
import { computeTimeSeriesFlops } from '../engine/adapters/time-series-adapter';
import { computeTabularFlops } from '../engine/adapters/tabular-adapter';
import { estimateClassicalTabular } from '../engine/adapters/classical-tabular-adapter';
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

const defaultTabConfig = {
  modelParameters: 100e6,
  numberOfPretrainingTasks: 1e6,
  rowsPerTask: 1024,
  columnsPerTask: 100,
  tokenizationMode: 'row' as const,
  epochs: 1,
  architectureType: 'row_transformer' as const,
  architectureFactor: 6,
  testTimeComputeMultiplier: 1,
  memoryBytesPerParameter: 16,
};

const defaultClassicalConfig = {
  algorithm: 'lightgbm' as const,
  rows: 1e6,
  columns: 100,
  boostingRounds: 1000,
  maxDepth: 8,
  bins: 256,
  cvFolds: 1,
  hyperparameterTrials: 1,
  cpuOrGpu: 'gpu' as const,
  throughputCoefficient: 0,
};

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

type TabConfig = typeof defaultTabConfig;
type ClassicalConfig = typeof defaultClassicalConfig;

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
  tabConfig: TabConfig;
  classicalConfig: ClassicalConfig;

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
  setArchitectureFactor: (factor: number) => void;
  setTrainingTokensOverride: (tokens: number | undefined) => void;
  addCustomGpu: (gpu: GpuSku) => void;
  removeCustomGpu: (id: string) => void;
  setScenario: (scenario: TrainingScenario) => void;
  setModelFamily: (family: ModelFamily) => void;
  setTsField: <K extends keyof ScenarioStore['tsConfig']>(key: K, value: ScenarioStore['tsConfig'][K]) => void;
  setTabField: <K extends keyof TabConfig>(key: K, value: TabConfig[K]) => void;
  setClassicalField: <K extends keyof ClassicalConfig>(key: K, value: ClassicalConfig[K]) => void;
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

function buildAdapterResults(
  adapterResult: { effectiveTokens: number; baseFlops: number; totalFlops: number; trace: string[]; warnings: string[] },
  modelParameters: number,
  memoryBytesPerParameter: number,
  scenario: TrainingScenario,
  customGpus: GpuSku[],
): EstimateResult[] {
  const h100 = getH100Reference();
  const allGpus = getAllGpus(customGpus);

  return scenario.selectedGpuIds
    .map((id) => allGpus.find((g) => g.id === id) ?? getGpuById(id))
    .filter((gpu): gpu is NonNullable<typeof gpu> => gpu != null)
    .map((gpu) => {
      const hwResult = estimateHardware(
        {
          totalFlops: adapterResult.totalFlops,
          modelParameters,
          memoryBytesPerParameter,
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

function computeResults(
  scenario: TrainingScenario,
  customGpus: GpuSku[],
  modelFamily: ModelFamily = 'llm',
  tsConfig: ScenarioStore['tsConfig'] = defaultTsConfig,
  tabConfig: TabConfig = defaultTabConfig,
  classicalConfig: ClassicalConfig = defaultClassicalConfig,
): EstimateResult[] {
  if (modelFamily === 'time_series_foundation') {
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
    return buildAdapterResults(adapterResult, tsConfig.modelParameters, tsConfig.memoryBytesPerParameter, scenario, customGpus);
  }

  if (modelFamily === 'tabular_foundation') {
    const tabScenario = {
      modelFamily: 'tabular_foundation' as const,
      ...tabConfig,
      trainingWindowSeconds: scenario.trainingWindowSeconds,
      precision: scenario.precision,
      selectedGpuIds: scenario.selectedGpuIds,
      mfuByGpuId: scenario.mfuByGpuId,
      availability: scenario.availability,
      overheadFactor: scenario.overheadFactor,
    };
    const adapterResult = computeTabularFlops(tabScenario);
    return buildAdapterResults(adapterResult, tabConfig.modelParameters, tabConfig.memoryBytesPerParameter, scenario, customGpus);
  }

  if (modelFamily === 'classical_tabular') {
    const classicalScenario = {
      modelFamily: 'classical_tabular' as const,
      ...classicalConfig,
      trainingWindowSeconds: scenario.trainingWindowSeconds,
      precision: scenario.precision,
      selectedGpuIds: scenario.selectedGpuIds,
      mfuByGpuId: scenario.mfuByGpuId,
      availability: scenario.availability,
      overheadFactor: scenario.overheadFactor,
    };
    const result = estimateClassicalTabular(classicalScenario);
    const requiredGpus = Math.max(1, Math.ceil(result.estimatedSeconds / scenario.trainingWindowSeconds));

    return scenario.selectedGpuIds.map((gpuId) => ({
      gpuId,
      tokens: result.workUnits,
      baseFlops: 0,
      totalFlops: 0,
      sustainedFlopsPerGpu: 0,
      requiredGpus,
      h100Equivalents: requiredGpus,
      memoryLowerBoundGpus: 1,
      warnings: result.warnings,
      trace: result.trace,
    }));
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

function recompute(state: ScenarioStore, overrides: Partial<Pick<ScenarioStore, 'scenario' | 'customGpus' | 'modelFamily' | 'tsConfig' | 'tabConfig' | 'classicalConfig'>> = {}) {
  const s = overrides.scenario ?? state.scenario;
  const cg = overrides.customGpus ?? state.customGpus;
  const mf = overrides.modelFamily ?? state.modelFamily;
  const ts = overrides.tsConfig ?? state.tsConfig;
  const tab = overrides.tabConfig ?? state.tabConfig;
  const cl = overrides.classicalConfig ?? state.classicalConfig;
  return computeResults(s, cg, mf, ts, tab, cl);
}

export const useScenarioStore = create<ScenarioStore>((set, get) => ({
  scenario: defaultScenario,
  results: computeResults(defaultScenario, initialCustomGpus),
  customGpus: initialCustomGpus,
  modelFamily: 'llm',
  tsConfig: { ...defaultTsConfig },
  tabConfig: { ...defaultTabConfig },
  classicalConfig: { ...defaultClassicalConfig },

  setModelParameters: (params) =>
    set((state) => {
      const scenario = { ...state.scenario, modelParameters: params };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setTokensPerParameter: (tpp) =>
    set((state) => {
      const scenario = { ...state.scenario, tokensPerParameter: tpp };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setTrainingWindowSeconds: (seconds) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingWindowSeconds: seconds };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setPrecision: (precision) =>
    set((state) => {
      const scenario = { ...state.scenario, precision };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setTrainingMode: (mode) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingMode: mode };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setSelectedGpuIds: (ids) =>
    set((state) => {
      const scenario = { ...state.scenario, selectedGpuIds: ids };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setMfuForGpu: (gpuId, mfu) =>
    set((state) => {
      const scenario = {
        ...state.scenario,
        mfuByGpuId: { ...state.scenario.mfuByGpuId, [gpuId]: mfu },
      };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setAvailability: (availability) =>
    set((state) => {
      const scenario = { ...state.scenario, availability };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setOverheadFactor: (overhead) =>
    set((state) => {
      const scenario = { ...state.scenario, overheadFactor: overhead };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setMemoryBytesPerParameter: (bytes) =>
    set((state) => {
      const scenario = { ...state.scenario, memoryBytesPerParameter: bytes };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setArchitectureFactor: (factor) =>
    set((state) => {
      const scenario = { ...state.scenario, architectureFactor: factor };
      return { scenario, results: recompute(state, { scenario }) };
    }),

  setTrainingTokensOverride: (tokens) =>
    set((state) => {
      const scenario = { ...state.scenario, trainingTokensOverride: tokens };
      return { scenario, results: recompute(state, { scenario }) };
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
      return { customGpus, scenario, results: recompute(state, { scenario, customGpus }) };
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
      return { customGpus, scenario, results: recompute(state, { scenario, customGpus }) };
    }),

  setScenario: (scenario) =>
    set((state) => ({ scenario, results: recompute(state, { scenario }) })),

  setModelFamily: (family) =>
    set((state) => ({
      modelFamily: family,
      results: recompute(state, { modelFamily: family }),
    })),

  setTsField: (key, value) =>
    set((state) => {
      const tsConfig = { ...state.tsConfig, [key]: value };
      return { tsConfig, results: recompute(state, { tsConfig }) };
    }),

  setTabField: (key, value) =>
    set((state) => {
      const tabConfig = { ...state.tabConfig, [key]: value };
      return { tabConfig, results: recompute(state, { tabConfig }) };
    }),

  setClassicalField: (key, value) =>
    set((state) => {
      const classicalConfig = { ...state.classicalConfig, [key]: value };
      return { classicalConfig, results: recompute(state, { classicalConfig }) };
    }),
}));
