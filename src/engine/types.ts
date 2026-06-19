export type Precision = 'BF16_DENSE' | 'FP8_DENSE';

export type TrainingMode =
  | 'FULL_PRETRAINING'
  | 'CONTINUED_PRETRAINING'
  | 'SFT'
  | 'LORA'
  | 'RLHF'
  | 'DISTILLATION';

export type GpuSku = {
  id: string;
  label: string;
  bf16DenseFlops: number;   // FLOP/s per GPU (dense, not sparse)
  fp8DenseFlops?: number;   // FLOP/s per GPU
  memoryGb: number;
  bandwidthTbps?: number;
  defaultMfu: number;       // 0-1 fraction
  notes?: string;
};

export type TrainingScenario = {
  modelParameters: number;         // absolute count (not billions)
  tokensPerParameter: number;
  trainingWindowSeconds: number;
  precision: Precision;
  selectedGpuIds: string[];
  mfuByGpuId: Record<string, number>;  // 0-1 fraction per GPU
  availability: number;            // 0-1 fraction
  overheadFactor: number;          // >= 1.0
  trainingMode: TrainingMode;
  memoryBytesPerParameter: number;
  architectureFactor?: number;     // default 6
  trainingTokensOverride?: number; // optional, overrides TPP × N
};

export type EstimateResult = {
  gpuId: string;
  tokens: number;
  baseFlops: number;
  totalFlops: number;
  sustainedFlopsPerGpu: number;
  requiredGpus: number;
  h100Equivalents: number;
  memoryLowerBoundGpus: number;
  warnings: string[];
  trace: string[];
};

// === Model Family Extension Types ===

export type ModelFamily = 'llm' | 'time_series_foundation' | 'tabular_foundation' | 'classical_tabular';

export type ConfidenceLevel = 'high' | 'medium' | 'medium-low' | 'low';

export type TimeSeriesTokenizationMode = 'channel_compressed' | 'channel_expanded' | 'custom';

export type TimeSeriesArchitectureType =
  | 'decoder_transformer'
  | 'encoder_transformer'
  | 'encoder_decoder'
  | 'patch_transformer'
  | 'custom';

export type BaseHardwareConfig = {
  trainingWindowSeconds: number;
  precision: Precision;
  selectedGpuIds: string[];
  mfuByGpuId: Record<string, number>;
  availability: number;
  overheadFactor: number;
};

export type LlmConfig = {
  modelFamily: 'llm';
  modelParameters: number;
  tokensPerParameter: number;
  trainingTokensOverride?: number;
  architectureFactor: number;
  trainingMode: TrainingMode;
  memoryBytesPerParameter: number;
};

export type TimeSeriesConfig = {
  modelFamily: 'time_series_foundation';
  modelParameters: number;
  numberOfSeries: number;
  averageTimestepsPerSeries: number;
  variablesPerSeries: number;
  lookbackWindow: number;
  forecastHorizon: number;
  stride: number;
  patchSize: number;
  tokenizationMode: TimeSeriesTokenizationMode;
  customTokensPerWindow?: number;
  epochs: number;
  architectureType: TimeSeriesArchitectureType;
  architectureFactor: number;
  memoryBytesPerParameter: number;
};

export type TabularTokenizationMode = 'row' | 'cell' | 'axial' | 'custom';

export type TabularArchitectureType =
  | 'row_transformer'
  | 'cell_transformer'
  | 'axial_transformer'
  | 'tabpfn_icl'
  | 'custom';

export type TabularFoundationConfig = {
  modelFamily: 'tabular_foundation';
  modelParameters: number;
  numberOfPretrainingTasks: number;
  rowsPerTask: number;
  columnsPerTask: number;
  tokenizationMode: TabularTokenizationMode;
  customTokensPerTask?: number;
  epochs: number;
  architectureType: TabularArchitectureType;
  architectureFactor: number;
  testTimeComputeMultiplier: number;
  memoryBytesPerParameter: number;
};

export type Scenario =
  | (BaseHardwareConfig & LlmConfig)
  | (BaseHardwareConfig & TimeSeriesConfig)
  | (BaseHardwareConfig & TabularFoundationConfig);

export type AdapterResult = {
  effectiveTokens: number;
  baseFlops: number;
  totalFlops: number;
  trace: string[];
  warnings: string[];
  confidence: ConfidenceLevel;
  dataBreakdown?: Record<string, string | number>;
};
