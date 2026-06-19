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
