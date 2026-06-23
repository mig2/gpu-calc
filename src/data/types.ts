export type ApiModel = {
  model: string;
  inputPer1M: number;    // $ per 1M input tokens
  outputPer1M: number;   // $ per 1M output tokens
  contextWindow: number;
};

export type ApiProvider = {
  provider: string;
  models: ApiModel[];
};

export type ApiPricingData = {
  lastUpdated: string;
  providers: ApiProvider[];
};

export type CloudGpuInstance = {
  instance: string;
  gpu: string;
  gpuCount: number;
  onDemandPerHr: number;
  reservedPerHr: number | null;
  spotPerHr: number | null;
};

export type CloudGpuProvider = {
  provider: string;
  instances: CloudGpuInstance[];
};

export type CloudGpuPricingData = {
  lastUpdated: string;
  providers: CloudGpuProvider[];
};

export type SelfHostEntry = {
  model: string;
  parameters: string;
  gpu: string;
  gpuCount: number;
  quantization: string;
  framework: string;
  outputTokensPerSec: number;
  maxConcurrentRequests: number;
};

export type SelfHostThroughputData = {
  lastUpdated: string;
  note: string;
  entries: SelfHostEntry[];
};
