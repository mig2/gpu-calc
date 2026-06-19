import type { TrainingScenario, EstimateResult } from './types';
import { getGpuById } from './gpu-data';

export function exportToJson(
  scenario: TrainingScenario,
  results: EstimateResult[],
  modelFamily: string = 'llm',
): string {
  return JSON.stringify({ schema_version: '2.0', model_family: modelFamily, scenario, results }, null, 2);
}

export function exportToCsv(results: EstimateResult[]): string {
  const headers = ['GPU', 'Required GPUs', 'H100 Equivalents', 'Tokens', 'Base FLOPs', 'Total FLOPs', 'Sustained FLOP/s', 'Memory LB GPUs'];
  const rows = results.map((r) => {
    const gpu = getGpuById(r.gpuId);
    return [
      gpu?.label ?? r.gpuId,
      r.requiredGpus,
      r.h100Equivalents.toFixed(1),
      r.tokens,
      r.baseFlops,
      r.totalFlops,
      r.sustainedFlopsPerGpu,
      r.memoryLowerBoundGpus,
    ].join(',');
  });
  return [headers.join(','), ...rows].join('\n');
}

export function exportToMarkdown(scenario: TrainingScenario, results: EstimateResult[]): string {
  const lines: string[] = [
    '# GPU Calculator Results',
    '',
    `**Model:** ${formatParams(scenario.modelParameters)}`,
    `**Tokens:** ${formatParams(scenario.tokensPerParameter * scenario.modelParameters)}`,
    `**Training window:** ${(scenario.trainingWindowSeconds / 86_400).toFixed(0)} days`,
    `**Precision:** ${scenario.precision}`,
    `**Availability:** ${(scenario.availability * 100).toFixed(0)}%`,
    `**Overhead:** ${scenario.overheadFactor}x`,
    '',
    '| GPU | Required GPUs | H100 Equiv. |',
    '|-----|--------------|-------------|',
  ];

  for (const r of results) {
    const gpu = getGpuById(r.gpuId);
    lines.push(`| ${gpu?.label ?? r.gpuId} | ${r.requiredGpus.toLocaleString()} | ${r.h100Equivalents.toFixed(1)} |`);
  }

  return lines.join('\n');
}

function formatParams(n: number): string {
  if (n >= 1e12) return `${(n / 1e12).toFixed(1)}T`;
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  return n.toLocaleString();
}

export function encodeScenarioToHash(scenario: TrainingScenario, modelFamily: string = 'llm'): string {
  const params = new URLSearchParams();
  params.set('family', modelFamily);
  const paramB = scenario.modelParameters / 1e9;
  params.set('model', `${paramB}B`);
  params.set('days', String(scenario.trainingWindowSeconds / 86_400));
  params.set('tpp', String(scenario.tokensPerParameter));
  params.set('gpus', scenario.selectedGpuIds.join(','));
  params.set('precision', scenario.precision);
  params.set('avail', String(scenario.availability));
  params.set('overhead', String(scenario.overheadFactor));
  params.set('mode', scenario.trainingMode);
  params.set('membytes', String(scenario.memoryBytesPerParameter));

  for (const gpuId of scenario.selectedGpuIds) {
    const mfu = scenario.mfuByGpuId[gpuId];
    if (mfu != null) params.set(`mfu_${gpuId}`, String(mfu));
  }

  return '#' + params.toString();
}

export function decodeScenarioFromHash(hash: string): { scenario: Partial<TrainingScenario>; modelFamily?: string } | null {
  if (!hash || hash === '#') return null;
  try {
    const params = new URLSearchParams(hash.replace(/^#/, ''));
    const result: Partial<TrainingScenario> = {};
    const modelFamily = params.get('family') ?? 'llm';

    const model = params.get('model');
    if (model) {
      const match = model.match(/^(\d+(?:\.\d+)?)\s*([MBT])?$/i);
      if (match) {
        const num = parseFloat(match[1]);
        const unit = (match[2] || 'B').toUpperCase();
        result.modelParameters = unit === 'T' ? num * 1e12 : unit === 'M' ? num * 1e6 : num * 1e9;
      }
    }

    const days = params.get('days');
    if (days) result.trainingWindowSeconds = parseFloat(days) * 86_400;

    const tpp = params.get('tpp');
    if (tpp) result.tokensPerParameter = parseFloat(tpp);

    const gpus = params.get('gpus');
    if (gpus) result.selectedGpuIds = gpus.split(',');

    const precision = params.get('precision');
    if (precision === 'BF16_DENSE' || precision === 'FP8_DENSE') result.precision = precision;

    const avail = params.get('avail');
    if (avail) result.availability = parseFloat(avail);

    const overhead = params.get('overhead');
    if (overhead) result.overheadFactor = parseFloat(overhead);

    const mode = params.get('mode');
    if (mode) result.trainingMode = mode as TrainingScenario['trainingMode'];

    const membytes = params.get('membytes');
    if (membytes) result.memoryBytesPerParameter = parseFloat(membytes);

    if (result.selectedGpuIds) {
      const mfuByGpuId: Record<string, number> = {};
      for (const gpuId of result.selectedGpuIds) {
        const mfu = params.get(`mfu_${gpuId}`);
        if (mfu) mfuByGpuId[gpuId] = parseFloat(mfu);
      }
      if (Object.keys(mfuByGpuId).length > 0) result.mfuByGpuId = mfuByGpuId;
    }

    return { scenario: result, modelFamily };
  } catch {
    return null;
  }
}
