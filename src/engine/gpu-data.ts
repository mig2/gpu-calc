import type { GpuSku } from './types';

export const GPU_SKUS: GpuSku[] = [
  {
    id: 'h100-sxm',
    label: 'NVIDIA H100 SXM',
    bf16DenseFlops: 0.9895e15,
    fp8DenseFlops: 1.979e15,
    memoryGb: 80,
    bandwidthTbps: 3.35,
    defaultMfu: 0.40,
    notes: 'Dense values derived by halving NVIDIA sparse tensor-core specs.',
  },
  {
    id: 'h200-sxm',
    label: 'NVIDIA H200 SXM',
    bf16DenseFlops: 0.9895e15,
    fp8DenseFlops: 1.979e15,
    memoryGb: 141,
    bandwidthTbps: 4.8,
    defaultMfu: 0.45,
    notes: 'Same raw Hopper compute class as H100; more HBM and bandwidth.',
  },
  {
    id: 'b200-sxm',
    label: 'NVIDIA B200 SXM / HGX B200',
    bf16DenseFlops: 2.25e15,
    fp8DenseFlops: 4.5e15,
    memoryGb: 175,
    defaultMfu: 0.40,
    notes: 'Per-GPU value derived from HGX B200 8-GPU specs.',
  },
  {
    id: 'gb200-nvl72-gpu',
    label: 'GB200 NVL72 Blackwell GPU equivalent',
    bf16DenseFlops: 2.5e15,
    fp8DenseFlops: 5.0e15,
    memoryGb: 186,
    bandwidthTbps: 8.0,
    defaultMfu: 0.40,
    notes: 'Per-GPU value derived from 72-GPU NVL72 dense rack specs.',
  },
];

export const H100_REFERENCE_ID = 'h100-sxm';

export function getGpuById(id: string): GpuSku | undefined {
  return GPU_SKUS.find((g) => g.id === id);
}

export function getH100Reference(): GpuSku {
  const h100 = getGpuById(H100_REFERENCE_ID);
  if (!h100) throw new Error('H100 reference GPU not found in SKU table');
  return h100;
}

/** Spec date for the GPU data table, shown in the UI */
export const GPU_DATA_SPEC_DATE = '2026-06-18';
