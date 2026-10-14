import type { GpuSku } from './types';

/**
 * GPU/accelerator SKU table, transcribed from
 * `docs/research/2026-10-14-llm-training-research.md` §4.2.
 *
 * All compute figures are DENSE. NVIDIA publishes tensor-core throughput with
 * 2:4 structural sparsity (marked with an asterisk on its datasheets); dense is
 * exactly half. Structural sparsity is not used in pretraining, so every row
 * here halves NVIDIA's headline numbers. AMD publishes dense FP16/BF16 directly
 * but sparse FP8/FP4; Google and AWS publish dense. See §4.1 for the full
 * derivation note.
 *
 * `defaultMfu` is the BF16 baseline MFU. FP8/FP4 are NOT modeled as a free peak
 * swap — see PRECISION_MFU_FACTORS in hardware-estimator.ts.
 */
export const GPU_SKUS: GpuSku[] = [
  {
    id: 'h100-sxm',
    label: 'NVIDIA H100 SXM',
    bf16DenseFlops: 0.9895e15,
    fp8DenseFlops: 1.979e15,
    memoryGb: 80,
    bandwidthTbps: 3.35,
    tdpWatts: 700,
    scaleUpGbps: 900,
    defaultMfu: 0.40,
    notes:
      'Dense = half of NVIDIA sparse spec: 1,979 TFLOPS BF16* -> 989.5 dense; 3,958 FP8* -> 1,979 dense. No FP4 tensor cores on Hopper. MFU 0.40 anchored on Llama 3.1 405B (43% at 8,192 GPUs). Source: NVIDIA H100 datasheet (nvidia.com/en-us/data-center/h100/).',
  },
  {
    id: 'h200-sxm',
    label: 'NVIDIA H200 SXM',
    bf16DenseFlops: 0.9895e15,
    fp8DenseFlops: 1.979e15,
    memoryGb: 141,
    bandwidthTbps: 4.8,
    tdpWatts: 700,
    scaleUpGbps: 900,
    defaultMfu: 0.42,
    notes:
      'Same Hopper compute dies as H100, so identical dense derivation (1,979 BF16* -> 989.5 dense; 3,958 FP8* -> 1,979 dense). Advantage is 141 GB HBM3e at 4.8 TB/s, which raises achievable MFU via larger microbatches. defaultMfu lowered 0.45 -> 0.42 (§4.3): no published pretraining MFU for any Hopper part exceeds Llama 3 405B at 43% on H100, so 0.45 was an unsourced claim. Source: NVIDIA H200 datasheet (nvidia.com/en-us/data-center/h200/).',
  },
  {
    id: 'b200-sxm',
    label: 'NVIDIA B200 SXM / HGX B200',
    bf16DenseFlops: 2.25e15,
    fp8DenseFlops: 4.5e15,
    fp4DenseFlops: 9.0e15,
    memoryGb: 180,
    bandwidthTbps: 7.7,
    tdpWatts: 1000,
    scaleUpGbps: 1800,
    defaultMfu: 0.39,
    notes:
      'Derived per-GPU from HGX B200 8-GPU specs: 72 PFLOPS FP8 dense / 8 = 4.5 PF; BF16 = half of FP8 = 2.25 PF; FP4 72 PF dense / 8 = 9 PF. Memory 1,440 GB / 8 = 180 GB (corrected from an unsourced 175, §4.3). Sources: Lenovo HGX B200 product guide (lenovopress.lenovo.com/lp2226.pdf), NVIDIA DGX B200.',
  },
  {
    id: 'b300-sxm',
    label: 'NVIDIA B300 SXM (Blackwell Ultra)',
    bf16DenseFlops: 2.25e15,
    fp8DenseFlops: 4.5e15,
    fp4DenseFlops: 13.5e15,
    memoryGb: 288,
    bandwidthTbps: 8.0,
    tdpWatts: 1300,
    scaleUpGbps: 900,
    defaultMfu: 0.40,
    notes:
      'HONESTY MARKER: BF16 (2.25 PF) and FP8 (4.5 PF) are IDENTICAL to B200 — Blackwell Ultra\'s uplift is 1.5x FP4 (13.5 PF dense) and 49% more memory (288 GB, ~268 GB usable), nothing else. A user selecting B300 for BF16 or FP8 work gets no compute gain over B200. TDP 1,200-1,400 W (1,300 W midpoint used). Source: SemiAnalysis B300 (inferencex.semianalysis.com/chips/b300).',
  },
  {
    id: 'gb200-nvl72-gpu',
    label: 'GB200 NVL72 per-GPU',
    bf16DenseFlops: 2.5e15,
    fp8DenseFlops: 5.0e15,
    fp4DenseFlops: 10.0e15,
    memoryGb: 186,
    bandwidthTbps: 8.0,
    tdpWatts: 1200,
    scaleUpGbps: 1800,
    defaultMfu: 0.42,
    notes:
      'Derived per-GPU from the 72-GPU rack: 360 PFLOPS FP16/BF16* / 72 / 2 = 2.5 PF dense; FP8 = 2x BF16 = 5.0 PF; NVFP4 720 PF dense / 72 = 10 PF; HBM 13.4 TB / 72 = 186 GB. defaultMfu raised 0.40 -> 0.42 (§4.3, medium confidence): the 72-GPU NVLink domain at 130 TB/s aggregate lets TP groups span many more GPUs without crossing a slow link — the one SKU where topology should buy MFU. Source: NVIDIA GB200 NVL72 (nvidia.com/en-us/data-center/gb200-nvl72/).',
  },
  {
    id: 'gb300-nvl72-gpu',
    label: 'GB300 NVL72 per-GPU (Ultra)',
    bf16DenseFlops: 2.5e15,
    fp8DenseFlops: 5.0e15,
    fp4DenseFlops: 15.0e15,
    memoryGb: 288,
    bandwidthTbps: 8.0,
    tdpWatts: 1400,
    scaleUpGbps: 1800,
    defaultMfu: 0.41,
    notes:
      'Blackwell Ultra in the NVL72 rack: BF16/FP8 unchanged from GB200 (2.5 / 5.0 PF dense per GPU, same halved-sparse derivation); FP4 rises to 15 PF dense and memory to 288 GB. Source: Flopper GB300 (flopper.io) — an aggregator, not a first-party datasheet, so treat the FP4 and TDP figures as medium confidence.',
  },
  {
    id: 'mi300x',
    label: 'AMD Instinct MI300X',
    bf16DenseFlops: 1.3074e15,
    fp8DenseFlops: 1.3074e15,
    memoryGb: 192,
    bandwidthTbps: 5.3,
    tdpWatts: 750,
    scaleUpGbps: 896,
    defaultMfu: 0.30,
    notes:
      'AMD publishes dense FP16/BF16 directly (1,307.4 TFLOPS), so no halving applied. FP8 equals BF16 on MI300X — the matrix cores gain no rate at FP8, only memory footprint. No FP4 support. MFU 0.30 reflects a less mature training software stack. Sources: AMD Instinct spec summary (Wikipedia AMD Instinct), ASBIS MI300 (promo.asbis.com/amd_instinct_mi300).',
  },
  {
    id: 'mi325x',
    label: 'AMD Instinct MI325X',
    bf16DenseFlops: 1.3074e15,
    fp8DenseFlops: 2.6149e15,
    memoryGb: 256,
    bandwidthTbps: 6.0,
    tdpWatts: 1000,
    scaleUpGbps: 896,
    defaultMfu: 0.31,
    notes:
      'Same CDNA3 compute as MI300X for BF16 (1,307.4 TFLOPS dense, published dense by AMD) but FP8 doubles to 2,614.9 TFLOPS dense. Memory grows to 256 GB HBM3e at 6.0 TB/s. No FP4. Source: SemiAnalysis MI325X (inferencex.semianalysis.com/chips/mi325x).',
  },
  {
    id: 'mi355x',
    label: 'AMD Instinct MI355X',
    bf16DenseFlops: 2.5e15,
    fp8DenseFlops: 5.0e15,
    fp4DenseFlops: 10.1e15,
    memoryGb: 288,
    bandwidthTbps: 8.0,
    tdpWatts: 1400,
    scaleUpGbps: 1075,
    defaultMfu: 0.33,
    notes:
      'AMD gives 2.5 PF dense FP16/BF16 and 5.0 PF dense FP8 directly (no halving needed); FP4 10.1 PF. Read AMD datasheets carefully — AMD publishes dense for FP16/BF16 but sparse for FP8/FP4, and these are the dense figures. Sources: Oracle OCI MI355X performance post (blogs.oracle.com/cloud-infrastructure), SemiAnalysis MI355X.',
  },
  {
    id: 'mi450',
    label: 'AMD Instinct MI450',
    bf16DenseFlops: 5.0e15,
    fp8DenseFlops: 20e15,
    fp4DenseFlops: 40e15,
    memoryGb: 432,
    bandwidthTbps: 19.6,
    tdpWatts: 1800,
    scaleUpGbps: 3600,
    defaultMfu: 0.30,
    notes:
      'HONESTY MARKER: the BF16 figure (5.0 PF) is EXTRAPOLATED, NOT PUBLISHED. AMD released only FP8 (20 PF) and FP4 (40 PF); BF16 at half of FP8 is the conventional ratio but AMD has never confirmed it for MI450. Treat BF16 results on this SKU as an estimate, not a spec. Memory 432 GB HBM4; bandwidth reported as 19.6-23.3 TB/s (lower bound used); TDP ~1,800 W. LOW confidence overall. Source: WCCFTech MI400 preview (wccftech.com) — an aggregator ahead of the datasheet.',
  },
  {
    id: 'tpu-v6e',
    label: 'Google TPU v6e (Trillium)',
    bf16DenseFlops: 0.918e15,
    memoryGb: 32,
    bandwidthTbps: 1.638,
    scaleUpGbps: 1200,
    defaultMfu: 0.45,
    notes:
      'Google publishes dense figures, so 918 TFLOPS BF16 is used as-is. No native FP8 and no FP4 — BF16 is the only training precision, so FP8/FP4 selections fall back to the BF16 peak. scaleUpGbps is the 1.2 TB/s ICI interconnect expressed in GB/s. MFU 0.45 reflects PaLM-class TPU results (46.2% MFU on TPU v4). TDP not published. Source: Google TPU v6e docs (docs.cloud.google.com/tpu/docs/v6e).',
  },
  {
    id: 'tpu-v7',
    label: 'Google TPU v7 (Ironwood)',
    bf16DenseFlops: 2.307e15,
    fp8DenseFlops: 4.614e15,
    memoryGb: 192,
    bandwidthTbps: 7.37,
    scaleUpGbps: 1200,
    defaultMfu: 0.40,
    notes:
      'Google-published dense figures, used as-is: 2,307 TFLOPS BF16, 4,614 TFLOPS FP8 (exactly 2x). No FP4. 192 GB HBM at 7.37 TB/s; 1.2 TB/s ICI recorded as scaleUpGbps. TDP not published. Source: Google TPU7x docs (docs.cloud.google.com/tpu/docs/tpu7x).',
  },
  {
    id: 'trainium2',
    label: 'AWS Trainium2',
    bf16DenseFlops: 0.667e15,
    fp8DenseFlops: 1.299e15,
    memoryGb: 96,
    bandwidthTbps: 2.9,
    defaultMfu: 0.35,
    notes:
      'AWS publishes dense, used as-is: 667 TFLOPS BF16, 1,299 TFLOPS FP8 (slightly under 2x). No FP4. Memory is 96 GiB (recorded as 96 GB, a ~7% conservative understatement of the byte count). No scale-up fabric figure published in GB/s terms. TDP not published. Source: AWS Neuron Trainium2 architecture docs.',
  },
  {
    id: 'trainium3',
    label: 'AWS Trainium3',
    bf16DenseFlops: 0.671e15,
    fp8DenseFlops: 2.517e15,
    fp4DenseFlops: 2.517e15,
    memoryGb: 144,
    bandwidthTbps: 4.9,
    defaultMfu: 0.35,
    notes:
      'AWS-published dense figures: 671 TFLOPS BF16, 2,517 TFLOPS MXFP8. MXFP4 is quoted at the SAME 2,517 TFLOPS as MXFP8 — unlike NVIDIA/AMD, Trainium3 gains no rate at 4-bit, only memory footprint, so FP4 buys nothing in compute here. BF16 is unusually low relative to FP8 (3.75x gap). Memory 144 GiB (recorded as 144 GB). TDP not published. Source: AWS Neuron Trainium3 architecture docs.',
  },
  {
    id: 'gaudi3',
    label: 'Intel Gaudi 3',
    bf16DenseFlops: 1.835e15,
    fp8DenseFlops: 1.835e15,
    memoryGb: 128,
    bandwidthTbps: 3.7,
    tdpWatts: 900,
    scaleUpGbps: 1200,
    defaultMfu: 0.30,
    notes:
      'Dense 1,835 TFLOPS for both BF16 and FP8 — Gaudi 3 matrix engines run FP8 at the same rate as BF16, so FP8 saves memory and bandwidth but not time. No FP4. 128 GB HBM2e at 3.7 TB/s, 900 W, 1.2 TB/s scale-up Ethernet. MFU 0.30 reflects stack maturity. Source: Hot Chips 2024 Gaudi 3 presentation (hc2024.hotchips.org).',
  },
];

// Deliberately excluded (§4.2): NVIDIA Vera Rubin VR200 / NVL144. Rack-level
// figures are published (3.6 EF FP4, 1.2 EF FP8 across 144 GPUs) but per-GPU
// dense BF16 is not, and sources conflict on whether the headline
// "50 PFLOPS FP4" is per-package or per-die. Adding a row on that basis would
// be invention. Add it when the datasheet lands.

export const H100_REFERENCE_ID = 'h100-sxm';

/** Optional extra GPUs to search (set by the store for custom GPUs) */
let _extraGpus: GpuSku[] = [];

export function setExtraGpus(gpus: GpuSku[]) {
  _extraGpus = gpus;
}

export function getGpuById(id: string): GpuSku | undefined {
  return GPU_SKUS.find((g) => g.id === id) ?? _extraGpus.find((g) => g.id === id);
}

export function getH100Reference(): GpuSku {
  const h100 = getGpuById(H100_REFERENCE_ID);
  if (!h100) throw new Error('H100 reference GPU not found in SKU table');
  return h100;
}

/** Spec date for the GPU data table, shown in the UI */
export const GPU_DATA_SPEC_DATE = '2026-10-14';
