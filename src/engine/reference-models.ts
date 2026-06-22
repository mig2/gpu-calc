export type ReferenceModel = {
  name: string;
  parameters: number;
  trainingTokens: number;
  estimatedFlops: number;
  source: string;
};

/**
 * Known training runs for comparison. FLOPs are approximate 6ND estimates
 * unless a more precise figure is publicly available.
 */
export const REFERENCE_MODELS: ReferenceModel[] = [
  {
    name: 'GPT-3 175B',
    parameters: 175e9,
    trainingTokens: 300e9,
    estimatedFlops: 3.15e23,
    source: 'Brown et al. 2020 (estimated)',
  },
  {
    name: 'Chinchilla 70B',
    parameters: 70e9,
    trainingTokens: 1.4e12,
    estimatedFlops: 5.88e23,
    source: 'Hoffmann et al. 2022',
  },
  {
    name: 'Llama 2 7B',
    parameters: 7e9,
    trainingTokens: 2e12,
    estimatedFlops: 8.4e19,
    source: 'Touvron et al. 2023',
  },
  {
    name: 'Llama 2 13B',
    parameters: 13e9,
    trainingTokens: 2e12,
    estimatedFlops: 1.56e20,
    source: 'Touvron et al. 2023',
  },
  {
    name: 'Llama 2 70B',
    parameters: 70e9,
    trainingTokens: 2e12,
    estimatedFlops: 8.4e23,
    source: 'Touvron et al. 2023',
  },
  {
    name: 'Llama 3 8B',
    parameters: 8e9,
    trainingTokens: 15e12,
    estimatedFlops: 7.2e23,
    source: 'Meta 2024',
  },
  {
    name: 'Llama 3 70B',
    parameters: 70e9,
    trainingTokens: 15e12,
    estimatedFlops: 6.3e24,
    source: 'Meta 2024',
  },
  {
    name: 'Llama 3 405B',
    parameters: 405e9,
    trainingTokens: 15e12,
    estimatedFlops: 3.645e25,
    source: 'Meta 2024',
  },
  {
    name: 'Marin 8B',
    parameters: 8e9,
    trainingTokens: 1.6e12,
    estimatedFlops: 7.68e22,
    source: 'Marin Community 2025',
  },
  {
    name: 'Marin 32B',
    parameters: 32e9,
    trainingTokens: 3.2e12,
    estimatedFlops: 6.144e23,
    source: 'Marin Community 2025',
  },
];

/**
 * Find the closest reference models to a given FLOP count.
 * Returns the nearest smaller and nearest larger model.
 */
export function findNearestReferences(totalFlops: number): {
  smaller: ReferenceModel | null;
  larger: ReferenceModel | null;
  closest: ReferenceModel;
} {
  const sorted = [...REFERENCE_MODELS].sort((a, b) => a.estimatedFlops - b.estimatedFlops);

  let smaller: ReferenceModel | null = null;
  let larger: ReferenceModel | null = null;

  for (const model of sorted) {
    if (model.estimatedFlops <= totalFlops) {
      smaller = model;
    } else if (!larger) {
      larger = model;
    }
  }

  const closest = sorted.reduce((prev, curr) =>
    Math.abs(curr.estimatedFlops - totalFlops) < Math.abs(prev.estimatedFlops - totalFlops)
      ? curr : prev
  );

  return { smaller, larger, closest };
}

export function formatFlopsShort(n: number): string {
  if (n >= 1e24) return `${(n / 1e24).toFixed(1)}e24`;
  if (n >= 1e23) return `${(n / 1e23).toFixed(1)}e23`;
  if (n >= 1e22) return `${(n / 1e22).toFixed(1)}e22`;
  if (n >= 1e21) return `${(n / 1e21).toFixed(1)}e21`;
  if (n >= 1e20) return `${(n / 1e20).toFixed(1)}e20`;
  if (n >= 1e19) return `${(n / 1e19).toFixed(1)}e19`;
  return n.toExponential(1);
}
