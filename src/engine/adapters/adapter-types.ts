import type { AdapterResult, Scenario } from '../types';

export type DomainAdapter<T extends Scenario = Scenario> = {
  computeFlops: (scenario: T) => AdapterResult;
};
