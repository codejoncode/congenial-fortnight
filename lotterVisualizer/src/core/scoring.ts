// src/core/scoring.ts
import type { ComboExplanation } from './types';

export const calculateConfidenceIndex = (
  lift?: number,
  recurrenceDetails?: ComboExplanation['recurrence'],
  regimeCenter?: number,
  sum?: number,
  regimeTol?: number
): number => {
  const liftScore = lift !== undefined ? Math.max(0, Math.min(40, lift)) : 0;

  let recurrenceScore = 0;
  if (recurrenceDetails && recurrenceDetails.length) {
    const avgProb = recurrenceDetails.reduce((acc: number, r) => acc + Math.max(r.pRepeat, r.pRebound), 0) / recurrenceDetails.length;
    recurrenceScore = Math.min(40, avgProb * 40);
  }

  let regimeScore = 0;
  if (regimeCenter !== undefined && sum !== undefined && regimeTol !== undefined) {
    const dist = Math.abs(sum - regimeCenter);
    regimeScore = dist <= regimeTol ? 20 * (1 - dist / regimeTol) : 0;
  }

  return Math.round(liftScore + recurrenceScore + regimeScore);
};
