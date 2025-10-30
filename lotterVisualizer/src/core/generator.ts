// src/core/generator.ts
import type { ComboExplanation } from './types';

export const generateExplainedCombosFast = (
  pool: number[],
  pickCount: number,
  explainFn: (combo: number[]) => ComboExplanation | null,
  minSum: number,
  maxSum: number
): ComboExplanation[] => {
  const results: ComboExplanation[] = [];
  const combo: number[] = [];

  const backtrack = (start: number, sumSoFar: number) => {
    if (combo.length === pickCount) {
      const expl = explainFn(combo);
      if (expl) results.push(expl);
      return;
    }
    for (let i = start; i < pool.length; i++) {
      const n = pool[i];
      const newSum = sumSoFar + n;

      const remaining = pickCount - combo.length - 1;
      if (newSum + (remaining * pool[pool.length - 1]) < minSum) continue;
      if (newSum + (remaining * pool[0]) > maxSum) continue;

      combo.push(n);
      backtrack(i + 1, newSum);
      combo.pop();
    }
  };

  backtrack(0, 0);
  return results;
};
