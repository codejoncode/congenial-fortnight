// src/core/compiledFilter.ts
import type { FilterConfig, TypeStats, ComboExplanation } from './types';
import { checkRecurrenceWithExplain } from './recurrence';
import { evaluateComboHistory } from './history';

export const buildExplainingFilterFn = (args: {
  keys: number[];
  minSum: number;
  maxSum: number;
  evenCount?: number;
  oddCount?: number;
  regimeCenter?: number;
  regimeTol?: number;
  recurrenceRules?: FilterConfig['recurrenceRules'];
  recurrenceStats?: Record<string, TypeStats>;
  lastDraw?: number[];
  baseline?: { hitRate: number; avgResidue: number };
  drawHistory?: number[][];
  sumConstraint: (draw: number[], a: number, b: number) => boolean;
  parityConstraint: (draw: number[], e?: number, o?: number) => boolean;
  generatePoolForDraw: (history: number[][], idx: number, filterFn: (draw: number[]) => boolean) => number[][];
}) => {
  const { keys, minSum, maxSum, evenCount, oddCount, regimeCenter, regimeTol,
    recurrenceRules, recurrenceStats, lastDraw, baseline, drawHistory,
    sumConstraint, parityConstraint, generatePoolForDraw } = args;

  return (combo: number[]): ComboExplanation | null => {
    // Handle invalid inputs gracefully
    if (!combo || !Array.isArray(combo) || combo.length === 0) {
      return null;
    }

    const reasons: string[] = [];
    let sum = 0, evens = 0;

    for (const k of keys) { if (!combo.includes(k)) return null; }
    if (keys.length) reasons.push(`Contains locked keys: ${keys.join(', ')}`);

    for (const n of combo) { sum += n; if (n % 2 === 0) evens++; }
    if (sum < minSum || sum > maxSum) return null;
    reasons.push(`Sum ${sum} within corridor ${minSum}–${maxSum}`);

    if (evenCount !== undefined && evens !== evenCount) return null;
    if (oddCount !== undefined && (combo.length - evens) !== oddCount) return null;
    if (evenCount !== undefined || oddCount !== undefined)
      reasons.push(`Parity ${evens}E-${combo.length - evens}O matches`);

    if (regimeCenter !== undefined && regimeTol !== undefined) {
      if (Math.abs(sum - regimeCenter) > regimeTol) return null;
      reasons.push(`Regime: ${regimeCenter}±${regimeTol}`);
    }

    let recurrenceDetails: ComboExplanation['recurrence'] = [];
    if (recurrenceRules && recurrenceStats && lastDraw) {
      const { passed, details } = checkRecurrenceWithExplain(lastDraw, recurrenceRules, recurrenceStats);
      recurrenceDetails = details;
      if (!passed) return null;
    }

    let history: ComboExplanation['history'] = { testedDraws: 0, hits: 0, hitRate: 0, avgResidue: 0 };
    let lift = 0;
    if (drawHistory && baseline) {
      const filterFn = (draw: number[]) =>
        sumConstraint(draw, minSum, maxSum) &&
        parityConstraint(draw, evenCount, oddCount) &&
        keys.every(k => draw.includes(k));
      history = evaluateComboHistory(combo, drawHistory, filterFn, generatePoolForDraw);
      lift = baseline.hitRate ? ((history.hitRate - baseline.hitRate) / baseline.hitRate) * 100 : 0;
    }

    return {
      combo,
      reasons,
      sum,
      parity: { evens, odds: combo.length - evens },
      regimeMatch: regimeCenter,
      keysMatched: keys,
      history,
      baseline,
      lift,
      recurrence: recurrenceDetails
    };
  };
};
