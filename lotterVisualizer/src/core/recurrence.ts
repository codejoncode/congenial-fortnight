// src/core/recurrence.ts
import type { TypeStats, FilterConfig, ComboExplanation } from './types';

export const analyzeTypeRecurrence = (
  draws: number[][],
  typeFn: (draw: number[]) => boolean
): TypeStats => {
  let hits = 0, repeats = 0, totalGaps = 0, maxGap = 0, streaks = 0;
  const gapDist: Record<number, number> = {};
  let lastHitIndex: number | null = null;
  let prevHit = false;

  draws.forEach((draw, idx) => {
    const hit = typeFn(draw);
    if (hit) {
      hits++;
      if (prevHit) streaks++;
      if (lastHitIndex !== null) {
        const gap = idx - lastHitIndex - 1;
        gapDist[gap] = (gapDist[gap] || 0) + 1;
        totalGaps += gap;
        if (gap > maxGap) maxGap = gap;
        if (gap === 0) repeats++;
      }
      lastHitIndex = idx;
    }
    prevHit = hit;
  });

  const hitPct = hits / draws.length;
  const repeatPct = hits ? repeats / hits : 0;
  const avgGap = hits > 1 ? totalGaps / (hits - 1) : 0;
  const streakPct = hits ? streaks / hits : 0;

  let hitAfterHit = 0, hitAfterMiss = 0, hitCountAfterHit = 0, hitCountAfterMiss = 0;
  for (let i = 0; i < draws.length - 1; i++) {
    const nowHit = typeFn(draws[i]);
    const nextHit = typeFn(draws[i+1]);
    if (nowHit) { hitCountAfterHit++; if (nextHit) hitAfterHit++; }
    else { hitCountAfterMiss++; if (nextHit) hitAfterMiss++; }
  }

  return {
    typeId: typeFn.name || 'anonType',
    hitPct,
    repeatPct,
    avgGap,
    gapDist,
    maxGap,
    streakPct,
    pRepeat: hitCountAfterHit ? hitAfterHit / hitCountAfterHit : 0,
    pRebound: hitCountAfterMiss ? hitAfterMiss / hitCountAfterMiss : 0
  };
};

export const buildRecurrenceStats = (
  draws: number[][],
  types: { id: string; fn: (draw: number[]) => boolean }[]
): Record<string, TypeStats> => {
  const stats: Record<string, TypeStats> = {};
  types.forEach(t => { stats[t.id] = analyzeTypeRecurrence(draws, t.fn); });
  return stats;
};

export const checkRecurrenceWithExplain = (
  lastDraw: number[],
  recurrenceRules: FilterConfig['recurrenceRules'],
  stats: Record<string, TypeStats>
): { passed: boolean; details: ComboExplanation['recurrence'] } => {
  if (!recurrenceRules) return { passed: true, details: [] };
  const { types, minRepeatProb, minReboundProb, mode } = recurrenceRules;
  let passed = true;
  const details: ComboExplanation['recurrence'] = [];

  const pct = (n?: number) => n !== undefined ? `${(n * 100).toFixed(1)}%` : 'n/a';

  for (const t of types) {
    const typeHitLastDraw = t.fn(lastDraw);
    const s = stats[t.id];
    if (!s) continue;

    let typePassed = true;
    let reason = '';

    if ((mode === 'excludeLowRepeat' || mode === 'both') && typeHitLastDraw) {
      if (s.pRepeat < (minRepeatProb ?? 0)) { typePassed = false;
        reason = `Just hit, repeat ${pct(s.pRepeat)} < min ${pct(minRepeatProb)}`; }
      else { reason = `Just hit, repeat ${pct(s.pRepeat)} ≥ min ${pct(minRepeatProb)}`; }
    }

    if ((mode === 'includeHighRebound' || mode === 'both') && !typeHitLastDraw) {
      if (s.pRebound < (minReboundProb ?? 0)) { typePassed = false;
        reason = `Missed last, rebound ${pct(s.pRebound)} < min ${pct(minReboundProb)}`; }
      else { reason = `Missed last, rebound ${pct(s.pRebound)} ≥ min ${pct(minReboundProb)}`; }
    }

    if (!typePassed) passed = false;
    details.push({ typeId: t.id, pRepeat: s.pRepeat, pRebound: s.pRebound, passed: typePassed, reason });
  }
  return { passed, details };
};
