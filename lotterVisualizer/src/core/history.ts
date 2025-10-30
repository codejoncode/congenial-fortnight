// src/core/history.ts
export const evaluateComboHistory = (
  combo: number[],
  drawHistory: number[][],
  filterFn: (draw: number[]) => boolean,
  generatePoolForDraw: (history: number[][], idx: number, filterFn: (draw: number[]) => boolean) => number[][]
) => {
  let testedDraws = 0;
  let hits = 0;
  let totalResidue = 0;
  let lastHitIndex: number | undefined;

  drawHistory.forEach((draw, idx) => {
    const pool = generatePoolForDraw(drawHistory, idx, filterFn);
    if (pool.some(c => arraysEqual(c, combo))) {
      testedDraws++;
      totalResidue += pool.length;
      if (arraysEqual(draw, combo)) {
        hits++;
        if (lastHitIndex === undefined) lastHitIndex = idx;
      }
    }
  });

  const hitRate = testedDraws ? hits / testedDraws : 0;
  const avgResidue = testedDraws ? totalResidue / testedDraws : 0;
  const drawsOut = lastHitIndex !== undefined ? lastHitIndex : undefined;
  return { testedDraws, hits, hitRate, avgResidue, lastHitIndex, drawsOut };
};

const arraysEqual = (a: number[], b: number[]) =>
  a.length === b.length && a.every((n, i) => n === b[i]);

// Total Draw Points (post‑Q4 start)
// src/core/history.ts (continued)
export const buildNumberPointsTable = (
  drawHistory: number[][],
  startIndex: number
): Record<number, number> => {
  const counts: Record<number, number> = {};
  const draws = drawHistory.slice(startIndex);
  draws.forEach(draw => draw.forEach(num => { counts[num] = (counts[num] || 0) + 1; }));
  const maxHits = Math.max(1, ...Object.values(counts));
  Object.keys(counts).forEach(n => { counts[+n] = (counts[+n] / maxHits) * 10; });
  return counts;
};

export const calculateTotalDrawPoints = (
  combo: number[],
  pointsTable: Record<number, number>
): number => combo.reduce((acc, n) => acc + (pointsTable[n] || 0), 0);
