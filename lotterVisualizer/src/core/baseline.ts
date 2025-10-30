// src/core/baseline.ts
export const calculateBaselineMetrics = (
  drawHistory: number[][],
  filterFn: (draw: number[]) => boolean,
  generatePoolForDraw: (history: number[][], idx: number, filterFn: (draw: number[]) => boolean) => number[][]
) => {
  let testedDraws = 0;
  let hits = 0;
  let totalResidue = 0;

  drawHistory.forEach((draw, idx) => {
    const pool = generatePoolForDraw(drawHistory, idx, filterFn);
    if (pool.length) {
      testedDraws++;
      totalResidue += pool.length;
      if (pool.some(c => arraysEqual(c, draw))) hits++;
    }
  });

  const hitRate = testedDraws ? hits / testedDraws : 0;
  const avgResidue = testedDraws ? totalResidue / testedDraws : 0;
  return { hitRate, avgResidue };
};

const arraysEqual = (a: number[], b: number[]) =>
  a.length === b.length && a.every((n, i) => n === b[i]);
