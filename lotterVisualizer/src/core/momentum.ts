// src/core/momentum.ts
export const calculateMomentumAdjustedConfidence = (
  baseConfidence: number,
  trail: ComboMomentumPoint[],
  lookback: number
): number => {
  if (trail.length < 2) return baseConfidence;
  const recent = trail.slice(-lookback);
  const first = recent[0].confidence;
  const last = recent[recent.length - 1].confidence;
  const delta = last - first;
  const maxDelta = 20; // tune
  const momentumBonus = Math.max(-20, Math.min(20, (delta / maxDelta) * 20));
  return Math.round(baseConfidence + momentumBonus);
};

export const calculateFreshnessMultiplier = (
  drawsOut: number,
  maxPenaltyGap: number = 20
): number => {
  if (drawsOut <= 2) return 1.15;
  if (drawsOut >= maxPenaltyGap) return 0.85;
  const range = maxPenaltyGap - 2;
  const factor = 1.15 - ((drawsOut - 2) / range) * (1.15 - 0.85);
  return factor;
};

export const calculateHeatScore = (
  baseConfidence: number,
  trail: ComboMomentumPoint[],
  lookback: number,
  drawsOut: number
): number => {
  const momentumConfidence = calculateMomentumAdjustedConfidence(baseConfidence, trail, lookback);
  const freshnessFactor = calculateFreshnessMultiplier(drawsOut);
  return Math.round(momentumConfidence * freshnessFactor);
};

import type { ComboExplanation, ComboMomentumPoint } from './types';

export const buildMomentumTrail = (
  combo: number[],
  draws: number[][],
  lookback: number = 10
): ComboMomentumPoint[] => {
  const trail: ComboMomentumPoint[] = [];

  for (let i = Math.max(0, draws.length - lookback); i < draws.length; i++) {
    const confidence = calculateComboConfidence(combo, draws.slice(0, i + 1));
    trail.push({
      drawIndex: i,
      confidence,
      totalPoints: combo.reduce((sum, num) => sum + num, 0)
    });
  }

  return trail;
};

export const calculateComboConfidence = (
  combo: number[],
  history: number[][]
): number => {
  // Simple confidence calculation based on historical performance
  // This is a placeholder - you might want to use more sophisticated logic
  const hits = history.filter(draw =>
    combo.every(num => draw.includes(num))
  ).length;

  return Math.round((hits / history.length) * 100);
};

export const enhanceCombosWithMomentum = (
  combos: ComboExplanation[],
  draws: number[][],
  lookback: number = 10
): ComboExplanation[] => {
  return combos.map(combo => {
    const trail = buildMomentumTrail(combo.combo, draws, lookback);
    const baseConfidence = combo.confidence || 0;
    const drawsOut = combo.history.drawsOut || 0;

    const heatScore = calculateHeatScore(baseConfidence, trail, lookback, drawsOut);

    // Calculate momentum delta
    let momentumDelta = 0;
    if (trail.length >= 2) {
      const recent = trail.slice(-lookback);
      const first = recent[0].confidence;
      const last = recent[recent.length - 1].confidence;
      momentumDelta = last - first;
    }

    return {
      ...combo,
      baseConfidence,
      heatScore,
      // Store momentum delta for display
      momentumDelta
    };
  });
};
