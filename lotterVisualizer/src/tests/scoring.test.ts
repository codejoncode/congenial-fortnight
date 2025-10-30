// src/tests/scoring.test.ts
import { describe, it, expect } from 'vitest';
import { calculateConfidenceIndex } from '../core/scoring';
import { calculateMomentumAdjustedConfidence, calculateHeatScore, calculateFreshnessMultiplier } from '../core/momentum';
import type { ComboMomentumPoint } from '../core/types';

describe('Scoring Functions', () => {
  describe('calculateConfidenceIndex', () => {
    it('should calculate confidence with lift score', () => {
      const result = calculateConfidenceIndex(15, [], undefined, 150, 10);
      expect(result).toBeGreaterThan(0);
      expect(result).toBeLessThanOrEqual(100);
    });

    it('should handle undefined lift', () => {
      const result = calculateConfidenceIndex(undefined, [], undefined, 150, 10);
      expect(result).toBeGreaterThanOrEqual(0);
    });

    it('should include recurrence score', () => {
      const recurrenceDetails = [
        { typeId: 'test-type', pRepeat: 0.8, pRebound: 0.3, passed: true, reason: 'Good recurrence' }
      ];
      const result = calculateConfidenceIndex(10, recurrenceDetails, undefined, 150, 10);
      expect(result).toBeGreaterThan(10); // Should include recurrence bonus
    });

    it('should include regime score', () => {
      const result = calculateConfidenceIndex(10, [], 150, 150, 5);
      expect(result).toBeGreaterThan(10); // Should include regime bonus
    });
  });

  describe('calculateFreshnessMultiplier', () => {
    it('should give bonus for very recent hits', () => {
      const result = calculateFreshnessMultiplier(1);
      expect(result).toBeGreaterThan(1);
    });

    it('should give penalty for old hits', () => {
      const result = calculateFreshnessMultiplier(25);
      expect(result).toBeLessThan(1);
    });

    it('should handle edge cases', () => {
      expect(calculateFreshnessMultiplier(0)).toBe(1.15);
      expect(calculateFreshnessMultiplier(30)).toBe(0.85);
    });
  });

  describe('calculateMomentumAdjustedConfidence', () => {
    it('should return base confidence with insufficient data', () => {
      const trail: ComboMomentumPoint[] = [];
      const result = calculateMomentumAdjustedConfidence(50, trail, 5);
      expect(result).toBe(50);
    });

    it('should apply positive momentum bonus', () => {
      const trail: ComboMomentumPoint[] = [
        { drawIndex: 0, confidence: 40, totalPoints: 150 },
        { drawIndex: 1, confidence: 45, totalPoints: 152 },
        { drawIndex: 2, confidence: 50, totalPoints: 148 },
        { drawIndex: 3, confidence: 55, totalPoints: 155 },
        { drawIndex: 4, confidence: 60, totalPoints: 153 }
      ];
      const result = calculateMomentumAdjustedConfidence(50, trail, 5);
      expect(result).toBeGreaterThan(50);
    });

    it('should apply negative momentum penalty', () => {
      const trail: ComboMomentumPoint[] = [
        { drawIndex: 0, confidence: 60, totalPoints: 150 },
        { drawIndex: 1, confidence: 55, totalPoints: 152 },
        { drawIndex: 2, confidence: 50, totalPoints: 148 },
        { drawIndex: 3, confidence: 45, totalPoints: 155 },
        { drawIndex: 4, confidence: 40, totalPoints: 153 }
      ];
      const result = calculateMomentumAdjustedConfidence(50, trail, 5);
      expect(result).toBeLessThan(50);
    });
  });

  describe('calculateHeatScore', () => {
    it('should combine momentum and freshness', () => {
      const trail: ComboMomentumPoint[] = [
        { drawIndex: 0, confidence: 50, totalPoints: 150 },
        { drawIndex: 1, confidence: 52, totalPoints: 152 }
      ];
      const result = calculateHeatScore(50, trail, 5, 3);
      expect(result).toBeGreaterThan(0);
      expect(typeof result).toBe('number');
    });

    it('should handle fresh combos', () => {
      const trail: ComboMomentumPoint[] = [
        { drawIndex: 0, confidence: 50, totalPoints: 150 }
      ];
      const result = calculateHeatScore(50, trail, 5, 1);
      expect(result).toBeGreaterThan(50); // Freshness bonus
    });
  });
});
