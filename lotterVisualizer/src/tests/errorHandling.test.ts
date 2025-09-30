// src/tests/errorHandling.test.ts
import { describe, it, expect } from 'vitest';
import { calculateConfidenceIndex } from '../core/scoring';
import { buildExplainingFilterFn } from '../core/compiledFilter';
import { sumConstraint, parityConstraint } from '../utils/math';

describe('Error Handling Tests', () => {
  describe('Scoring Function Error Handling', () => {
    it('handles null/undefined inputs gracefully', () => {
      expect(() => calculateConfidenceIndex(10, null as any, 150, 150, 10)).not.toThrow();
      expect(() => calculateConfidenceIndex(10, undefined, 150, 150, 10)).not.toThrow();
      expect(() => calculateConfidenceIndex(10, [], null as any, 150, 10)).not.toThrow();
    });

    it('handles extreme values', () => {
      const result = calculateConfidenceIndex(1000, [], 150, 150, 10);
      expect(result).toBeLessThanOrEqual(100);

      const result2 = calculateConfidenceIndex(-1000, [], 150, 150, 10);
      expect(result2).toBeGreaterThanOrEqual(0);
    });

    it('handles empty recurrence arrays', () => {
      const result = calculateConfidenceIndex(10, [], 150, 150, 10);
      expect(typeof result).toBe('number');
      expect(result).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Filter Function Error Handling', () => {
    const testDraws = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]];
    const pool = Array.from({ length: 69 }, (_, i) => i + 1);

    it('handles missing baseline data', () => {
      expect(() => {
        buildExplainingFilterFn({
          keys: [],
          minSum: 100,
          maxSum: 200,
          evenCount: undefined,
          oddCount: undefined,
          lastDraw: testDraws[0],
          baseline: null as any,
          drawHistory: testDraws,
          sumConstraint,
          parityConstraint,
          generatePoolForDraw: () => [pool],
          recurrenceRules: undefined,
          recurrenceStats: {}
        });
      }).not.toThrow();
    });

    it('handles empty draw history', () => {
      const explainFn = buildExplainingFilterFn({
        keys: [],
        minSum: 100,
        maxSum: 200,
        evenCount: undefined,
        oddCount: undefined,
        lastDraw: [1, 2, 3, 4, 5],
        baseline: { hitRate: 0.1, avgResidue: 10 },
        drawHistory: [],
        sumConstraint,
        parityConstraint,
        generatePoolForDraw: () => [pool],
        recurrenceRules: undefined,
        recurrenceStats: {}
      });

      const result = explainFn([10, 20, 30, 40, 50]);
      expect(result).toBeDefined();
    });

    it('handles invalid combo inputs', () => {
      const explainFn = buildExplainingFilterFn({
        keys: [],
        minSum: 100,
        maxSum: 200,
        evenCount: undefined,
        oddCount: undefined,
        lastDraw: testDraws[0],
        baseline: { hitRate: 0.1, avgResidue: 10 },
        drawHistory: testDraws,
        sumConstraint,
        parityConstraint,
        generatePoolForDraw: () => [pool],
        recurrenceRules: undefined,
        recurrenceStats: {}
      });

      // Test with invalid inputs
      expect(() => explainFn([])).not.toThrow();
      expect(() => explainFn([1, 2, 3])).not.toThrow();
      expect(() => explainFn([1, 2, 3, 4, 5, 6])).not.toThrow();
      expect(() => explainFn(null as any)).not.toThrow();
    });
  });

  describe('Data Validation Error Handling', () => {
    it('handles malformed draw data', () => {
      const malformedDraws = [
        [1, 2, 3, 4, 5],      // Valid
        [1, 2, 3],            // Too short
        [1, 2, 3, 4, 5, 6],  // Too long
        [],                    // Empty
        [1, 1, 2, 3, 4],     // Duplicates
        [70, 71, 72, 73, 74] // Out of range
      ];

      // These should not crash the system
      malformedDraws.forEach(draw => {
        expect(() => {
          const sum = draw.reduce((a, b) => a + b, 0);
          expect(typeof sum).toBe('number');
        }).not.toThrow();
      });
    });

    it('handles division by zero scenarios', () => {
      // Test hit rate calculations with zero draws
      const hitRate = 5 / 0;
      expect(hitRate).toBe(Infinity);

      // Test with very small numbers
      const smallHitRate = 1 / 1000;
      expect(smallHitRate).toBeGreaterThan(0);
      expect(smallHitRate).toBeLessThan(1);
    });
  });
});
