// src/tests/filters.test.ts
import { describe, it, expect } from 'vitest';
import { buildExplainingFilterFn } from '../core/compiledFilter';
import { sumConstraint, parityConstraint } from '../utils/math';
import { generateTestDraws } from './testData';

describe('Filter Functions', () => {
  const testDraws = generateTestDraws(20);
  const lastDraw = testDraws[testDraws.length - 1];

  describe('buildExplainingFilterFn', () => {
    it('should filter by sum constraints', () => {
      const explainFn = buildExplainingFilterFn({
        keys: [],
        minSum: 100,
        maxSum: 200,
        evenCount: undefined,
        oddCount: undefined,
        lastDraw,
        baseline: { hitRate: 0.1, avgResidue: 10 },
        drawHistory: testDraws,
        sumConstraint,
        parityConstraint,
        generatePoolForDraw: () => [Array.from({ length: 69 }, (_, i) => i + 1)],
        recurrenceRules: undefined,
        recurrenceStats: {}
      });

      const validCombo = [10, 20, 30, 40, 50]; // Sum = 150
      const invalidCombo = [1, 2, 3, 4, 5]; // Sum = 15

      const validResult = explainFn(validCombo);
      const invalidResult = explainFn(invalidCombo);

      expect(validResult).toBeTruthy();
      expect(invalidResult).toBeFalsy();
    });

    it('should filter by parity constraints', () => {
      const explainFn = buildExplainingFilterFn({
        keys: [],
        minSum: 20,  // Lower min sum to ensure combo passes
        maxSum: 300,
        evenCount: 3,
        oddCount: 2,
        lastDraw,
        baseline: { hitRate: 0.1, avgResidue: 10 },
        drawHistory: testDraws,
        sumConstraint,
        parityConstraint,
        generatePoolForDraw: () => [Array.from({ length: 69 }, (_, i) => i + 1)],
        recurrenceRules: undefined,
        recurrenceStats: {}
      });

      const validCombo = [2, 4, 6, 7, 9]; // 3 evens, 2 odds, sum = 28
      const invalidCombo = [1, 3, 5, 7, 9]; // 0 evens, 5 odds, sum = 25

      const validResult = explainFn(validCombo);
      const invalidResult = explainFn(invalidCombo);

      expect(validResult).toBeTruthy();
      expect(invalidResult).toBeFalsy();
    });

    it('should include keys in explanation', () => {
      const explainFn = buildExplainingFilterFn({
        keys: [5, 10],
        minSum: 50,
        maxSum: 300,
        evenCount: undefined,
        oddCount: undefined,
        lastDraw,
        baseline: { hitRate: 0.1, avgResidue: 10 },
        drawHistory: testDraws,
        sumConstraint,
        parityConstraint,
        generatePoolForDraw: () => [Array.from({ length: 69 }, (_, i) => i + 1)],
        recurrenceRules: undefined,
        recurrenceStats: {}
      });

      const comboWithKeys = [5, 10, 15, 20, 25];
      const comboWithoutKeys = [1, 2, 3, 4, 6];

      const resultWithKeys = explainFn(comboWithKeys);
      explainFn(comboWithoutKeys); // Test that it doesn't crash

      expect(resultWithKeys).toBeTruthy();
      expect(resultWithKeys?.keysMatched).toContain(5);
      expect(resultWithKeys?.keysMatched).toContain(10);
    });
  });

  describe('Combo Generation Integration', () => {
    it('should generate combos that pass filters', () => {
      const explainFn = buildExplainingFilterFn({
        keys: [],
        minSum: 100,
        maxSum: 200,
        evenCount: undefined,
        oddCount: undefined,
        lastDraw,
        baseline: { hitRate: 0.1, avgResidue: 10 },
        drawHistory: testDraws,
        sumConstraint,
        parityConstraint,
        generatePoolForDraw: () => [Array.from({ length: 69 }, (_, i) => i + 1)],
        recurrenceRules: undefined,
        recurrenceStats: {}
      });

      // Test a few sample combos
      const testCombos = [
        [10, 20, 30, 40, 50], // Sum = 150 (valid)
        [1, 2, 3, 4, 5],     // Sum = 15 (invalid)
        [60, 61, 62, 63, 64] // Sum = 310 (invalid)
      ];

      const results = testCombos.map(combo => explainFn(combo));

      expect(results[0]).toBeTruthy(); // Valid combo
      expect(results[1]).toBeFalsy();  // Too low sum
      expect(results[2]).toBeFalsy();  // Too high sum
    });
  });
});
