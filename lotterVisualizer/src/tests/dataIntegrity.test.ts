// src/tests/dataIntegrity.test.ts
import { describe, it, expect } from 'vitest';
import { generateTestDraws, generateTestCombos } from './testData';

describe('Data Integrity Tests', () => {
  describe('Combo Data Structure Validation', () => {
    it('validates combo array structure', () => {
      const testCombos = generateTestCombos(100);

      testCombos.forEach((combo, index) => {
        // Basic structure validation
        expect(Array.isArray(combo)).toBe(true);
        expect(combo).toHaveLength(5);
        expect(combo.every(num => typeof num === 'number')).toBe(true);

        // Value range validation
        combo.forEach(num => {
          expect(num).toBeGreaterThanOrEqual(1);
          expect(num).toBeLessThanOrEqual(69);
          expect(Number.isInteger(num)).toBe(true);
        });

        // Uniqueness validation
        const uniqueNumbers = new Set(combo);
        expect(uniqueNumbers.size).toBe(5);

        // Sorting validation
        for (let i = 1; i < combo.length; i++) {
          expect(combo[i]).toBeGreaterThanOrEqual(combo[i - 1]);
        }
      });
    });

    it('validates draw data structure', () => {
      const testDraws = generateTestDraws(100);

      testDraws.forEach((draw, index) => {
        // Basic structure validation
        expect(Array.isArray(draw)).toBe(true);
        expect(draw).toHaveLength(5);
        expect(draw.every(num => typeof num === 'number')).toBe(true);

        // Value range validation
        draw.forEach(num => {
          expect(num).toBeGreaterThanOrEqual(1);
          expect(num).toBeLessThanOrEqual(69);
          expect(Number.isInteger(num)).toBe(true);
        });

        // Uniqueness validation
        const uniqueNumbers = new Set(draw);
        expect(uniqueNumbers.size).toBe(5);

        // Sorting validation
        for (let i = 1; i < draw.length; i++) {
          expect(draw[i]).toBeGreaterThanOrEqual(draw[i - 1]);
        }
      });
    });
  });

  describe('Mathematical Consistency Tests', () => {
    it('validates sum calculations', () => {
      const testCombos = generateTestCombos(50);

      testCombos.forEach(combo => {
        const sum1 = combo.reduce((a, b) => a + b, 0);
        const sum2 = combo[0] + combo[1] + combo[2] + combo[3] + combo[4];

        expect(sum1).toBe(sum2);
        expect(sum1).toBeGreaterThanOrEqual(15); // 1+2+3+4+5
        expect(sum1).toBeLessThanOrEqual(345); // 65+66+67+68+69
      });
    });

    it('validates parity calculations', () => {
      const testCombos = generateTestCombos(50);

      testCombos.forEach(combo => {
        const evens = combo.filter(n => n % 2 === 0).length;
        const odds = combo.filter(n => n % 2 === 1).length;

        expect(evens + odds).toBe(5);
        expect(evens).toBeGreaterThanOrEqual(0);
        expect(evens).toBeLessThanOrEqual(5);
        expect(odds).toBeGreaterThanOrEqual(0);
        expect(odds).toBeLessThanOrEqual(5);
      });
    });

    it('validates statistical calculations', () => {
      const testDraws = generateTestDraws(200);
      const testCombo = [10, 20, 30, 40, 50];

      // Hit rate calculation
      const hits = testDraws.filter(draw =>
        testCombo.every(num => draw.includes(num))
      ).length;

      const hitRate = hits / testDraws.length;

      expect(hitRate).toBeGreaterThanOrEqual(0);
      expect(hitRate).toBeLessThanOrEqual(1);
      expect(typeof hitRate).toBe('number');
      expect(isNaN(hitRate)).toBe(false);
    });
  });

  describe('Data Transformation Tests', () => {
    it('validates combo sorting consistency', () => {
      const unsortedCombos = [
        [30, 10, 50, 20, 40],
        [15, 5, 35, 25, 45],
        [60, 20, 40, 10, 30]
      ];

      unsortedCombos.forEach(combo => {
        const sorted = [...combo].sort((a, b) => a - b);

        // Verify sorting is correct
        for (let i = 1; i < sorted.length; i++) {
          expect(sorted[i]).toBeGreaterThanOrEqual(sorted[i - 1]);
        }

        // Verify all original numbers are present
        expect(sorted.length).toBe(combo.length);
        combo.forEach(num => {
          expect(sorted.includes(num)).toBe(true);
        });
      });
    });

    it('validates data type consistency', () => {
      const testDraws = generateTestDraws(20);
      const testCombos = generateTestCombos(20);

      // All data should be numbers
      testDraws.forEach(draw => {
        draw.forEach(num => {
          expect(typeof num).toBe('number');
          expect(Number.isInteger(num)).toBe(true);
        });
      });

      testCombos.forEach(combo => {
        combo.forEach(num => {
          expect(typeof num).toBe('number');
          expect(Number.isInteger(num)).toBe(true);
        });
      });
    });

    it('validates range boundary conditions', () => {
      const testCombos = generateTestCombos(200);

      let minNumber = 69;
      let maxNumber = 1;

      testCombos.forEach(combo => {
        combo.forEach(num => {
          minNumber = Math.min(minNumber, num);
          maxNumber = Math.max(maxNumber, num);
        });
      });

      // Should use full range of available numbers
      expect(minNumber).toBeLessThanOrEqual(10); // Should include low numbers
      expect(maxNumber).toBeGreaterThanOrEqual(60); // Should include high numbers
    });
  });

  describe('Regression Prevention Tests', () => {
    it('prevents duplicate combo generation', () => {
      const testCombos = generateTestCombos(200);
      const comboStrings = testCombos.map(combo => combo.join('-'));

      // Check for duplicates
      const uniqueCombos = new Set(comboStrings);
      expect(uniqueCombos.size).toBe(testCombos.length);
    });

    it('prevents duplicate draw generation', () => {
      const testDraws = generateTestDraws(200);
      const drawStrings = testDraws.map(draw => draw.join('-'));

      // Check for duplicates (very unlikely but possible)
      const uniqueDraws = new Set(drawStrings);
      expect(uniqueDraws.size).toBe(testDraws.length);
    });

    it('validates consistent data structure over time', () => {
      // Generate data multiple times to ensure consistency
      const batch1 = generateTestCombos(10);
      const batch2 = generateTestCombos(10);
      const batch3 = generateTestCombos(10);

      [batch1, batch2, batch3].forEach(batch => {
        expect(batch).toHaveLength(10);
        batch.forEach(combo => {
          expect(combo).toHaveLength(5);
          expect(new Set(combo).size).toBe(5);
        });
      });
    });
  });
});
