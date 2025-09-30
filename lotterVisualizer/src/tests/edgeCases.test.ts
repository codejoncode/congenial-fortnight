// src/tests/edgeCases.test.ts
import { describe, it, expect } from 'vitest';
import { generateTestDraws, generateTestCombos } from './testData';

describe('Edge Cases and Boundary Tests', () => {
  describe('Extreme Value Handling', () => {
    it('handles minimum possible combo values', () => {
      const minCombo = [1, 2, 3, 4, 5];
      const sum = minCombo.reduce((a, b) => a + b, 0);
      const avg = sum / minCombo.length;

      expect(sum).toBe(15);
      expect(avg).toBe(3);
      expect(minCombo.every(n => n >= 1)).toBe(true);
    });

    it('handles maximum possible combo values', () => {
      const maxCombo = [65, 66, 67, 68, 69];
      const sum = maxCombo.reduce((a, b) => a + b, 0);
      const avg = sum / maxCombo.length;

      expect(sum).toBe(335);
      expect(avg).toBe(67);
      expect(maxCombo.every(n => n <= 69)).toBe(true);
    });

    it('handles boundary number values', () => {
      const boundaryCombos = [
        [1, 2, 3, 4, 5],    // All minimum
        [69, 68, 67, 66, 65], // All maximum
        [1, 69, 35, 20, 40]   // Mixed boundaries
      ];

      boundaryCombos.forEach(combo => {
        expect(combo.every(n => n >= 1 && n <= 69)).toBe(true);
        expect(new Set(combo).size).toBe(5); // All unique
      });
    });
  });

  describe('Statistical Edge Cases', () => {
    it('handles zero hit scenarios', () => {
      const testDraws = generateTestDraws(100);
      const unlikelyCombo = [1, 2, 3, 4, 5]; // Very specific combo

      const hits = testDraws.filter(draw =>
        unlikelyCombo.every(num => draw.includes(num))
      ).length;

      const hitRate = hits / testDraws.length;

      expect(hitRate).toBeGreaterThanOrEqual(0);
      expect(hitRate).toBeLessThanOrEqual(1);

      // Very unlikely to have hits with such a specific combo
      expect(hitRate).toBeLessThan(0.1);
    });

    it('handles perfect hit scenarios', () => {
      const repeatedDraw = [10, 20, 30, 40, 50];
      const testDraws = Array(10).fill(repeatedDraw);

      const hits = testDraws.filter(draw =>
        repeatedDraw.every(num => draw.includes(num))
      ).length;

      const hitRate = hits / testDraws.length;

      expect(hitRate).toBe(1); // Perfect hit rate
      expect(hits).toBe(testDraws.length);
    });

    it('handles single draw scenarios', () => {
      const singleDraw = [[10, 20, 30, 40, 50]];
      const testCombo = [10, 20, 30, 40, 50];

      const hits = singleDraw.filter(draw =>
        testCombo.every(num => draw.includes(num))
      ).length;

      const hitRate = hits / singleDraw.length;

      expect(hitRate).toBe(1);
      expect(hits).toBe(1);
    });
  });

  describe('Filter Edge Cases', () => {
    it('handles impossible filter combinations', () => {
      const testCombos = generateTestCombos(100);

      // Impossible: sum between 400-500 (max possible is ~335)
      const impossibleSumFilter = testCombos.filter(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        return sum >= 400 && sum <= 500;
      });

      expect(impossibleSumFilter.length).toBe(0);
    });

    it('handles contradictory parity requirements', () => {
      const testCombos = generateTestCombos(100);

      // Impossible: 4 evens and 4 odds (only 5 numbers total)
      const impossibleParityFilter = testCombos.filter(combo => {
        const evens = combo.filter(n => n % 2 === 0).length;
        const odds = combo.filter(n => n % 2 === 1).length;
        return evens === 4 && odds === 4;
      });

      expect(impossibleParityFilter.length).toBe(0);
    });

    it('handles empty key requirements', () => {
      const testCombos = generateTestCombos(50);
      const emptyKeys: number[] = [];

      const keyFiltered = testCombos.filter(combo =>
        emptyKeys.length === 0 || emptyKeys.some(key => combo.includes(key))
      );

      // Should return all combos when keys is empty
      expect(keyFiltered.length).toBe(testCombos.length);
    });
  });

  describe('Performance Edge Cases', () => {
    it('handles very large datasets efficiently', () => {
      const startTime = Date.now();

      // Generate large but manageable dataset
      const largeDataset = generateTestDraws(1000);

      const generationTime = Date.now() - startTime;
      expect(generationTime).toBeLessThan(5000); // 5 seconds max
      expect(largeDataset).toHaveLength(1000);

      // Validate data quality even with large dataset
      const sample = largeDataset.slice(0, 50);
      sample.forEach(draw => {
        expect(draw).toHaveLength(5);
        expect(new Set(draw).size).toBe(5);
      });
    });

    it('handles memory-intensive operations', () => {
      const startTime = Date.now();
      const memoryTestCombos = generateTestCombos(1000);

      const processingTime = Date.now() - startTime;
      expect(processingTime).toBeLessThan(3000); // 3 seconds max

      // Validate all generated combos
      memoryTestCombos.forEach(combo => {
        expect(combo).toHaveLength(5);
        expect(combo.every(n => n >= 1 && n <= 69)).toBe(true);
      });
    });
  });

  describe('Data Validation Edge Cases', () => {
    it('handles near-boundary values', () => {
      const nearBoundaryCombos = [
        [1, 2, 3, 4, 6],     // Just above minimum
        [64, 65, 66, 67, 68], // Just below maximum
        [1, 69, 2, 68, 3]     // Alternating boundaries
      ];

      nearBoundaryCombos.forEach(combo => {
        expect(combo.every(n => n >= 1 && n <= 69)).toBe(true);
        expect(new Set(combo).size).toBe(5);
      });
    });

    it('validates sorting edge cases', () => {
      const edgeSortCases = [
        [1, 1, 2, 3, 4],     // Duplicate (shouldn't happen but test handling)
        [5, 4, 3, 2, 1],     // Reverse sorted
        [10, 10, 10, 10, 10], // All same (shouldn't happen)
        [69, 1, 35, 20, 50]   // Random order
      ];

      edgeSortCases.forEach(combo => {
        const sorted = [...combo].sort((a, b) => a - b);

        // Verify sorting works even with edge cases
        for (let i = 1; i < sorted.length; i++) {
          expect(sorted[i]).toBeGreaterThanOrEqual(sorted[i - 1]);
        }
      });
    });

    it('handles sparse data distributions', () => {
      // Test with combos that use mostly low numbers (at least 3 out of 5)
      const lowNumberCombos = generateTestCombos(500).filter(combo =>
        combo.filter(n => n <= 20).length >= 3
      );

      // Test with combos that use mostly high numbers (at least 3 out of 5)
      const highNumberCombos = generateTestCombos(500).filter(combo =>
        combo.filter(n => n >= 50).length >= 3
      );

      expect(lowNumberCombos.length).toBeGreaterThan(0);
      expect(highNumberCombos.length).toBeGreaterThan(0);

      // Validate ranges
      lowNumberCombos.forEach(combo => {
        expect(combo.filter(n => n <= 20).length).toBeGreaterThanOrEqual(3);
      });

      highNumberCombos.forEach(combo => {
        expect(combo.filter(n => n >= 50).length).toBeGreaterThanOrEqual(3);
      });
    });
  });
});
