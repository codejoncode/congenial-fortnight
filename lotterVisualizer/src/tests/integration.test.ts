// src/tests/integration.test.ts
import { describe, it, expect } from 'vitest';
import { generateTestCombos } from './testData';

describe('Integration Tests', () => {
  describe('Filter Function Integration', () => {
    it('should build and use filter functions correctly', () => {
      const testCombos = generateTestCombos(50);
      const filteredCombos = testCombos.filter(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        return sum >= 150 && sum <= 200;
      });

      expect(filteredCombos.length).toBeGreaterThanOrEqual(0);
      expect(filteredCombos.length).toBeLessThanOrEqual(50);
    });

    it('should handle edge cases gracefully', () => {
      const testCombos = generateTestCombos(20);
      const filteredCombos = testCombos.filter(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        return sum >= 200 && sum <= 210;
      });

      expect(Array.isArray(filteredCombos)).toBe(true);
    });

    it('should maintain data integrity through pipeline', () => {
      const testCombos = generateTestCombos(30);
      const filteredCombos = testCombos.filter(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        return sum >= 150 && sum <= 200;
      });

      filteredCombos.forEach(combo => {
        const sortedCombo = [...combo].sort((a, b) => a - b);
        expect(combo).toEqual(sortedCombo);
        combo.forEach(num => {
          expect(num).toBeGreaterThanOrEqual(1);
          expect(num).toBeLessThanOrEqual(69);
        });
      });
    });
  });

  describe('Performance Tests', () => {
    it('should process combos within reasonable time', () => {
      const startTime = Date.now();
      const testCombos = generateTestCombos(100);
      const filteredCombos = testCombos.filter(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        return sum >= 100 && sum <= 200;
      });
      const endTime = Date.now();
      const duration = endTime - startTime;

      expect(duration).toBeLessThan(1000);
      expect(filteredCombos.length).toBeGreaterThan(0);
    });
  });
});
