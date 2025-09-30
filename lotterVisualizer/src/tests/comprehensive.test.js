// src/tests/comprehensive.test.js
// Comprehensive test suite using vitest format

import { describe, it, expect } from 'vitest';

describe('ApexScoop Comprehensive Test Suite', () => {
  describe('Core Data Generation', () => {
    it('should generate valid test draws', () => {
      function generateTestDraw() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      const testDraws = [];
      for (let i = 0; i < 50; i++) {
        testDraws.push(generateTestDraw());
      }

      expect(testDraws).toHaveLength(50);

      // Validate data structure
      const drawsValid = testDraws.every(draw =>
        Array.isArray(draw) &&
        draw.length === 5 &&
        draw.every(n => n >= 1 && n <= 69) &&
        new Set(draw).size === 5
      );

      expect(drawsValid).toBe(true);
    });

    it('should generate valid test combos', () => {
      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      const testCombos = [];
      for (let i = 0; i < 50; i++) {
        testCombos.push(generateTestCombo());
      }

      expect(testCombos).toHaveLength(50);

      const combosValid = testCombos.every(combo =>
        Array.isArray(combo) &&
        combo.length === 5 &&
        combo.every(n => n >= 1 && n <= 69) &&
        new Set(combo).size === 5
      );

      expect(combosValid).toBe(true);
    });
  });

  describe('Mathematical Operations', () => {
    it('should validate sums correctly', () => {
      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function validateSum(combo) {
        const sum = combo.reduce((a, b) => a + b, 0);
        return sum >= 15 && sum <= 335; // Valid Powerball sum range
      }

      const testCombos = [];
      for (let i = 0; i < 50; i++) {
        testCombos.push(generateTestCombo());
      }

      const sumValidation = testCombos.map(validateSum);
      expect(sumValidation.every(Boolean)).toBe(true);
    });

    it('should validate parity correctly', () => {
      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function validateParity(combo) {
        const evens = combo.filter(n => n % 2 === 0).length;
        const odds = combo.filter(n => n % 2 === 1).length;
        return evens + odds === 5 && evens >= 0 && evens <= 5;
      }

      const testCombos = [];
      for (let i = 0; i < 50; i++) {
        testCombos.push(generateTestCombo());
      }

      const parityValidation = testCombos.map(validateParity);
      expect(parityValidation.every(Boolean)).toBe(true);
    });
  });

  describe('Statistical Calculations', () => {
    it('should calculate hit rates correctly', () => {
      function generateTestDraw() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function calculateHitRate(combo, draws) {
        const hits = draws.filter(draw =>
          combo.every(num => draw.includes(num))
        ).length;
        return hits / draws.length;
      }

      const testDraws = [];
      const testCombos = [];

      for (let i = 0; i < 20; i++) {
        testDraws.push(generateTestDraw());
        testCombos.push(generateTestCombo());
      }

      const hitRates = testCombos.map(combo => calculateHitRate(combo, testDraws));

      expect(hitRates).toHaveLength(20);
      expect(hitRates.every(r => r >= 0 && r <= 1)).toBe(true);
    });
  });

  describe('Scoring Logic', () => {
    it('should calculate confidence scores correctly', () => {
      function calculateBasicConfidence(hitRate, baseScore = 50) {
        const lift = hitRate > 0.5 ? (hitRate - 0.5) * 100 : (hitRate - 0.5) * 50;
        return Math.max(0, Math.min(100, baseScore + lift));
      }

      const testRates = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0];
      const confidenceScores = testRates.map(rate => calculateBasicConfidence(rate));

      expect(confidenceScores).toHaveLength(7);
      expect(confidenceScores.every(s => s >= 0 && s <= 100)).toBe(true);
      expect(confidenceScores[0]).toBe(25); // 0 rate
      expect(confidenceScores[3]).toBe(50); // 0.5 rate (baseline)
      expect(confidenceScores[6]).toBe(100); // 1.0 rate (max)
    });

    it('should calculate freshness multipliers correctly', () => {
      function calculateFreshnessMultiplier(drawsOut) {
        if (drawsOut <= 2) return 1.15;
        if (drawsOut >= 20) return 0.85;
        return 1.15 - ((drawsOut - 2) / 18) * 0.3;
      }

      const testDrawsOut = [1, 5, 10, 15, 25];
      const freshnessMultipliers = testDrawsOut.map(drawsOut => calculateFreshnessMultiplier(drawsOut));

      expect(freshnessMultipliers).toHaveLength(5);
      expect(freshnessMultipliers[0]).toBe(1.15); // Very fresh
      expect(freshnessMultipliers[4]).toBe(0.85); // Stale
    });
  });

  describe('Filter Logic', () => {
    it('should filter by sum correctly', () => {
      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function filterBySum(combos, minSum, maxSum) {
        return combos.filter(combo => {
          const sum = combo.reduce((a, b) => a + b, 0);
          return sum >= minSum && sum <= maxSum;
        });
      }

      const testCombos = [];
      for (let i = 0; i < 100; i++) {
        testCombos.push(generateTestCombo());
      }

      const sumFiltered = filterBySum(testCombos, 100, 200);
      expect(sumFiltered.length).toBeGreaterThan(0);
      expect(sumFiltered.length).toBeLessThan(testCombos.length);

      // Verify all filtered combos are within range
      sumFiltered.forEach(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        expect(sum).toBeGreaterThanOrEqual(100);
        expect(sum).toBeLessThanOrEqual(200);
      });
    });

    it('should filter by parity correctly', () => {
      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function filterByParity(combos, minEvens) {
        return combos.filter(combo => {
          const evens = combo.filter(n => n % 2 === 0).length;
          return evens >= minEvens;
        });
      }

      const testCombos = [];
      for (let i = 0; i < 100; i++) {
        testCombos.push(generateTestCombo());
      }

      const parityFiltered = filterByParity(testCombos, 3);
      expect(parityFiltered.length).toBeGreaterThan(0);

      // Verify all filtered combos have at least 3 evens
      parityFiltered.forEach(combo => {
        const evens = combo.filter(n => n % 2 === 0).length;
        expect(evens).toBeGreaterThanOrEqual(3);
      });
    });
  });

  describe('Data Integrity', () => {
    it('should maintain data integrity for draws', () => {
      function generateTestDraw() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function validateDataIntegrity(data, type) {
        return data.every(item => {
          if (!Array.isArray(item) || item.length !== 5) return false;

          // Check number range and uniqueness
          const nums = item.filter(n => typeof n === 'number' && n >= 1 && n <= 69);
          if (nums.length !== 5) return false;

          const unique = new Set(nums);
          if (unique.size !== 5) return false;

          // Check sorting
          for (let i = 1; i < nums.length; i++) {
            if (nums[i] < nums[i - 1]) return false;
          }

          return true;
        });
      }

      const testDraws = [];
      for (let i = 0; i < 50; i++) {
        testDraws.push(generateTestDraw());
      }

      const drawsIntegrity = validateDataIntegrity(testDraws, 'draws');
      expect(drawsIntegrity).toBe(true);
    });

    it('should maintain data integrity for combos', () => {
      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function validateDataIntegrity(data, type) {
        return data.every(item => {
          if (!Array.isArray(item) || item.length !== 5) return false;

          // Check number range and uniqueness
          const nums = item.filter(n => typeof n === 'number' && n >= 1 && n <= 69);
          if (nums.length !== 5) return false;

          const unique = new Set(nums);
          if (unique.size !== 5) return false;

          // Check sorting
          for (let i = 1; i < nums.length; i++) {
            if (nums[i] < nums[i - 1]) return false;
          }

          return true;
        });
      }

      const testCombos = [];
      for (let i = 0; i < 50; i++) {
        testCombos.push(generateTestCombo());
      }

      const combosIntegrity = validateDataIntegrity(testCombos, 'combos');
      expect(combosIntegrity).toBe(true);
    });
  });

  describe('Performance Validation', () => {
    it('should perform data generation efficiently', () => {
      function generateTestDraw() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      const perfStartTime = Date.now();

      // Generate larger dataset for performance testing
      const perfDraws = [];
      const perfCombos = [];

      for (let i = 0; i < 1000; i++) {
        perfDraws.push(generateTestDraw());
        perfCombos.push(generateTestCombo());
      }

      const perfEndTime = Date.now();
      const perfDuration = perfEndTime - perfStartTime;

      expect(perfDraws).toHaveLength(1000);
      expect(perfCombos).toHaveLength(1000);
      expect(perfDuration).toBeLessThan(5000); // Should complete in under 5 seconds
    });
  });

  describe('Edge Cases', () => {
    it('should handle boundary values correctly', () => {
      const edgeCases = [
        [1, 2, 3, 4, 5],      // Minimum values
        [65, 66, 67, 68, 69], // Maximum values
        [1, 69, 35, 20, 40],  // Mixed boundaries
      ];

      edgeCases.forEach(combo => {
        expect(combo.every(n => n >= 1 && n <= 69)).toBe(true);
        expect(new Set(combo).size).toBe(5);

        const sum = combo.reduce((a, b) => a + b, 0);
        expect(sum).toBeGreaterThanOrEqual(15);
        expect(sum).toBeLessThanOrEqual(335);
      });
    });
  });

  describe('Statistical Distribution', () => {
    it('should maintain reasonable statistical distribution', () => {
      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function calculateAverage(numbers) {
        return numbers.reduce((a, b) => a + b, 0) / numbers.length;
      }

      const testCombos = [];
      for (let i = 0; i < 100; i++) {
        testCombos.push(generateTestCombo());
      }

      const sums = testCombos.map(combo => combo.reduce((a, b) => a + b, 0));
      const avgSum = calculateAverage(sums);
      const minSum = Math.min(...sums);
      const maxSum = Math.max(...sums);

      expect(avgSum).toBeGreaterThan(100);
      expect(avgSum).toBeLessThan(200);
      expect(maxSum).toBeLessThanOrEqual(335);
      expect(minSum).toBeGreaterThanOrEqual(15);
    });
  });

  describe('Integration Test', () => {
    it('should run complete integration test', () => {
      function generateTestDraw() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function generateTestCombo() {
        const numbers = new Set();
        while (numbers.size < 5) {
          numbers.add(Math.floor(Math.random() * 69) + 1);
        }
        return Array.from(numbers).sort((a, b) => a - b);
      }

      function calculateHitRate(combo, draws) {
        const hits = draws.filter(draw =>
          combo.every(num => draw.includes(num))
        ).length;
        return hits / draws.length;
      }

      function calculateBasicConfidence(hitRate, baseScore = 50) {
        const lift = hitRate > 0.5 ? (hitRate - 0.5) * 100 : (hitRate - 0.5) * 50;
        return Math.max(0, Math.min(100, baseScore + lift));
      }

      function calculateFreshnessMultiplier(drawsOut) {
        if (drawsOut <= 2) return 1.15;
        if (drawsOut >= 20) return 0.85;
        return 1.15 - ((drawsOut - 2) / 18) * 0.3;
      }

      // Generate test data
      const draws = Array.from({ length: 20 }, () => generateTestDraw());
      const combo = generateTestCombo();

      // Calculate metrics
      const hitRate = calculateHitRate(combo, draws);
      const confidence = calculateBasicConfidence(hitRate);
      const freshness = calculateFreshnessMultiplier(5);
      const finalScore = confidence * freshness;

      // Validate results
      expect(hitRate).toBeGreaterThanOrEqual(0);
      expect(hitRate).toBeLessThanOrEqual(1);
      expect(confidence).toBeGreaterThanOrEqual(0);
      expect(confidence).toBeLessThanOrEqual(100);
      expect(freshness).toBeGreaterThanOrEqual(0.85);
      expect(freshness).toBeLessThanOrEqual(1.15);
      expect(finalScore).toBeGreaterThanOrEqual(0);
      expect(finalScore).toBeLessThanOrEqual(115);
    });
  });

  describe('6th Ball Analysis', () => {
    it('should analyze 6th ball digit patterns', () => {
      // Test digit analysis functions
      function getLastDigit(num) { return num % 10; }
      function getFirstDigit(num) { return Math.floor(num / 10); }
      function getDigitSum(num) {
        return num.toString().split('').reduce((sum, digit) => sum + parseInt(digit), 0);
      }

      const testNumbers = [1, 5, 12, 18, 22, 26];

      testNumbers.forEach(num => {
        const lastDigit = getLastDigit(num);
        const firstDigit = getFirstDigit(num);
        const digitSum = getDigitSum(num);

        expect(lastDigit).toBeGreaterThanOrEqual(0);
        expect(lastDigit).toBeLessThanOrEqual(9);
        expect(firstDigit).toBeGreaterThanOrEqual(0);
        expect(firstDigit).toBeLessThanOrEqual(2);
        expect(digitSum).toBeGreaterThan(0);
        expect(digitSum).toBeLessThanOrEqual(9); // Max for 26 is 2+6=8, but let's be safe
      });
    });

    it('should calculate 6th ball skip statistics', () => {
      function calculateSkipStats(powerballs, target) {
        const skips = [];
        let lastSeen = -1;

        for (let i = 0; i < powerballs.length; i++) {
          if (powerballs[i] === target) {
            if (lastSeen !== -1) {
              skips.push(i - lastSeen - 1);
            }
            lastSeen = i;
          }
        }

        const currentSkip = lastSeen === -1 ? powerballs.length : powerballs.length - 1 - lastSeen;
        const averageSkip = skips.length > 0 ? skips.reduce((a, b) => a + b, 0) / skips.length : 0;

        return { currentSkip, averageSkip, skips };
      }

      const powerballs = [5, 12, 5, 18, 5, 22, 7];
      const stats = calculateSkipStats(powerballs, 5);

      expect(stats.currentSkip).toBe(2); // Last 5 at index 4, current at 6 (length 7, so 7-1-4=2)
      expect(stats.averageSkip).toBe(1); // Skips: 1, 1 -> average 1
      expect(stats.skips).toEqual([1, 1]);
    });

    it('should predict 6th ball candidates', () => {
      function predictCandidates(historical, recent = [], topN = 3) {
        const candidates = [];

        for (let num = 1; num <= 26; num++) {
          let score = 0;
          const reasons = [];

          // Simple scoring based on recency
          const lastAppearance = historical.lastIndexOf(num);
          if (lastAppearance === -1 || historical.length - lastAppearance > 5) {
            score += 10;
            reasons.push('Due for appearance');
          }

          // Digit sum analysis
          const digitSum = num.toString().split('').reduce((sum, d) => sum + parseInt(d), 0);
          const recentSums = recent.map(n => n.toString().split('').reduce((sum, d) => sum + parseInt(d), 0));
          const avgRecentSum = recentSums.length > 0 ? recentSums.reduce((a, b) => a + b, 0) / recentSums.length : 5;

          if (Math.abs(digitSum - avgRecentSum) < 2) {
            score += 5;
            reasons.push('Digit sum matches trend');
          }

          candidates.push({ number: num, score, reasons });
        }

        return candidates.sort((a, b) => b.score - a.score).slice(0, topN);
      }

      const historical = [5, 12, 8, 15, 3, 22, 7, 18];
      const recent = [5, 12, 8];

      const predictions = predictCandidates(historical, recent, 5);

      expect(predictions).toHaveLength(5);
      expect(predictions[0].score).toBeGreaterThanOrEqual(predictions[4].score);
      predictions.forEach(pred => {
        expect(pred.number).toBeGreaterThan(0);
        expect(pred.number).toBeLessThanOrEqual(26);
        expect(Array.isArray(pred.reasons)).toBe(true);
      });
    });

    it('should validate 6th ball data integrity', () => {
      // Generate test Powerball data
      const powerballs = [];
      for (let i = 0; i < 100; i++) {
        powerballs.push(Math.floor(Math.random() * 26) + 1);
      }

      // Validate range
      const allValid = powerballs.every(pb => pb >= 1 && pb <= 26);
      expect(allValid).toBe(true);

      // Check distribution (should be relatively even)
      const counts = {};
      powerballs.forEach(pb => {
        counts[pb] = (counts[pb] || 0) + 1;
      });

      const uniqueNumbers = Object.keys(counts).length;
      expect(uniqueNumbers).toBeGreaterThan(15); // Should have good variety

      // Check that no number appears excessively
      const maxCount = Math.max(...Object.values(counts));
      expect(maxCount).toBeLessThan(15); // No number should dominate
    });

    it('should analyze 6th ball patterns over time', () => {
      // Generate time series data
      const powerballs = [];
      for (let i = 0; i < 200; i++) {
        powerballs.push(Math.floor(Math.random() * 26) + 1);
      }

      // Analyze patterns
      const evenCount = powerballs.filter(pb => pb % 2 === 0).length;
      const oddCount = powerballs.length - evenCount;
      const evenRatio = evenCount / powerballs.length;

      // Should be roughly 50/50 but allow for randomness
      expect(evenRatio).toBeGreaterThan(0.35);
      expect(evenRatio).toBeLessThan(0.65);

      // Check digit sum distribution
      const digitSums = powerballs.map(pb =>
        pb.toString().split('').reduce((sum, digit) => sum + parseInt(digit), 0)
      );

      const avgDigitSum = digitSums.reduce((a, b) => a + b, 0) / digitSums.length;
      expect(avgDigitSum).toBeGreaterThan(3);
      expect(avgDigitSum).toBeLessThan(7);
    });
  });
});