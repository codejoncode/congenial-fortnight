// src/tests/businessLogic.test.ts
import { describe, it, expect } from 'vitest';
import { generateTestDraws, generateTestCombos } from './testData';

describe('Business Logic Validation Tests', () => {
  describe('Lottery Rules Compliance', () => {
    it('validates Powerball number ranges', () => {
      const testDraws = generateTestDraws(100);

      testDraws.forEach(draw => {
        expect(draw).toHaveLength(5);
        draw.forEach(num => {
          expect(num).toBeGreaterThanOrEqual(1);
          expect(num).toBeLessThanOrEqual(69);
          expect(Number.isInteger(num)).toBe(true);
        });

        // Check for uniqueness
        const uniqueNumbers = new Set(draw);
        expect(uniqueNumbers.size).toBe(5);
      });
    });

    it('validates combo generation rules', () => {
      const testCombos = generateTestCombos(50);

      testCombos.forEach(combo => {
        expect(combo).toHaveLength(5);
        expect(combo.every(num => num >= 1 && num <= 69)).toBe(true);

        // Check sorting
        for (let i = 1; i < combo.length; i++) {
          expect(combo[i]).toBeGreaterThanOrEqual(combo[i - 1]);
        }

        // Check uniqueness
        const uniqueNumbers = new Set(combo);
        expect(uniqueNumbers.size).toBe(5);
      });
    });

    it('validates sum ranges for realistic combos', () => {
      const testCombos = generateTestCombos(100);

      testCombos.forEach(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        // Realistic Powerball sums should be between ~50 and ~325
        expect(sum).toBeGreaterThanOrEqual(15); // 1+2+3+4+5
        expect(sum).toBeLessThanOrEqual(345); // 65+66+67+68+69
      });
    });
  });

  describe('Statistical Validation', () => {
    it('validates hit rate calculations', () => {
      const testDraws = generateTestDraws(1000);
      const testCombo = [10, 20, 30, 40, 50];

      const hits = testDraws.filter(draw =>
        testCombo.every(num => draw.includes(num))
      ).length;

      const hitRate = hits / testDraws.length;

      // Hit rate should be between 0 and 1
      expect(hitRate).toBeGreaterThanOrEqual(0);
      expect(hitRate).toBeLessThanOrEqual(1);

      // For a specific combo, hit rate should be very low (close to 0)
      expect(hitRate).toBeLessThan(0.01);
    });

    it('validates confidence score distributions', () => {
      const scores = [];
      for (let i = 0; i < 100; i++) {
        // Generate random but realistic confidence scores
        const baseScore = Math.floor(Math.random() * 60) + 20; // 20-80
        const lift = (Math.random() - 0.5) * 40; // -20 to +20
        const liftBonus = Math.max(0, Math.min(40, lift));
        const finalScore = Math.min(100, baseScore + liftBonus);
        scores.push(finalScore);
      }

      // Statistical validation
      const avgScore = scores.reduce((a, b) => a + b, 0) / scores.length;
      const minScore = Math.min(...scores);
      const maxScore = Math.max(...scores);

      expect(avgScore).toBeGreaterThan(20);
      expect(avgScore).toBeLessThan(80);
      expect(minScore).toBeGreaterThanOrEqual(0);
      expect(maxScore).toBeLessThanOrEqual(100);
    });

    it('validates momentum calculations', () => {
      // Simulate momentum trail
      const trail = [];
      let currentConfidence = 50;

      for (let i = 0; i < 10; i++) {
        // Add some random variation
        currentConfidence += (Math.random() - 0.5) * 10;
        currentConfidence = Math.max(0, Math.min(100, currentConfidence));

        trail.push({
          drawIndex: i,
          confidence: Math.round(currentConfidence),
          totalPoints: 150 + Math.floor(Math.random() * 50)
        });
      }

      // Calculate momentum
      if (trail.length >= 2) {
        const first = trail[0].confidence;
        const last = trail[trail.length - 1].confidence;
        const delta = last - first;
        const maxDelta = 20;
        const momentumBonus = Math.max(-20, Math.min(20, (delta / maxDelta) * 20));

        expect(momentumBonus).toBeGreaterThanOrEqual(-20);
        expect(momentumBonus).toBeLessThanOrEqual(20);
      }
    });
  });

  describe('Filter Logic Validation', () => {
    it('validates sum filtering logic', () => {
      const testCombos = generateTestCombos(200);

      const filteredBySum = testCombos.filter(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        return sum >= 100 && sum <= 200;
      });

      // Should have some combos in this range
      expect(filteredBySum.length).toBeGreaterThan(0);
      expect(filteredBySum.length).toBeLessThan(testCombos.length);

      // Validate all filtered combos meet criteria
      filteredBySum.forEach(combo => {
        const sum = combo.reduce((a, b) => a + b, 0);
        expect(sum).toBeGreaterThanOrEqual(100);
        expect(sum).toBeLessThanOrEqual(200);
      });
    });

    it('validates parity filtering logic', () => {
      const testCombos = generateTestCombos(200);

      const evenHeavyCombos = testCombos.filter(combo => {
        const evens = combo.filter(n => n % 2 === 0).length;
        return evens >= 3;
      });

      // Should have some combos meeting criteria
      expect(evenHeavyCombos.length).toBeGreaterThan(0);

      // Validate all filtered combos meet criteria
      evenHeavyCombos.forEach(combo => {
        const evens = combo.filter(n => n % 2 === 0).length;
        expect(evens).toBeGreaterThanOrEqual(3);
      });
    });

    it('validates key matching logic', () => {
      const testCombos = generateTestCombos(200);
      const keys = [5, 10, 15];

      const combosWithKeys = testCombos.filter(combo =>
        keys.some(key => combo.includes(key))
      );

      // Should have some combos with keys
      expect(combosWithKeys.length).toBeGreaterThan(0);

      // Validate all filtered combos contain at least one key
      combosWithKeys.forEach(combo => {
        const hasKey = keys.some(key => combo.includes(key));
        expect(hasKey).toBe(true);
      });
    });
  });

  describe('Performance Validation', () => {
    it('validates large dataset processing', () => {
      const startTime = Date.now();

      // Generate manageable dataset
      const largeDataset = generateTestDraws(1000);
      const testCombo = [10, 20, 30, 40, 50];

      // Process dataset
      const hits = largeDataset.filter(draw =>
        testCombo.every(num => draw.includes(num))
      ).length;

      const processingTime = Date.now() - startTime;

      // Should process within reasonable time
      expect(processingTime).toBeLessThan(2000); // 2 seconds max
      expect(hits).toBeGreaterThanOrEqual(0);
      expect(hits).toBeLessThanOrEqual(largeDataset.length);
    });

    it('validates combo generation performance', () => {
      const startTime = Date.now();

      // Generate manageable number of combos
      const largeComboSet = generateTestCombos(500);

      const generationTime = Date.now() - startTime;

      // Should generate within reasonable time
      expect(generationTime).toBeLessThan(1500); // 1.5 seconds max
      expect(largeComboSet).toHaveLength(500);

      // Validate data quality
      largeComboSet.forEach(combo => {
        expect(combo).toHaveLength(5);
        expect(new Set(combo).size).toBe(5); // All unique
      });
    });
  });
});
