// src/tests/sixthBall.test.ts
import { describe, it, expect } from 'vitest';
import {
  getLastDigit,
  getFirstDigit,
  getDigitSum,
  getDigitDivision,
  analyze6thBallDigits,
  calculate6thBallSkipStats,
  predict6thBallCandidates,
  get6thBallHotCold,
  analyze6thBallDigitGroups
} from '../utils/math';

describe('6th Ball Digit Analysis', () => {
  describe('Basic Digit Functions', () => {
    it('should extract last digit correctly', () => {
      expect(getLastDigit(5)).toBe(5);
      expect(getLastDigit(15)).toBe(5);
      expect(getLastDigit(26)).toBe(6);
      expect(getLastDigit(1)).toBe(1);
    });

    it('should extract first digit correctly', () => {
      expect(getFirstDigit(5)).toBe(0);
      expect(getFirstDigit(15)).toBe(1);
      expect(getFirstDigit(26)).toBe(2);
      expect(getFirstDigit(1)).toBe(0);
    });

    it('should calculate digit sum correctly', () => {
      expect(getDigitSum(5)).toBe(5);
      expect(getDigitSum(15)).toBe(1 + 5);
      expect(getDigitSum(26)).toBe(2 + 6);
      expect(getDigitSum(123)).toBe(1 + 2 + 3);
    });

    it('should calculate digit division correctly', () => {
      expect(getDigitDivision(15)).toEqual({ quotient: 1, remainder: 5 });
      expect(getDigitDivision(26)).toEqual({ quotient: 2, remainder: 6 });
      expect(getDigitDivision(5)).toEqual({ quotient: 0, remainder: 5 });
    });
  });

  describe('6th Ball Analysis', () => {
    it('should analyze 6th ball digits comprehensively', () => {
      const analysis = analyze6thBallDigits(15);
      expect(analysis).toEqual({
        powerball: 15,
        lastDigit: 5,
        firstDigit: 1,
        digitSum: 6,
        quotient: 1,
        remainder: 5,
        isEvenSum: true,
        isEvenQuotient: false,
        isEvenRemainder: false,
        digitPattern: '15',
        sumParity: 'even'
      });
    });

    it('should handle single digit numbers', () => {
      const analysis = analyze6thBallDigits(5);
      expect(analysis).toEqual({
        powerball: 5,
        lastDigit: 5,
        firstDigit: 0,
        digitSum: 5,
        quotient: 0,
        remainder: 5,
        isEvenSum: false,
        isEvenQuotient: true,
        isEvenRemainder: false,
        digitPattern: '05',
        sumParity: 'odd'
      });
    });
  });

  describe('Skip Statistics', () => {
    it('should calculate skip statistics correctly', () => {
      const powerballs = [5, 12, 5, 18, 5, 22];
      const stats = calculate6thBallSkipStats(powerballs, 5);

      expect(stats.currentSkip).toBe(1); // Last 5 was at index 4, current is 5
      expect(stats.averageSkip).toBe(1); // Skips: 1 (after first), 1 (after second)
      expect(stats.maxSkip).toBe(1);
      expect(stats.skipDistribution).toEqual([1, 1]);
    });

    it('should handle number that never appeared', () => {
      const powerballs = [1, 2, 3, 4, 5];
      const stats = calculate6thBallSkipStats(powerballs, 10);

      expect(stats.currentSkip).toBe(5);
      expect(stats.averageSkip).toBe(0);
      expect(stats.maxSkip).toBe(0);
      expect(stats.skipDistribution).toEqual([]);
    });
  });

  describe('Prediction Algorithm', () => {
    it('should predict 6th ball candidates based on historical data', () => {
      const historicalPowerballs = [5, 12, 8, 15, 3, 22, 7, 18, 5, 11];
      const recentDraws = [5, 12, 8];

      const predictions = predict6thBallCandidates(historicalPowerballs, recentDraws, 3);

      expect(predictions).toHaveLength(3);
      expect(predictions[0].score).toBeGreaterThanOrEqual(predictions[1].score);
      expect(predictions[0].number).toBeGreaterThan(0);
      expect(predictions[0].number).toBeLessThanOrEqual(26);
      expect(Array.isArray(predictions[0].reasons)).toBe(true);
    });

    it('should handle empty historical data', () => {
      const predictions = predict6thBallCandidates([], [], 2);

      expect(predictions).toHaveLength(2);
      predictions.forEach(pred => {
        expect(pred.score).toBe(0); // No historical data means no scoring
        expect(pred.reasons).toHaveLength(0);
      });
    });

    it('should prioritize numbers due for appearance', () => {
      const historicalPowerballs = [1, 1, 1, 1, 1]; // 1 appears frequently
      const recentDraws = [1, 1, 1];

      const predictions = predict6thBallCandidates(historicalPowerballs, recentDraws, 5);

      // Numbers other than 1 should have higher scores due to being "due"
      const nonOnePredictions = predictions.filter(p => p.number !== 1);
      expect(nonOnePredictions.length).toBeGreaterThan(0);
      expect(nonOnePredictions[0].score).toBeGreaterThan(0);
    });
  });

  describe('Integration with Lottery Data', () => {
    it('should work with realistic Powerball ranges', () => {
      // Test with typical Powerball numbers
      const powerballNumbers = [1, 5, 12, 18, 22, 26];

      powerballNumbers.forEach(num => {
        const analysis = analyze6thBallDigits(num);
        expect(analysis.powerball).toBe(num);
        expect(analysis.lastDigit).toBeGreaterThanOrEqual(0);
        expect(analysis.lastDigit).toBeLessThanOrEqual(9);
        expect(analysis.digitSum).toBeGreaterThan(0);
        expect(analysis.digitSum).toBeLessThanOrEqual(18); // Max for 26 is 2+6=8
      });
    });

    it('should handle edge cases in digit analysis', () => {
      // Test boundary values
      expect(analyze6thBallDigits(1).digitPattern).toBe('01');
      expect(analyze6thBallDigits(26).digitPattern).toBe('26');
      expect(analyze6thBallDigits(10).digitSum).toBe(1); // 1+0=1
    });
  });

  describe('Hot & Cold Analysis', () => {
    it('should identify hot and cold numbers correctly', () => {
      const powerballs = [5, 5, 5, 12, 12, 18, 18, 18, 18, 22];
      const { hot, cold } = get6thBallHotCold(powerballs, 3);

      expect(hot).toHaveLength(3);
      expect(hot[0].number).toBe(18);
      expect(hot[0].frequency).toBe(4);
      expect(hot[1].number).toBe(5);
      expect(hot[1].frequency).toBe(3);

      expect(cold).toHaveLength(3);
      expect(cold[0].frequency).toBe(0); // Numbers not appeared
    });
  });

  describe('Digit Group Analysis', () => {
    it('should analyze digit group skips', () => {
      const powerballs = [5, 2, 15, 12, 25]; // Last digits: 5,2,5,2,5
      const groups = analyze6thBallDigitGroups(powerballs);

      expect(groups.lastDigitSkips[5].currentSkip).toBe(0); // Last appearance was recent
      expect(groups.lastDigitSkips[5].averageSkip).toBeGreaterThan(0);
    });
  });
});
