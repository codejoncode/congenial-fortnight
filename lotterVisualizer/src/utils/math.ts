// src/utils/math.ts
export const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

export const sumConstraint = (draw: number[], min: number, max: number): boolean => {
  const sum = draw.reduce((a, b) => a + b, 0);
  return sum >= min && sum <= max;
};

export const parityConstraint = (draw: number[], evenCount?: number, oddCount?: number): boolean => {
  const evens = draw.filter(n => n % 2 === 0).length;
  const odds = draw.length - evens;
  return (evenCount === undefined || evens === evenCount) && (oddCount === undefined || odds === oddCount);
};

export const generatePoolForDraw = (history: number[][], idx: number, filterFn: (draw: number[]) => boolean): number[][] => {
  // For simplicity, return all previous draws that match the filter
  return history.slice(0, idx).filter(filterFn);
};

// 6th Ball Digit Analysis Functions
export const getLastDigit = (num: number): number => num % 10;

export const getFirstDigit = (num: number): number => Math.floor(num / 10);

export const getDigitSum = (num: number): number => {
  return num.toString().split('').reduce((sum, digit) => sum + parseInt(digit), 0);
};

export const getDigitDivision = (num: number): { quotient: number; remainder: number } => {
  const firstDigit = getFirstDigit(num);
  const lastDigit = getLastDigit(num);
  return {
    quotient: firstDigit,
    remainder: lastDigit
  };
};

export const analyze6thBallDigits = (powerball: number) => {
  const lastDigit = getLastDigit(powerball);
  const firstDigit = getFirstDigit(powerball);
  const digitSum = getDigitSum(powerball);
  const { quotient, remainder } = getDigitDivision(powerball);
  const isEvenSum = digitSum % 2 === 0;
  const isEvenQuotient = quotient % 2 === 0;
  const isEvenRemainder = remainder % 2 === 0;

  return {
    powerball,
    lastDigit,
    firstDigit,
    digitSum,
    quotient,
    remainder,
    isEvenSum,
    isEvenQuotient,
    isEvenRemainder,
    digitPattern: `${firstDigit}${lastDigit}`,
    sumParity: isEvenSum ? 'even' : 'odd'
  };
};

export const calculate6thBallSkipStats = (powerballs: number[], target: number): {
  currentSkip: number;
  averageSkip: number;
  maxSkip: number;
  skipDistribution: number[];
} => {
  const skips: number[] = [];
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
  const maxSkip = skips.length > 0 ? Math.max(...skips) : 0;

  return {
    currentSkip,
    averageSkip,
    maxSkip,
    skipDistribution: skips
  };
};

export const predict6thBallCandidates = (
  historicalPowerballs: number[],
  recentDraws: number[] = [],
  topCandidates: number = 5
): Array<{ number: number; score: number; reasons: string[] }> => {
  if (historicalPowerballs.length === 0) {
    return Array.from({ length: topCandidates }, (_, i) => ({
      number: i + 1,
      score: 0,
      reasons: []
    }));
  }

  const candidates: Array<{ number: number; score: number; reasons: string[] }> = [];

  // Analyze each possible Powerball number (1-26 for Powerball)
  for (let num = 1; num <= 26; num++) {
    let score = 0;
    const reasons: string[] = [];

    // Skip analysis
    const skipStats = calculate6thBallSkipStats(historicalPowerballs, num);
    if (skipStats.currentSkip > skipStats.averageSkip * 1.5) {
      score += 20;
      reasons.push(`Due for appearance (skip: ${skipStats.currentSkip})`);
    }

    // Digit pattern analysis
    const digitAnalysis = analyze6thBallDigits(num);
    const recentDigits = recentDraws.slice(-10).map(pb => analyze6thBallDigits(pb));

    // Check if digit sum is trending
    if (recentDigits.length > 0) {
      const avgRecentSum = recentDigits.reduce((sum, d) => sum + d.digitSum, 0) / recentDigits.length;
      if (Math.abs(digitAnalysis.digitSum - avgRecentSum) < 2) {
        score += 15;
        reasons.push(`Digit sum matches recent trend (${digitAnalysis.digitSum})`);
      }
    }

    // Check last digit frequency
    if (recentDigits.length > 0) {
      const lastDigitFreq = recentDigits.filter(d => d.lastDigit === digitAnalysis.lastDigit).length;
      if (lastDigitFreq < 3) {
        score += 10;
        reasons.push(`Last digit ${digitAnalysis.lastDigit} is due`);
      }
    }

    // Even/odd balance
    if (recentDigits.length > 0) {
      const evenCount = recentDigits.filter(d => d.isEvenSum).length;
      const oddCount = recentDigits.length - evenCount;
      if ((digitAnalysis.isEvenSum && evenCount < oddCount) ||
          (!digitAnalysis.isEvenSum && oddCount < evenCount)) {
        score += 12;
        reasons.push(`Balances ${digitAnalysis.sumParity} sum parity`);
      }
    }

    candidates.push({ number: num, score, reasons });
  }

  // Sort by score and return top candidates
  return candidates
    .sort((a, b) => b.score - a.score)
    .slice(0, topCandidates);
};

export const get6thBallHotCold = (powerballs: number[], topN: number = 5): {
  hot: Array<{ number: number; frequency: number }>;
  cold: Array<{ number: number; frequency: number }>;
} => {
  const freq: Record<number, number> = {};
  powerballs.forEach(pb => {
    freq[pb] = (freq[pb] || 0) + 1;
  });

  const allNumbers = Array.from({ length: 26 }, (_, i) => i + 1);
  const withFreq = allNumbers.map(num => ({ number: num, frequency: freq[num] || 0 }));

  const hot = withFreq.sort((a, b) => b.frequency - a.frequency).slice(0, topN);
  const cold = withFreq.sort((a, b) => a.frequency - b.frequency).slice(0, topN);

  return { hot, cold };
};

export const analyze6thBallDigitGroups = (powerballs: number[]): {
  lastDigitSkips: Record<number, { currentSkip: number; averageSkip: number }>;
  firstDigitSkips: Record<number, { currentSkip: number; averageSkip: number }>;
  sumSkips: Record<number, { currentSkip: number; averageSkip: number }>;
} => {
  const lastDigitSkips: Record<number, { currentSkip: number; averageSkip: number }> = {};
  const firstDigitSkips: Record<number, { currentSkip: number; averageSkip: number }> = {};
  const sumSkips: Record<number, { currentSkip: number; averageSkip: number }> = {};

  // For last digits 0-9
  for (let d = 0; d <= 9; d++) {
    const lastDigits = powerballs.map(pb => getLastDigit(pb));
    const stats = calculate6thBallSkipStats(lastDigits, d);
    lastDigitSkips[d] = { currentSkip: stats.currentSkip, averageSkip: stats.averageSkip };
  }

  // For first digits 0-2 (since 1-26)
  for (let d = 0; d <= 2; d++) {
    const firstDigits = powerballs.map(pb => getFirstDigit(pb));
    const stats = calculate6thBallSkipStats(firstDigits, d);
    firstDigitSkips[d] = { currentSkip: stats.currentSkip, averageSkip: stats.averageSkip };
  }

  // For digit sums 1-18 (1+9=10, 2+6=8, etc.)
  for (let s = 1; s <= 18; s++) {
    const sums = powerballs.map(pb => getDigitSum(pb));
    const stats = calculate6thBallSkipStats(sums, s);
    sumSkips[s] = { currentSkip: stats.currentSkip, averageSkip: stats.averageSkip };
  }

  return { lastDigitSkips, firstDigitSkips, sumSkips };
};
