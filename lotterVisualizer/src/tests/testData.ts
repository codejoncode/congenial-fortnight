// src/tests/testData.ts
export const generateTestDraws = (count: number = 100): number[][] => {
  const draws: number[][] = [];

  for (let i = 0; i < count; i++) {
    // Generate 5 unique numbers between 1-69
    const numbers = new Set<number>();
    while (numbers.size < 5) {
      numbers.add(Math.floor(Math.random() * 69) + 1);
    }
    draws.push(Array.from(numbers).sort((a, b) => a - b));
  }

  return draws;
};

export const generateTestCombos = (count: number = 50): number[][] => {
  const combos: number[][] = [];

  for (let i = 0; i < count; i++) {
    const numbers = new Set<number>();
    while (numbers.size < 5) {
      numbers.add(Math.floor(Math.random() * 69) + 1);
    }
    combos.push(Array.from(numbers).sort((a, b) => a - b));
  }

  return combos;
};

export const mockComboExplanation = (combo: number[], draws: number[][]) => {
  const hits = draws.filter(draw =>
    combo.every(num => draw.includes(num))
  ).length;

  const hitRate = hits / draws.length;
  const avgResidue = Math.floor(Math.random() * 20) + 5; // Mock residue
  const lastHitIndex = Math.floor(Math.random() * draws.length);
  const drawsOut = draws.length - lastHitIndex;

  return {
    combo,
    reasons: [`Combo ${combo.join('-')} analysis`],
    sum: combo.reduce((a, b) => a + b, 0),
    parity: {
      evens: combo.filter(n => n % 2 === 0).length,
      odds: combo.filter(n => n % 2 === 1).length
    },
    keysMatched: combo.slice(0, 2), // Mock keys
    history: {
      testedDraws: draws.length,
      hits,
      hitRate,
      avgResidue,
      lastHitIndex,
      drawsOut
    },
    lift: hitRate > 0.5 ? Math.random() * 20 : -Math.random() * 10,
    confidence: Math.floor(Math.random() * 100),
    totalPoints: combo.reduce((a, b) => a + b, 0),
    heatScore: Math.floor(Math.random() * 100),
    momentumDelta: Math.floor(Math.random() * 20) - 10
  };
};

export const generateTestPowerballs = (count: number = 100): number[] => {
  const powerballs: number[] = [];

  for (let i = 0; i < count; i++) {
    // Powerball numbers are typically 1-26
    powerballs.push(Math.floor(Math.random() * 26) + 1);
  }

  return powerballs;
};

export const generateFullLotteryDraws = (count: number = 100): Array<{ mainNumbers: number[]; powerball: number }> => {
  const draws: Array<{ mainNumbers: number[]; powerball: number }> = [];

  for (let i = 0; i < count; i++) {
    // Generate 5 unique main numbers (1-69)
    const mainNumbers = new Set<number>();
    while (mainNumbers.size < 5) {
      mainNumbers.add(Math.floor(Math.random() * 69) + 1);
    }

    // Generate Powerball (1-26)
    const powerball = Math.floor(Math.random() * 26) + 1;

    draws.push({
      mainNumbers: Array.from(mainNumbers).sort((a, b) => a - b),
      powerball
    });
  }

  return draws;
};

export const generateTestPowerballCombos = (count: number = 50): number[] => {
  const combos: number[] = [];

  for (let i = 0; i < count; i++) {
    combos.push(Math.floor(Math.random() * 26) + 1);
  }

  return combos;
};
