// src/tests/runTests.js
// Simple test runner for immediate testing (JavaScript version)

import { generateTestDraws, generateTestCombos } from './testData.js';
import { calculateConfidenceIndex } from '../core/scoring.js';
import { buildExplainingFilterFn } from '../core/compiledFilter.js';
import { sumConstraint, parityConstraint } from '../utils/math.js';

console.log('🧪 Running ApexScoop Test Suite...\n');

// Test 1: Data Generation
console.log('📊 Test 1: Data Generation');
const testDraws = generateTestDraws(20);
const testCombos = generateTestCombos(10);

console.log(`✅ Generated ${testDraws.length} test draws`);
console.log(`✅ Generated ${testCombos.length} test combos`);

// Verify data quality
const allNumbersValid = testDraws.every(draw =>
  draw.length === 5 &&
  draw.every(num => num >= 1 && num <= 69) &&
  new Set(draw).size === 5 // No duplicates
);
console.log(`✅ All test data valid: ${allNumbersValid}\n`);

// Test 2: Scoring Functions
console.log('🎯 Test 2: Scoring Functions');
const confidenceScore = calculateConfidenceIndex(15, [], undefined, 150, 10);
console.log(`✅ Confidence score calculation: ${confidenceScore}`);
console.log(`✅ Score in valid range (0-100): ${confidenceScore >= 0 && confidenceScore <= 100}\n`);

// Test 3: Filter Functions
console.log('🔍 Test 3: Filter Functions');
const pool = Array.from({ length: 69 }, (_, i) => i + 1);

try {
  const explainFn = buildExplainingFilterFn({
    keys: [],
    minSum: 100,
    maxSum: 200,
    evenCount: undefined,
    oddCount: undefined,
    lastDraw: testDraws[testDraws.length - 1],
    baseline: { hitRate: 0.1, avgResidue: 10 },
    drawHistory: testDraws,
    sumConstraint,
    parityConstraint,
    generatePoolForDraw: () => [pool],
    recurrenceRules: undefined,
    recurrenceStats: {}
  });

  const validCombo = [10, 20, 30, 40, 50]; // Sum = 150
  const invalidCombo = [1, 2, 3, 4, 5]; // Sum = 15

  const validResult = explainFn(validCombo);
  const invalidResult = explainFn(invalidCombo);

  console.log(`✅ Valid combo (sum=150) passes filter: ${!!validResult}`);
  console.log(`✅ Invalid combo (sum=15) fails filter: ${!invalidResult}`);
} catch (error) {
  console.log(`❌ Filter test failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
}

console.log('\n🎉 Basic test suite completed!');
console.log('\n📋 Next Steps:');
console.log('1. Run: npm install');
console.log('2. Run: npm test (for full test suite)');
console.log('3. Run: npm run test:coverage (for coverage report)');
