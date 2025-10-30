# ApexScoop Testing Guide

## Overview

This comprehensive testing suite ensures the reliability, accuracy, and performance of the ApexScoop Picks lottery visualization system. The tests cover all core functionality including combo generation, filtering, scoring, momentum analysis, and data integrity.

## Test Structure

```
src/tests/
├── testData.ts          # Test data generation utilities
├── scoring.test.ts      # Unit tests for scoring functions
├── filters.test.ts      # Unit tests for filter functions
├── integration.test.ts  # End-to-end integration tests
└── README.md           # This testing guide
```

## Test Categories

### 1. Unit Tests (`scoring.test.ts`, `filters.test.ts`)

#### Scoring Function Tests
- **calculateConfidenceIndex**: Tests lift score, recurrence score, and regime score calculations
- **calculateFreshnessMultiplier**: Tests freshness bonuses and penalties based on draws-out
- **calculateMomentumAdjustedConfidence**: Tests momentum bonus/penalty calculations
- **calculateHeatScore**: Tests the combination of momentum and freshness scoring

#### Filter Function Tests
- **buildExplainingFilterFn**: Tests sum constraints, parity constraints, and key matching
- **Combo validation**: Tests that generated combos meet specified criteria

### 2. Integration Tests (`integration.test.ts`)

#### End-to-End Pipeline Tests
- **Full combo generation pipeline**: Tests the complete flow from generation to scoring
- **Data integrity verification**: Ensures data consistency throughout the pipeline
- **Edge case handling**: Tests restrictive filters and boundary conditions
- **Performance validation**: Ensures operations complete within acceptable time limits

## Running Tests

### Prerequisites

```bash
npm install --save-dev vitest @testing-library/react @testing-library/jest-dom jsdom
```

### Configuration

Add to `package.json`:

```json
{
  "scripts": {
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage"
  }
}
```

Create `vitest.config.ts`:

```typescript
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
  },
})
```

### Execution

```bash
# Run all tests
npm test

# Run with UI
npm run test:ui

# Run with coverage
npm run test:coverage

# Run specific test file
npm test scoring.test.ts

# Run tests in watch mode
npm test -- --watch
```

## Test Data Generation

The `testData.ts` module provides utilities for generating realistic test data:

### `generateTestDraws(count)`
Generates random but valid lottery draws:
- 5 unique numbers between 1-69
- Numbers are sorted
- Mimics real Powerball draw format

### `generateTestCombos(count)`
Generates random combo arrays for testing:
- 5 unique numbers between 1-69
- Sorted for consistency
- Can be used to test combo processing functions

### `mockComboExplanation(combo, draws)`
Creates mock ComboExplanation objects for testing:
- Realistic hit rates and residue values
- Proper history statistics
- Confidence and heat scores

## Key Test Scenarios

### 1. Filter Validation

```typescript
// Test sum constraints
const explainFn = buildExplainingFilterFn({
  minSum: 100,
  maxSum: 200,
  // ... other params
});

const validCombo = [10, 20, 30, 40, 50]; // Sum = 150
const invalidCombo = [1, 2, 3, 4, 5];   // Sum = 15

expect(explainFn(validCombo)).toBeTruthy();
expect(explainFn(invalidCombo)).toBeFalsy();
```

### 2. Scoring Accuracy

```typescript
// Test momentum calculation
const trail = [
  { drawIndex: 0, confidence: 40, totalPoints: 150 },
  { drawIndex: 1, confidence: 50, totalPoints: 152 },
  { drawIndex: 2, confidence: 60, totalPoints: 148 }
];

const momentumScore = calculateMomentumAdjustedConfidence(50, trail, 5);
expect(momentumScore).toBeGreaterThan(50); // Positive momentum
```

### 3. Data Integrity

```typescript
// Test that combos maintain integrity through pipeline
const combos = generateExplainedCombosFast(pool, 5, explainFn, 100, 200);
const enhanced = enhanceCombosWithMomentum(combos, testDraws);

enhanced.forEach(combo => {
  // Verify all numbers are valid
  combo.combo.forEach(num => {
    expect(num).toBeGreaterThanOrEqual(1);
    expect(num).toBeLessThanOrEqual(69);
  });

  // Verify sum calculation
  const calculatedSum = combo.combo.reduce((a, b) => a + b, 0);
  expect(combo.sum).toBe(calculatedSum);
});
```

## Performance Benchmarks

### Expected Performance
- **Combo Generation**: < 5 seconds for 1000+ combos
- **Filtering**: < 1 second for 100 combos
- **Scoring**: < 2 seconds for 100 combos with momentum
- **Memory Usage**: < 50MB for typical datasets

### Performance Test Example

```typescript
it('should generate combos within reasonable time', () => {
  const startTime = Date.now();

  const combos = generateExplainedCombosFast(pool, 5, explainFn, 100, 200);

  const duration = Date.now() - startTime;
  expect(duration).toBeLessThan(5000); // 5 second limit
});
```

## Coverage Goals

### Unit Test Coverage
- **Core Functions**: 90%+ coverage
- **Edge Cases**: All boundary conditions tested
- **Error Handling**: Invalid inputs properly handled

### Integration Coverage
- **End-to-End Flows**: All major user journeys tested
- **Data Pipeline**: Complete data transformation validation
- **API Contracts**: All function interfaces verified

## Test Data Validation

### Statistical Validation
- **Hit Rate Distribution**: Should follow expected probability distributions
- **Score Ranges**: Confidence scores should be within 0-100 range
- **Momentum Calculations**: Should reflect realistic trend analysis

### Data Quality Checks
- **Uniqueness**: No duplicate combos in generated sets
- **Validity**: All numbers within lottery bounds (1-69)
- **Consistency**: Combo arrays should be sorted
- **Completeness**: All required fields populated

## Continuous Integration

### GitHub Actions Example

```yaml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '18'
      - run: npm ci
      - run: npm test
      - run: npm run test:coverage
```

## Debugging Failed Tests

### Common Issues

1. **Type Errors**: Check TypeScript types and imports
2. **Async Operations**: Ensure proper await handling
3. **Random Data**: Use seeded random for reproducible tests
4. **Performance**: Check for infinite loops or excessive computations

### Debugging Tools

```typescript
// Add debug logging
console.log('Debug info:', { combo, result, expected });

// Use test.only for focused testing
it.only('should debug specific case', () => {
  // Focused test logic
});
```

## Maintenance

### Adding New Tests

1. **Identify Test Category**: Unit, Integration, or E2E
2. **Create Test File**: Follow naming convention `*.test.ts`
3. **Write Descriptive Tests**: Clear test names and assertions
4. **Update Documentation**: Add to this README if needed

### Test Data Updates

- **Regular Updates**: Refresh test data periodically
- **Edge Cases**: Add more boundary condition tests
- **Performance Baselines**: Update performance expectations

## Troubleshooting

### Test Failures

1. **Check Dependencies**: Ensure all dev dependencies are installed
2. **Environment**: Verify Node.js version compatibility
3. **Data Consistency**: Check test data hasn't changed unexpectedly
4. **Type Issues**: Verify TypeScript configuration

### Performance Issues

1. **Profiling**: Use browser dev tools for performance analysis
2. **Optimization**: Identify bottlenecks in test setup
3. **Parallelization**: Consider running tests in parallel
4. **Mocking**: Use mocks for expensive operations

## Conclusion

This comprehensive testing suite ensures the ApexScoop system maintains high quality, performance, and reliability. Regular test execution and maintenance are crucial for catching regressions and ensuring the lottery analysis engine works correctly under all conditions.

For questions or issues with the testing suite, refer to the main project documentation or create an issue in the repository.
