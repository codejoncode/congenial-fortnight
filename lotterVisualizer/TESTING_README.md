# ApexScoop Testing & Validation Suite

## 🎯 Test Suite Overview

This comprehensive testing suite validates the ApexScoop Picks lottery visualization system, ensuring all filters, scoring algorithms, and data processing functions work correctly.

## 📁 Test Structure

```
src/tests/
├── testData.ts           # Test data generation utilities
├── scoring.test.ts       # Unit tests for scoring functions
├── filters.test.ts       # Unit tests for filter functions
├── integration.test.ts   # End-to-end integration tests
├── runTests.ts          # Simple test runner (no dependencies)
├── setup.ts             # Test configuration
└── README.md            # Comprehensive testing guide
```

## 🚀 Quick Start

### Option 1: Simple Test Runner (No Dependencies)
```bash
cd src/tests
node runTests.ts
```

### Option 2: Full Test Suite (Requires Dependencies)
```bash
npm install
npm test
```

## 🧪 Test Categories

### 1. **Data Generation Tests**
- ✅ Random lottery draw generation (1-69, 5 numbers)
- ✅ Combo generation with uniqueness validation
- ✅ Mock data creation for testing scenarios

### 2. **Scoring Function Tests**
- ✅ `calculateConfidenceIndex` - Lift, recurrence, regime scoring
- ✅ `calculateFreshnessMultiplier` - Recency bonuses/penalties
- ✅ `calculateMomentumAdjustedConfidence` - Trend analysis
- ✅ `calculateHeatScore` - Combined momentum + freshness

### 3. **Filter Function Tests**
- ✅ Sum constraints (min/max validation)
- ✅ Parity constraints (even/odd count)
- ✅ Key matching validation
- ✅ Combo generation with filters applied

### 4. **Integration Tests**
- ✅ End-to-end combo generation pipeline
- ✅ Data integrity through all processing stages
- ✅ Performance validation (< 5 seconds)
- ✅ Edge case handling

## 📊 Test Data Validation

### Statistical Checks
- **Hit Rate Distribution**: Validates probability calculations
- **Score Ranges**: Ensures confidence scores are 0-100
- **Data Integrity**: Verifies combo sorting and uniqueness
- **Boundary Conditions**: Tests edge cases and limits

### Quality Assurance
- **Uniqueness**: No duplicate combos in generated sets
- **Validity**: All numbers within lottery bounds (1-69)
- **Consistency**: Proper data structure maintenance
- **Completeness**: All required fields populated

## 🎮 Manual Testing Scenarios

### Filter Testing
1. **Sum Range**: Set min=100, max=200
   - ✅ Combos with sum 150 should pass
   - ❌ Combos with sum 50 should fail

2. **Parity**: Set evens=3, odds=2
   - ✅ [2,4,6,7,9] should pass (3 evens, 2 odds)
   - ❌ [1,3,5,7,9] should fail (0 evens, 5 odds)

3. **Keys**: Set keys=[5,10]
   - ✅ [5,10,15,20,25] should pass
   - ❌ [1,2,3,4,6] should fail

### Scoring Validation
1. **Fresh Combos**: drawsOut ≤ 2
   - Expected: 15% bonus multiplier

2. **Stale Combos**: drawsOut ≥ 20
   - Expected: 15% penalty multiplier

3. **Positive Momentum**: Increasing confidence trend
   - Expected: Bonus added to base score

4. **Negative Momentum**: Decreasing confidence trend
   - Expected: Penalty applied to base score

## 📈 Performance Benchmarks

| Operation | Expected Time | Status |
|-----------|---------------|--------|
| Generate 1000 combos | < 3 seconds | ✅ |
| Apply filters | < 1 second | ✅ |
| Calculate scores | < 2 seconds | ✅ |
| Full pipeline | < 5 seconds | ✅ |

## 🔧 Test Configuration

### Vitest Setup
```typescript
// vitest.config.ts
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/tests/setup.ts'],
  },
})
```

### Package.json Scripts
```json
{
  "scripts": {
    "test": "vitest",
    "test:ui": "vitest --ui",
    "test:coverage": "vitest --coverage",
    "test:watch": "vitest --watch"
  }
}
```

## 🎯 Validation Checklist

### Core Functionality
- [x] Combo generation with constraints
- [x] Filter application and validation
- [x] Scoring algorithm accuracy
- [x] Momentum calculation
- [x] Freshness weighting
- [x] Data integrity preservation

### Edge Cases
- [x] Empty filter sets
- [x] Restrictive constraints (no matches)
- [x] Boundary values
- [x] Invalid input handling

### Performance
- [x] Reasonable execution times
- [x] Memory usage within limits
- [x] Scalability with larger datasets

## 🚨 Known Test Limitations

1. **Random Data**: Tests use random generation - results may vary between runs
2. **Mock Data**: Some tests use simplified mock data for performance
3. **External Dependencies**: Full test suite requires additional npm packages

## 🔄 Continuous Testing

### Automated Testing
```bash
# Run tests on every commit
npm test

# Generate coverage reports
npm run test:coverage

# Watch mode for development
npm run test:watch
```

### Manual Validation
1. **UI Testing**: Verify dashboard displays correctly
2. **Filter Testing**: Test all filter combinations manually
3. **Export Testing**: Verify shortlist export functionality
4. **Performance Testing**: Monitor execution times

## 📋 Test Maintenance

### Adding New Tests
1. Create test file in `src/tests/`
2. Follow naming convention: `*.test.ts`
3. Use descriptive test names
4. Include edge cases and error conditions

### Updating Test Data
- Refresh random data periodically
- Add more boundary condition tests
- Update performance baselines as needed

## 🎉 Success Criteria

The test suite is successful when:
- ✅ All core functions work as expected
- ✅ Filters correctly validate combos
- ✅ Scoring algorithms produce reasonable results
- ✅ Performance meets benchmarks
- ✅ Data integrity is maintained
- ✅ Edge cases are handled gracefully

## 📞 Troubleshooting

### Common Issues
1. **Import Errors**: Ensure all dependencies are installed
2. **Type Errors**: Check TypeScript configuration
3. **Performance Issues**: Verify test environment
4. **Random Failures**: Check for flaky random data

### Debug Mode
```bash
# Run with debug logging
DEBUG=true npm test

# Run specific test
npm test -- scoring.test.ts
```

---

## 🎊 Conclusion

This testing suite provides comprehensive validation of the ApexScoop lottery analysis system. The tests ensure that all filtering, scoring, and data processing functions work correctly and efficiently.

**Ready to run tests?** Choose your preferred method:

- **Quick validation**: `node src/tests/runTests.ts`
- **Full test suite**: `npm install && npm test`

All tests should pass, confirming the system is working correctly! 🚀
