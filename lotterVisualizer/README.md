# ApexScoop Lottery Analytics Visualizer

A comprehensive lottery analytics web application built with modern web technologies to help users analyze lottery draw patterns, predict potential winning combinations, and visualize data insights.

## 🚀 Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Backend**: Node.js with TypeScript
- **Testing**: Vitest + React Testing Library
- **State Management**: Zustand
- **Visualization**: D3.js
- **Build Tool**: Vite
- **Package Manager**: npm

## 📋 Prerequisites

- Node.js (v16 or higher)
- npm (v7 or higher)

## 🛠️ Installation & Setup

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd /path/to/lottervisualizer
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm run dev
   ```

   The application will be available at `http://localhost:5173` (default Vite port).

4. **Build for production**:
   ```bash
   npm run build
   ```

5. **Preview production build**:
   ```bash
   npm run preview
   ```

## 🧪 Running Tests

The project includes comprehensive test coverage using Vitest.

### Run all tests:
```bash
npm test
# or
npx vitest run
```

### Run tests in watch mode:
```bash
npm run test:watch
# or
npx vitest
```

### Run tests with coverage:
```bash
npx vitest run --coverage
```

### Run specific test files:
```bash
npx vitest run src/tests/sixthBall.test.ts
```

## ✨ Available Features

### Core Analytics Engine
- **Skip Tracking**: Calculate number of draws since last appearance (NODS)
- **Hit Frequency Analysis**: Track how often number combinations appear
- **Pair & Triple Extraction**: Analyze adjacent number patterns
- **Draw Sum Analysis**: Statistical analysis of draw totals
- **Recurrence Scoring**: Predict based on historical patterns

### 6th Ball (Powerball) Analysis
- **Digit Pattern Analysis**:
  - Last digit frequency and trends
  - First digit patterns
  - Digit sum calculations
  - Division-based analysis (quotient/remainder)
- **Skip Statistics**: Current skips, average skips, skip distribution
- **Predictive Algorithms**: Score-based candidate prediction using:
  - Skip analysis (due for appearance)
  - Digit sum trending
  - Last digit frequency balancing
  - Even/odd parity balance
- **Hot/Cold Analysis**: Most and least frequent 6th ball numbers

### User Interface Components
- **Dashboard**: Main analytics overview
- **Data Visualization**: Charts and graphs for patterns
- **Filter System**: Parity, sum corridor, regime alignment
- **6th Ball Analytics Panel**: Dedicated prediction and analysis display
- **Interactive Filters**: Real-time combination filtering

### Testing & Quality Assurance
- **Unit Tests**: 101 comprehensive tests covering all functions
- **Integration Tests**: API and component interaction validation
- **UI Component Tests**: React component behavior verification
- **Edge Case Coverage**: Boundary condition and error handling
- **Performance Testing**: Efficiency validation for large datasets

### Data Management
- **Test Data Generation**: Factory-based realistic lottery data creation
- **Data Integrity Checks**: Validation of draw data consistency
- **Mocking & Stubbing**: Isolated testing of components
- **Error Handling**: Robust error management throughout the application

## 🎯 Usage Examples

### Analyzing 6th Ball Patterns
```typescript
import { analyze6thBallDigits, predict6thBallCandidates } from './src/utils/math';

// Analyze a Powerball number
const analysis = analyze6thBallDigits(15);
// Returns: { powerball: 15, lastDigit: 5, firstDigit: 1, digitSum: 6, ... }

// Get predictions
const predictions = predict6thBallCandidates(historicalData, recentDraws, 5);
// Returns top 5 predicted numbers with scores and reasons
```

### Running Analytics on Draw Data
```typescript
import { calculateSkipStats, extractPairs } from './src/utils/math';

// Calculate skip statistics
const stats = calculateSkipStats(drawHistory, targetNumber);

// Extract number pairs
const pairs = extractPairs([3, 15, 27, 42]);
```

## 📁 Project Structure

```
src/
├── ui/                 # React components
│   ├── Dashboard.tsx
│   ├── SixthBallAnalytics.tsx
│   └── ...
├── utils/
│   ├── math.ts         # Core analytics functions
│   └── ...
├── tests/              # Test files
│   ├── sixthBall.test.ts
│   ├── comprehensive.test.js
│   └── ...
├── types/              # TypeScript type definitions
└── ...
```

## 🔧 Development

### Adding New Features
1. Implement core logic in `src/utils/math.ts`
2. Create React components in `src/ui/`
3. Add comprehensive tests in `src/tests/`
4. Update UI integration in Dashboard

### Testing Strategy
- Write tests first (TDD approach)
- Cover happy path, edge cases, and error conditions
- Use factory functions for test data generation
- Maintain >95% code coverage

## 📊 Performance

- Optimized for large datasets (1000+ draws)
- Efficient algorithms for real-time analysis
- Lazy loading for heavy computations
- Responsive UI for various screen sizes

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Implement the feature
5. Ensure all tests pass
6. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙋 Support

For questions or issues:
- Check the test suite for examples
- Review the comprehensive testing strategy document
- Examine the analytics functions in `src/utils/math.ts`

---

**Ready to analyze some lottery data?** Start the dev server and explore the powerful analytics features! 🎰📈
