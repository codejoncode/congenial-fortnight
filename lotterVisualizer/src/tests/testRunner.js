// src/tests/testRunner.js
// Test runner that executes all test files

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

console.log('🚀 ApexScoop Test Runner\n');

const testDir = path.join(__dirname);
const testFiles = fs.readdirSync(testDir)
  .filter(file => file.endsWith('.test.js') || file.endsWith('.test.ts') || file.endsWith('.test.tsx'))
  .sort();

console.log(`📁 Found ${testFiles.length} test files:\n`);

// Categorize tests
const jsTests = testFiles.filter(file => file.endsWith('.test.js'));
const tsTests = testFiles.filter(file => file.endsWith('.test.ts') || file.endsWith('.test.tsx'));

console.log('📄 JavaScript Tests (no dependencies):');
jsTests.forEach(file => console.log(`  • ${file}`));

console.log('\n📄 TypeScript Tests (require vitest):');
tsTests.forEach(file => console.log(`  • ${file}`));

// Run JavaScript tests first
console.log('\n🧪 Running JavaScript Tests...\n');

let jsTestsPassed = 0;
let jsTestsFailed = 0;

jsTests.forEach(testFile => {
  try {
    console.log(`▶️  Running ${testFile}...`);
    const output = execSync(`node "${path.join(testDir, testFile)}"`, {
      encoding: 'utf8',
      timeout: 30000
    });

    // Check if test passed
    if (output.includes('ALL TESTS PASSED') || output.includes('passed')) {
      console.log(`✅ ${testFile} - PASSED\n`);
      jsTestsPassed++;
    } else {
      console.log(`❌ ${testFile} - FAILED\n`);
      console.log('Output:', output.slice(-500)); // Last 500 chars
      jsTestsFailed++;
    }
  } catch (error) {
    console.log(`❌ ${testFile} - ERROR: ${error.message}\n`);
    jsTestsFailed++;
  }
});

// Check if vitest is available
let vitestAvailable = false;
try {
  execSync('npx vitest --version', { stdio: 'ignore' });
  vitestAvailable = true;
} catch (error) {
  vitestAvailable = false;
}

if (vitestAvailable) {
  console.log('\n🧪 Running TypeScript Tests with Vitest...\n');

  let tsTestsPassed = 0;
  let tsTestsFailed = 0;

  tsTests.forEach(testFile => {
    try {
      console.log(`▶️  Running ${testFile}...`);
      const output = execSync(`npx vitest run "${path.join(testDir, testFile)}"`, {
        encoding: 'utf8',
        timeout: 30000
      });

      if (output.includes('✓') && !output.includes('✗')) {
        console.log(`✅ ${testFile} - PASSED\n`);
        tsTestsPassed++;
      } else {
        console.log(`❌ ${testFile} - FAILED\n`);
        console.log('Output:', output.slice(-500));
        tsTestsFailed++;
      }
    } catch (error) {
      console.log(`❌ ${testFile} - ERROR: ${error.message}\n`);
      tsTestsFailed++;
    }
  });

  console.log('\n📊 TypeScript Test Results:');
  console.log(`✅ Passed: ${tsTestsPassed}`);
  console.log(`❌ Failed: ${tsTestsFailed}`);
  console.log(`📈 Success Rate: ${((tsTestsPassed / tsTests.length) * 100).toFixed(1)}%`);
} else {
  console.log('\n⚠️  Vitest not available. TypeScript tests require dependencies.');
  console.log('Run: npm install');
  console.log('Then: npm test');
}

console.log('\n📊 JavaScript Test Results:');
console.log(`✅ Passed: ${jsTestsPassed}`);
console.log(`❌ Failed: ${jsTestsFailed}`);
console.log(`📈 Success Rate: ${((jsTestsPassed / jsTests.length) * 100).toFixed(1)}%`);

const totalTests = jsTests.length + (vitestAvailable ? tsTests.length : 0);
const totalPassed = jsTestsPassed + (vitestAvailable ? (tsTests.length - (typeof tsTestsFailed !== 'undefined' ? tsTestsFailed : 0)) : 0);
const totalFailed = jsTestsFailed + (vitestAvailable ? (typeof tsTestsFailed !== 'undefined' ? tsTestsFailed : 0) : 0);

console.log('\n🏆 Overall Test Results:');
console.log(`✅ Total Passed: ${totalPassed}`);
console.log(`❌ Total Failed: ${totalFailed}`);
console.log(`📊 Overall Success Rate: ${((totalPassed / totalTests) * 100).toFixed(1)}%`);

if (totalPassed === totalTests) {
  console.log('\n🎉 ALL TESTS PASSED! The ApexScoop system is working correctly.');
} else {
  console.log('\n⚠️  Some tests failed. Please review the output above.');
}

console.log('\n💡 Next Steps:');
if (!vitestAvailable) {
  console.log('1. Install dependencies: npm install');
  console.log('2. Run full test suite: npm test');
}
console.log('3. Review test results and fix any issues');
console.log('4. Consider adding more test cases for edge scenarios');
