// src/tests/setup.ts
// Test setup and global configurations
import { expect, afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

// Cleanup after each test
afterEach(() => {
  cleanup();
});

// Global test utilities
global.testUtils = {
  // Add any global test utilities here
};
