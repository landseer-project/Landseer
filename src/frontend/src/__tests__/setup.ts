/**
 * Vitest test setup file
 * 
 * This file is run before each test file and sets up the testing environment.
 */

import '@testing-library/jest-dom/vitest';
import { afterEach, beforeAll, afterAll } from 'vitest';

// Recharts uses ResizeObserver in jsdom tests.
class ResizeObserverMock {
  observe() {}
  unobserve() {}
  disconnect() {}
}

if (!(globalThis as { ResizeObserver?: unknown }).ResizeObserver) {
  (globalThis as { ResizeObserver?: unknown }).ResizeObserver = ResizeObserverMock;
}

// Clean up after each test
afterEach(() => {
  // Clear any mocks
});

// Global setup
beforeAll(() => {
  // Setup global test state if needed
});

// Global teardown
afterAll(() => {
  // Cleanup global test state
});
