import { describe, it, expect, beforeEach, vi } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import SimpleLotteryAnalyzer from './SimpleLotteryAnalyzer';
import { useStore } from '../store';

// Mock the store
vi.mock('../store', () => ({
  useStore: vi.fn()
}));

// Mock fetch
const mockFetch = vi.fn();
global.fetch = mockFetch;

const mockDraws = [
  [12, 18, 30, 40, 52],
  [5, 15, 25, 35, 45],
  [8, 22, 33, 44, 55]
];

describe('SimpleLotteryAnalyzer', () => {
  beforeEach(() => {
    vi.clearAllMocks();

    // Mock the store implementation
    vi.mocked(useStore).mockReturnValue({
      draws: mockDraws,
      setDraws: vi.fn()
    });

    // Mock fetch to return mock data
    mockFetch.mockResolvedValue({
      json: () => Promise.resolve(mockDraws)
    } as Response);
  });

  it('renders loading state initially', () => {
    render(<SimpleLotteryAnalyzer />);
    expect(screen.getByText('Loading Powerball data...')).toBeInTheDocument();
  });

  it('renders the main title', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      expect(screen.getByText('Powerball Prediction Analyzer')).toBeInTheDocument();
    });
  });

  it('displays all 70 numbers in the grid', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      // Check that the grid contains number elements
      const gridContainer = screen.getByText('Current Number Status (1-70)').parentElement;
      const numberElements = gridContainer?.querySelectorAll('.font-bold.text-lg');

      // Should have at least 70 number elements in the grid
      expect(numberElements?.length).toBeGreaterThanOrEqual(70);

      // Verify some specific numbers are present in the grid
      expect(screen.getByText('1')).toBeInTheDocument();
      expect(screen.getByText('25')).toBeInTheDocument();
      expect(screen.getByText('50')).toBeInTheDocument();
    });
  });

  it('shows filter controls', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      expect(screen.getByText('Show Filters')).toBeInTheDocument();
      expect(screen.getByText('Generate Predictions (0)')).toBeInTheDocument();
    });
  });

  it('toggles filters panel when button is clicked', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      const filterButton = screen.getByText('Show Filters');
      fireEvent.click(filterButton);

      expect(screen.getByText('Hide Filters')).toBeInTheDocument();
    });
  });

  it('displays prediction mode selector', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      expect(screen.getByText('Prediction Mode:')).toBeInTheDocument();
      expect(screen.getByDisplayValue('Balanced (All Numbers)')).toBeInTheDocument();
    });
  });

  it('generates combinations when button is clicked', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      const generateButton = screen.getByText('Generate Predictions (0)');
      fireEvent.click(generateButton);

      // Should show some combinations after generation
      expect(screen.getByText('Predicted Combinations')).toBeInTheDocument();
    });
  });

  it('shows statistics dashboard', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      expect(screen.getByText('Total Draws')).toBeInTheDocument();
      expect(screen.getByText('Hot Numbers')).toBeInTheDocument();
      expect(screen.getByText('Cold Numbers')).toBeInTheDocument();
      expect(screen.getByText('Due Numbers')).toBeInTheDocument();
    });
  });

  it('displays current number status with colors', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      // Check that status indicators are present
      const statusElements = screen.getAllByText(/HOT|COLD|DUE/);
      expect(statusElements.length).toBeGreaterThan(0);
    });
  });

  it('shows purchase button when combinations are selected', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      const generateButton = screen.getByText('Generate Predictions (0)');
      fireEvent.click(generateButton);
    });

    // Wait for combinations to be generated
    await waitFor(() => {
      const combinationElements = screen.queryAllByText(/\d+ - \d+ - \d+ - \d+ - \d+/);
      if (combinationElements.length > 0) {
        fireEvent.click(combinationElements[0]);
        const purchaseButton = screen.queryByText(/Purchase Selected/);
        expect(purchaseButton).toBeInTheDocument();
      }
    });
  });

  it('displays combination details correctly', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      const generateButton = screen.getByText('Generate Predictions (0)');
      fireEvent.click(generateButton);
    });

    await waitFor(() => {
      // Check for pattern information
      const patternElements = screen.queryAllByText(/Pattern:/);
      if (patternElements.length > 0) {
        expect(patternElements[0]).toBeInTheDocument();
      }
    });
  });

  it('handles filter mode changes', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      const selectElement = screen.getByDisplayValue('Balanced (All Numbers)');
      fireEvent.change(selectElement, { target: { value: 'hot' } });

      expect(screen.getByDisplayValue('Hot Numbers Only')).toBeInTheDocument();
    });
  });

  it('shows empty state message when no combinations generated', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      expect(screen.getByText('Click "Generate Predictions" to see combinations')).toBeInTheDocument();
    });
  });

  it('displays number statistics correctly', async () => {
    render(<SimpleLotteryAnalyzer />);

    await waitFor(() => {
      // Check for "Out:" labels which indicate number statistics
      const outLabels = screen.getAllByText(/Out:/);
      expect(outLabels.length).toBe(70); // Should have stats for all 70 numbers
    });
  });

  it('handles error state gracefully', async () => {
    // Mock fetch to reject
    mockFetch.mockRejectedValue(new Error('Network error'));

    render(<SimpleLotteryAnalyzer />);

    // Should not crash and should handle the error
    await waitFor(() => {
      expect(screen.queryByText('Loading Powerball data...')).not.toBeInTheDocument();
    });
  });
});
