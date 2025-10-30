import React, { useState, useMemo } from 'react';
import { useStore } from '../store';

interface PatternFilter {
  id: string;
  name: string;
  pattern: string;
  count: number;
  lastSeen: number;
  frequency: number;
}

interface AdvancedFiltersProps {
  selectedNumbers: Set<number>;
  filterMode: 'include' | 'exclude';
  onFilterChange: (filteredCombos: number[][]) => void;
}

const AdvancedFilters: React.FC<AdvancedFiltersProps> = ({
  selectedNumbers,
  filterMode,
  onFilterChange
}) => {
  const { draws } = useStore();
  const [activeFilters, setActiveFilters] = useState<Set<string>>(new Set());

  // Pattern analysis for recent draws
  const patternAnalysis = useMemo(() => {
    const patterns: PatternFilter[] = [];
    const recentDraws = draws.slice(-10); // Last 10 draws

    // Analyze patterns in recent draws
    recentDraws.forEach((draw, index) => {
      const sorted = [...draw].sort((a,b) => a-b);

      // HHHHH, LLLLL, etc.
      const highLowPattern = sorted.map(n => n > 35 ? 'H' : 'L').join('');
      const evenOddPattern = sorted.map(n => n % 2 === 0 ? 'E' : 'O').join('');

      // Add to patterns
      patterns.push({
        id: `hl_${index}`,
        name: `High/Low: ${highLowPattern}`,
        pattern: highLowPattern,
        count: 1,
        lastSeen: index,
        frequency: 0.1
      });

      patterns.push({
        id: `eo_${index}`,
        name: `Even/Odd: ${evenOddPattern}`,
        pattern: evenOddPattern,
        count: 1,
        lastSeen: index,
        frequency: 0.1
      });
    });

    return patterns;
  }, [draws]);

  // Generate combinations based on filters
  const generateFilteredCombos = useMemo(() => {
    let baseNumbers = Array.from({ length: 70 }, (_, i) => i + 1);

    // Apply custom list filter
    if (selectedNumbers.size > 0) {
      if (filterMode === 'include') {
        baseNumbers = Array.from(selectedNumbers);
      } else {
        baseNumbers = baseNumbers.filter(n => !selectedNumbers.has(n));
      }
    }

    // Generate combinations (simplified for demo - in real app use proper combo generation)
    const combos: number[][] = [];
    for (let i = 0; i < Math.min(baseNumbers.length - 4, 100); i++) {
      for (let j = i + 1; j < Math.min(baseNumbers.length - 3, 100); j++) {
        for (let k = j + 1; k < Math.min(baseNumbers.length - 2, 100); k++) {
          for (let l = k + 1; l < Math.min(baseNumbers.length - 1, 100); l++) {
            for (let m = l + 1; m < Math.min(baseNumbers.length, 100); m++) {
              combos.push([
                baseNumbers[i],
                baseNumbers[j],
                baseNumbers[k],
                baseNumbers[l],
                baseNumbers[m]
              ]);
            }
          }
        }
      }
    }

    return combos;
  }, [selectedNumbers, filterMode]);

  const toggleFilter = (filterId: string) => {
    const newFilters = new Set(activeFilters);
    if (newFilters.has(filterId)) {
      newFilters.delete(filterId);
    } else {
      newFilters.add(filterId);
    }
    setActiveFilters(newFilters);
  };

  const applyFilters = () => {
    onFilterChange(generateFilteredCombos);
  };

  return (
    <div className="p-6 bg-gray-900 min-h-screen">
      <h2 className="text-3xl font-bold mb-6">Advanced Pattern Filters</h2>

      {/* Pattern Analysis Section */}
      <div className="mb-8">
        <h3 className="text-2xl font-semibold mb-4">Recent Pattern Analysis</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {patternAnalysis.slice(0, 10).map(pattern => (
            <div key={pattern.id} className="bg-gray-800 p-4 rounded">
              <div className="flex justify-between items-center mb-2">
                <span className="font-semibold">{pattern.name}</span>
                <button
                  onClick={() => toggleFilter(pattern.id)}
                  className={`px-3 py-1 rounded text-sm ${
                    activeFilters.has(pattern.id)
                      ? 'bg-blue-600 hover:bg-blue-700'
                      : 'bg-gray-600 hover:bg-gray-500'
                  }`}
                >
                  {activeFilters.has(pattern.id) ? 'Active' : 'Add'}
                </button>
              </div>
              <div className="text-sm text-gray-400">
                Last seen: {pattern.lastSeen} draws ago<br/>
                Frequency: {(pattern.frequency * 100).toFixed(1)}%
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Filter Controls */}
      <div className="mb-8">
        <h3 className="text-2xl font-semibold mb-4">Filter Controls</h3>

        {/* Sum Range Filter */}
        <div className="mb-4 p-4 bg-gray-800 rounded">
          <h4 className="font-semibold mb-2">Sum Range</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm mb-1">Min Sum</label>
              <input
                type="number"
                className="w-full bg-gray-700 border border-gray-600 rounded p-2"
                defaultValue="100"
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Max Sum</label>
              <input
                type="number"
                className="w-full bg-gray-700 border border-gray-600 rounded p-2"
                defaultValue="250"
              />
            </div>
          </div>
        </div>

        {/* Parity Filter */}
        <div className="mb-4 p-4 bg-gray-800 rounded">
          <h4 className="font-semibold mb-2">Even/Odd Balance</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm mb-1">Even Count</label>
              <select className="w-full bg-gray-700 border border-gray-600 rounded p-2">
                <option value="">Any</option>
                {[0,1,2,3,4,5].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm mb-1">Odd Count</label>
              <select className="w-full bg-gray-700 border border-gray-600 rounded p-2">
                <option value="">Any</option>
                {[0,1,2,3,4,5].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* High/Low Filter */}
        <div className="mb-4 p-4 bg-gray-800 rounded">
          <h4 className="font-semibold mb-2">High/Low Balance (35+ = High)</h4>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm mb-1">High Count</label>
              <select className="w-full bg-gray-700 border border-gray-600 rounded p-2">
                <option value="">Any</option>
                {[0,1,2,3,4,5].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm mb-1">Low Count</label>
              <select className="w-full bg-gray-700 border border-gray-600 rounded p-2">
                <option value="">Any</option>
                {[0,1,2,3,4,5].map(n => (
                  <option key={n} value={n}>{n}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Skip Count Filters */}
        <div className="mb-4 p-4 bg-gray-800 rounded">
          <h4 className="font-semibold mb-2">Skip Count Filters</h4>
          <div className="space-y-2">
            <label className="flex items-center">
              <input type="checkbox" className="mr-2" />
              <span>Include at least one number with skip ≥ 10</span>
            </label>
            <label className="flex items-center">
              <input type="checkbox" className="mr-2" />
              <span>Include at least one number with skip ≥ 20</span>
            </label>
            <label className="flex items-center">
              <input type="checkbox" className="mr-2" />
              <span>Include at least one triple with skip ≥ 100</span>
            </label>
          </div>
        </div>
      </div>

      {/* Active Filters Summary */}
      <div className="mb-8 p-4 bg-gray-800 rounded">
        <h3 className="text-xl font-semibold mb-4">Active Filters</h3>
        {activeFilters.size === 0 ? (
          <p className="text-gray-400">No filters active</p>
        ) : (
          <div className="space-y-2">
            {Array.from(activeFilters).map(filterId => (
              <div key={filterId} className="flex justify-between items-center bg-gray-700 p-2 rounded">
                <span>{patternAnalysis.find(p => p.id === filterId)?.name || filterId}</span>
                <button
                  onClick={() => toggleFilter(filterId)}
                  className="text-red-400 hover:text-red-300"
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Results Summary */}
      <div className="mb-8 p-4 bg-gray-800 rounded">
        <h3 className="text-xl font-semibold mb-4">Filter Results</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-2xl font-bold text-blue-400">{generateFilteredCombos.length.toLocaleString()}</p>
            <p className="text-sm text-gray-400">Combinations match filters</p>
          </div>
          <div>
            <p className="text-2xl font-bold text-green-400">
              {selectedNumbers.size > 0 ? `${selectedNumbers.size} numbers` : 'All 70 numbers'}
            </p>
            <p className="text-sm text-gray-400">Base numbers used</p>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-4">
        <button
          onClick={applyFilters}
          className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold"
        >
          Generate Combinations
        </button>
        <button
          onClick={() => setActiveFilters(new Set())}
          className="px-6 py-3 bg-gray-600 hover:bg-gray-500 rounded-lg font-semibold"
        >
          Clear All Filters
        </button>
      </div>
    </div>
  );
};

export default AdvancedFilters;
