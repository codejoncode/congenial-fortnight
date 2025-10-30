import React, { useState, useEffect } from 'react';
import { useStore } from '../store';
import EnhancedNumberGrid from './EnhancedNumberGrid';
import AdvancedFilters from './AdvancedFilters';
import PatternTracker from './PatternTracker';

type ViewType = 'grid' | 'filters' | 'patterns' | 'combinations';

const LotteryAnalyticsDashboard: React.FC = () => {
  const { draws, setDraws } = useStore();
  const [currentView, setCurrentView] = useState<ViewType>('grid');
  const [selectedNumbers] = useState<Set<number>>(new Set());
  const [filterMode] = useState<'include' | 'exclude'>('include');
  const [filteredCombinations, setFilteredCombinations] = useState<number[][]>([]);
  const [selectedPattern, setSelectedPattern] = useState<string>('');

  // Load data on component mount
  useEffect(() => {
    const loadData = async () => {
      try {
        const response = await fetch('/src/data/draws.json');
        const data = await response.json();
        setDraws(data);
      } catch (error) {
        console.error('Failed to load lottery data:', error);
      }
    };

    if (draws.length === 0) {
      loadData();
    }
  }, [draws.length, setDraws]);

  const handleFilterChange = (combos: number[][]) => {
    setFilteredCombinations(combos);
    setCurrentView('combinations');
  };

  const handlePatternSelect = (pattern: string) => {
    setSelectedPattern(pattern);
    // Could automatically apply pattern-based filters here
  };

  const navigationItems = [
    { id: 'grid' as ViewType, label: 'Number Grid', icon: '📊' },
    { id: 'filters' as ViewType, label: 'Advanced Filters', icon: '🔍' },
    { id: 'patterns' as ViewType, label: 'Pattern Tracker', icon: '📈' },
    { id: 'combinations' as ViewType, label: 'Combinations', icon: '🎯' }
  ];

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700 p-4">
        <div className="max-w-7xl mx-auto">
          <h1 className="text-3xl font-bold text-blue-400">Lottery Analytics Dashboard</h1>
          <p className="text-gray-400 mt-1">
            Advanced pattern analysis and combination generation for lottery strategy
          </p>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex space-x-8">
            {navigationItems.map(item => (
              <button
                key={item.id}
                onClick={() => setCurrentView(item.id)}
                className={`
                  flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm
                  ${currentView === item.id
                    ? 'border-blue-400 text-blue-400'
                    : 'border-transparent text-gray-400 hover:text-white hover:border-gray-300'
                  }
                `}
              >
                <span>{item.icon}</span>
                <span>{item.label}</span>
              </button>
            ))}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto py-6 px-4">
        {currentView === 'grid' && (
          <EnhancedNumberGrid />
        )}

        {currentView === 'filters' && (
          <AdvancedFilters
            selectedNumbers={selectedNumbers}
            filterMode={filterMode}
            onFilterChange={handleFilterChange}
          />
        )}

        {currentView === 'patterns' && (
          <PatternTracker onPatternSelect={handlePatternSelect} />
        )}

        {currentView === 'combinations' && (
          <div className="p-6">
            <h2 className="text-3xl font-bold mb-6">Generated Combinations</h2>

            {filteredCombinations.length === 0 ? (
              <div className="text-center py-12">
                <div className="text-6xl mb-4">🎯</div>
                <h3 className="text-xl font-semibold mb-2">No Combinations Generated</h3>
                <p className="text-gray-400 mb-4">
                  Use the Advanced Filters to generate lottery combinations based on your criteria.
                </p>
                <button
                  onClick={() => setCurrentView('filters')}
                  className="px-6 py-3 bg-blue-600 hover:bg-blue-700 rounded-lg font-semibold"
                >
                  Go to Filters
                </button>
              </div>
            ) : (
              <>
                <div className="mb-6 p-4 bg-gray-800 rounded">
                  <h3 className="text-xl font-semibold mb-2">Results Summary</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-2xl font-bold text-blue-400">
                        {filteredCombinations.length.toLocaleString()}
                      </p>
                      <p className="text-sm text-gray-400">Total Combinations</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-green-400">
                        {selectedNumbers.size || 70}
                      </p>
                      <p className="text-sm text-gray-400">Numbers Used</p>
                    </div>
                    <div>
                      <p className="text-2xl font-bold text-purple-400">
                        {selectedPattern || 'None'}
                      </p>
                      <p className="text-sm text-gray-400">Active Pattern</p>
                    </div>
                  </div>
                </div>

                <div className="bg-gray-800 rounded max-h-96 overflow-y-auto">
                  <div className="p-4">
                    <h3 className="text-lg font-semibold mb-4">Combinations</h3>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                      {filteredCombinations.slice(0, 50).map((combo, index) => (
                        <div
                          key={index}
                          className="bg-gray-700 p-3 rounded font-mono text-sm hover:bg-gray-600 transition-colors"
                        >
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-gray-400">#{index + 1}</span>
                            <span className="text-xs text-gray-500">
                              Sum: {combo.reduce((a,b) => a+b, 0)}
                            </span>
                          </div>
                          <div className="flex flex-wrap gap-1">
                            {combo.map((num, numIndex) => (
                              <span
                                key={numIndex}
                                className={`
                                  inline-block w-8 h-8 rounded-full text-center leading-8 text-xs font-bold
                                  ${num > 35 ? 'bg-blue-600' : 'bg-green-600'}
                                `}
                              >
                                {num}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>

                    {filteredCombinations.length > 50 && (
                      <div className="mt-4 text-center text-gray-400">
                        ... and {filteredCombinations.length - 50} more combinations
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 border-t border-gray-700 mt-12">
        <div className="max-w-7xl mx-auto py-4 px-4">
          <div className="flex justify-between items-center text-sm text-gray-400">
            <div>
              Total Draws Analyzed: {draws.length}
            </div>
            <div>
              Last Updated: {new Date().toLocaleDateString()}
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default LotteryAnalyticsDashboard;
