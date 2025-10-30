import React, { useState, useEffect } from 'react';
import { useStore } from '../store';

interface NumberStat {
  number: number;
  out: number;
  hits: number;
  avg: number;
  status: string;
  lastSeen: number;
}

interface PowerballCombo {
  id: string;
  numbers: number[];
  sum: number;
  evens: number;
  odds: number;
  highs: number;
  lows: number;
  pattern: string;
  neverHit: boolean;
  selected: boolean;
}

const SimpleLotteryAnalyzer: React.FC = () => {
  const { draws, setDraws } = useStore();
  const [isLoading, setIsLoading] = useState(true);
  const [numberStats, setNumberStats] = useState<NumberStat[]>([]);
  const [showFilters, setShowFilters] = useState(false);
  const [generatedCombos, setGeneratedCombos] = useState<PowerballCombo[]>([]);
  const [selectedCombos, setSelectedCombos] = useState<string[]>([]);
  const [filterMode, setFilterMode] = useState<'hot' | 'cold' | 'due' | 'balanced'>('balanced');

  useEffect(() => {
    fetch('/src/data/draws.json')
      .then(response => response.json())
      .then(data => {
        setDraws(data);
        calculateStats(data);
        setIsLoading(false);
      })
      .catch(error => {
        console.error('Error loading data:', error);
        setIsLoading(false);
      });
  }, [setDraws]);

  const calculateStats = (drawData: number[][]) => {
    const stats: NumberStat[] = [];
    for (let num = 1; num <= 70; num++) {
      let hits = 0;
      let lastHit = -1;
      const gaps: number[] = [];

      drawData.forEach((draw, index) => {
        if (draw.includes(num)) {
          hits++;
          if (lastHit !== -1) {
            gaps.push(index - lastHit);
          }
          lastHit = index;
        }
      });

      const out = lastHit === -1 ? drawData.length : drawData.length - 1 - lastHit;
      const avg = gaps.length > 0 ? gaps.reduce((a, b) => a + b, 0) / gaps.length : 0;

      let status = 'normal';
      if (out <= 3) status = 'HOT';
      else if (out >= 15) status = 'COLD';
      else if (out >= 8) status = 'DUE';

      stats.push({
        number: num,
        out,
        hits,
        avg: Math.round(avg * 10) / 10,
        status,
        lastSeen: lastHit
      });
    }
    setNumberStats(stats);
  };

  const generatePowerballCombos = () => {
    const combos: PowerballCombo[] = [];
    const existingCombos = new Set<string>();

    // Create set of existing combinations for uniqueness check
    draws.forEach(draw => {
      const sorted = [...draw].sort((a, b) => a - b);
      existingCombos.add(sorted.join(','));
    });

    let attempts = 0;
    const maxAttempts = 1000;

    while (combos.length < 20 && attempts < maxAttempts) {
      attempts++;
      const combo: number[] = [];

      // Generate based on filter mode
      if (filterMode === 'hot') {
        // Favor hot numbers (out <= 3)
        const hotNumbers = numberStats.filter(n => n.out <= 3).map(n => n.number);
        while (combo.length < 5) {
          const randomIndex = Math.floor(Math.random() * hotNumbers.length);
          const num = hotNumbers[randomIndex];
          if (!combo.includes(num)) combo.push(num);
        }
      } else if (filterMode === 'cold') {
        // Favor cold numbers (out >= 15)
        const coldNumbers = numberStats.filter(n => n.out >= 15).map(n => n.number);
        while (combo.length < 5) {
          const randomIndex = Math.floor(Math.random() * coldNumbers.length);
          const num = coldNumbers[randomIndex];
          if (!combo.includes(num)) combo.push(num);
        }
      } else if (filterMode === 'due') {
        // Favor due numbers (out 8-14)
        const dueNumbers = numberStats.filter(n => n.out >= 8 && n.out <= 14).map(n => n.number);
        while (combo.length < 5) {
          const randomIndex = Math.floor(Math.random() * dueNumbers.length);
          const num = dueNumbers[randomIndex];
          if (!combo.includes(num)) combo.push(num);
        }
      } else {
        // Balanced - mix of all types
        while (combo.length < 5) {
          const num = Math.floor(Math.random() * 70) + 1;
          if (!combo.includes(num)) combo.push(num);
        }
      }

      combo.sort((a, b) => a - b);
      const comboStr = combo.join(',');
      const neverHit = !existingCombos.has(comboStr);

      const sum = combo.reduce((a, b) => a + b, 0);
      const evens = combo.filter(n => n % 2 === 0).length;
      const odds = 5 - evens;
      const highs = combo.filter(n => n > 35).length;
      const lows = 5 - highs;

      const pattern = `${evens}E${odds}O-${highs}H${lows}L`;

      combos.push({
        id: `combo_${combos.length + 1}`,
        numbers: combo,
        sum,
        evens,
        odds,
        highs,
        lows,
        pattern,
        neverHit,
        selected: false
      });
    }

    setGeneratedCombos(combos);
  };

  const toggleComboSelection = (comboId: string) => {
    setSelectedCombos(prev =>
      prev.includes(comboId)
        ? prev.filter(id => id !== comboId)
        : [...prev, comboId]
    );
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'HOT': return 'text-green-400';
      case 'COLD': return 'text-red-400';
      case 'DUE': return 'text-yellow-400';
      default: return 'text-gray-400';
    }
  };

  if (isLoading) {
    return (
      <div className="flex justify-center items-center h-screen text-xl bg-gray-900 text-white">
        Loading Powerball data...
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto p-4 font-mono text-sm bg-gray-900 text-white min-h-screen">
      <h1 className="text-4xl font-bold mb-6 text-center text-blue-400">Powerball Prediction Analyzer</h1>

      {/* Control Panel */}
      <div className="mb-6 p-6 border border-gray-600 rounded bg-gray-800">
        <div className="flex gap-4 justify-center flex-wrap mb-4">
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded border"
          >
            {showFilters ? 'Hide' : 'Show'} Filters
          </button>
          <button
            onClick={generatePowerballCombos}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded border"
          >
            Generate Predictions ({generatedCombos.length})
          </button>
          {selectedCombos.length > 0 && (
            <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded border">
              Purchase Selected ({selectedCombos.length})
            </button>
          )}
        </div>

        {/* Filter Mode Selection */}
        <div className="text-center">
          <label className="mr-4 text-lg">Prediction Mode:</label>
          <select
            value={filterMode}
            onChange={(e) => setFilterMode(e.target.value as 'hot' | 'cold' | 'due' | 'balanced')}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded"
          >
            <option value="balanced">Balanced (All Numbers)</option>
            <option value="hot">Hot Numbers Only</option>
            <option value="cold">Cold Numbers Only</option>
            <option value="due">Due Numbers Only</option>
          </select>
        </div>
      </div>

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Left Side - Numbers 1-70 */}
        <div>
          <h2 className="text-2xl font-bold mb-4 text-center">Current Number Status (1-70)</h2>
          <div className="grid grid-cols-10 gap-1 max-h-96 overflow-y-auto border border-gray-600 p-2 rounded bg-gray-800">
            {numberStats.map(stat => (
              <div key={stat.number} className="p-2 border border-gray-500 text-center bg-gray-700 rounded hover:bg-gray-600 transition-colors">
                <div className="font-bold text-lg">{stat.number}</div>
                <div className="text-xs">Out: {stat.out}</div>
                <div className="text-xs">Hits: {stat.hits}</div>
                <div className={`text-xs font-bold ${getStatusColor(stat.status)}`}>
                  {stat.status}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right Side - Generated Combinations */}
        <div>
          <h2 className="text-2xl font-bold mb-4 text-center">Predicted Combinations</h2>
          <div className="border border-gray-600 p-4 rounded max-h-96 overflow-y-auto bg-gray-800">
            {generatedCombos.length === 0 ? (
              <div className="text-center py-8">
                <p className="text-gray-400 text-lg mb-4">
                  Click "Generate Predictions" to see combinations
                </p>
                <p className="text-sm text-gray-500">
                  Choose your prediction mode above to filter by hot, cold, due, or balanced numbers
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {generatedCombos.map((combo) => (
                  <div
                    key={combo.id}
                    className={`p-3 border rounded cursor-pointer transition-all ${
                      selectedCombos.includes(combo.id)
                        ? 'border-green-400 bg-green-900'
                        : 'border-gray-500 bg-gray-700 hover:bg-gray-600'
                    }`}
                    onClick={() => toggleComboSelection(combo.id)}
                  >
                    <div className="flex justify-between items-center mb-2">
                      <div className="font-bold text-lg">
                        {combo.numbers.join(' - ')}
                      </div>
                      <div className="flex gap-2">
                        {combo.neverHit && (
                          <span className="px-2 py-1 bg-purple-600 text-xs rounded">NEW</span>
                        )}
                        {selectedCombos.includes(combo.id) && (
                          <span className="px-2 py-1 bg-green-600 text-xs rounded">SELECTED</span>
                        )}
                      </div>
                    </div>
                    <div className="text-sm text-gray-300 grid grid-cols-2 gap-2">
                      <div>Sum: {combo.sum}</div>
                      <div>Pattern: {combo.pattern}</div>
                      <div>Evens: {combo.evens}, Odds: {combo.odds}</div>
                      <div>Highs: {combo.highs}, Lows: {combo.lows}</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Filters Panel */}
      {showFilters && (
        <div className="mt-8 p-6 border border-gray-600 rounded bg-gray-800">
          <h3 className="text-xl font-bold mb-4 text-center">Advanced Filters</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm mb-1">Min Sum</label>
              <input type="number" className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded" defaultValue="100" />
            </div>
            <div>
              <label className="block text-sm mb-1">Max Sum</label>
              <input type="number" className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded" defaultValue="250" />
            </div>
            <div>
              <label className="block text-sm mb-1">Even Count</label>
              <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded">
                <option value="">Any</option>
                {[0,1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm mb-1">Odd Count</label>
              <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded">
                <option value="">Any</option>
                {[0,1,2,3,4,5].map(n => <option key={n} value={n}>{n}</option>)}
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Statistics Summary */}
      <div className="mt-8 grid grid-cols-2 md:grid-cols-5 gap-4">
        <div className="p-4 border border-gray-600 rounded bg-gray-800 text-center">
          <div className="text-2xl font-bold text-blue-400">{draws.length}</div>
          <div className="text-sm text-gray-400">Total Draws</div>
        </div>
        <div className="p-4 border border-gray-600 rounded bg-gray-800 text-center">
          <div className="text-2xl font-bold text-green-400">{numberStats.filter(n => n.status === 'HOT').length}</div>
          <div className="text-sm text-gray-400">Hot Numbers</div>
        </div>
        <div className="p-4 border border-gray-600 rounded bg-gray-800 text-center">
          <div className="text-2xl font-bold text-red-400">{numberStats.filter(n => n.status === 'COLD').length}</div>
          <div className="text-sm text-gray-400">Cold Numbers</div>
        </div>
        <div className="p-4 border border-gray-600 rounded bg-gray-800 text-center">
          <div className="text-2xl font-bold text-yellow-400">{numberStats.filter(n => n.status === 'DUE').length}</div>
          <div className="text-sm text-gray-400">Due Numbers</div>
        </div>
        <div className="p-4 border border-gray-600 rounded bg-gray-800 text-center">
          <div className="text-2xl font-bold text-purple-400">{generatedCombos.filter(c => c.neverHit).length}</div>
          <div className="text-sm text-gray-400">New Combos</div>
        </div>
      </div>
    </div>
  );
};

export default SimpleLotteryAnalyzer;
