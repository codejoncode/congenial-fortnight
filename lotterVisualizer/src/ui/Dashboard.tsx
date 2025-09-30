import { useState } from 'react';

const Dashboard = () => {
  const [sortBy, setSortBy] = useState('heatScore');
  const [filterParity, setFilterParity] = useState('all');

  // Mock data for demonstration
  const combos = [
    { combo: [5, 12, 23, 34, 45], heatScore: 85, momentumDelta: 5, totalPoints: 90, lift: 15.2, sum: 119, drawsOut: 3, parity: 'mixed' },
    { combo: [3, 15, 27, 39, 48], heatScore: 78, momentumDelta: -2, totalPoints: 82, lift: 12.8, sum: 132, drawsOut: 7, parity: 'odd' },
    { combo: [8, 19, 31, 42, 55], heatScore: 92, momentumDelta: 8, totalPoints: 95, lift: 18.5, sum: 155, drawsOut: 1, parity: 'mixed' },
    { combo: [2, 14, 26, 38, 50], heatScore: 76, momentumDelta: 3, totalPoints: 88, lift: 14.1, sum: 130, drawsOut: 5, parity: 'even' },
    { combo: [7, 19, 31, 43, 55], heatScore: 89, momentumDelta: 6, totalPoints: 93, lift: 16.7, sum: 155, drawsOut: 2, parity: 'odd' },
  ];

  const filteredCombos = combos.filter(combo => {
    if (filterParity === 'all') return true;
    return combo.parity === filterParity;
  });

  const sortedCombos = [...filteredCombos].sort((a, b) => {
    switch (sortBy) {
      case 'heatScore': return b.heatScore - a.heatScore;
      case 'totalPoints': return b.totalPoints - a.totalPoints;
      case 'lift': return b.lift - a.lift;
      case 'drawsOut': return a.drawsOut - b.drawsOut;
      default: return 0;
    }
  });

  return (
    <div>
      <h2 className="text-2xl mb-4">Dashboard</h2>

      {/* Filters */}
      <div className="mb-4 flex gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Sort By</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-700 border-gray-600 rounded p-2"
          >
            <option value="heatScore">Heat Score</option>
            <option value="totalPoints">Total Points</option>
            <option value="lift">Lift %</option>
            <option value="drawsOut">Draws Out</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Parity</label>
          <select
            value={filterParity}
            onChange={(e) => setFilterParity(e.target.value)}
            className="bg-gray-700 border-gray-600 rounded p-2"
          >
            <option value="all">All</option>
            <option value="even">Even</option>
            <option value="odd">Odd</option>
            <option value="mixed">Mixed</option>
          </select>
        </div>
      </div>

      {/* Top Combinations */}
      <div className="bg-gray-700 p-4 rounded mb-4">
        <h3 className="text-lg mb-2">Top Combinations ({sortedCombos.length})</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-600">
                <th className="text-left p-2">Rank</th>
                <th className="text-left p-2">Combo</th>
                <th className="text-left p-2">Heat Score</th>
                <th className="text-left p-2">Momentum Δ</th>
                <th className="text-left p-2">Total Points</th>
                <th className="text-left p-2">Lift %</th>
                <th className="text-left p-2">Sum</th>
                <th className="text-left p-2">Draws Out</th>
                <th className="text-left p-2">Parity</th>
              </tr>
            </thead>
            <tbody>
              {sortedCombos.slice(0, 10).map((combo, idx) => (
                <tr key={idx} className="border-b border-gray-600 hover:bg-gray-600">
                  <td className="p-2">{idx + 1}</td>
                  <td className="p-2 font-mono">{combo.combo.join('-')}</td>
                  <td className="p-2">{combo.heatScore}</td>
                  <td className="p-2">{combo.momentumDelta > 0 ? `+${combo.momentumDelta}` : combo.momentumDelta}</td>
                  <td className="p-2">{combo.totalPoints}</td>
                  <td className="p-2">{combo.lift}%</td>
                  <td className="p-2">{combo.sum}</td>
                  <td className="p-2">{combo.drawsOut}</td>
                  <td className="p-2">{combo.parity}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Quick Stats */}
      <div className="grid grid-cols-3 gap-4">
        <div className="bg-gray-700 p-4 rounded">
          <h3 className="text-lg mb-2">Quick Stats</h3>
          <p>Total Draws Analyzed: 100</p>
          <p>Average Sum: 125</p>
          <p>Hot Numbers: 12, 23, 34</p>
        </div>
        <div className="bg-gray-700 p-4 rounded">
          <h3 className="text-lg mb-2">Pattern Insights</h3>
          <p>Most Common Sum: 130-140</p>
          <p>Hot Parity: Mixed</p>
          <p>Avg Draws Out: 4.2</p>
        </div>
        <div className="bg-gray-700 p-4 rounded">
          <h3 className="text-lg mb-2">Predictions</h3>
          <p>Next Hot Combo: 8-19-31-42-55</p>
          <p>Confidence: 92%</p>
          <p>Expected Lift: 18.5%</p>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
