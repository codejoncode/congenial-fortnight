import React from 'react';
import { useStore } from '../store';

const Controls: React.FC = () => {
  const { filters, setFilters, combos } = useStore();

  const updateFilter = (key: keyof typeof filters, value: unknown) => {
    setFilters({ ...filters, [key]: value });
  };

  const exportShortlist = () => {
    const shortlist = combos.slice(0, 20).map(combo => ({
      combo: combo.combo,
      heatScore: combo.heatScore,
      confidence: combo.confidence,
      totalPoints: combo.totalPoints,
      lift: combo.lift,
      momentumDelta: combo.momentumDelta,
      drawsOut: combo.history.drawsOut
    }));

    const dataStr = JSON.stringify(shortlist, null, 2);
    const dataBlob = new Blob([dataStr], { type: 'application/json' });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement('a');
    link.href = url;
    link.download = `apexscoop-shortlist-${new Date().toISOString().split('T')[0]}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="controls">
      <div className="top-bar">
        <select value="Live" onChange={() => {}}>
          <option>Live</option>
          <option>Historical</option>
        </select>
        <div className="filters">
          <label>
            Min Heat:
            <input
              type="range"
              min="0"
              max="100"
              value={filters.minHeat || 0}
              onChange={(e) => updateFilter('minHeat', +e.target.value)}
            />
          </label>
          <label>
            Min Points:
            <input
              type="range"
              min="0"
              max="50"
              value={filters.minPoints || 0}
              onChange={(e) => updateFilter('minPoints', +e.target.value)}
            />
          </label>
          <label>
            <input
              type="checkbox"
              checked={filters.regimeMatch || false}
              onChange={(e) => updateFilter('regimeMatch', e.target.checked)}
            />
            Regime match
          </label>
          <label>
            <input
              type="checkbox"
              checked={filters.positiveMomentum || false}
              onChange={(e) => updateFilter('positiveMomentum', e.target.checked)}
            />
            Positive momentum
          </label>
          <label>
            <input
              type="checkbox"
              checked={filters.passesRecurrence || false}
              onChange={(e) => updateFilter('passesRecurrence', e.target.checked)}
            />
            Passes recurrence
          </label>
        </div>
        <select value="Heat Score" onChange={(e) => updateFilter('sortBy', e.target.value)}>
          <option>Heat Score</option>
          <option>Confidence</option>
          <option>Total Points</option>
          <option>Lift %</option>
        </select>
        <button onClick={exportShortlist} className="export-btn">
          Export Shortlist
        </button>
      </div>
    </div>
  );
};

export default Controls;
