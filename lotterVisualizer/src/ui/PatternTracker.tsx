import React, { useState, useMemo } from 'react';
import { useStore } from '../store';

interface PatternStats {
  pattern: string;
  type: 'highLow' | 'evenOdd' | 'sum' | 'parity' | 'digit';
  occurrences: number;
  lastSeen: number;
  avgGap: number;
  maxGap: number;
  currentGap: number;
  isHot: boolean;
  isCold: boolean;
  hitRate: number;
  nextExpected: number;
}

interface PatternTrackerProps {
  onPatternSelect: (pattern: string) => void;
}

const PatternTracker: React.FC<PatternTrackerProps> = ({ onPatternSelect }) => {
  const { draws } = useStore();
  const [selectedType, setSelectedType] = useState<string>('all');
  const [sortBy, setSortBy] = useState<string>('occurrences');

  // Comprehensive pattern analysis
  const patternStats = useMemo((): PatternStats[] => {
    const stats: PatternStats[] = [];
    const patternMap = new Map<string, { occurrences: number; lastSeen: number; gaps: number[] }>();

    draws.forEach((draw, index) => {
      const sorted = [...draw].sort((a,b) => a-b);
      const sum = sorted.reduce((a,b) => a+b, 0);
      const evens = sorted.filter(n => n % 2 === 0).length;
      const highs = sorted.filter(n => n > 35).length;

      // High/Low patterns (HHHHH, LLLLL, etc.)
      const hlPattern = sorted.map(n => n > 35 ? 'H' : 'L').join('');
      updatePattern(`HL_${hlPattern}`, index, patternMap);

      // Even/Odd patterns (EEOOO, OOOEE, etc.)
      const eoPattern = sorted.map(n => n % 2 === 0 ? 'E' : 'O').join('');
      updatePattern(`EO_${eoPattern}`, index, patternMap);

      // Sum ranges
      const sumRange = Math.floor(sum / 25) * 25;
      updatePattern(`SUM_${sumRange}-${sumRange+24}`, index, patternMap);

      // Parity counts
      updatePattern(`PARITY_${evens}E${5-evens}O`, index, patternMap);

      // High/Low counts
      updatePattern(`HLCOUNT_${highs}H${5-highs}L`, index, patternMap);
    });

    // Convert to stats
    patternMap.forEach((data, pattern) => {
      const gaps = data.gaps;
      const avgGap = gaps.length > 0 ? gaps.reduce((a,b) => a+b, 0) / gaps.length : draws.length;
      const maxGap = Math.max(...gaps, 0);
      const currentGap = draws.length - 1 - data.lastSeen;
      const hitRate = data.occurrences / draws.length;

      // Determine hot/cold status
      const isHot = currentGap <= Math.max(2, avgGap * 0.5);
      const isCold = currentGap >= Math.max(8, avgGap * 1.5);

      // Pattern type
      let type: PatternStats['type'] = 'highLow';
      if (pattern.startsWith('EO_')) type = 'evenOdd';
      else if (pattern.startsWith('SUM_')) type = 'sum';
      else if (pattern.startsWith('PARITY_')) type = 'parity';
      else if (pattern.startsWith('HLCOUNT_')) type = 'digit';

      stats.push({
        pattern: pattern.replace(/^(HL_|EO_|SUM_|PARITY_|HLCOUNT_)/, ''),
        type,
        occurrences: data.occurrences,
        lastSeen: data.lastSeen,
        avgGap,
        maxGap,
        currentGap,
        isHot,
        isCold,
        hitRate,
        nextExpected: Math.max(0, avgGap - currentGap)
      });
    });

    return stats;
  }, [draws]);

  const updatePattern = (pattern: string, index: number, patternMap: Map<string, { occurrences: number; lastSeen: number; gaps: number[] }>) => {
    if (!patternMap.has(pattern)) {
      patternMap.set(pattern, { occurrences: 0, lastSeen: -1, gaps: [] });
    }

    const data = patternMap.get(pattern)!;
    data.occurrences += 1;

    if (data.lastSeen !== -1) {
      data.gaps.push(index - data.lastSeen);
    }
    data.lastSeen = index;
  };

  // Filter and sort patterns
  const filteredPatterns = useMemo(() => {
    let filtered = patternStats;

    if (selectedType !== 'all') {
      filtered = filtered.filter(p => p.type === selectedType);
    }

    return filtered.sort((a, b) => {
      switch (sortBy) {
        case 'occurrences': return b.occurrences - a.occurrences;
        case 'lastSeen': return a.lastSeen - b.lastSeen;
        case 'currentGap': return b.currentGap - a.currentGap;
        case 'hitRate': return b.hitRate - a.hitRate;
        default: return 0;
      }
    });
  }, [patternStats, selectedType, sortBy]);

  const getPatternColor = (stats: PatternStats): string => {
    if (stats.isHot) return 'bg-green-600 hover:bg-green-500';
    if (stats.isCold) return 'bg-red-600 hover:bg-red-500';
    return 'bg-gray-600 hover:bg-gray-500';
  };

  const getTypeLabel = (type: string): string => {
    switch (type) {
      case 'highLow': return 'High/Low';
      case 'evenOdd': return 'Even/Odd';
      case 'sum': return 'Sum Range';
      case 'parity': return 'Parity Count';
      case 'digit': return 'HL Count';
      default: return type;
    }
  };

  return (
    <div className="p-6">
      <h2 className="text-3xl font-bold mb-6">Pattern Analysis & Tracking</h2>

      {/* Controls */}
      <div className="mb-6 flex gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Pattern Type</label>
          <select
            value={selectedType}
            onChange={(e) => setSelectedType(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded p-2"
          >
            <option value="all">All Types</option>
            <option value="highLow">High/Low Patterns</option>
            <option value="evenOdd">Even/Odd Patterns</option>
            <option value="sum">Sum Ranges</option>
            <option value="parity">Parity Counts</option>
            <option value="digit">High/Low Counts</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium mb-1">Sort By</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
            className="bg-gray-700 border border-gray-600 rounded p-2"
          >
            <option value="occurrences">Most Common</option>
            <option value="lastSeen">Recently Seen</option>
            <option value="currentGap">Longest Gap</option>
            <option value="hitRate">Highest Hit Rate</option>
          </select>
        </div>
      </div>

      {/* Pattern Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {filteredPatterns.slice(0, 24).map((stats, index) => (
          <div
            key={`${stats.type}_${stats.pattern}_${index}`}
            onClick={() => onPatternSelect(stats.pattern)}
            className={`
              p-4 rounded-lg cursor-pointer transition-all duration-200
              ${getPatternColor(stats)}
            `}
          >
            <div className="font-bold text-lg mb-2">{stats.pattern}</div>
            <div className="text-sm space-y-1">
              <div>Type: {getTypeLabel(stats.type)}</div>
              <div>Occurrences: {stats.occurrences}</div>
              <div>Last Seen: {stats.lastSeen} draws ago</div>
              <div>Current Gap: {stats.currentGap}</div>
              <div>Avg Gap: {stats.avgGap.toFixed(1)}</div>
              <div>Hit Rate: {(stats.hitRate * 100).toFixed(1)}%</div>
              {stats.nextExpected > 0 && (
                <div className="text-yellow-300">Due in: {stats.nextExpected.toFixed(1)} draws</div>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Pattern Insights */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-xl font-semibold mb-3">Hot Patterns</h3>
          <div className="space-y-2">
            {patternStats.filter(p => p.isHot).slice(0, 5).map((p, i) => (
              <div key={i} className="text-sm">
                <span className="font-semibold">{p.pattern}</span>
                <span className="text-gray-400 ml-2">({p.occurrences} times)</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-xl font-semibold mb-3">Cold Patterns</h3>
          <div className="space-y-2">
            {patternStats.filter(p => p.isCold).slice(0, 5).map((p, i) => (
              <div key={i} className="text-sm">
                <span className="font-semibold">{p.pattern}</span>
                <span className="text-gray-400 ml-2">({p.currentGap} gap)</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-gray-800 p-4 rounded">
          <h3 className="text-xl font-semibold mb-3">Pattern Trends</h3>
          <div className="space-y-2 text-sm">
            <div>Total Patterns: {patternStats.length}</div>
            <div>Most Common: {patternStats[0]?.pattern || 'N/A'}</div>
            <div>Avg Pattern Gap: {patternStats.reduce((sum, p) => sum + p.avgGap, 0) / patternStats.length || 0}</div>
            <div>Patterns Due: {patternStats.filter(p => p.nextExpected > 0).length}</div>
          </div>
        </div>
      </div>

      {/* Recent Pattern History */}
      <div className="mt-8">
        <h3 className="text-2xl font-semibold mb-4">Recent Pattern History</h3>
        <div className="bg-gray-800 p-4 rounded max-h-64 overflow-y-auto">
          <div className="space-y-2">
            {draws.slice(-10).reverse().map((draw, index) => {
              const sorted = [...draw].sort((a,b) => a-b);
              const sum = sorted.reduce((a,b) => a+b, 0);
              const hlPattern = sorted.map(n => n > 35 ? 'H' : 'L').join('');
              const eoPattern = sorted.map(n => n % 2 === 0 ? 'E' : 'O').join('');
              const evens = sorted.filter(n => n % 2 === 0).length;

              return (
                <div key={index} className="flex justify-between items-center py-2 border-b border-gray-700">
                  <div className="font-mono">
                    Draw #{draws.length - index}: [{sorted.join(', ')}]
                  </div>
                  <div className="text-sm text-gray-400">
                    Sum: {sum} | HL: {hlPattern} | EO: {eoPattern} | {evens}E{(5-evens)}O
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PatternTracker;
