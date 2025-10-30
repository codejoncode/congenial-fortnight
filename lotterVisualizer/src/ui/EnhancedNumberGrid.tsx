import React, { useState, useMemo } from 'react';
import { useStore } from '../store';

// Enhanced number analysis utilities
const getLastDigit = (num: number): number => num % 10;
const getFirstDigit = (num: number): number => Math.floor(num / 10);
const getDigitSum = (num: number): number => {
  return num.toString().split('').reduce((sum, digit) => sum + parseInt(digit), 0);
};

const isHigh = (num: number): boolean => num > 35;
const isLow = (num: number): boolean => num <= 35;
const isEven = (num: number): boolean => num % 2 === 0;
const isOdd = (num: number): boolean => num % 2 !== 0;

interface NumberStats {
  number: number;
  drawsOut: number;
  totalHits: number;
  lastHitIndex: number;
  avgSkip: number;
  maxSkip: number;
  currentSkip: number;
  isHot: boolean;
  isCold: boolean;
  isDue: boolean;
  lastDigit: number;
  firstDigit: number;
  digitSum: number;
  isHigh: boolean;
  isLow: boolean;
  isEven: boolean;
  isOdd: boolean;
  pattern: string; // H/L + E/O + digit pattern
}

const EnhancedNumberGrid: React.FC = () => {
  const { draws } = useStore();
  const [selectedNumbers, setSelectedNumbers] = useState<Set<number>>(new Set());
  const [filterMode, setFilterMode] = useState<'include' | 'exclude'>('include');

  // Calculate comprehensive stats for each number
  const numberStats = useMemo((): NumberStats[] => {
    const stats: NumberStats[] = [];

    for (let num = 1; num <= 70; num++) {
      let totalHits = 0;
      let lastHitIndex = -1;
      const skips: number[] = [];
      let currentSkip = 0;

      // Analyze all draws
      draws.forEach((draw, drawIndex) => {
        if (draw.includes(num)) {
          totalHits++;
          if (lastHitIndex !== -1) {
            skips.push(drawIndex - lastHitIndex);
          }
          lastHitIndex = drawIndex;
        }
      });

      // Calculate current skip (draws since last hit)
      if (lastHitIndex !== -1) {
        currentSkip = draws.length - 1 - lastHitIndex;
      } else {
        currentSkip = draws.length; // Never hit
      }

      // Calculate statistics
      const avgSkip = skips.length > 0 ? skips.reduce((a, b) => a + b, 0) / skips.length : draws.length;
      const maxSkip = Math.max(...skips, 0);

      // Determine hot/cold/due status
      const isHot = currentSkip <= Math.max(2, avgSkip * 0.5);
      const isCold = currentSkip >= Math.max(10, avgSkip * 2);
      const isDue = !isHot && !isCold && currentSkip > avgSkip;

      // Pattern analysis
      const lastDigit = getLastDigit(num);
      const firstDigit = getFirstDigit(num);
      const digitSum = getDigitSum(num);
      const high = isHigh(num);
      const low = isLow(num);
      const even = isEven(num);
      const odd = isOdd(num);

      const pattern = `${high ? 'H' : 'L'}${even ? 'E' : 'O'}${lastDigit}`;

      stats.push({
        number: num,
        drawsOut: currentSkip,
        totalHits,
        lastHitIndex,
        avgSkip,
        maxSkip,
        currentSkip,
        isHot,
        isCold,
        isDue,
        lastDigit,
        firstDigit,
        digitSum,
        isHigh: high,
        isLow: low,
        isEven: even,
        isOdd: odd,
        pattern
      });
    }

    return stats;
  }, [draws]);

  // Color coding function
  const getNumberColor = (stats: NumberStats): string => {
    if (stats.isHot) return 'bg-green-600 hover:bg-green-500';
    if (stats.isCold) return 'bg-red-600 hover:bg-red-500';
    if (stats.isDue) return 'bg-yellow-600 hover:bg-yellow-500';
    return 'bg-gray-600 hover:bg-gray-500';
  };

  const toggleNumberSelection = (num: number) => {
    const newSelection = new Set(selectedNumbers);
    if (newSelection.has(num)) {
      newSelection.delete(num);
    } else {
      newSelection.add(num);
    }
    setSelectedNumbers(newSelection);
  };

  return (
    <div className="p-6">
      <div className="mb-6">
        <h2 className="text-3xl font-bold mb-4">Complete Number Grid (1-70)</h2>

        {/* Legend */}
        <div className="mb-4 p-4 bg-gray-800 rounded">
          <h3 className="text-lg font-semibold mb-2">Color Legend</h3>
          <div className="flex flex-wrap gap-4 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-green-600 rounded"></div>
              <span>Hot (Recent hits)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-red-600 rounded"></div>
              <span>Cold (Long time no hit)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-yellow-600 rounded"></div>
              <span>Due (Above average skip)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-4 h-4 bg-gray-600 rounded"></div>
              <span>Neutral</span>
            </div>
          </div>
        </div>

        {/* Selection Controls */}
        <div className="mb-4 p-4 bg-gray-800 rounded">
          <h3 className="text-lg font-semibold mb-2">Custom List Builder</h3>
          <div className="flex gap-4 mb-2">
            <button
              onClick={() => setFilterMode('include')}
              className={`px-4 py-2 rounded ${filterMode === 'include' ? 'bg-blue-600' : 'bg-gray-600'}`}
            >
              Include Selected
            </button>
            <button
              onClick={() => setFilterMode('exclude')}
              className={`px-4 py-2 rounded ${filterMode === 'exclude' ? 'bg-blue-600' : 'bg-gray-600'}`}
            >
              Exclude Selected
            </button>
            <button
              onClick={() => setSelectedNumbers(new Set())}
              className="px-4 py-2 bg-gray-600 hover:bg-gray-500 rounded"
            >
              Clear Selection
            </button>
          </div>
          <p className="text-sm text-gray-400">
            Selected: {Array.from(selectedNumbers).sort((a,b)=>a-b).join(', ')} ({selectedNumbers.size} numbers)
          </p>
        </div>
      </div>

      {/* Number Grid */}
      <div className="grid grid-cols-10 gap-2 max-h-screen overflow-y-auto">
        {numberStats.map(stats => (
          <div
            key={stats.number}
            onClick={() => toggleNumberSelection(stats.number)}
            className={`
              p-3 rounded-lg text-center text-xs cursor-pointer transition-all duration-200
              ${getNumberColor(stats)}
              ${selectedNumbers.has(stats.number) ? 'ring-2 ring-blue-400 ring-opacity-75' : ''}
            `}
          >
            <div className="font-bold text-lg mb-1">{stats.number}</div>

            {/* Core Stats */}
            <div className="space-y-1">
              <div>Out: {stats.drawsOut}</div>
              <div>Hits: {stats.totalHits}</div>
              <div>Avg: {stats.avgSkip.toFixed(1)}</div>
              <div>Max: {stats.maxSkip}</div>
            </div>

            {/* Pattern Info */}
            <div className="mt-2 pt-2 border-t border-gray-500 text-xs">
              <div>{stats.pattern}</div>
              <div>Sum: {stats.digitSum}</div>
            </div>

            {/* Hover tooltip would show more details */}
            <div className="absolute invisible group-hover:visible bg-black p-2 rounded text-xs z-10">
              Last hit: Draw #{stats.lastHitIndex + 1}<br/>
              Pattern: {stats.isHigh ? 'High' : 'Low'}, {stats.isEven ? 'Even' : 'Odd'}<br/>
              Digit: {stats.firstDigit}{stats.lastDigit}
            </div>
          </div>
        ))}
      </div>

      {/* Summary Stats */}
      <div className="mt-6 grid grid-cols-4 gap-4">
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="font-bold">Hot Numbers</h3>
          <p>{numberStats.filter(s => s.isHot).length} numbers</p>
          <p className="text-sm">{numberStats.filter(s => s.isHot).slice(0,5).map(s => s.number).join(', ')}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="font-bold">Cold Numbers</h3>
          <p>{numberStats.filter(s => s.isCold).length} numbers</p>
          <p className="text-sm">{numberStats.filter(s => s.isCold).slice(0,5).map(s => s.number).join(', ')}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="font-bold">Due Numbers</h3>
          <p>{numberStats.filter(s => s.isDue).length} numbers</p>
          <p className="text-sm">{numberStats.filter(s => s.isDue).slice(0,5).map(s => s.number).join(', ')}</p>
        </div>
        <div className="bg-gray-800 p-4 rounded">
          <h3 className="font-bold">Never Hit</h3>
          <p>{numberStats.filter(s => s.totalHits === 0).length} numbers</p>
          <p className="text-sm">{numberStats.filter(s => s.totalHits === 0).slice(0,5).map(s => s.number).join(', ')}</p>
        </div>
      </div>
    </div>
  );
};

export default EnhancedNumberGrid;
