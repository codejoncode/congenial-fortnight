import { useState } from 'react';

const Filters = () => {
  const [parity, setParity] = useState('all');
  const [sumMin, setSumMin] = useState(100);
  const [sumMax, setSumMax] = useState(200);
  const [evenCount, setEvenCount] = useState('');
  const [oddCount, setOddCount] = useState('');

  const applyFilters = () => {
    console.log('Applying filters:', { parity, sumMin, sumMax, evenCount, oddCount });
    // In real app, this would update the store
  };

  const resetFilters = () => {
    setParity('all');
    setSumMin(100);
    setSumMax(200);
    setEvenCount('');
    setOddCount('');
  };

  return (
    <div>
      <h2 className="text-2xl mb-4">Filters & Controls</h2>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Parity Filter */}
        <div className="bg-gray-700 p-4 rounded">
          <h3 className="text-lg mb-3">Parity</h3>
          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="radio"
                name="parity"
                value="all"
                checked={parity === 'all'}
                onChange={(e) => setParity(e.target.value)}
                className="mr-2"
              />
              All
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="parity"
                value="even"
                checked={parity === 'even'}
                onChange={(e) => setParity(e.target.value)}
                className="mr-2"
              />
              Even Dominant
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="parity"
                value="odd"
                checked={parity === 'odd'}
                onChange={(e) => setParity(e.target.value)}
                className="mr-2"
              />
              Odd Dominant
            </label>
            <label className="flex items-center">
              <input
                type="radio"
                name="parity"
                value="mixed"
                checked={parity === 'mixed'}
                onChange={(e) => setParity(e.target.value)}
                className="mr-2"
              />
              Mixed
            </label>
          </div>
        </div>

        {/* Sum Range Filter */}
        <div className="bg-gray-700 p-4 rounded">
          <h3 className="text-lg mb-3">Sum Range</h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm mb-1">Minimum Sum: {sumMin}</label>
              <input
                type="range"
                min="50"
                max="250"
                value={sumMin}
                onChange={(e) => setSumMin(Number(e.target.value))}
                className="w-full"
              />
            </div>
            <div>
              <label className="block text-sm mb-1">Maximum Sum: {sumMax}</label>
              <input
                type="range"
                min="50"
                max="250"
                value={sumMax}
                onChange={(e) => setSumMax(Number(e.target.value))}
                className="w-full"
              />
            </div>
          </div>
        </div>

        {/* Even/Odd Count Filter */}
        <div className="bg-gray-700 p-4 rounded">
          <h3 className="text-lg mb-3">Even/Odd Balance</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm mb-1">Even Count</label>
              <select
                value={evenCount}
                onChange={(e) => setEvenCount(e.target.value)}
                className="w-full bg-gray-600 border-gray-500 rounded p-2"
              >
                <option value="">Any</option>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
              </select>
            </div>
            <div>
              <label className="block text-sm mb-1">Odd Count</label>
              <select
                value={oddCount}
                onChange={(e) => setOddCount(e.target.value)}
                className="w-full bg-gray-600 border-gray-500 rounded p-2"
              >
                <option value="">Any</option>
                <option value="0">0</option>
                <option value="1">1</option>
                <option value="2">2</option>
                <option value="3">3</option>
                <option value="4">4</option>
                <option value="5">5</option>
              </select>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="bg-gray-700 p-4 rounded">
          <h3 className="text-lg mb-3">Actions</h3>
          <div className="space-y-2">
            <button
              onClick={applyFilters}
              className="w-full bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
            >
              Apply Filters
            </button>
            <button
              onClick={resetFilters}
              className="w-full bg-gray-600 hover:bg-gray-500 px-4 py-2 rounded"
            >
              Reset Filters
            </button>
          </div>
          <div className="mt-4 text-sm text-gray-400">
            <p>Active Filters:</p>
            <p>Parity: {parity}</p>
            <p>Sum: {sumMin} - {sumMax}</p>
            <p>Even: {evenCount || 'Any'}, Odd: {oddCount || 'Any'}</p>
          </div>
        </div>
      </div>

      {/* Filter Presets */}
      <div className="mt-6 bg-gray-700 p-4 rounded">
        <h3 className="text-lg mb-3">Quick Presets</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          <button
            onClick={() => { setParity('even'); setSumMin(120); setSumMax(160); }}
            className="bg-green-600 hover:bg-green-700 px-3 py-2 rounded text-sm"
          >
            Hot Even
          </button>
          <button
            onClick={() => { setParity('odd'); setSumMin(100); setSumMax(140); }}
            className="bg-purple-600 hover:bg-purple-700 px-3 py-2 rounded text-sm"
          >
            Lucky Odd
          </button>
          <button
            onClick={() => { setParity('mixed'); setSumMin(130); setSumMax(170); }}
            className="bg-yellow-600 hover:bg-yellow-700 px-3 py-2 rounded text-sm"
          >
            Balanced Mix
          </button>
          <button
            onClick={() => { setParity('all'); setSumMin(80); setSumMax(200); }}
            className="bg-red-600 hover:bg-red-700 px-3 py-2 rounded text-sm"
          >
            All Options
          </button>
        </div>
      </div>
    </div>
  );
};

export default Filters;
