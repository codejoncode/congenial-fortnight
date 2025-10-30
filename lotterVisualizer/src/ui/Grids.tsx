import { analyze6thBallDigits, calculate6thBallSkipStats } from '../utils/math';

const Grids = () => {
  // Mock data for demonstration - in real app, this would come from store
  const mockPowerballs = [5, 12, 8, 15, 3, 22, 7, 18, 25, 11];

  const grids = Array.from({ length: 35 }, (_, i) => {
    const num = i + 1;
    const analysis = analyze6thBallDigits(num);
    const skipStats = calculate6thBallSkipStats(mockPowerballs, num);

    return {
      number: num,
      analysis,
      skipStats,
      isHot: skipStats.currentSkip <= 2, // Mock hot/cold logic
    };
  });

  return (
    <div>
      <h2 className="text-2xl mb-4">Grids & Patterns (35 Numbers)</h2>
      <div className="mb-4">
        <p className="text-sm text-gray-400">Color coding: Green = Hot (recent), Red = Cold (due), Gray = Neutral</p>
      </div>
      <div className="grid grid-cols-7 gap-2 overflow-y-auto max-h-screen">
        {grids.map(grid => (
          <div
            key={grid.number}
            className={`p-2 rounded text-center text-xs ${
              grid.isHot ? 'bg-green-600' : grid.skipStats.currentSkip > 5 ? 'bg-red-600' : 'bg-gray-600'
            }`}
          >
            <div className="font-bold">{grid.number}</div>
            <div>Last: {grid.analysis.lastDigit}</div>
            <div>First: {grid.analysis.firstDigit}</div>
            <div>Sum: {grid.analysis.digitSum}</div>
            <div>Skip: {grid.skipStats.currentSkip}</div>
          </div>
        ))}
      </div>

      {/* Digit Pattern Summary */}
      <div className="mt-6">
        <h3 className="text-lg mb-2">Digit Pattern Analysis</h3>
        <div className="grid grid-cols-4 gap-4">
          <div className="bg-gray-700 p-4 rounded">
            <h4 className="font-bold">Last Digits</h4>
            <div className="text-sm">
              {Array.from({ length: 10 }, (_, i) => (
                <span key={i} className="inline-block w-6 text-center">
                  {mockPowerballs.filter(pb => pb % 10 === i).length}
                </span>
              ))}
            </div>
          </div>
          <div className="bg-gray-700 p-4 rounded">
            <h4 className="font-bold">First Digits</h4>
            <div className="text-sm">
              {Array.from({ length: 3 }, (_, i) => (
                <span key={i} className="inline-block w-6 text-center">
                  {mockPowerballs.filter(pb => Math.floor(pb / 10) === i).length}
                </span>
              ))}
            </div>
          </div>
          <div className="bg-gray-700 p-4 rounded">
            <h4 className="font-bold">Digit Sums</h4>
            <div className="text-sm">
              Avg: {Math.round(mockPowerballs.reduce((sum, pb) => sum + (pb % 10 + Math.floor(pb / 10)), 0) / mockPowerballs.length)}
            </div>
          </div>
          <div className="bg-gray-700 p-4 rounded">
            <h4 className="font-bold">Skip Distribution</h4>
            <div className="text-sm">
              Hot: {grids.filter(g => g.isHot).length}<br />
              Cold: {grids.filter(g => g.skipStats.currentSkip > 5).length}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Grids;
