import { analyze6thBallDigits } from '../utils/math';

const LastDraws = () => {
  // Mock last draws data
  const draws = [
    { date: '2025-09-09', numbers: [5, 12, 23, 34, 45], powerball: 7, jackpot: '$150M' },
    { date: '2025-09-06', numbers: [3, 15, 27, 39, 48], powerball: 12, jackpot: '$130M' },
    { date: '2025-09-02', numbers: [8, 19, 31, 42, 55], powerball: 3, jackpot: '$110M' },
    { date: '2025-08-30', numbers: [11, 22, 33, 44, 55], powerball: 18, jackpot: '$90M' },
    { date: '2025-08-26', numbers: [2, 14, 26, 38, 50], powerball: 25, jackpot: '$70M' },
  ];

  const powerballAnalysis = draws.map(draw => ({
    ...draw,
    analysis: analyze6thBallDigits(draw.powerball)
  }));

  return (
    <div>
      <h2 className="text-2xl mb-4">Last Draws & Powerball Analysis</h2>

      {/* Summary Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-700 p-4 rounded text-center">
          <h3 className="text-lg font-bold">Total Draws</h3>
          <p className="text-2xl">{draws.length}</p>
        </div>
        <div className="bg-gray-700 p-4 rounded text-center">
          <h3 className="text-lg font-bold">Avg Powerball</h3>
          <p className="text-2xl">{Math.round(powerballAnalysis.reduce((sum, d) => sum + d.powerball, 0) / draws.length)}</p>
        </div>
        <div className="bg-gray-700 p-4 rounded text-center">
          <h3 className="text-lg font-bold">Most Common Last Digit</h3>
          <p className="text-2xl">
            {powerballAnalysis
              .map(d => d.analysis.lastDigit)
              .sort((a, b) => powerballAnalysis.filter(d => d.analysis.lastDigit === a).length - powerballAnalysis.filter(d => d.analysis.lastDigit === b).length)
              .pop()}
          </p>
        </div>
        <div className="bg-gray-700 p-4 rounded text-center">
          <h3 className="text-lg font-bold">Avg Jackpot</h3>
          <p className="text-xl">$110M</p>
        </div>
      </div>

      {/* Detailed Draw List */}
      <div className="space-y-4">
        {powerballAnalysis.map((draw, index) => (
          <div key={index} className="bg-gray-700 p-4 rounded">
            <div className="flex justify-between items-start mb-2">
              <div>
                <h3 className="text-lg font-bold">{draw.date}</h3>
                <p className="text-sm text-gray-400">Jackpot: {draw.jackpot}</p>
              </div>
              <div className="text-right">
                <p className="text-sm">Powerball Analysis</p>
                <p className="text-xs">Last: {draw.analysis.lastDigit} | First: {draw.analysis.firstDigit}</p>
                <p className="text-xs">Sum: {draw.analysis.digitSum} | {draw.analysis.sumParity}</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h4 className="font-semibold mb-2">Main Numbers</h4>
                <div className="flex gap-2">
                  {draw.numbers.map((num, idx) => (
                    <span
                      key={idx}
                      className={`inline-block w-8 h-8 rounded-full text-center leading-8 text-xs font-bold ${
                        num % 2 === 0 ? 'bg-blue-600' : 'bg-purple-600'
                      }`}
                    >
                      {num}
                    </span>
                  ))}
                </div>
                <p className="text-sm mt-1">Sum: {draw.numbers.reduce((a, b) => a + b, 0)}</p>
              </div>

              <div>
                <h4 className="font-semibold mb-2">Powerball</h4>
                <div className="flex items-center gap-4">
                  <span className="inline-block w-12 h-12 bg-red-600 rounded-full text-center leading-12 text-lg font-bold text-white">
                    {draw.powerball}
                  </span>
                  <div className="text-sm">
                    <p>Pattern: {draw.analysis.digitPattern}</p>
                    <p>Division: {draw.analysis.quotient} ÷ {draw.analysis.remainder}</p>
                    <p>Parity: {draw.analysis.isEvenSum ? 'Even' : 'Odd'} Sum</p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Pattern Insights */}
      <div className="mt-6 bg-gray-700 p-4 rounded">
        <h3 className="text-lg mb-3">Pattern Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <h4 className="font-semibold mb-2">Last Digit Trends</h4>
            <div className="text-sm space-y-1">
              {Array.from({ length: 10 }, (_, i) => {
                const count = powerballAnalysis.filter(d => d.analysis.lastDigit === i).length;
                return (
                  <div key={i} className="flex justify-between">
                    <span>{i}:</span>
                    <span>{count} times</span>
                  </div>
                );
              })}
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2">Sum Distribution</h4>
            <div className="text-sm space-y-1">
              {Array.from({ length: 18 }, (_, i) => i + 1).map(sum => {
                const count = powerballAnalysis.filter(d => d.analysis.digitSum === sum).length;
                if (count > 0) {
                  return (
                    <div key={sum} className="flex justify-between">
                      <span>Sum {sum}:</span>
                      <span>{count} times</span>
                    </div>
                  );
                }
                return null;
              })}
            </div>
          </div>

          <div>
            <h4 className="font-semibold mb-2">Parity Balance</h4>
            <div className="text-sm space-y-1">
              <div className="flex justify-between">
                <span>Even Sum:</span>
                <span>{powerballAnalysis.filter(d => d.analysis.isEvenSum).length} draws</span>
              </div>
              <div className="flex justify-between">
                <span>Odd Sum:</span>
                <span>{powerballAnalysis.filter(d => !d.analysis.isEvenSum).length} draws</span>
              </div>
              <div className="flex justify-between">
                <span>Even Last:</span>
                <span>{powerballAnalysis.filter(d => d.analysis.isEvenRemainder).length} draws</span>
              </div>
              <div className="flex justify-between">
                <span>Odd Last:</span>
                <span>{powerballAnalysis.filter(d => !d.analysis.isEvenRemainder).length} draws</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LastDraws;
