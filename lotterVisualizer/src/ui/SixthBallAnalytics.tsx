import React from 'react';
import { analyze6thBallDigits, predict6thBallCandidates, get6thBallHotCold, calculate6thBallSkipStats, analyze6thBallDigitGroups } from '../utils/math';

interface SixthBallAnalyticsProps {
  powerballs: number[];
  recentDraws?: number[];
}

const SixthBallAnalytics: React.FC<SixthBallAnalyticsProps> = ({ powerballs, recentDraws = [] }) => {
  const predictions = predict6thBallCandidates(powerballs, recentDraws, 5);
  const { hot, cold } = get6thBallHotCold(powerballs, 5);
  const digitGroups = React.useMemo(() => {
    if (powerballs.length === 0) return null;
    return analyze6thBallDigitGroups(powerballs);
  }, [powerballs]);

  const digitAnalysis = React.useMemo(() => {
    if (powerballs.length === 0) return null;

    const last10 = powerballs.slice(-10);
    const digitStats = last10.map(pb => analyze6thBallDigits(pb));

    const lastDigitFreq: Record<number, number> = {};
    const firstDigitFreq: Record<number, number> = {};
    const sumFreq: Record<number, number> = {};

    digitStats.forEach(analysis => {
      lastDigitFreq[analysis.lastDigit] = (lastDigitFreq[analysis.lastDigit] || 0) + 1;
      firstDigitFreq[analysis.firstDigit] = (firstDigitFreq[analysis.firstDigit] || 0) + 1;
      sumFreq[analysis.digitSum] = (sumFreq[analysis.digitSum] || 0) + 1;
    });

    return { lastDigitFreq, firstDigitFreq, sumFreq, digitStats };
  }, [powerballs]);

  if (!digitAnalysis) return null;

  return (
    <div className="sixth-ball-analytics">
      <h3>6th Ball Analytics</h3>

      <div className="analytics-grid">
        <div className="predictions-section">
          <h4>Top Predictions</h4>
          <div className="predictions-list">
            {predictions.map((pred) => {
              const skipStats = calculate6thBallSkipStats(powerballs, pred.number);
              return (
                <div key={pred.number} className="prediction-item">
                  <span className="number">{pred.number}</span>
                  <span className="score">Score: {pred.score}</span>
                  <div className="skip-info">
                    <span>Current Skip: {skipStats.currentSkip}</span>
                    <span>Avg Skip: {skipStats.averageSkip.toFixed(1)}</span>
                  </div>
                  <div className="reasons">
                    {pred.reasons.map((reason, i) => (
                      <span key={i} className="reason">{reason}</span>
                    ))}
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        <div className="digit-analysis-section">
          <h4>Digit Analysis (Last 10 Draws)</h4>

          <div className="digit-stats">
            <div className="stat-group">
              <h5>Last Digit Frequency</h5>
              <div className="frequency-bars">
                {Object.entries(digitAnalysis.lastDigitFreq)
                  .sort(([,a], [,b]) => (b as number) - (a as number))
                  .slice(0, 5)
                  .map(([digit, count]) => (
                    <div key={digit} className="frequency-bar">
                      <span className="digit">{digit}</span>
                      <div
                        className="bar"
                        style={{ width: `${((count as number) / 10) * 100}%` }}
                      ></div>
                      <span className="count">{count}</span>
                    </div>
                  ))}
              </div>
            </div>

            <div className="stat-group">
              <h5>Digit Sum Distribution</h5>
              <div className="frequency-bars">
                {Object.entries(digitAnalysis.sumFreq)
                  .sort(([,a], [,b]) => (b as number) - (a as number))
                  .map(([sum, count]) => (
                    <div key={sum} className="frequency-bar">
                      <span className="digit">{sum}</span>
                      <div
                        className="bar"
                        style={{ width: `${((count as number) / 10) * 100}%` }}
                      ></div>
                      <span className="count">{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          </div>
        </div>

        <div className="pattern-analysis-section">
          <h4>Pattern Insights</h4>
          <div className="insights">
            <div className="insight">
              <span className="label">Even Sum Ratio:</span>
              <span className="value">
                {((digitAnalysis.digitStats.filter(d => d.isEvenSum).length / digitAnalysis.digitStats.length) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="insight">
              <span className="label">Avg Digit Sum:</span>
              <span className="value">
                {(digitAnalysis.digitStats.reduce((sum, d) => sum + d.digitSum, 0) / digitAnalysis.digitStats.length).toFixed(1)}
              </span>
            </div>
            <div className="insight">
              <span className="label">Most Common Last Digit:</span>
              <span className="value">
                {Object.entries(digitAnalysis.lastDigitFreq)
                  .sort(([,a], [,b]) => (b as number) - (a as number))[0][0]}
              </span>
            </div>
          </div>
        </div>

        <div className="hot-cold-section">
          <h4>Hot & Cold Numbers</h4>
          <div className="hot-cold-grid">
            <div className="hot-numbers">
              <h5>Hottest (Most Frequent)</h5>
              <div className="number-list">
                {hot.map(({ number, frequency }) => (
                  <div key={number} className="number-item hot">
                    <span className="number">{number}</span>
                    <span className="freq">{frequency} times</span>
                  </div>
                ))}
              </div>
            </div>
            <div className="cold-numbers">
              <h5>Coldest (Least Frequent)</h5>
              <div className="number-list">
                {cold.map(({ number, frequency }) => (
                  <div key={number} className="number-item cold">
                    <span className="number">{number}</span>
                    <span className="freq">{frequency} times</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>

        {digitGroups && (
          <div className="digit-groups-section">
            <h4>Digit Group Skip Analysis</h4>
            <div className="digit-groups-grid">
              <div className="group">
                <h5>Last Digit Skips</h5>
                <div className="skip-list">
                  {Object.entries(digitGroups.lastDigitSkips)
                    .sort(([,a], [,b]) => b.currentSkip - a.currentSkip)
                    .slice(0, 5)
                    .map(([digit, stats]) => (
                      <div key={digit} className="skip-item">
                        <span>Digit {digit}:</span>
                        <span>Current: {stats.currentSkip}, Avg: {stats.averageSkip.toFixed(1)}</span>
                      </div>
                    ))}
                </div>
              </div>
              <div className="group">
                <h5>Digit Sum Skips</h5>
                <div className="skip-list">
                  {Object.entries(digitGroups.sumSkips)
                    .sort(([,a], [,b]) => b.currentSkip - a.currentSkip)
                    .slice(0, 5)
                    .map(([sum, stats]) => (
                      <div key={sum} className="skip-item">
                        <span>Sum {sum}:</span>
                        <span>Current: {stats.currentSkip}, Avg: {stats.averageSkip.toFixed(1)}</span>
                      </div>
                    ))}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SixthBallAnalytics;
