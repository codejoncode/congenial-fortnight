#!/usr/bin/env python3
"""
Pip-Based Quality Signal System

Focus: High-quality setups only
- Risk:Reward minimum 1:2 (risk 20 pips, target 40+ pips)
- Win rate target: 75%+
- Trade frequency: Only when optimal conditions exist
- No trading in ranging/unclear markets

Comprehensive pip tracking:
- Average pips won per winning trade
- Average pips lost per losing trade
- Total pips gained over backtest period
- Risk:Reward ratio per trade
- Win rate percentage
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import json
from datetime import datetime
import talib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipBasedSignalSystem:
    """
    Generates high-quality signals with pip-based risk management
    """
    
    def __init__(self, min_risk_reward: float = 2.0, min_confidence: float = 0.75):
        """
        Args:
            min_risk_reward: Minimum R:R ratio (2.0 = risk 1 to make 2)
            min_confidence: Minimum model confidence (0.75 = 75%)
        """
        self.min_risk_reward = min_risk_reward
        self.min_confidence = min_confidence
        self.pip_values = {
            'EURUSD': 0.0001,  # 1 pip = 0.0001
            'GBPUSD': 0.0001,
            'USDJPY': 0.01,    # 1 pip = 0.01
            'XAUUSD': 0.10,    # 1 pip = 0.10 for gold
            'USDCAD': 0.0001,
            'AUDUSD': 0.0001,
            'NZDUSD': 0.0001,
            'USDCHF': 0.0001
        }
        
        # Typical spreads (in pips)
        self.typical_spreads = {
            'EURUSD': 1.0,
            'GBPUSD': 1.5,
            'USDJPY': 1.0,
            'XAUUSD': 30.0,  # Gold has wider spreads
            'USDCAD': 1.5,
            'AUDUSD': 1.2,
            'NZDUSD': 1.5,
            'USDCHF': 1.5
        }
        
        self.backtest_results = []
    
    def calculate_pips(self, pair: str, entry: float, exit: float, 
                       direction: str = 'long') -> float:
        """
        Calculate pips gained/lost on a trade
        
        Args:
            pair: Currency pair
            entry: Entry price
            exit: Exit price
            direction: 'long' or 'short'
            
        Returns:
            Pips gained (positive) or lost (negative)
        """
        pip_value = self.pip_values.get(pair, 0.0001)
        price_diff = exit - entry
        
        if direction == 'short':
            price_diff = -price_diff
        
        pips = price_diff / pip_value
        return pips
    
    def detect_quality_setup(self, df: pd.DataFrame, pair: str, 
                            model_prediction: Dict) -> Dict:
        """
        Detect if current conditions represent a QUALITY setup
        
        Returns signal only if:
        1. Model confidence >= 75%
        2. Market regime is trending (not ranging)
        3. Risk:Reward ratio >= 2:1
        4. Clear support/resistance levels
        5. Momentum aligned with direction
        
        Returns:
            {
                'signal': 'long'/'short'/None,
                'confidence': 0.0-1.0,
                'entry': float,
                'stop_loss': float,
                'take_profit': float,
                'risk_pips': float,
                'reward_pips': float,
                'risk_reward_ratio': float,
                'setup_quality': 'excellent'/'good'/None,
                'reasoning': str
            }
        """
        
        # Step 1: Check model confidence
        model_confidence = model_prediction.get('confidence', 0.0)
        if model_confidence < self.min_confidence:
            return self._no_signal_response(
                f"Model confidence {model_confidence:.1%} below minimum {self.min_confidence:.1%}"
            )
        
        # Step 2: Analyze market regime
        regime = self._analyze_market_regime(df)
        if not regime['is_trending']:
            return self._no_signal_response(
                f"Market is {regime['state']} - only trading trending markets"
            )
        
        # Step 3: Detect support/resistance levels
        sr_levels = self._detect_support_resistance(df, pair)
        
        # Step 4: Calculate ATR for dynamic stops
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Step 5: Determine signal direction
        predicted_direction = model_prediction.get('direction', None)
        if predicted_direction not in ['long', 'short']:
            return self._no_signal_response("No clear directional bias")
        
        # Step 6: Calculate entry, stop, and target levels
        signal_details = self._calculate_trade_levels(
            pair=pair,
            direction=predicted_direction,
            current_price=current_price,
            atr=atr,
            sr_levels=sr_levels,
            regime=regime
        )
        
        # Step 7: Verify risk:reward ratio
        if signal_details['risk_reward_ratio'] < self.min_risk_reward:
            return self._no_signal_response(
                f"R:R {signal_details['risk_reward_ratio']:.2f} below minimum {self.min_risk_reward}"
            )
        
        # Step 8: Check for momentum alignment
        momentum_aligned = self._check_momentum_alignment(df, predicted_direction)
        if not momentum_aligned:
            return self._no_signal_response(
                "Momentum not aligned with predicted direction"
            )
        
        # Step 9: Assess setup quality
        quality_score = self._assess_setup_quality(
            confidence=model_confidence,
            risk_reward=signal_details['risk_reward_ratio'],
            regime_strength=regime['strength'],
            momentum_aligned=momentum_aligned
        )
        
        # Step 10: Generate signal with full details
        signal = {
            'signal': predicted_direction,
            'confidence': model_confidence,
            'entry': signal_details['entry'],
            'stop_loss': signal_details['stop_loss'],
            'take_profit': signal_details['take_profit'],
            'risk_pips': signal_details['risk_pips'],
            'reward_pips': signal_details['reward_pips'],
            'risk_reward_ratio': signal_details['risk_reward_ratio'],
            'setup_quality': quality_score['quality'],
            'quality_score': quality_score['score'],
            'reasoning': self._generate_reasoning(
                direction=predicted_direction,
                confidence=model_confidence,
                regime=regime,
                risk_reward=signal_details['risk_reward_ratio'],
                quality=quality_score['quality']
            ),
            'timestamp': df.index[-1],
            'pair': pair
        }
        
        logger.info(f"✅ QUALITY SETUP DETECTED: {pair} {predicted_direction.upper()}")
        logger.info(f"   Risk: {signal_details['risk_pips']:.1f} pips | "
                   f"Reward: {signal_details['reward_pips']:.1f} pips | "
                   f"R:R {signal_details['risk_reward_ratio']:.2f}")
        
        return signal
    
    def _analyze_market_regime(self, df: pd.DataFrame) -> Dict:
        """
        Determine if market is trending or ranging
        
        Returns:
            {
                'is_trending': bool,
                'state': 'strong_trend'/'weak_trend'/'ranging',
                'direction': 'bullish'/'bearish'/None,
                'strength': 0.0-1.0
            }
        """
        # ADX for trend strength
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        
        # EMA alignment
        ema_20 = talib.EMA(df['close'], timeperiod=20).iloc[-1]
        ema_50 = talib.EMA(df['close'], timeperiod=50).iloc[-1]
        ema_200 = talib.EMA(df['close'], timeperiod=200).iloc[-1]
        current_price = df['close'].iloc[-1]
        
        # Determine trend state
        if adx > 25:  # Strong trend
            direction = 'bullish' if plus_di > minus_di else 'bearish'
            strength = min(adx / 50.0, 1.0)  # Normalize to 0-1
            
            # Verify EMA alignment
            if direction == 'bullish' and current_price > ema_20 > ema_50:
                return {
                    'is_trending': True,
                    'state': 'strong_trend',
                    'direction': 'bullish',
                    'strength': strength
                }
            elif direction == 'bearish' and current_price < ema_20 < ema_50:
                return {
                    'is_trending': True,
                    'state': 'strong_trend',
                    'direction': 'bearish',
                    'strength': strength
                }
        
        # Ranging market
        return {
            'is_trending': False,
            'state': 'ranging',
            'direction': None,
            'strength': 0.0
        }
    
    def _detect_support_resistance(self, df: pd.DataFrame, pair: str) -> Dict:
        """
        Detect key support and resistance levels
        """
        lookback = min(50, len(df))
        recent_highs = df['high'].iloc[-lookback:]
        recent_lows = df['low'].iloc[-lookback:]
        
        # Resistance: highest high in lookback period
        resistance = recent_highs.max()
        
        # Support: lowest low in lookback period
        support = recent_lows.min()
        
        # Pivot levels
        pivot = (df['high'].iloc[-1] + df['low'].iloc[-1] + df['close'].iloc[-1]) / 3
        
        return {
            'resistance': resistance,
            'support': support,
            'pivot': pivot,
            'distance_to_resistance': resistance - df['close'].iloc[-1],
            'distance_to_support': df['close'].iloc[-1] - support
        }
    
    def _calculate_trade_levels(self, pair: str, direction: str, 
                                current_price: float, atr: float,
                                sr_levels: Dict, regime: Dict) -> Dict:
        """
        Calculate entry, stop loss, and take profit levels
        
        Strategy:
        - Entry: Current price (or slightly better)
        - Stop: ATR-based or just beyond S/R level
        - Target: Minimum 2x risk, ideally to next S/R level
        """
        pip_value = self.pip_values[pair]
        spread_pips = self.typical_spreads.get(pair, 1.0)
        spread_price = spread_pips * pip_value
        
        # ATR multiplier based on pair volatility
        if pair == 'XAUUSD':
            atr_stop_multiplier = 2.0  # Gold is more volatile
            atr_target_multiplier = 5.0
        else:
            atr_stop_multiplier = 1.5
            atr_target_multiplier = 4.0
        
        if direction == 'long':
            # Entry at ask price (current + spread)
            entry = current_price + spread_price
            
            # Stop loss: Below support or ATR-based
            atr_stop = entry - (atr * atr_stop_multiplier)
            support_stop = sr_levels['support'] - (2 * pip_value)  # 2 pips buffer
            stop_loss = max(atr_stop, support_stop)  # Use closer stop
            
            # Take profit: Above resistance or ATR-based
            atr_target = entry + (atr * atr_target_multiplier)
            resistance_target = sr_levels['resistance'] - (2 * pip_value)
            
            # Choose target that gives better R:R but is realistic
            risk = entry - stop_loss
            min_target = entry + (risk * self.min_risk_reward)
            take_profit = max(atr_target, min_target)
            
            # Don't exceed distant resistance
            if take_profit > sr_levels['resistance']:
                take_profit = resistance_target
        
        else:  # short
            # Entry at bid price
            entry = current_price - spread_price
            
            # Stop loss: Above resistance or ATR-based
            atr_stop = entry + (atr * atr_stop_multiplier)
            resistance_stop = sr_levels['resistance'] + (2 * pip_value)
            stop_loss = min(atr_stop, resistance_stop)
            
            # Take profit: Below support or ATR-based
            atr_target = entry - (atr * atr_target_multiplier)
            support_target = sr_levels['support'] + (2 * pip_value)
            
            risk = stop_loss - entry
            min_target = entry - (risk * self.min_risk_reward)
            take_profit = min(atr_target, min_target)
            
            if take_profit < sr_levels['support']:
                take_profit = support_target
        
        # Calculate pips
        risk_pips = abs(self.calculate_pips(pair, entry, stop_loss, direction))
        reward_pips = abs(self.calculate_pips(pair, entry, take_profit, direction))
        
        return {
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_pips': risk_pips,
            'reward_pips': reward_pips,
            'risk_reward_ratio': reward_pips / risk_pips if risk_pips > 0 else 0
        }
    
    def _check_momentum_alignment(self, df: pd.DataFrame, direction: str) -> bool:
        """
        Check if momentum indicators align with predicted direction
        """
        # RSI
        rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
        
        # MACD
        macd, signal, hist = talib.MACD(df['close'])
        macd_current = macd.iloc[-1]
        signal_current = signal.iloc[-1]
        
        # Stochastic
        slowk, slowd = talib.STOCH(df['high'], df['low'], df['close'])
        stoch_k = slowk.iloc[-1]
        
        if direction == 'long':
            # Bullish momentum checks
            rsi_ok = rsi > 40 and rsi < 70  # Not oversold or overbought
            macd_ok = macd_current > signal_current  # MACD above signal
            stoch_ok = stoch_k > 20  # Not oversold
            
            return sum([rsi_ok, macd_ok, stoch_ok]) >= 2  # At least 2 of 3
        
        else:  # short
            # Bearish momentum checks
            rsi_ok = rsi < 60 and rsi > 30
            macd_ok = macd_current < signal_current
            stoch_ok = stoch_k < 80
            
            return sum([rsi_ok, macd_ok, stoch_ok]) >= 2
    
    def _assess_setup_quality(self, confidence: float, risk_reward: float,
                              regime_strength: float, momentum_aligned: bool) -> Dict:
        """
        Assess overall setup quality on a scale
        
        Returns:
            {
                'quality': 'excellent'/'good'/'fair',
                'score': 0.0-100.0
            }
        """
        # Score components (0-100 each)
        confidence_score = confidence * 100
        rr_score = min(risk_reward / 3.0, 1.0) * 100  # Max score at 3:1 R:R
        regime_score = regime_strength * 100
        momentum_score = 100 if momentum_aligned else 50
        
        # Weighted average
        total_score = (
            confidence_score * 0.35 +
            rr_score * 0.30 +
            regime_score * 0.20 +
            momentum_score * 0.15
        )
        
        # Classify quality
        if total_score >= 80:
            quality = 'excellent'
        elif total_score >= 65:
            quality = 'good'
        else:
            quality = 'fair'
        
        return {
            'quality': quality,
            'score': total_score
        }
    
    def _generate_reasoning(self, direction: str, confidence: float,
                           regime: Dict, risk_reward: float, quality: str) -> str:
        """Generate human-readable reasoning for the signal"""
        
        reasons = []
        reasons.append(f"Model confidence: {confidence:.1%}")
        reasons.append(f"Market regime: {regime['state']} ({regime['direction']})")
        reasons.append(f"Risk:Reward ratio: 1:{risk_reward:.2f}")
        reasons.append(f"Setup quality: {quality.upper()}")
        
        if regime['strength'] > 0.7:
            reasons.append("Strong trend detected ✓")
        
        reasoning = f"{direction.upper()} Signal - " + " | ".join(reasons)
        return reasoning
    
    def _no_signal_response(self, reason: str) -> Dict:
        """Return no-signal response with reason"""
        logger.info(f"⏸️  No signal: {reason}")
        return {
            'signal': None,
            'confidence': 0.0,
            'reasoning': reason
        }
    
    def backtest_with_pip_tracking(self, df: pd.DataFrame, pair: str,
                                   signals: List[Dict]) -> Dict:
        """
        Backtest signals with detailed pip tracking
        
        Args:
            df: Historical OHLC data
            pair: Currency pair
            signals: List of signal dictionaries from detect_quality_setup
            
        Returns:
            {
                'total_trades': int,
                'winning_trades': int,
                'losing_trades': int,
                'win_rate': float,
                'total_pips': float,
                'avg_win_pips': float,
                'avg_loss_pips': float,
                'largest_win_pips': float,
                'largest_loss_pips': float,
                'avg_risk_reward': float,
                'total_days': int,
                'trades_per_month': float,
                'trade_results': List[Dict]
            }
        """
        
        trade_results = []
        total_pips = 0
        winning_pips = []
        losing_pips = []
        risk_reward_ratios = []
        
        for signal in signals:
            if signal['signal'] is None:
                continue
            
            # Get signal details
            entry = signal['entry']
            stop_loss = signal['stop_loss']
            take_profit = signal['take_profit']
            direction = signal['signal']
            entry_time = signal['timestamp']
            
            # Find what happened after entry
            try:
                future_data = df[df.index > entry_time].head(100)  # Look ahead max 100 candles
            except:
                continue
            
            if len(future_data) == 0:
                continue
            
            # Determine trade outcome
            outcome = self._simulate_trade_outcome(
                future_data, entry, stop_loss, take_profit, direction
            )
            
            # Calculate pips
            pips = self.calculate_pips(pair, entry, outcome['exit_price'], direction)
            
            # Account for spread
            spread_pips = self.typical_spreads.get(pair, 1.0)
            pips -= spread_pips  # Every trade pays the spread
            
            # Record result
            trade_result = {
                'entry_time': entry_time,
                'exit_time': outcome['exit_time'],
                'direction': direction,
                'entry': entry,
                'exit': outcome['exit_price'],
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'outcome': outcome['result'],
                'pips': pips,
                'risk_pips': signal['risk_pips'],
                'reward_pips': signal['reward_pips'],
                'risk_reward_ratio': signal['risk_reward_ratio'],
                'confidence': signal['confidence'],
                'quality': signal['setup_quality']
            }
            
            trade_results.append(trade_result)
            total_pips += pips
            risk_reward_ratios.append(signal['risk_reward_ratio'])
            
            if outcome['result'] == 'win':
                winning_pips.append(pips)
            else:
                losing_pips.append(pips)
        
        # Calculate statistics
        total_trades = len(trade_results)
        winning_trades = len(winning_pips)
        losing_trades = len(losing_pips)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Calculate time span
        if len(df) > 0:
            start_date = df.index[0]
            end_date = df.index[-1]
            total_days = (end_date - start_date).days
            trades_per_month = (total_trades / total_days) * 30 if total_days > 0 else 0
        else:
            total_days = 0
            trades_per_month = 0
        
        results = {
            'pair': pair,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pips': total_pips,
            'avg_win_pips': np.mean(winning_pips) if winning_pips else 0,
            'avg_loss_pips': np.mean(losing_pips) if losing_pips else 0,
            'largest_win_pips': max(winning_pips) if winning_pips else 0,
            'largest_loss_pips': min(losing_pips) if losing_pips else 0,
            'avg_risk_reward': np.mean(risk_reward_ratios) if risk_reward_ratios else 0,
            'total_days': total_days,
            'trades_per_month': trades_per_month,
            'trade_results': trade_results
        }
        
        self.backtest_results.append(results)
        return results
    
    def _simulate_trade_outcome(self, future_data: pd.DataFrame, entry: float,
                                stop_loss: float, take_profit: float,
                                direction: str) -> Dict:
        """
        Simulate trade outcome based on future price action
        """
        for i, (timestamp, row) in enumerate(future_data.iterrows()):
            high = row['high']
            low = row['low']
            
            if direction == 'long':
                # Check if stop hit
                if low <= stop_loss:
                    return {
                        'result': 'loss',
                        'exit_price': stop_loss,
                        'exit_time': timestamp,
                        'bars_held': i + 1
                    }
                # Check if target hit
                if high >= take_profit:
                    return {
                        'result': 'win',
                        'exit_price': take_profit,
                        'exit_time': timestamp,
                        'bars_held': i + 1
                    }
            
            else:  # short
                # Check if stop hit
                if high >= stop_loss:
                    return {
                        'result': 'loss',
                        'exit_price': stop_loss,
                        'exit_time': timestamp,
                        'bars_held': i + 1
                    }
                # Check if target hit
                if low <= take_profit:
                    return {
                        'result': 'win',
                        'exit_price': take_profit,
                        'exit_time': timestamp,
                        'bars_held': i + 1
                    }
        
        # Trade still open at end of data
        return {
            'result': 'open',
            'exit_price': future_data['close'].iloc[-1],
            'exit_time': future_data.index[-1],
            'bars_held': len(future_data)
        }
    
    def print_backtest_summary(self, results: Dict):
        """Print formatted backtest results with pip statistics"""
        
        print("\n" + "="*80)
        print(f"📊 PIP-BASED BACKTEST RESULTS - {results['pair']}")
        print("="*80)
        
        print(f"\n📅 PERIOD:")
        print(f"   Total Days: {results['total_days']}")
        print(f"   Trades Per Month: {results['trades_per_month']:.1f}")
        
        print(f"\n📈 TRADE STATISTICS:")
        print(f"   Total Trades: {results['total_trades']}")
        print(f"   Winning Trades: {results['winning_trades']}")
        print(f"   Losing Trades: {results['losing_trades']}")
        print(f"   Win Rate: {results['win_rate']*100:.1f}%")
        
        print(f"\n💰 PIP PERFORMANCE:")
        print(f"   Total Pips: {results['total_pips']:+.1f}")
        print(f"   Avg Win: {results['avg_win_pips']:+.1f} pips")
        print(f"   Avg Loss: {results['avg_loss_pips']:+.1f} pips")
        print(f"   Largest Win: {results['largest_win_pips']:+.1f} pips")
        print(f"   Largest Loss: {results['largest_loss_pips']:+.1f} pips")
        print(f"   Avg Risk:Reward: 1:{results['avg_risk_reward']:.2f}")
        
        # Calculate expectancy
        if results['total_trades'] > 0:
            expectancy = (
                results['win_rate'] * results['avg_win_pips'] +
                (1 - results['win_rate']) * results['avg_loss_pips']
            )
            print(f"   Expectancy: {expectancy:+.2f} pips per trade")
        
        print("\n" + "="*80)
    
    def save_detailed_results(self, output_dir: str = 'output'):
        """Save detailed backtest results to CSV and JSON"""
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for results in self.backtest_results:
            pair = results['pair']
            
            # Save trade-by-trade results to CSV
            if results['trade_results']:
                df = pd.DataFrame(results['trade_results'])
                csv_path = output_path / f"{pair}_pip_trades_{timestamp}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"✅ Saved detailed trades to {csv_path}")
            
            # Save summary to JSON
            summary = {k: v for k, v in results.items() if k != 'trade_results'}
            json_path = output_path / f"{pair}_pip_summary_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            logger.info(f"✅ Saved summary to {json_path}")
        
        # Save combined summary
        combined_summary = {
            'timestamp': timestamp,
            'pairs': [r['pair'] for r in self.backtest_results],
            'results': [
                {k: v for k, v in r.items() if k != 'trade_results'}
                for r in self.backtest_results
            ]
        }
        combined_path = output_path / f"pip_backtest_summary_{timestamp}.json"
        with open(combined_path, 'w') as f:
            json.dump(combined_summary, f, indent=2, default=str)
        
        logger.info(f"✅ Saved combined summary to {combined_path}")


# Example usage
if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         PIP-BASED QUALITY SIGNAL SYSTEM                      ║
    ╚══════════════════════════════════════════════════════════════╝
    
    This system focuses on HIGH-QUALITY setups only:
    
    ✅ Minimum 2:1 Risk:Reward ratio
    ✅ 75%+ model confidence required
    ✅ Only trades in trending markets
    ✅ Comprehensive pip tracking
    ✅ Won't trade every day (quality over quantity)
    
    Features:
    - Average pips won per winning trade
    - Average pips lost per losing trade
    - Total pips over backtest period
    - Win rate percentage
    - Risk:Reward ratio per trade
    - Trades per month frequency
    
    Usage:
    1. Generate model predictions
    2. Pass to detect_quality_setup()
    3. System returns signal only if quality criteria met
    4. Backtest with backtest_with_pip_tracking()
    5. Review pip statistics
    """)
