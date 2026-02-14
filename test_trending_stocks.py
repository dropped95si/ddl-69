"""Test complete system on top 10 trending stocks - swing trading"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json

from ddl69.backtesting.walkforward import MultiStrategyComparator
from ddl69.experts.prediction_expert import EnsembleExpertPortfolio
from ddl69.trading.bot import PaperTradingBot
from ddl69.data.loaders import DataLoader
from ddl69.retraining.scheduler import AutoRetrainingScheduler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Top 10 trending stocks (swing trading candidates)
TOP_10_TRENDING = [
    "NVDA",   # AI leader
    "PLTR",   # Palantir - trending
    "TSLA",   # Tesla
    "MSTR",   # Microstrategy - Bitcoin play
    "SOUN",   # SoundHound - AI speech
    "AVGO",   # Broadcom - semi
    "COIN",   # Coinbase - crypto
    "RIOT",   # Riot - Bitcoin mining
    "MARA",   # Marathon - Bitcoin mining
    "ARM",    # Arm Holdings - semi
]


def test_system():
    """Full system test on trending stocks"""

    print("\n" + "="*80)
    print("DDL-69 PROBABILITY ENGINE - TRENDING STOCKS TEST")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols: {', '.join(TOP_10_TRENDING)}")
    print("="*80 + "\n")

    # ============ STEP 1: BACKTEST ALL SYMBOLS ============
    print("[STEP 1] Running Walk-Forward Backtests (5-fold CV)...\n")

    comparator = MultiStrategyComparator(TOP_10_TRENDING)
    backtest_results = comparator.run_all()

    print("\nBacktest Results by Sharpe Ratio (Best Risk-Adjusted Returns):\n")
    print(f"{'Rank':<5} {'Symbol':<8} {'Sharpe':<10} {'Accuracy':<12} {'Status':<15}")
    print("-" * 55)

    for rank, (symbol, sharpe, accuracy) in enumerate(backtest_results['ranking'], 1):
        status = "✅ STRONG" if sharpe > 1.5 else "✓ GOOD" if sharpe > 1.0 else "⚠ WEAK"
        print(f"{rank:<5} {symbol:<8} {sharpe:<10.3f} {accuracy:<12.1%} {status:<15}")

    # ============ STEP 2: GET LIVE SIGNALS ============
    print("\n[STEP 2] Generating Live Trading Signals...\n")

    portfolio = EnsembleExpertPortfolio(TOP_10_TRENDING)
    predictions = portfolio.predict_all()

    print(f"{'Symbol':<8} {'Signal':<8} {'Probability':<14} {'Confidence':<12} {'Price est':<12}")
    print("-" * 60)

    buy_signals = []
    sell_signals = []
    hold_signals = []

    for symbol in TOP_10_TRENDING:
        if predictions.get(symbol) is None:
            continue

        pred = predictions[symbol]
        signal = pred['signal']
        prob = pred['raw_probability']
        conf = pred['confidence']
        price = pred.get('supporting_indicators', ['$0.00'])[-1]

        print(f"{symbol:<8} {signal:<8} {prob:<14.1%} {conf:<12.1%} {price:<12}")

        if signal == "BUY":
            buy_signals.append((symbol, prob, conf))
        elif signal == "SELL":
            sell_signals.append((symbol, prob, conf))
        else:
            hold_signals.append(symbol)

    print(f"\nSignal Summary: {len(buy_signals)} BUY, {len(sell_signals)} SELL, {len(hold_signals)} HOLD\n")

    # ============ STEP 3: SIMULATE PAPER TRADING ============
    print("[STEP 3] Simulating Paper Trading (100k initial capital)...\n")

    bot = PaperTradingBot(
        initial_capital=100000,
        max_position_size=0.05,  # 5% per position (swing trading)
        risk_per_trade=0.02,
    )

    # Mock current prices (use close prices from latest bars)
    current_prices = {}
    for symbol in TOP_10_TRENDING:
        # Generate realistic prices
        base_prices = {
            "NVDA": 850, "PLTR": 45, "TSLA": 250, "MSTR": 420, "SOUN": 15,
            "AVGO": 140, "COIN": 180, "RIOT": 25, "MARA": 30, "ARM": 165,
        }
        # Add noise
        current_prices[symbol] = base_prices.get(symbol, 100) * np.random.uniform(0.98, 1.02)

    trades_executed = 0
    for symbol, pred in predictions.items():
        if pred is None:
            continue

        order = bot.execute_signal(
            symbol=symbol,
            signal=pred['signal'],
            probability=pred['raw_probability'],
            confidence=pred['confidence'],
            current_price=current_prices[symbol],
        )

        if order:
            trades_executed += 1

    portfolio_status = bot.get_portfolio_status(current_prices)

    print(f"Executed: {trades_executed} orders")
    print(f"Portfolio Value: ${portfolio_status['portfolio_value']:,.2f}")
    print(f"Cash: ${portfolio_status['cash']:,.2f}")
    print(f"Total P&L: ${portfolio_status['total_pnl']:,.2f} ({portfolio_status['total_pnl_pct']:.2%})")
    print(f"Active Positions: {sum(1 for p in portfolio_status['positions'].values() if p['quantity'] != 0)}")

    # ============ STEP 4: DETECTION CHECK ============
    print("\n[STEP 4] Model Drift Detection...\n")

    drift_status = {}
    for symbol in TOP_10_TRENDING[:5]:  # Check first 5 for drift
        scheduler = AutoRetrainingScheduler(symbol)
        drift = scheduler.drift_detector.detect_drift()
        if drift.get('has_drifted'):
            drift_status[symbol] = f"⚠ {drift.get('degradation_pct')} degradation"
        else:
            drift_status[symbol] = "✅ Stable"

    for symbol, status in drift_status.items():
        print(f"{symbol}: {status}")

    # ============ STEP 5: ASSESSMENT ============
    print("\n" + "="*80)
    print("SYSTEM ASSESSMENT")
    print("="*80 + "\n")

    assessment = {
        "backtests_completed": len(backtest_results['ranking']),
        "avg_sharpe": np.mean([s for _, s, _ in backtest_results['ranking']]),
        "best_performer": backtest_results['ranking'][0][0] if backtest_results['ranking'] else None,
        "best_sharpe": backtest_results['ranking'][0][1] if backtest_results['ranking'] else 0,
        "best_accuracy": backtest_results['ranking'][0][2] if backtest_results['ranking'] else 0,
        "signals_generated": len(buy_signals) + len(sell_signals),
        "buy_signals": len(buy_signals),
        "sell_signals": len(sell_signals),
        "trades_simulated": trades_executed,
        "portfolio_value": portfolio_status['portfolio_value'],
        "portfolio_pnl_pct": portfolio_status['total_pnl_pct'],
        "drift_detected": len([s for s, st in drift_status.items() if "⚠" in st]),
    }

    print(f"✅ Backtests Completed: {assessment['backtests_completed']}/10")
    print(f"📊 Average Sharpe Ratio: {assessment['avg_sharpe']:.3f}")
    print(f"🏆 Best Performer: {assessment['best_performer']} (Sharpe: {assessment['best_sharpe']:.3f}, Accuracy: {assessment['best_accuracy']:.1%})")
    print(f"📈 Signals Generated: {assessment['signals_generated']} ({assessment['buy_signals']} BUY / {assessment['sell_signals']} SELL)")
    print(f"💼 Paper Trades: {assessment['trades_simulated']}")
    print(f"💰 Portfolio Value: ${assessment['portfolio_value']:,.2f}")
    print(f"📉 P&L: {assessment['portfolio_pnl_pct']:+.2%}")
    print(f"⚠️  Drift Alerts: {assessment['drift_detected']}")

    print("\n" + "="*80)

    # PASS/FAIL CRITERIA
    criteria = {
        "backtest_coverage": assessment['backtests_completed'] >= 8,  # At least 8/10
        "signal_quality": assessment['avg_sharpe'] > 0.8,  # Average Sharpe > 0.8
        "signal_generation": assessment['signals_generated'] > 0,  # At least 1 signal
        "model_stability": assessment['drift_detected'] == 0,  # No drift
    }

    status = "✅ PASS - READY FOR DEPLOYMENT" if all(criteria.values()) else "⚠️  WARNING - REVIEW RESULTS"

    print(f"\nDeployment Status: {status}")
    print("\nCriteria:")
    for criterion, passed in criteria.items():
        mark = "✅" if passed else "❌"
        print(f"  {mark} {criterion}")

    print("\n" + "="*80 + "\n")

    return assessment, criteria, all(criteria.values())


if __name__ == "__main__":
    assessment, criteria, ready_for_deploy = test_system()

    # Save assessment
    report = {
        "timestamp": datetime.now().isoformat(),
        "test_symbols": TOP_10_TRENDING,
        "assessment": assessment,
        "criteria": criteria,
        "ready_for_deployment": ready_for_deploy,
    }

    with open(".artifacts/test_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"Test report saved to .artifacts/test_report.json\n")

    if ready_for_deploy:
        print("🚀 System PASSED all criteria! Ready for production deployment.\n")
        print("Next step: vercel deploy --prod\n")
    else:
        print("⚠️  System requires review before deployment.\n")
        print("Check .artifacts/test_report.json for details.\n")
