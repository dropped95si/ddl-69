"""Serverless functions for live trading predictions

Usage:
  /api/predictions?symbol=SPY
  /api/portfolio?symbols=SPY,QQQ,AAPL
  /api/refresh?symbol=SPY
"""

import json
import logging
from datetime import datetime
from typing import Optional
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

logger = logging.getLogger(__name__)

# Global pipeline cache (in-memory for this serverless instance)
_pipeline_cache = {}


def _load_or_train_pipeline(symbol: str, artifact_root: Optional[str] = None):
    """Lazy load or train pipeline for symbol"""
    cache_key = f"{symbol}_{artifact_root or 'default'}"

    if cache_key not in _pipeline_cache:
        try:
            from ddl69.data.loaders import DataLoader
            from ddl69.core.real_pipeline import RealDataPipeline
            from ddl69.agents.sklearn_ensemble import SklearnEnsemble
            import numpy as np

            loader = DataLoader(artifact_root=artifact_root)
            df = loader.load(symbol)

            pipeline = RealDataPipeline(artifact_root=artifact_root)
            df = pipeline._preprocess(df)
            df = pipeline._add_indicators(df, symbol)
            df = pipeline._create_labels(df)

            # Quick train on all data
            split_idx = int(len(df) * 0.7)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

            feature_cols = [
                c for c in df.columns
                if c not in [
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "returns",
                    "label",
                    "forward_return",
                ]
                and df[c].dtype in ["float64", "int64"]
            ]

            X_train = train_df[feature_cols].fillna(0)
            y_train = (train_df["label"] > 0).astype(int)

            X_test = test_df[feature_cols].fillna(0)
            y_test = (test_df["label"] > 0).astype(int)

            pipeline.ensemble = SklearnEnsemble(
                task="classification",
                models=["rf", "xgb", "lgb"],
                voting="soft",
            )
            pipeline.ensemble.fit(X_train, y_train)

            # Store metrics
            pipeline._test_accuracy = pipeline.ensemble.score(X_test, y_test)
            pipeline._test_sharpe = (
                np.mean(X_test.iloc[:, 0]) / (np.std(X_test.iloc[:, 0]) + 1e-8) * np.sqrt(252)
                if len(X_test) > 1
                else 0.0
            )
            pipeline._df = df
            pipeline._feature_cols = feature_cols

            _pipeline_cache[cache_key] = pipeline
            logger.info(f"Trained pipeline for {symbol}")

        except Exception as e:
            logger.error(f"Failed to load/train pipeline for {symbol}: {e}")
            raise

    return _pipeline_cache[cache_key]


def handler(request):
    """Main prediction handler - routes to signal/portfolio/refresh

    Routes:
      GET /api/predictions?symbol=SPY
      GET /api/portfolio?symbols=SPY,QQQ
      POST /api/refresh?symbol=SPY
    """
    action = request.path.split('/')[-1] if request.path else 'signal'

    if action == 'signal' or request.args.get('symbol'):
        return signal_handler(request)
    elif action == 'portfolio' or 'portfolio' in request.path:
        return portfolio_handler(request)
    elif action == 'refresh' or 'refresh' in request.path:
        return refresh_handler(request)
    else:
        return {
            "statusCode": 404,
            "body": json.dumps({"error": "Unknown action"}),
        }


def signal_handler(request):
    """Get latest trading signal for a symbol

    Query params:
      - symbol: (required) Ticker symbol
      - artifact_root: (optional) Path to parquet artifacts
    """
    try:
        symbol = request.args.get('symbol', '').upper()
        artifact_root = request.args.get('artifact_root')

        if not symbol:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "symbol parameter required"}),
            }

        pipeline = _load_or_train_pipeline(symbol, artifact_root)

        # Get latest bar prediction
        latest_row = pipeline._df.iloc[-1:].copy()
        pred = pipeline._predict(latest_row, symbol)

        if "error" in pred:
            return {
                "statusCode": 500,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": pred["error"]}),
            }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "symbol": symbol,
                "signal": pred["signal"],
                "probability": pred["buy_probability"],
                "confidence": pred["confidence"],
                "price": pred["current_price"],
                "timestamp": str(pred["timestamp"]),
                "accuracy": float(getattr(pipeline, "_test_accuracy", 0.0)),
                "sharpe": float(getattr(pipeline, "_test_sharpe", 0.0)),
            }),
        }

    except Exception as e:
        logger.error(f"Error in signal_handler: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }


def portfolio_handler(request):
    """Get signals for entire portfolio

    Query params:
      - symbols: (optional) Comma-separated list (default: SPY,QQQ,AAPL)
      - artifact_root: (optional) Path to parquet artifacts
    """
    try:
        symbols_str = request.args.get('symbols', 'SPY,QQQ,AAPL')
        artifact_root = request.args.get('artifact_root')

        symbol_list = [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

        signals = []
        buy_count = sell_count = hold_count = 0
        sharpe_scores = []

        for symbol in symbol_list:
            try:
                pipeline = _load_or_train_pipeline(symbol, artifact_root)

                latest_row = pipeline._df.iloc[-1:].copy()
                pred = pipeline._predict(latest_row, symbol)

                if "error" not in pred:
                    sig_obj = {
                        "symbol": symbol,
                        "signal": pred["signal"],
                        "probability": pred["buy_probability"],
                        "confidence": pred["confidence"],
                        "price": pred["current_price"],
                        "timestamp": str(pred["timestamp"]),
                        "accuracy": float(getattr(pipeline, "_test_accuracy", 0.0)),
                        "sharpe": float(getattr(pipeline, "_test_sharpe", 0.0)),
                    }
                    signals.append(sig_obj)

                    if pred["signal"] == "BUY":
                        buy_count += 1
                    elif pred["signal"] == "SELL":
                        sell_count += 1
                    else:
                        hold_count += 1

                    if getattr(pipeline, "_test_sharpe", 0.0):
                        sharpe_scores.append(pipeline._test_sharpe)

            except Exception as e:
                logger.warning(f"Failed to get signal for {symbol}: {e}")
                continue

        portfolio_sharpe = (
            sum(sharpe_scores) / len(sharpe_scores)
            if sharpe_scores else 0.0
        )

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "timestamp": datetime.utcnow().isoformat(),
                "total_symbols": len(signals),
                "buy_count": buy_count,
                "sell_count": sell_count,
                "hold_count": hold_count,
                "signals": signals,
                "portfolio_sharpe": float(portfolio_sharpe),
            }),
        }

    except Exception as e:
        logger.error(f"Error in portfolio_handler: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }


def refresh_handler(request):
    """Manually refresh cached model for symbol

    Query params:
      - symbol: (required) Ticker symbol
      - artifact_root: (optional) Path to parquet artifacts
    """
    try:
        symbol = request.args.get('symbol', '').upper()
        artifact_root = request.args.get('artifact_root')

        if not symbol:
            return {
                "statusCode": 400,
                "headers": {"Content-Type": "application/json"},
                "body": json.dumps({"error": "symbol parameter required"}),
            }

        cache_key = f"{symbol}_{artifact_root or 'default'}"

        # Remove from cache to force retrain
        if cache_key in _pipeline_cache:
            del _pipeline_cache[cache_key]
            logger.info(f"Cleared cache for {symbol}")

        # Reload and retrain
        pipeline = _load_or_train_pipeline(symbol, artifact_root)

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({
                "status": "success",
                "symbol": symbol,
                "message": f"Refreshed model for {symbol}",
            }),
        }

    except Exception as e:
        logger.error(f"Error in refresh_handler: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
