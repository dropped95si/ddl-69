"""Real data loaders: Parquet + Polygon + Alpaca + Yahoo (fallback)"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Literal
import logging

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class DataLoader:
    """Load market data from multiple sources with fallback chain."""

    def __init__(
        self,
        artifact_root: Optional[str] = None,
        polygon_key: Optional[str] = None,
        alpaca_key: Optional[str] = None,
        alpaca_secret: Optional[str] = None,
    ):
        self.artifact_root = Path(artifact_root or os.getenv("ARTIFACT_ROOT", ".artifacts"))
        self.polygon_key = polygon_key or os.getenv("POLYGON_API_KEY")
        self.alpaca_key = alpaca_key or os.getenv("APCA_API_KEY_ID")
        self.alpaca_secret = alpaca_secret or os.getenv("APCA_API_SECRET_KEY")

        self._polygon_client = None
        self._alpaca_client = None

    @property
    def polygon_client(self):
        """Lazy load Polygon client."""
        if not self._polygon_client and self.polygon_key:
            try:
                from polygon import RESTClient
                self._polygon_client = RESTClient(api_key=self.polygon_key)
            except ImportError:
                logger.warning("polygon-api-client not installed, skipping Polygon")
        return self._polygon_client

    @property
    def alpaca_client(self):
        """Lazy load Alpaca client."""
        if not self._alpaca_client and self.alpaca_key and self.alpaca_secret:
            try:
                from alpaca.data.historical import StockHistoricalDataClient
                self._alpaca_client = StockHistoricalDataClient(
                    api_key=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                )
            except ImportError:
                logger.warning("alpaca-trade-api not installed, skipping Alpaca")
        return self._alpaca_client

    def load_parquet(
        self, symbol: str, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        """Load from local parquet artifacts."""
        bars_dir = self.artifact_root / "bars"
        if not bars_dir.exists():
            return None

        # Try multiple parquet patterns
        patterns = [
            f"bars_polygon_{symbol}_*.parquet",
            f"bars_csv_{symbol}_*.parquet",
            "bars.parquet",
        ]

        for pattern in patterns:
            files = list(bars_dir.glob(pattern))
            if files:
                # Use most recent file
                file_path = max(files, key=lambda p: p.stat().st_mtime)
                try:
                    df = pd.read_parquet(file_path)
                    logger.info(f"Loaded {len(df)} rows from {file_path.name}")
                    return self._filter_dates(df, start_date, end_date)
                except Exception as e:
                    logger.warning(f"Failed to load {file_path}: {e}")

        return None

    def load_polygon(
        self, symbol: str, start_date: str, end_date: str, timespan: str = "day"
    ) -> Optional[pd.DataFrame]:
        """Load from Polygon.io REST API."""
        if not self.polygon_client:
            return None

        try:
            bars = []
            for agg in self.polygon_client.list_aggs(
                symbol,
                1,
                timespan,
                start_date,
                end_date,
                limit=50000,
            ):
                bars.append(
                    {
                        "timestamp": pd.Timestamp(agg.timestamp, unit="ms"),
                        "open": agg.open,
                        "high": agg.high,
                        "low": agg.low,
                        "close": agg.close,
                        "volume": agg.volume,
                        "vwap": agg.vwap,
                        "trades_count": agg.n if hasattr(agg, "n") else None,
                    }
                )

            if not bars:
                return None

            df = pd.DataFrame(bars)
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Loaded {len(df)} bars from Polygon: {symbol}")
            return df

        except Exception as e:
            logger.warning(f"Polygon failed for {symbol}: {e}")
            return None

    def load_alpaca(
        self, symbol: str, start_date: str, end_date: str, timeframe: str = "1Day"
    ) -> Optional[pd.DataFrame]:
        """Load from Alpaca historical data API."""
        if not self.alpaca_client:
            return None

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=TimeFrame.Day if timeframe == "1Day" else TimeFrame.Hour,
                start=pd.Timestamp(start_date),
                end=pd.Timestamp(end_date),
            )

            bars = self.alpaca_client.get_stock_bars(request_params)
            if bars is None or bars.empty:
                return None

            df = bars.reset_index()
            df.columns = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Loaded {len(df)} bars from Alpaca: {symbol}")
            return df

        except Exception as e:
            logger.warning(f"Alpaca failed for {symbol}: {e}")
            return None

    def load_yahoo(
        self, symbol: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Load from Yahoo Finance (fallback)."""
        try:
            import yfinance as yf

            df = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                progress=False,
            )

            if df.empty:
                return None

            df = df.reset_index()
            df.columns = ["timestamp", "open", "high", "low", "close", "volume"]
            df = df[["timestamp", "open", "high", "low", "close", "volume"]]
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp").reset_index(drop=True)
            logger.info(f"Loaded {len(df)} bars from Yahoo: {symbol}")
            return df

        except Exception as e:
            logger.warning(f"Yahoo failed for {symbol}: {e}")
            return None

    def load(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        sources: list[Literal["parquet", "polygon", "alpaca", "yahoo"]] = None,
    ) -> pd.DataFrame:
        """Load data with fallback chain: parquet -> Polygon -> Alpaca -> Yahoo."""
        if sources is None:
            sources = ["parquet", "polygon", "alpaca", "yahoo"]

        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        for source in sources:
            logger.info(f"Trying {source} for {symbol}...")

            if source == "parquet":
                df = self.load_parquet(symbol, start_date, end_date)
            elif source == "polygon":
                df = self.load_polygon(symbol, start_date, end_date)
            elif source == "alpaca":
                df = self.load_alpaca(symbol, start_date, end_date)
            elif source == "yahoo":
                df = self.load_yahoo(symbol, start_date, end_date)
            else:
                continue

            if df is not None and len(df) > 0:
                return self._standardize(df)

        raise ValueError(f"No data available for {symbol} from sources: {sources}")

    def load_multiple(
        self,
        symbols: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """Load data for multiple symbols."""
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.load(symbol, start_date, end_date)
            except ValueError as e:
                logger.error(f"Failed to load {symbol}: {e}")
        return data

    def save_parquet(
        self, df: pd.DataFrame, symbol: str, kind: str = "bars"
    ) -> Path:
        """Save dataframe to parquet artifact."""
        bars_dir = self.artifact_root / kind
        bars_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = bars_dir / f"{kind}_{symbol}_{timestamp}.parquet"

        df.to_parquet(file_path, index=False)
        logger.info(f"Saved {len(df)} rows to {file_path}")
        return file_path

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize OHLCV columns."""
        # Ensure timestamp column
        if "timestamp" not in df.columns:
            if "index" in df.columns:
                df = df.rename(columns={"index": "timestamp"})
            elif df.index.name in ["timestamp", "date", "time"]:
                df = df.reset_index()

        # Normalize datetime
        if df["timestamp"].dtype == "object":
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Ensure required columns are numeric
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop NaN rows
        df = df.dropna(subset=["close", "volume"])

        return df.sort_values("timestamp").reset_index(drop=True)

    def _filter_dates(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Filter dataframe by date range."""
        if "timestamp" in df.columns:
            if start_date:
                df = df[df["timestamp"] >= pd.Timestamp(start_date)]
            if end_date:
                df = df[df["timestamp"] <= pd.Timestamp(end_date)]
        return df


class SupabaseCache:
    """Cache data in Supabase for live model updates."""

    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        self.url = url or os.getenv("SUPABASE_URL")
        self.key = key or os.getenv("SUPABASE_KEY")
        self._client = None

    @property
    def client(self):
        """Lazy load Supabase client."""
        if not self._client and self.url and self.key:
            try:
                from supabase import create_client
                self._client = create_client(self.url, self.key)
            except ImportError:
                logger.warning("supabase not installed")
        return self._client

    def save_bars(self, symbol: str, df: pd.DataFrame) -> bool:
        """Save OHLCV bars to Supabase."""
        if not self.client:
            return False

        try:
            # Convert df to records
            records = df.to_dict("records")
            timestamp_col = df[[c for c in df.columns if "timestamp" in c.lower()][0]]

            # Upsert with composite key: symbol + timestamp
            for record in records:
                record["symbol"] = symbol

            self.client.table("bars").upsert(records).execute()
            logger.info(f"Saved {len(records)} bars for {symbol} to Supabase")
            return True

        except Exception as e:
            logger.warning(f"Failed to save to Supabase: {e}")
            return False

    def get_latest_timestamp(self, symbol: str) -> Optional[datetime]:
        """Get last timestamp for a symbol from Supabase."""
        if not self.client:
            return None

        try:
            result = (
                self.client.table("bars")
                .select("timestamp")
                .eq("symbol", symbol)
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )

            if result.data:
                return pd.Timestamp(result.data[0]["timestamp"])
            return None

        except Exception as e:
            logger.warning(f"Failed to query Supabase: {e}")
            return None
