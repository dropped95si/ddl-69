"""
TA-Lib wrapper with fallback to pandas/numpy implementations
Full suite of technical indicators
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# Try to import talib, fall back to custom implementations
try:
    import talib as ta
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


class TALibWrapper:
    """Unified TA interface with talib or custom fallback"""

    def __init__(self, use_talib: bool = True):
        self.use_talib = use_talib and HAS_TALIB

    # Overlap Studies
    def SMA(self, close: pd.Series, period: int = 30) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.SMA(close.values, timeperiod=period), index=close.index)
        return close.rolling(period).mean()

    def EMA(self, close: pd.Series, period: int = 30) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.EMA(close.values, timeperiod=period), index=close.index)
        return close.ewm(span=period, adjust=False).mean()

    def BBANDS(self, close: pd.Series, period: int = 20, nbdevup: float = 2, nbdevdn: float = 2):
        if self.use_talib:
            upper, middle, lower = ta.BBANDS(close.values, timeperiod=period,
                                            nbdevup=nbdevup, nbdevdn=nbdevdn)
            return pd.DataFrame({
                'upper': upper,
                'middle': middle,
                'lower': lower
            }, index=close.index)
        middle = close.rolling(period).mean()
        std = close.rolling(period).std()
        return pd.DataFrame({
            'upper': middle + nbdevup * std,
            'middle': middle,
            'lower': middle - nbdevdn * std
        }, index=close.index)

    def DEMA(self, close: pd.Series, period: int = 30) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.DEMA(close.values, timeperiod=period), index=close.index)
        ema1 = close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2

    def TEMA(self, close: pd.Series, period: int = 30) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.TEMA(close.values, timeperiod=period), index=close.index)
        ema1 = close.ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3

    def KAMA(self, close: pd.Series, period: int = 30) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.KAMA(close.values, timeperiod=period), index=close.index)
        # Simplified KAMA fallback
        return close.ewm(span=period, adjust=False).mean()

    # Momentum Indicators
    def RSI(self, close: pd.Series, period: int = 14) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.RSI(close.values, timeperiod=period), index=close.index)
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def MACD(self, close: pd.Series, fastperiod: int = 12, slowperiod: int = 26, signalperiod: int = 9):
        if self.use_talib:
            macd, signal, hist = ta.MACD(close.values, fastperiod=fastperiod,
                                        slowperiod=slowperiod, signalperiod=signalperiod)
            return pd.DataFrame({
                'macd': macd,
                'signal': signal,
                'hist': hist
            }, index=close.index)
        fast = close.ewm(span=fastperiod, adjust=False).mean()
        slow = close.ewm(span=slowperiod, adjust=False).mean()
        macd = fast - slow
        signal = macd.ewm(span=signalperiod, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({'macd': macd, 'signal': signal, 'hist': hist}, index=close.index)

    def STOCH(self, high: pd.Series, low: pd.Series, close: pd.Series,
              fastk_period: int = 5, slowk_period: int = 3, slowd_period: int = 3):
        if self.use_talib:
            slowk, slowd = ta.STOCH(high.values, low.values, close.values,
                                   fastk_period=fastk_period, slowk_period=slowk_period,
                                   slowd_period=slowd_period)
            return pd.DataFrame({'slowk': slowk, 'slowd': slowd}, index=close.index)
        lowest_low = low.rolling(fastk_period).min()
        highest_high = high.rolling(fastk_period).max()
        fastk = 100 * (close - lowest_low) / (highest_high - lowest_low)
        slowk = fastk.rolling(slowk_period).mean()
        slowd = slowk.rolling(slowd_period).mean()
        return pd.DataFrame({'slowk': slowk, 'slowd': slowd}, index=close.index)

    def ADX(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
        # Simplified ADX fallback
        tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return (atr / close * 100).fillna(0)

    def CCI(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.CCI(high.values, low.values, close.values, timeperiod=period), index=close.index)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(period).mean()
        mad = (tp - sma_tp).abs().rolling(period).mean()
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.fillna(0)

    # Volume Indicators
    def OBV(self, close: pd.Series, volume: pd.Series) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.OBV(close.values, volume.values), index=close.index)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        return obv

    def AD(self, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.AD(high.values, low.values, close.values, volume.values), index=close.index)
        clv = ((close - low) - (high - close)) / (high - low)
        clv = clv.fillna(0)
        ad = (clv * volume).cumsum()
        return ad

    # Volatility Indicators
    def ATR(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.ATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(alpha=1/period, adjust=False).mean()

    def NATR(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        if self.use_talib:
            return pd.Series(ta.NATR(high.values, low.values, close.values, timeperiod=period), index=close.index)
        atr = self.ATR(high, low, close, period)
        return (atr / close * 100).fillna(0)


# Convenience function
def compute_all_indicators(df: pd.DataFrame, use_talib: bool = True) -> pd.DataFrame:
    """
    Compute full suite of indicators
    Expected columns: open, high, low, close, volume
    """
    ta_wrap = TALibWrapper(use_talib)
    result = df.copy()

    # Overlap
    result['sma_5'] = ta_wrap.SMA(df['close'], 5)
    result['sma_10'] = ta_wrap.SMA(df['close'], 10)
    result['sma_20'] = ta_wrap.SMA(df['close'], 20)
    result['sma_50'] = ta_wrap.SMA(df['close'], 50)
    result['sma_200'] = ta_wrap.SMA(df['close'], 200)
    result['ema_12'] = ta_wrap.EMA(df['close'], 12)
    result['ema_26'] = ta_wrap.EMA(df['close'], 26)

    bb = ta_wrap.BBANDS(df['close'], 20)
    result['bb_upper'] = bb['upper']
    result['bb_middle'] = bb['middle']
    result['bb_lower'] = bb['lower']

    # Momentum
    result['rsi_14'] = ta_wrap.RSI(df['close'], 14)
    macd_df = ta_wrap.MACD(df['close'])
    result['macd'] = macd_df['macd']
    result['macd_signal'] = macd_df['signal']
    result['macd_hist'] = macd_df['hist']

    stoch = ta_wrap.STOCH(df['high'], df['low'], df['close'])
    result['stoch_k'] = stoch['slowk']
    result['stoch_d'] = stoch['slowd']

    result['adx_14'] = ta_wrap.ADX(df['high'], df['low'], df['close'], 14)
    result['cci_14'] = ta_wrap.CCI(df['high'], df['low'], df['close'], 14)

    # Volume
    if 'volume' in df.columns:
        result['obv'] = ta_wrap.OBV(df['close'], df['volume'])
        result['ad'] = ta_wrap.AD(df['high'], df['low'], df['close'], df['volume'])

    # Volatility
    result['atr_14'] = ta_wrap.ATR(df['high'], df['low'], df['close'], 14)
    result['natr_14'] = ta_wrap.NATR(df['high'], df['low'], df['close'], 14)

    return result
