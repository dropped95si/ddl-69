"""
Lopez de Prado methods from Advances in Financial Machine Learning
- Fractional differentiation
- Purged K-Fold CV
- Meta-labeling
- Sample weights
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from typing import Optional


def frac_diff_ffd(series: pd.Series, d: float, thres: float = 0.01) -> pd.Series:
    """
    Fractionally differentiated features (FFD) from Lopez de Prado
    Args:
        series: price series
        d: differentiation order (0.4-0.6 typical)
        thres: threshold for weight cutoff
    """
    weights = _get_weights_ffd(d, thres, len(series))
    width = len(weights) - 1
    out = pd.Series(index=series.index, dtype=float)
    for i in range(width, len(series)):
        out.iloc[i] = np.dot(weights.T, series.iloc[i - width:i + 1])
    return out


def _get_weights_ffd(d: float, thres: float, lim: int) -> np.ndarray:
    """Calculate FFD weights"""
    w = [1.0]
    k = 1
    while k < lim:
        w_ = -w[-1] / k * (d - k + 1)
        if abs(w_) < thres:
            break
        w.append(w_)
        k += 1
    return np.array(w[::-1]).reshape(-1, 1)


class PurgedKFold:
    """
    Purged K-Fold CV from Lopez de Prado
    Purges train samples that overlap with test period + embargo
    """
    def __init__(self, n_splits: int = 5, embargo_pct: float = 0.01):
        self.n_splits = n_splits
        self.embargo_pct = embargo_pct

    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None,
              t1: Optional[pd.Series] = None):
        """
        Args:
            X: features with DatetimeIndex
            y: labels
            t1: end times for each sample (Series with same index as X)
        """
        if t1 is None:
            t1 = pd.Series(X.index, index=X.index)

        indices = np.arange(len(X))
        embargo = int(len(X) * self.embargo_pct)
        test_ranges = [(i * len(X) // self.n_splits,
                       (i + 1) * len(X) // self.n_splits)
                       for i in range(self.n_splits)]

        for test_start, test_end in test_ranges:
            test_indices = indices[test_start:test_end]

            # Purge train samples that overlap with test period
            if len(test_indices) == 0:
                continue
            test_times = X.index[test_indices]
            t1_test = t1.iloc[test_indices]

            train_indices = []
            for i in indices:
                if i in test_indices:
                    continue
                # Purge if sample's end time overlaps with test start
                if t1.iloc[i] > test_times.min():
                    if X.index[i] < test_times.max():
                        continue
                train_indices.append(i)

            # Apply embargo after test end
            if embargo > 0:
                max_test_idx = test_end
                train_indices = [i for i in train_indices
                               if i < test_start or i >= max_test_idx + embargo]

            yield np.array(train_indices), np.array(test_indices)


def get_sample_weights(t1: pd.Series, num_concurrent: pd.Series) -> pd.Series:
    """
    Sample weights based on label uniqueness (Lopez de Prado Ch4)
    Args:
        t1: end time for each sample
        num_concurrent: number of concurrent labels at each time
    Returns:
        weights normalized to sum to 1
    """
    weights = t1.map(lambda x: 1.0 / num_concurrent.get(x, 1.0))
    weights = weights / weights.sum()
    return weights


def meta_labeling(
    primary_pred: pd.Series,
    actual_labels: pd.Series,
    bet_size: Optional[pd.Series] = None
) -> pd.DataFrame:
    """
    Meta-labeling: use ML to size bets on primary model signals
    Args:
        primary_pred: primary model predictions (1/-1/0)
        actual_labels: true labels (1/-1)
        bet_size: optional existing bet sizes
    Returns:
        DataFrame with meta_label (0=skip, 1=take position)
    """
    meta = pd.DataFrame(index=primary_pred.index)
    meta['primary_pred'] = primary_pred
    meta['actual'] = actual_labels

    # Meta-label: 1 if primary correct, 0 otherwise
    meta['meta_label'] = (
        (meta['primary_pred'] > 0) & (meta['actual'] > 0)
    ) | (
        (meta['primary_pred'] < 0) & (meta['actual'] < 0)
    )
    meta['meta_label'] = meta['meta_label'].astype(int)

    if bet_size is not None:
        meta['bet_size'] = bet_size

    return meta


def get_num_concurrent_events(close_times: pd.Series) -> pd.Series:
    """
    Count number of concurrent events at each timestamp
    Used for sample weighting
    """
    # Expand each event to all timestamps it covers
    events = []
    for idx, end_time in close_times.items():
        events.append({'start': idx, 'end': end_time})

    # Count overlaps
    all_times = pd.concat([
        pd.Series(close_times.index),
        pd.Series(close_times.values)
    ]).unique()
    all_times = pd.Series(sorted(all_times))

    counts = {}
    for t in all_times:
        count = sum(1 for e in events if e['start'] <= t <= e['end'])
        counts[t] = count

    return pd.Series(counts)
