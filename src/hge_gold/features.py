from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .config import FeatureConfig

ID_COLUMNS = ["row_id", "date", "close", "volume"]


def _hurst_rs_single_scale_legacy(values: np.ndarray) -> float:
    """Return the historical one-scale R/S transform for explicit legacy comparison.

    This is not a multi-scale Hurst regression. It remains available only so formal
    ablations can compare the previous project feature with the DFA1 estimator.
    """

    values = np.asarray(values, dtype=float)
    if values.size < 32 or not np.isfinite(values).all():
        return float("nan")
    increments = np.diff(values)
    std = increments.std(ddof=1)
    if std <= 1e-12:
        return float("nan")
    path = np.cumsum(increments - increments.mean())
    rescaled_range = (path.max() - path.min()) / std
    if rescaled_range <= 0:
        return float("nan")
    return float(np.log(rescaled_range) / np.log(len(increments)))


def _dfa1_scales(n_increments: int) -> tuple[int, ...]:
    """Return the fixed DFA1 scale grid: powers of two from 4 through n // 3."""

    scales: list[int] = []
    scale = 4
    maximum = n_increments // 3
    while scale <= maximum:
        scales.append(scale)
        scale *= 2
    return tuple(scales)


def _hurst_dfa1(values: np.ndarray) -> float:
    """Estimate a DFA1 exponent from log prices using their stationary increments.

    The return increments are demeaned and integrated into a DFA profile. For each
    deterministic power-of-two scale, forward and backward non-overlapping segments
    are linearly detrended. The reported exponent is the OLS slope of log fluctuation
    against log scale. At least three usable scales are required.
    """

    values = np.asarray(values, dtype=float)
    if values.size < 32 or not np.isfinite(values).all():
        return float("nan")

    increments = np.diff(values)
    if increments.std(ddof=1) <= 1e-12:
        return float("nan")
    profile = np.cumsum(increments - increments.mean())
    scales = _dfa1_scales(len(increments))
    if len(scales) < 3:
        return float("nan")

    usable_scales: list[float] = []
    fluctuations: list[float] = []
    for scale in scales:
        n_segments = len(profile) // scale
        if n_segments < 3:
            continue
        remainder = len(profile) - n_segments * scale
        offsets = (0,) if remainder == 0 else (0, remainder)
        x = np.arange(scale, dtype=float)
        x -= x.mean()
        denominator = float(np.dot(x, x))
        residual_sum_squares = 0.0
        residual_count = 0
        for offset in offsets:
            for segment_index in range(n_segments):
                start = offset + segment_index * scale
                segment = profile[start : start + scale]
                centered = segment - segment.mean()
                slope = float(np.dot(x, centered) / denominator)
                residual = centered - slope * x
                residual_sum_squares += float(np.dot(residual, residual))
                residual_count += scale
        fluctuation = np.sqrt(residual_sum_squares / residual_count)
        if np.isfinite(fluctuation) and fluctuation > 1e-15:
            usable_scales.append(float(scale))
            fluctuations.append(float(fluctuation))

    if len(usable_scales) < 3:
        return float("nan")
    log_scales = np.log(np.asarray(usable_scales, dtype=float))
    log_fluctuations = np.log(np.asarray(fluctuations, dtype=float))
    log_scales -= log_scales.mean()
    denominator = float(np.dot(log_scales, log_scales))
    if denominator <= 0:
        return float("nan")
    return float(np.dot(log_scales, log_fluctuations - log_fluctuations.mean()) / denominator)


def _rolling_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or len(values) < 3:
        return float("nan")
    x = np.arange(len(values), dtype=float)
    x -= x.mean()
    denominator = float(np.dot(x, x))
    return float(np.dot(x, values - values.mean()) / denominator) if denominator else float("nan")


def _sign_entropy(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    positive = float((values > 0).mean())
    negative = 1.0 - positive
    terms = [p * np.log2(p) for p in (positive, negative) if p > 0]
    return float(-sum(terms))


def _rsi(ret: pd.Series, window: int = 14) -> pd.Series:
    gain = ret.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-ret.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _atr(frame: pd.DataFrame, window: int = 14) -> pd.Series:
    previous_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def _safe_zscore(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    minimum = min_periods or max(20, window // 2)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std()
    return (series - mean) / std.replace(0, np.nan)


def build_feature_matrix(
    frame: pd.DataFrame, config: FeatureConfig
) -> tuple[pd.DataFrame, list[str]]:
    result = frame[ID_COLUMNS].copy()
    close = frame["close"].astype(float)
    log_price = np.log(close)
    ret = log_price.diff()
    volume_log = np.log1p(frame["volume"].astype(float))

    for lag in config.return_lags:
        result[f"return_lag_{lag}"] = ret.shift(lag - 1)

    for window in config.windows:
        rolling_mean = ret.rolling(window, min_periods=window).mean()
        rolling_std = ret.rolling(window, min_periods=window).std()
        result[f"return_mean_{window}"] = rolling_mean
        result[f"return_std_{window}"] = rolling_std
        result[f"momentum_{window}"] = log_price - log_price.shift(window)
        result[f"price_zscore_{window}"] = _safe_zscore(log_price, window, window)
        result[f"drawdown_{window}"] = close / close.rolling(window, min_periods=window).max() - 1.0
        result[f"trend_slope_{window}"] = log_price.rolling(window, min_periods=window).apply(
            _rolling_slope, raw=True
        )
        result[f"sign_entropy_{window}"] = ret.rolling(window, min_periods=window).apply(
            _sign_entropy, raw=True
        )

    result["rsi_14"] = _rsi(ret, 14) / 100.0
    atr = _atr(frame, 14)
    result["atr_14_normalized"] = atr / close
    result["range_log"] = np.log(frame["high"] / frame["low"])
    result["close_open_log"] = np.log(frame["close"] / frame["open"])
    result["ewma_vol_20"] = ret.ewm(span=20, adjust=False, min_periods=20).std()
    result["ewma_vol_63"] = ret.ewm(span=63, adjust=False, min_periods=40).std()
    result["vol_ratio_20_63"] = result["ewma_vol_20"] / result["ewma_vol_63"].replace(0, np.nan)
    result["rolling_skew_20"] = ret.rolling(20, min_periods=20).skew()
    result["rolling_kurtosis_20"] = ret.rolling(20, min_periods=20).kurt()

    ema_fast = log_price.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_slow = log_price.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_fast - ema_slow
    result["macd"] = macd
    result["macd_signal"] = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    result["volume_change_1"] = volume_log.diff()
    result["volume_zscore_20"] = _safe_zscore(volume_log, 20, 20)
    result["volume_zscore_63"] = _safe_zscore(volume_log, 63, 40)
    result["return_volume_interaction"] = ret * result["volume_zscore_20"]

    for window in config.hurst_windows:
        result[f"hurst_rs_single_scale_legacy_{window}"] = log_price.rolling(
            window, min_periods=window
        ).apply(_hurst_rs_single_scale_legacy, raw=True)
        result[f"hurst_dfa1_{window}"] = log_price.rolling(window, min_periods=window).apply(
            _hurst_dfa1, raw=True
        )

    primary_hurst = result[f"hurst_dfa1_{max(config.hurst_windows)}"]
    result["hurst_dfa1_available"] = primary_hurst.notna().astype(float)
    regime_window = config.regime_window
    past_hurst = primary_hurst.shift(1)
    low_hurst = past_hurst.rolling(regime_window, min_periods=max(80, regime_window // 2)).quantile(
        0.33
    )
    high_hurst = past_hurst.rolling(
        regime_window, min_periods=max(80, regime_window // 2)
    ).quantile(0.67)
    hurst_regime_available = primary_hurst.notna() & low_hurst.notna() & high_hurst.notna()
    hurst_regime = pd.Series(np.nan, index=result.index, dtype=float)
    hurst_regime.loc[hurst_regime_available] = 0.0
    hurst_regime.loc[hurst_regime_available & (primary_hurst <= low_hurst)] = -1.0
    hurst_regime.loc[hurst_regime_available & (primary_hurst >= high_hurst)] = 1.0
    result["hurst_regime"] = hurst_regime
    result["hurst_regime_available"] = hurst_regime_available.astype(float)
    vol = result["ewma_vol_20"]
    result["volatility_zscore"] = _safe_zscore(
        vol.shift(1), regime_window, max(80, regime_window // 2)
    )
    result["trend_efficiency_20"] = (log_price - log_price.shift(20)).abs() / ret.abs().rolling(
        20, min_periods=20
    ).sum().replace(0, np.nan)
    trend_regime_available = result["trend_efficiency_20"].notna() & result["momentum_20"].notna()
    trend_regime = pd.Series(np.nan, index=result.index, dtype=float)
    trend_regime.loc[trend_regime_available] = 0.0
    trend_regime.loc[
        trend_regime_available
        & (result["trend_efficiency_20"] > 0.45)
        & (result["momentum_20"] > 0)
    ] = 1.0
    trend_regime.loc[
        trend_regime_available
        & (result["trend_efficiency_20"] > 0.45)
        & (result["momentum_20"] < 0)
    ] = -1.0
    result["trend_regime"] = trend_regime
    result["trend_regime_available"] = trend_regime_available.astype(float)

    day_of_week = frame["date"].dt.dayofweek.astype(float)
    month = frame["date"].dt.month.astype(float)
    result["dow_sin"] = np.sin(2 * np.pi * day_of_week / 5.0)
    result["dow_cos"] = np.cos(2 * np.pi * day_of_week / 5.0)
    result["month_sin"] = np.sin(2 * np.pi * (month - 1.0) / 12.0)
    result["month_cos"] = np.cos(2 * np.pi * (month - 1.0) / 12.0)

    result = result.replace([np.inf, -np.inf], np.nan)
    feature_columns = [column for column in result.columns if column not in ID_COLUMNS]
    result["feature_coverage"] = result[feature_columns].notna().mean(axis=1)
    return result, feature_columns


def assert_causal_features(
    original: pd.DataFrame,
    mutated: pd.DataFrame,
    columns: Iterable[str],
    inclusive_row: int,
    atol: float = 1e-12,
) -> None:
    left = original.loc[:inclusive_row, list(columns)].to_numpy(dtype=float)
    right = mutated.loc[:inclusive_row, list(columns)].to_numpy(dtype=float)
    if not np.allclose(left, right, equal_nan=True, atol=atol, rtol=0.0):
        raise AssertionError("Future source changes altered historical feature values")
