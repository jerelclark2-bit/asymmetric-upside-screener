from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

LOOKBACK = 126
PATH_POINTS = 32
VOLUME_POINTS = 16
CONTEXT_SCALES = np.array([0.65, 0.55, 0.45, 0.55, 1.10], dtype=np.float32)
STARTER_INDEX = (
    Path(__file__).resolve().parent.parent / "data" / "pattern_index_starter.npz"
)
CUSTOM_INDEX = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "cache"
    / "pattern_index_custom.npz"
)


@dataclass
class PatternIndex:
    tickers: np.ndarray
    end_dates: np.ndarray
    paths: np.ndarray
    volumes: np.ndarray
    contexts: np.ndarray
    outcomes: np.ndarray
    future_paths: np.ndarray

    @property
    def size(self) -> int:
        return len(self.tickers)

    @property
    def stock_count(self) -> int:
        return len(np.unique(self.tickers))

    @property
    def first_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.end_dates.min(), unit="D")

    @property
    def last_date(self) -> pd.Timestamp:
        return pd.Timestamp(self.end_dates.max(), unit="D")


def _resample(values: np.ndarray, points: int) -> np.ndarray:
    old_x = np.linspace(0, 1, len(values))
    new_x = np.linspace(0, 1, points)
    return np.interp(new_x, old_x, values).astype(np.float32)


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.nanstd(values))
    if not np.isfinite(std) or std < 1e-8:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - np.nanmean(values)) / std).astype(np.float32)


def encode_pattern(
    close: np.ndarray, volume: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    close = np.asarray(close, dtype=float)
    if (
        len(close) < LOOKBACK
        or np.any(~np.isfinite(close[-LOOKBACK:]))
        or np.any(close[-LOOKBACK:] <= 0)
    ):
        raise ValueError("A pattern requires 126 valid positive closing prices")
    sample = close[-LOOKBACK:]
    raw_path = _resample(np.log(sample / sample[0]), PATH_POINTS)
    path = _zscore(raw_path)
    returns = np.diff(np.log(sample))
    running_high = np.maximum.accumulate(sample)
    context = np.array(
        [
            sample[-1] / sample[0] - 1,
            sample[-1] / sample[-63] - 1,
            float(np.std(returns) * np.sqrt(252)),
            sample[-1] / running_high.max() - 1,
            sample[-1] / sample.min() - 1,
        ],
        dtype=np.float32,
    )
    if volume is None or len(volume) < LOOKBACK:
        volume_path = np.zeros(VOLUME_POINTS, dtype=np.float32)
    else:
        volume_sample = np.asarray(volume[-LOOKBACK:], dtype=float)
        volume_sample = np.nan_to_num(volume_sample, nan=np.nanmedian(volume_sample))
        volume_path = _zscore(
            _resample(np.log1p(np.maximum(volume_sample, 0)), VOLUME_POINTS)
        )
    return path, volume_path, context


def _history_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "Close" not in frame:
        return pd.DataFrame()
    result = frame[[c for c in ["Close", "Volume"] if c in frame]].copy()
    result.index = pd.to_datetime(result.index).tz_localize(None)
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result.replace([np.inf, -np.inf], np.nan).dropna(subset=["Close"])


def build_pattern_index(
    histories: dict[str, pd.DataFrame],
    output_path: str | Path,
    step: int = 21,
    min_dollar_volume: float = 1_000_000,
) -> PatternIndex:
    records: list[tuple] = []
    for ticker, raw_frame in histories.items():
        frame = _history_frame(raw_frame)
        if len(frame) < LOOKBACK + 253:
            continue
        close = frame["Close"].to_numpy(dtype=float)
        volume = frame.get(
            "Volume", pd.Series(np.zeros(len(frame)), index=frame.index)
        ).to_numpy(dtype=float)
        for end in range(LOOKBACK - 1, len(frame) - 252, step):
            start = end - LOOKBACK + 1
            close_window = close[start : end + 1]
            volume_window = volume[start : end + 1]
            if np.any(~np.isfinite(close_window)) or np.any(close_window <= 0):
                continue
            dollar_volume = float(
                np.nanmedian(close_window[-20:] * volume_window[-20:])
            )
            if np.isfinite(dollar_volume) and dollar_volume < min_dollar_volume:
                continue
            try:
                path, volume_path, context = encode_pattern(close_window, volume_window)
            except ValueError:
                continue
            outcomes = np.array(
                [
                    close[end + 63] / close[end] - 1,
                    close[end + 126] / close[end] - 1,
                    close[end + 252] / close[end] - 1,
                ],
                dtype=np.float32,
            )
            future = _resample(np.log(close[end : end + 253] / close[end]), PATH_POINTS)
            records.append(
                (
                    ticker.upper(),
                    frame.index[end]
                    .to_datetime64()
                    .astype("datetime64[D]")
                    .astype(int),
                    path,
                    volume_path,
                    context,
                    outcomes,
                    future,
                )
            )
    if not records:
        raise ValueError(
            "No valid historical windows were available for the pattern index"
        )
    index = PatternIndex(
        tickers=np.asarray([r[0] for r in records], dtype="U24"),
        end_dates=np.asarray([r[1] for r in records], dtype=np.int32),
        paths=np.vstack([r[2] for r in records]).astype(np.float32),
        volumes=np.vstack([r[3] for r in records]).astype(np.float32),
        contexts=np.vstack([r[4] for r in records]).astype(np.float32),
        outcomes=np.vstack([r[5] for r in records]).astype(np.float32),
        future_paths=np.vstack([r[6] for r in records]).astype(np.float32),
    )
    save_pattern_index(index, output_path)
    return index


def save_pattern_index(index: PatternIndex, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.array([1], dtype=np.int16),
        tickers=index.tickers,
        end_dates=index.end_dates,
        paths=index.paths,
        volumes=index.volumes,
        contexts=index.contexts,
        outcomes=index.outcomes,
        future_paths=index.future_paths,
    )


def load_pattern_index(path: str | Path | None = None) -> PatternIndex:
    selected = (
        Path(path)
        if path
        else (CUSTOM_INDEX if CUSTOM_INDEX.exists() else STARTER_INDEX)
    )
    with np.load(selected, allow_pickle=False) as data:
        return PatternIndex(
            tickers=data["tickers"],
            end_dates=data["end_dates"],
            paths=data["paths"],
            volumes=data["volumes"],
            contexts=data["contexts"],
            outcomes=data["outcomes"],
            future_paths=data["future_paths"],
        )


def _dtw_similarity(left: np.ndarray, right: np.ndarray) -> float:
    rows, cols = len(left), len(right)
    costs = np.full((rows + 1, cols + 1), np.inf, dtype=float)
    costs[0, 0] = 0.0
    for i in range(1, rows + 1):
        for j in range(max(1, i - 4), min(cols, i + 4) + 1):
            cost = abs(float(left[i - 1] - right[j - 1]))
            costs[i, j] = cost + min(
                costs[i - 1, j], costs[i, j - 1], costs[i - 1, j - 1]
            )
    distance = costs[rows, cols] / (rows + cols)
    return float(np.exp(-distance / 0.32))


def find_pattern_matches(
    history: pd.DataFrame,
    index: PatternIndex,
    query_ticker: str = "",
    top_n: int = 12,
    candidate_count: int = 500,
) -> pd.DataFrame:
    frame = _history_frame(history)
    if len(frame) < LOOKBACK:
        return pd.DataFrame()
    close = frame["Close"].to_numpy(dtype=float)
    volume = frame.get(
        "Volume", pd.Series(np.zeros(len(frame)), index=frame.index)
    ).to_numpy(dtype=float)
    query_path, query_volume, query_context = encode_pattern(close, volume)

    correlations = np.clip(index.paths @ query_path / PATH_POINTS, -1, 1)
    correlation_scores = (correlations + 1) / 2
    context_distance = np.sqrt(
        np.mean(((index.contexts - query_context) / CONTEXT_SCALES) ** 2, axis=1)
    )
    context_scores = np.exp(-context_distance)
    volume_scores = np.clip(
        (index.volumes @ query_volume / VOLUME_POINTS + 1) / 2, 0, 1
    )
    preliminary = (
        0.65 * correlation_scores + 0.25 * context_scores + 0.10 * volume_scores
    )

    # Keep current or near-current windows out of the answer, especially for the query stock.
    cutoff = (
        (frame.index[-1] - pd.DateOffset(years=2))
        .to_datetime64()
        .astype("datetime64[D]")
        .astype(int)
    )
    valid = np.ones(index.size, dtype=bool)
    if query_ticker:
        valid &= ~(
            (index.tickers == query_ticker.upper()) & (index.end_dates >= cutoff)
        )
    preliminary[~valid] = -np.inf
    pool_size = min(candidate_count, int(valid.sum()))
    if pool_size <= 0:
        return pd.DataFrame()
    candidates = np.argpartition(preliminary, -pool_size)[-pool_size:]

    rescored: list[tuple[float, int, float]] = []
    for idx in candidates:
        dtw = _dtw_similarity(query_path, index.paths[idx])
        final = (
            0.45 * correlation_scores[idx]
            + 0.25 * dtw
            + 0.20 * context_scores[idx]
            + 0.10 * volume_scores[idx]
        )
        rescored.append((float(final), int(idx), dtw))
    rescored.sort(reverse=True)

    chosen: list[dict] = []
    ticker_dates: dict[str, list[int]] = {}
    for final, idx, dtw in rescored:
        ticker = str(index.tickers[idx])
        end_date = int(index.end_dates[idx])
        prior_dates = ticker_dates.setdefault(ticker, [])
        if len(prior_dates) >= 2 or any(
            abs(end_date - old) < LOOKBACK for old in prior_dates
        ):
            continue
        prior_dates.append(end_date)
        similarity = round(final * 100, 1)
        chosen.append(
            {
                "ticker": ticker,
                "end_date": pd.Timestamp(end_date, unit="D"),
                "similarity": similarity,
                "path_correlation": round(float(correlations[idx]) * 100, 1),
                "dtw_similarity": round(dtw * 100, 1),
                "context_similarity": round(float(context_scores[idx]) * 100, 1),
                "volume_similarity": round(float(volume_scores[idx]) * 100, 1),
                "forward_3m": float(index.outcomes[idx, 0]),
                "forward_6m": float(index.outcomes[idx, 1]),
                "forward_12m": float(index.outcomes[idx, 2]),
                "matched_path": index.paths[idx],
                "future_path": index.future_paths[idx],
                "current_path": query_path,
            }
        )
        if len(chosen) >= top_n:
            break
    return pd.DataFrame(chosen)


def outcome_summary(matches: pd.DataFrame) -> dict[str, float]:
    if matches.empty:
        return {}
    returns = matches["forward_12m"].dropna()
    weights = np.maximum(matches.loc[returns.index, "similarity"].to_numpy() - 50, 1)
    return {
        "positive_rate": float(
            np.average((returns > 0).astype(float), weights=weights)
        ),
        "plus_25_rate": float(
            np.average((returns >= 0.25).astype(float), weights=weights)
        ),
        "minus_20_rate": float(
            np.average((returns <= -0.20).astype(float), weights=weights)
        ),
        "median_12m": float(returns.median()),
        "best_12m": float(returns.max()),
        "worst_12m": float(returns.min()),
    }


def download_histories(
    tickers: Iterable[str],
    period: str = "max",
    batch_size: int = 40,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, pd.DataFrame]:
    symbols = list(
        dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip())
    )
    histories: dict[str, pd.DataFrame] = {}
    for offset in range(0, len(symbols), batch_size):
        batch = symbols[offset : offset + batch_size]
        try:
            data = yf.download(
                batch,
                period=period,
                auto_adjust=True,
                actions=False,
                group_by="ticker",
                threads=True,
                progress=False,
            )
        except Exception:
            data = pd.DataFrame()
        for ticker in batch:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.get_level_values(0):
                        frame = data[ticker]
                    elif ticker in data.columns.get_level_values(1):
                        frame = data.xs(ticker, axis=1, level=1)
                    else:
                        continue
                else:
                    frame = data
                frame = _history_frame(frame)
                if not frame.empty:
                    histories[ticker] = frame
            except (KeyError, ValueError):
                continue
        if progress:
            progress(min(offset + batch_size, len(symbols)), len(symbols))
    return histories
