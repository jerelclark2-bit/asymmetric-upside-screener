from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .scoring import price_features

CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "analog_catalog.csv"
FEATURES = [
    "return_3m",
    "return_6m",
    "return_12m",
    "volatility",
    "drawdown_from_52w_high",
    "distance_from_52w_low",
    "volume_trend",
]


def load_catalog() -> pd.DataFrame:
    return pd.read_csv(CATALOG_PATH)


def find_analogs(
    current: dict[str, float], histories: dict[str, pd.DataFrame], top_n: int = 3
) -> pd.DataFrame:
    catalog = load_catalog()
    rows: list[dict] = []
    scales = np.array([0.35, 0.55, 0.95, 0.40, 0.35, 0.80, 0.70])
    current_vector = np.array([current.get(key, 0.0) for key in FEATURES])
    for item in catalog.to_dict("records"):
        history = histories.get(item["ticker"], pd.DataFrame())
        historical = price_features(history, item["anchor_date"])
        if not historical:
            continue
        vector = np.array([historical.get(key, 0.0) for key in FEATURES])
        distance = float(np.sqrt(np.mean(((current_vector - vector) / scales) ** 2)))
        similarity = round(100 * np.exp(-distance), 1)
        close = (
            history["Close"].copy() if "Close" in history else pd.Series(dtype=float)
        )
        close.index = pd.to_datetime(close.index).tz_localize(None)
        anchor = pd.Timestamp(item["anchor_date"])
        before = close.loc[:anchor]
        after = close.loc[anchor:].head(253)
        outcome = (
            float(after.iloc[-1] / before.iloc[-1] - 1)
            if len(before) and len(after) > 100
            else np.nan
        )
        rows.append(
            {
                **item,
                "similarity": similarity,
                "historical_features": historical,
                "forward_12m": outcome,
            }
        )
    return (
        pd.DataFrame(rows).sort_values("similarity", ascending=False).head(top_n)
        if rows
        else pd.DataFrame()
    )


def comparison_points(
    current: dict[str, float], analog: dict
) -> tuple[list[str], list[str]]:
    old = analog["historical_features"]
    similarities: list[str] = []
    differences: list[str] = []
    labels = {
        "return_6m": "six-month return",
        "volatility": "annualized volatility",
        "drawdown_from_52w_high": "drawdown from the 52-week high",
        "distance_from_52w_low": "distance above the 52-week low",
    }
    tolerances = {
        "return_6m": 0.18,
        "volatility": 0.15,
        "drawdown_from_52w_high": 0.12,
        "distance_from_52w_low": 0.25,
    }
    for key, label in labels.items():
        cur, past = current.get(key, 0.0), old.get(key, 0.0)
        sentence = f"{label}: current {cur:.0%} vs. analog {past:.0%}"
        (similarities if abs(cur - past) <= tolerances[key] else differences).append(
            sentence
        )
    return similarities, differences
