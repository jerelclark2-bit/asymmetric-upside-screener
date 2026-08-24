from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def _number(value: Any, default: float = 0.0) -> float:
    try:
        value = float(value)
        return value if math.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def scale(value: Any, bad: float, good: float) -> float:
    """Map a metric to 0..100, supporting reversed bounds."""
    value = _number(value)
    if good == bad:
        return 50.0
    return clamp((value - bad) / (good - bad) * 100)


def price_features(
    history: pd.DataFrame, as_of: str | pd.Timestamp | None = None
) -> dict[str, float]:
    if history.empty or "Close" not in history:
        return {}
    frame = history.copy()
    frame.index = pd.to_datetime(frame.index).tz_localize(None)
    if as_of is not None:
        frame = frame.loc[: pd.Timestamp(as_of)]
    close = frame["Close"].dropna()
    if len(close) < 30:
        return {}
    window = close.tail(252)
    returns = window.pct_change().dropna()

    def trailing_return(days: int) -> float:
        sample = close.tail(days + 1)
        return (
            float(sample.iloc[-1] / sample.iloc[0] - 1)
            if len(sample) > days // 2
            else 0.0
        )

    high = float(window.max())
    low = float(window.min())
    last = float(close.iloc[-1])
    volume_trend = 0.0
    if "Volume" in frame and len(frame["Volume"].dropna()) >= 40:
        volume = frame["Volume"].dropna()
        recent = float(volume.tail(20).mean())
        prior = float(volume.tail(80).head(60).mean())
        volume_trend = recent / prior - 1 if prior else 0.0
    return {
        "return_1m": trailing_return(21),
        "return_3m": trailing_return(63),
        "return_6m": trailing_return(126),
        "return_12m": trailing_return(252),
        "volatility": float(returns.std() * np.sqrt(252)) if len(returns) else 0.0,
        "drawdown_from_52w_high": last / high - 1 if high else 0.0,
        "distance_from_52w_low": last / low - 1 if low else 0.0,
        "volume_trend": volume_trend,
        "last_price": last,
    }


@dataclass
class ScoreCard:
    ticker: str
    total: float
    components: dict[str, float]
    probabilities: dict[str, float]
    metrics: dict[str, Any]
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def label(self) -> str:
        if self.total >= 78:
            return "Explosive setup"
        if self.total >= 65:
            return "High-upside watch"
        if self.total >= 50:
            return "Speculative / mixed"
        return "Avoid or wait"


def _scenario_probabilities(score: float, volatility: float) -> dict[str, float]:
    """Heuristic 12-month scenario weights; these are not calibrated forecasts."""
    vol = clamp(volatility * 100, 10, 140)
    bull_logit = (score - 58) / 15 + (vol - 35) / 130
    bear_logit = (52 - score) / 15 + (vol - 35) / 100
    base_logit = 0.65 - abs(score - 58) / 42
    logits = np.array([bull_logit, base_logit, bear_logit], dtype=float)
    probs = np.exp(logits - logits.max())
    probs = probs / probs.sum() * 100
    return {
        "Bull (+25% or more)": round(float(probs[0]), 1),
        "Base (-20% to +25%)": round(float(probs[1]), 1),
        "Bear (-20% or worse)": round(float(probs[2]), 1),
    }


def score_stock(
    ticker: str, metrics: dict[str, Any], price: dict[str, float]
) -> ScoreCard:
    revenue_growth = _number(metrics.get("revenueGrowth"))
    earnings_growth = _number(metrics.get("earningsGrowth"))
    gross_margin = _number(metrics.get("grossMargins"))
    operating_margin = _number(metrics.get("operatingMargins"))
    debt_to_equity = _number(metrics.get("debtToEquity"))
    current_ratio = _number(metrics.get("currentRatio"))
    free_cashflow = _number(metrics.get("freeCashflow"))
    market_cap = _number(metrics.get("marketCap"))
    sector_momentum = _number(metrics.get("sectorMomentum6m"))
    analyst_growth = _number(metrics.get("earningsQuarterlyGrowth"))

    growth = (
        0.55 * scale(revenue_growth, -0.10, 0.45)
        + 0.3 * scale(earnings_growth, -0.25, 0.60)
        + 0.15 * scale(analyst_growth, -0.25, 0.50)
    )
    survival = (
        0.30 * scale(gross_margin, 0.05, 0.65)
        + 0.25 * scale(operating_margin, -0.25, 0.25)
        + 0.20 * scale(current_ratio, 0.5, 2.5)
        + 0.15 * scale(debt_to_equity, 260, 20)
        + 0.10 * (100 if free_cashflow > 0 else 20)
    )

    dlow = price.get("distance_from_52w_low", 1.0)
    r3 = price.get("return_3m", 0.0)
    r6 = price.get("return_6m", 0.0)
    r12 = price.get("return_12m", 0.0)
    volume_trend = price.get("volume_trend", 0.0)
    # Reward proximity to the low, but require some stabilization to avoid pure falling knives.
    low_proximity = scale(dlow, 0.80, 0.02)
    stabilization = 0.55 * scale(r3, -0.25, 0.25) + 0.45 * scale(
        volume_trend, -0.30, 0.70
    )
    setup = 0.60 * low_proximity + 0.40 * stabilization
    momentum = (
        0.45 * scale(r3, -0.35, 0.45)
        + 0.35 * scale(r6, -0.50, 0.80)
        + 0.20 * scale(r12, -0.65, 1.50)
    )
    sector = 0.70 * scale(sector_momentum, -0.30, 0.40) + 0.30 * scale(
        revenue_growth - sector_momentum / 2, -0.15, 0.40
    )

    components = {
        "Growth engine": round(growth, 1),
        "Alive-business check": round(survival, 1),
        "Buy-low setup": round(setup, 1),
        "Momentum ignition": round(momentum, 1),
        "Sector opportunity": round(sector, 1),
    }
    total = (
        0.30 * growth + 0.18 * survival + 0.22 * setup + 0.15 * momentum + 0.15 * sector
    )

    flags: list[str] = []
    penalty = 0.0
    if revenue_growth < -0.10:
        penalty += 10
        flags.append("Revenue is shrinking faster than 10%")
    if r6 < -0.45 and r3 < -0.15:
        penalty += 9
        flags.append("Falling-knife price action")
    if free_cashflow < 0 and operating_margin < 0:
        penalty += 7
        flags.append("Cash burn plus negative operating margin")
    if debt_to_equity > 300:
        penalty += 5
        flags.append("Very high reported leverage")
    if market_cap and market_cap < 75_000_000:
        penalty += 5
        flags.append("Nano-cap liquidity/manipulation risk")
    # yfinance's `dividendYield` is often expressed in percentage points
    # (0.47 means 0.47%), while the trailing field is a decimal ratio.
    trailing_yield = metrics.get("trailingAnnualDividendYield")
    dividend_yield = (
        _number(trailing_yield)
        if trailing_yield is not None
        else _number(metrics.get("dividendYield")) / 100
    )
    if dividend_yield > 0.06:
        penalty += 3
        flags.append("High yield may signal a mature or distressed setup")
    total = round(clamp(total - min(penalty, 25)), 1)

    pros: list[str] = []
    cons: list[str] = []
    if revenue_growth >= 0.20:
        pros.append(f"Revenue growth is running at {revenue_growth:.0%}")
    elif revenue_growth > 0:
        pros.append(f"Revenue is still expanding ({revenue_growth:.0%})")
    else:
        cons.append(f"Revenue growth is {revenue_growth:.0%}")
    if dlow <= 0.20:
        pros.append(f"Only {dlow:.0%} above its 52-week low")
    elif dlow > 1.0:
        cons.append(f"Already {dlow:.0%} above its 52-week low")
    if r3 > 0.12:
        pros.append(f"Three-month momentum has ignited (+{r3:.0%})")
    elif r3 < -0.15:
        cons.append(f"Three-month momentum remains weak ({r3:.0%})")
    if free_cashflow > 0:
        pros.append("Positive reported free cash flow supports survival")
    else:
        cons.append("Negative or unavailable free cash flow")
    if sector_momentum > 0.10:
        pros.append(f"Sector proxy is strong over six months (+{sector_momentum:.0%})")
    elif sector_momentum < -0.10:
        cons.append(
            f"Sector proxy is depressed ({sector_momentum:.0%} over six months)"
        )
    cons.extend(flags[:2])

    probabilities = _scenario_probabilities(total, price.get("volatility", 0.35))
    return ScoreCard(
        ticker.upper(),
        total,
        components,
        probabilities,
        {**metrics, **price},
        pros[:5],
        cons[:5],
        flags,
    )
