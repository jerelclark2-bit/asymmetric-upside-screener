import pandas as pd

from screener.scoring import price_features, score_stock


def test_price_features_identifies_52_week_low_proximity():
    index = pd.date_range("2024-01-01", periods=260, freq="B")
    close = [100 - i * 0.1 for i in range(240)] + [76 + i * 0.2 for i in range(20)]
    history = pd.DataFrame({"Close": close, "Volume": [1_000_000] * 260}, index=index)
    features = price_features(history)
    assert 0 <= features["distance_from_52w_low"] < 0.1
    assert features["drawdown_from_52w_high"] < -0.15


def test_growth_and_survival_score_beats_dying_business():
    history_features = {
        "distance_from_52w_low": 0.1,
        "return_3m": 0.1,
        "return_6m": 0.05,
        "return_12m": -0.1,
        "volatility": 0.5,
        "volume_trend": 0.2,
    }
    strong = score_stock(
        "UP",
        {
            "revenueGrowth": 0.5,
            "earningsGrowth": 0.6,
            "grossMargins": 0.65,
            "operatingMargins": 0.2,
            "currentRatio": 2.0,
            "debtToEquity": 20,
            "freeCashflow": 10,
            "sectorMomentum6m": 0.15,
            "marketCap": 2e9,
        },
        history_features,
    )
    dying = score_stock(
        "DOWN",
        {
            "revenueGrowth": -0.3,
            "earningsGrowth": -0.5,
            "grossMargins": 0.05,
            "operatingMargins": -0.4,
            "currentRatio": 0.4,
            "debtToEquity": 500,
            "freeCashflow": -10,
            "sectorMomentum6m": -0.2,
            "marketCap": 2e9,
        },
        history_features,
    )
    assert strong.total > dying.total + 35
    assert "Revenue is shrinking faster than 10%" in dying.flags


def test_probabilities_sum_to_100_with_rounding_tolerance():
    card = score_stock("X", {}, {"volatility": 0.5})
    assert abs(sum(card.probabilities.values()) - 100) <= 0.2


def test_percentage_point_dividend_yield_does_not_trigger_false_distress_flag():
    card = score_stock("X", {"dividendYield": 0.47}, {"volatility": 0.5})
    assert "High yield may signal a mature or distressed setup" not in card.flags


def test_stock_far_above_52_week_low_gets_hard_buy_high_tax():
    fundamentals = {
        "revenueGrowth": 0.7,
        "earningsGrowth": 0.8,
        "grossMargins": 0.7,
        "operatingMargins": 0.3,
        "currentRatio": 2.5,
        "debtToEquity": 10,
        "freeCashflow": 100,
        "sectorMomentum6m": 0.4,
        "marketCap": 50e9,
    }
    common = {
        "return_3m": 0.3,
        "return_6m": 0.6,
        "return_12m": 1.2,
        "volatility": 0.6,
        "volume_trend": 0.2,
    }
    low_entry = score_stock(
        "LOW",
        fundamentals,
        {
            **common,
            "distance_from_52w_low": 0.12,
            "drawdown_from_52w_high": -0.35,
        },
    )
    chased = score_stock(
        "CHASE",
        fundamentals,
        {
            **common,
            "distance_from_52w_low": 2.1,
            "drawdown_from_52w_high": -0.02,
        },
    )
    assert chased.total <= low_entry.total - 35
    assert any(flag.startswith("Buy-high tax") for flag in chased.flags)
