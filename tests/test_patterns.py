import numpy as np
import pandas as pd

from screener.patterns import (
    build_pattern_index,
    encode_pattern,
    find_pattern_matches,
    load_pattern_index,
    merge_pattern_indexes,
)


def synthetic_history(seed: int, periods: int = 850) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recurring = np.sin(np.arange(periods) / 22) * 0.006
    returns = 0.0004 + recurring + rng.normal(0, 0.012, periods)
    close = 20 * np.exp(np.cumsum(returns))
    volume = rng.lognormal(14, 0.25, periods)
    return pd.DataFrame(
        {"Close": close, "Volume": volume},
        index=pd.date_range("2018-01-01", periods=periods, freq="B"),
    )


def test_pattern_encoding_is_scale_invariant():
    history = synthetic_history(1)
    first = encode_pattern(history["Close"].to_numpy(), history["Volume"].to_numpy())
    second = encode_pattern(
        (history["Close"] * 100).to_numpy(), history["Volume"].to_numpy()
    )
    np.testing.assert_allclose(first[0], second[0], atol=1e-6)


def test_index_roundtrip_and_search(tmp_path):
    histories = {"AAA": synthetic_history(2), "BBB": synthetic_history(3)}
    path = tmp_path / "patterns.npz"
    built = build_pattern_index(histories, path, step=21, min_dollar_volume=0)
    loaded = load_pattern_index(path)
    matches = find_pattern_matches(histories["AAA"], loaded, "AAA", top_n=5)
    assert built.size == loaded.size
    assert len(matches) == 2  # The query ticker is excluded; two eras remain for BBB.
    assert "AAA" not in matches["ticker"].tolist()
    assert matches["similarity"].is_monotonic_decreasing
    assert {"forward_3m", "forward_6m", "forward_12m"}.issubset(matches.columns)


def test_custom_index_extends_starter_without_same_ticker_matches(tmp_path):
    starter = build_pattern_index(
        {"AAA": synthetic_history(2), "BBB": synthetic_history(3)},
        tmp_path / "starter.npz",
        step=21,
        min_dollar_volume=0,
    )
    custom = build_pattern_index(
        {"AAA": synthetic_history(4)},
        tmp_path / "custom.npz",
        step=21,
        min_dollar_volume=0,
    )
    combined = merge_pattern_indexes(starter, custom)
    matches = find_pattern_matches(
        synthetic_history(2), combined, query_ticker="AAA", top_n=10
    )
    assert not matches.empty
    assert set(matches["ticker"]) == {"BBB"}
