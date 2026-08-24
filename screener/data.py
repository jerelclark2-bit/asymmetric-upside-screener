from __future__ import annotations

import io
import os
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache

import pandas as pd
import requests
import yfinance as yf

NASDAQ_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"

SECTOR_ETFS = {
    "Technology": "XLK",
    "Healthcare": "XLV",
    "Financial Services": "XLF",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
    "Communication Services": "XLC",
}

DISCOVERY_SCREENS = [
    "aggressive_small_caps",
    "growth_technology_stocks",
    "undervalued_growth_stocks",
    "small_cap_gainers",
    "most_actives",
    "day_gainers",
]


def _clean_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return frame


def fetch_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    return _clean_history(
        yf.Ticker(ticker).history(period=period, auto_adjust=True, actions=False)
    )


def fetch_stock(ticker: str) -> tuple[dict, pd.DataFrame]:
    asset = yf.Ticker(ticker.strip().upper())
    history = _clean_history(
        asset.history(period="5y", auto_adjust=True, actions=False)
    )
    try:
        info = asset.info or {}
    except Exception:
        info = {}
    info["symbol"] = info.get("symbol", ticker.upper())
    return info, history


@lru_cache(maxsize=32)
def sector_momentum(sector: str) -> float:
    proxy = SECTOR_ETFS.get(sector)
    if not proxy:
        return 0.0
    history = fetch_history(proxy, "1y")
    close = history.get("Close", pd.Series(dtype=float)).dropna()
    return float(close.iloc[-1] / close.iloc[-126] - 1) if len(close) >= 126 else 0.0


def enrich_stock(ticker: str) -> tuple[dict, pd.DataFrame]:
    info, history = fetch_stock(ticker)
    info["sectorMomentum6m"] = sector_momentum(info.get("sector", ""))
    return info, history


def yahoo_discovery(limit: int = 75) -> list[str]:
    symbols: list[str] = []
    each = max(10, min(50, limit // len(DISCOVERY_SCREENS) + 5))
    for name in DISCOVERY_SCREENS:
        try:
            response = yf.screen(name, count=each)
            quotes = response.get("quotes", []) if isinstance(response, dict) else []
            for quote in quotes:
                if quote.get("quoteType") == "EQUITY" and quote.get("symbol"):
                    symbols.append(quote["symbol"])
        except Exception:
            continue
    return list(dict.fromkeys(symbols))[:limit]


def us_listed_universe() -> pd.DataFrame:
    headers = {
        "User-Agent": os.getenv("SEC_USER_AGENT", "AsymmetricUpsideScreener/1.0")
    }
    frames = []
    for url, symbol_col in [(NASDAQ_LISTED, "Symbol"), (OTHER_LISTED, "ACT Symbol")]:
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        frame = pd.read_csv(io.StringIO(response.text), sep="|")
        frame = frame[
            ~frame[symbol_col].astype(str).str.startswith("File Creation Time")
        ]
        frame = frame.rename(columns={symbol_col: "symbol", "Security Name": "name"})
        if "Test Issue" in frame:
            frame = frame[frame["Test Issue"] != "Y"]
        frames.append(frame[["symbol", "name"]])
    result = pd.concat(frames, ignore_index=True).drop_duplicates("symbol")
    # Exclude obvious funds, warrants, rights, units, preferreds and test instruments.
    excluded = r"ETF|ETN|Warrant|Right|Unit|Preferred|Depositary Shares|Notes due"
    return result[
        ~result["name"].str.contains(excluded, case=False, na=False)
    ].reset_index(drop=True)


def fetch_many(
    tickers: Iterable[str], workers: int = 6
) -> dict[str, tuple[dict, pd.DataFrame]]:
    unique = list(
        dict.fromkeys(str(t).strip().upper() for t in tickers if str(t).strip())
    )
    results: dict[str, tuple[dict, pd.DataFrame]] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(unique)))) as pool:
        futures = {pool.submit(enrich_stock, ticker): ticker for ticker in unique}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                results[ticker] = future.result()
            except Exception:
                continue
    return results
