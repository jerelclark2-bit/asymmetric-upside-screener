from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from screener.data import (
    fetch_many,
    fetch_stock,
    sector_momentum,
    us_listed_universe,
    yahoo_discovery,
)
from screener.patterns import (
    CUSTOM_INDEX,
    STARTER_INDEX,
    build_pattern_index,
    download_histories,
    find_pattern_matches,
    load_pattern_index,
    outcome_summary,
)
from screener.scoring import ScoreCard, price_features, score_stock

load_dotenv()
st.set_page_config(page_title="Asymmetric Upside", page_icon="↗", layout="wide")

st.markdown(
    """
    <style>
    .block-container {max-width: 1320px; padding-top: 2rem;}
    .hero {padding: 1.5rem 1.7rem; border: 1px solid #24593A; border-radius: 18px;
           background: radial-gradient(circle at top right, #173C29, #09140E 65%); margin-bottom: 1rem;}
    .eyebrow {color:#38E07B; text-transform:uppercase; letter-spacing:.18em; font-size:.72rem; font-weight:800;}
    .hero h1 {font-size:clamp(2rem,5vw,4.2rem); line-height:.98; margin:.45rem 0 .65rem; letter-spacing:-.04em;}
    .hero p {color:#B9C8BF; max-width:760px; font-size:1.03rem;}
    .score {font-size:3.6rem; line-height:1; font-weight:850; color:#38E07B;}
    .card {border:1px solid #1E4931; border-radius:14px; padding:1rem; background:#0C1B13; min-height:130px;}
    .fine {font-size:.78rem; color:#93A59A;}
    div[data-testid="stMetric"] {background:#0C1B13; border:1px solid #1E4931; padding:.75rem; border-radius:12px;}
    </style>
    <div class="hero">
      <div class="eyebrow">Asymmetric Upside // Aggressive Equity Intelligence</div>
      <h1>Buy low, sell high.<br>Find asymmetric upside.</h1>
      <p>A growth-first stock screener for finding alive businesses in hated, emerging, or accelerating sectors—then stress-testing the upside against the downside.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.warning(
    "Speculative research only—not personalized investment advice. Scenario probabilities are heuristic model estimates, not calibrated odds or promises.",
    icon="⚠️",
)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_stock(ticker: str):
    info, history = fetch_stock(ticker)
    info["sectorMomentum6m"] = sector_momentum(info.get("sector", ""))
    return info, history


@st.cache_data(ttl=3600, show_spinner=False)
def cached_discovery(limit: int):
    return yahoo_discovery(limit)


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_universe():
    return us_listed_universe()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_many(tickers: tuple[str, ...]):
    return fetch_many(tickers)


@st.cache_resource(show_spinner=False)
def cached_pattern_index(starter_modified: float, custom_modified: float):
    del starter_modified, custom_modified  # Modification times form the cache key.
    return load_pattern_index()


def active_pattern_index():
    custom_modified = CUSTOM_INDEX.stat().st_mtime if CUSTOM_INDEX.exists() else 0.0
    return cached_pattern_index(STARTER_INDEX.stat().st_mtime, custom_modified)


def money(value) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    for suffix, divisor in [("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)]:
        if abs(value) >= divisor:
            return f"${value / divisor:,.1f}{suffix}"
    return f"${value:,.2f}"


def pct(value) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "—"


def stock_card(ticker: str, info: dict, history: pd.DataFrame) -> ScoreCard | None:
    features = price_features(history)
    return score_stock(ticker, info, features) if features else None


def price_chart(history: pd.DataFrame, ticker: str):
    frame = history.tail(756).copy()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=frame.index,
            y=frame["Close"],
            name=ticker,
            line={"color": "#38E07B", "width": 2},
        )
    )
    if len(frame) >= 200:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["Close"].rolling(200).mean(),
                name="200-day",
                line={"color": "#F2B84B", "width": 1.2, "dash": "dot"},
            )
        )
    fig.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=25, b=10),
        template="plotly_dark",
        paper_bgcolor="#07110D",
        plot_bgcolor="#07110D",
        legend=dict(orientation="h"),
        yaxis_title="Adjusted price",
    )
    return fig


def match_quality(score: float) -> str:
    if score >= 90:
        return "Dead-on"
    if score >= 82:
        return "Tight"
    if score >= 74:
        return "Strong"
    return "Loose"


def pattern_overlay_chart(matches: pd.DataFrame, ticker: str):
    fig = go.Figure()
    days = list(range(-125, 1, 4))[:32]
    current = matches.iloc[0]["current_path"]
    fig.add_trace(
        go.Scatter(
            x=days,
            y=current,
            name=f"{ticker} now",
            line={"color": "#FFFFFF", "width": 4},
        )
    )
    colors = ["#38E07B", "#F2B84B", "#8B9DFF"]
    for color, (_, match) in zip(colors, matches.head(3).iterrows(), strict=False):
        fig.add_trace(
            go.Scatter(
                x=days,
                y=match["matched_path"],
                name=f"{match['ticker']} · {match['end_date']:%Y-%m-%d}",
                line={"color": color, "width": 2},
            )
        )
    fig.update_layout(
        height=450,
        template="plotly_dark",
        paper_bgcolor="#07110D",
        plot_bgcolor="#07110D",
        margin=dict(l=20, r=20, t=105, b=70),
        xaxis_title="Trading days before the match date",
        yaxis_title="Standardized price shape",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.08,
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(7,17,13,0.92)",
        ),
        xaxis=dict(title_standoff=22, automargin=True),
        yaxis=dict(title_standoff=14, automargin=True),
    )
    return fig


def future_paths_chart(matches: pd.DataFrame):
    fig = go.Figure()
    days = list(range(0, 253, 8))[:32]
    for _, match in matches.head(5).iterrows():
        future_return = (
            np.exp(np.asarray(match["future_path"], dtype=float)) - 1
        ) * 100
        fig.add_trace(
            go.Scatter(
                x=days,
                y=future_return,
                name=f"{match['ticker']} · {match['end_date']:%Y-%m-%d}",
            )
        )
    fig.add_hline(y=0, line_dash="dot", line_color="#718078")
    fig.update_layout(
        height=475,
        template="plotly_dark",
        paper_bgcolor="#07110D",
        plot_bgcolor="#07110D",
        margin=dict(l=20, r=20, t=125, b=75),
        xaxis_title="Trading days after the historical match",
        yaxis_title="Return after match (%)",
        legend=dict(
            orientation="h",
            x=0,
            xanchor="left",
            y=1.10,
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(7,17,13,0.92)",
        ),
        xaxis=dict(title_standoff=24, automargin=True),
        yaxis=dict(title_standoff=14, automargin=True),
    )
    return fig


def render_score(card: ScoreCard, info: dict, history: pd.DataFrame):
    left, right = st.columns([1, 2.2])
    with left:
        st.markdown(
            f"<div class='card'><div class='eyebrow'>Aggressive upside score</div><div class='score'>{card.total:.0f}</div><b>{card.label}</b><div class='fine'>0–100, after survival and falling-knife penalties</div></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        st.caption("12-month scenario weights")
        for label, probability in card.probabilities.items():
            st.write(f"**{label}** · {probability:.1f}%")
            st.progress(probability / 100)
    with right:
        component_frame = pd.DataFrame(
            {"Component": card.components.keys(), "Score": card.components.values()}
        )
        fig = go.Figure(
            go.Bar(
                x=component_frame["Score"],
                y=component_frame["Component"],
                orientation="h",
                marker_color="#38E07B",
            )
        )
        fig.update_layout(
            height=330,
            xaxis_range=[0, 100],
            margin=dict(l=10, r=10, t=15, b=10),
            template="plotly_dark",
            paper_bgcolor="#07110D",
            plot_bgcolor="#07110D",
        )
        st.plotly_chart(fig, width="stretch")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Price", money(card.metrics.get("last_price")))
    m2.metric("Market cap", money(info.get("marketCap")))
    m3.metric("Revenue growth", pct(info.get("revenueGrowth")))
    m4.metric("Above 52w low", pct(card.metrics.get("distance_from_52w_low")))
    m5.metric("6m sector proxy", pct(info.get("sectorMomentum6m")))

    pro_col, con_col = st.columns(2)
    with pro_col:
        st.subheader("Why it could rip")
        for item in card.pros or [
            "No decisive upside signal available from the free dataset."
        ]:
            st.success(item, icon="↗")
    with con_col:
        st.subheader("What could break")
        for item in card.cons or [
            "Free data may omit material company-specific risks."
        ]:
            st.error(item, icon="↘")
    st.plotly_chart(price_chart(history, card.ticker), width="stretch")


discover_tab, deep_tab, pattern_tab, methodology_tab = st.tabs(
    [
        "⚡ Opportunity Radar",
        "◎ Stock Deep Dive",
        "⌁ Pattern Lab",
        "◇ Model & Limits",
    ]
)

with discover_tab:
    st.subheader("Search the aggressive edge")
    st.caption(
        "Discovery builds a stock-only candidate pool, then ranks it using growth, survival, 52-week-low setup, momentum ignition, and sector opportunity. Dividends add no points."
    )
    source = st.radio(
        "Candidate source",
        ["Aggressive global discovery", "Paste / upload tickers", "U.S. listed sample"],
        horizontal=True,
    )
    col_a, col_b = st.columns([2, 1])
    with col_a:
        pasted = st.text_area(
            "Ticker list",
            placeholder="RKLB, SOFI, TSM, 9988.HK, ASML.AS",
            disabled=source != "Paste / upload tickers",
        )
        upload = st.file_uploader(
            "Or upload a CSV with a symbol/ticker column",
            type="csv",
            disabled=source != "Paste / upload tickers",
        )
    with col_b:
        pool_size = st.slider("Candidate pool", 10, 150, 40, 10)
        min_score = st.slider("Minimum score", 0, 90, 45, 5)
        low_only = st.toggle(
            "Near 52-week low only",
            value=False,
            help="Keep stocks no more than 25% above their 52-week low.",
        )
        run_scan = st.button("Run aggressive scan", type="primary", width="stretch")

    if run_scan:
        if source == "Aggressive global discovery":
            tickers = cached_discovery(pool_size)
        elif source == "U.S. listed sample":
            universe = cached_universe()
            # Rotate the deterministic sample daily so repeated scans remain reproducible.
            day_seed = int(datetime.now(UTC).strftime("%Y%m%d"))
            tickers = universe.sample(
                min(pool_size, len(universe)), random_state=day_seed
            )["symbol"].tolist()
            st.info(
                "This mode samples the official U.S. directory. Use an uploaded universe for exhaustive batch research; free quote sources rate-limit thousands of simultaneous fundamentals requests."
            )
        else:
            tickers = [
                x.strip().upper()
                for x in pasted.replace("\n", ",").split(",")
                if x.strip()
            ]
            if upload is not None:
                uploaded = pd.read_csv(upload)
                candidate = next(
                    (c for c in uploaded.columns if c.lower() in {"symbol", "ticker"}),
                    None,
                )
                if candidate:
                    tickers.extend(
                        uploaded[candidate].dropna().astype(str).str.upper().tolist()
                    )
            tickers = list(dict.fromkeys(tickers))[:pool_size]
        if not tickers:
            st.error(
                "No equity tickers were found. Add a ticker list or try discovery again."
            )
        else:
            with st.spinner(f"Scoring {len(tickers)} stocks…"):
                bundles = cached_many(tuple(tickers))
            rows = []
            for ticker, (info, history) in bundles.items():
                card = stock_card(ticker, info, history)
                if not card or card.total < min_score:
                    continue
                if low_only and card.metrics.get("distance_from_52w_low", 99) > 0.25:
                    continue
                rows.append(
                    {
                        "Ticker": ticker,
                        "Company": info.get("shortName", ticker),
                        "Score": card.total,
                        "Setup": card.label,
                        "Sector": info.get("sector", "Unknown"),
                        "Price": card.metrics.get("last_price"),
                        "Above 52w low": card.metrics.get("distance_from_52w_low", 0)
                        * 100,
                        "Revenue growth": (info.get("revenueGrowth") * 100)
                        if info.get("revenueGrowth") is not None
                        else None,
                        "3m return": card.metrics.get("return_3m", 0) * 100,
                        "Bull case": card.probabilities["Bull (+25% or more)"],
                        "Market cap": info.get("marketCap"),
                        "Red flags": "; ".join(card.flags),
                    }
                )
            result = (
                pd.DataFrame(rows).sort_values("Score", ascending=False)
                if rows
                else pd.DataFrame()
            )
            if result.empty:
                st.warning(
                    "Nothing cleared the current filters. Lower the minimum score or expand the pool."
                )
            else:
                st.session_state["last_scan"] = result
                st.dataframe(
                    result,
                    width="stretch",
                    hide_index=True,
                    column_config={
                        "Score": st.column_config.ProgressColumn(
                            min_value=0, max_value=100, format="%.0f"
                        ),
                        "Price": st.column_config.NumberColumn(format="$%.2f"),
                        "Above 52w low": st.column_config.NumberColumn(format="%.1f%%"),
                        "Revenue growth": st.column_config.NumberColumn(
                            format="%.1f%%"
                        ),
                        "3m return": st.column_config.NumberColumn(format="%.1f%%"),
                        "Bull case": st.column_config.NumberColumn(format="%.1f%%"),
                        "Market cap": st.column_config.NumberColumn(format="$%.0f"),
                    },
                )
                st.download_button(
                    "Download ranked CSV",
                    result.to_csv(index=False),
                    "asymmetric_upside_scan.csv",
                    "text/csv",
                )

with deep_tab:
    ticker = (
        st.text_input(
            "Enter any Yahoo Finance equity ticker",
            value="NVDA",
            placeholder="Examples: NVDA, TSM, 9988.HK, ASML.AS",
        )
        .strip()
        .upper()
    )
    analyze = st.button("Analyze stock", type="primary")
    if analyze and ticker:
        with st.spinner(f"Building the {ticker} thesis…"):
            try:
                info, history = cached_stock(ticker)
            except Exception as exc:
                st.error(f"Could not load {ticker}: {exc}")
                info, history = {}, pd.DataFrame()
        card = stock_card(ticker, info, history)
        if card is None:
            st.error(
                "No usable equity price history was returned. Check the exchange suffix and ticker."
            )
        else:
            st.header(
                f"{info.get('longName') or info.get('shortName') or ticker} · {ticker}"
            )
            st.caption(
                f"{info.get('sector', 'Unknown sector')} / {info.get('industry', 'Unknown industry')} · Data timestamp: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}"
            )
            render_score(card, info, history)
            st.subheader("Precision historical pattern search")
            st.caption(
                "The engine searches rolling six-month windows—not a hand-picked story list. It combines exact path correlation, time-warped shape, volatility/drawdown context, and volume behavior."
            )
            with st.spinner("Searching historical regimes…"):
                pattern_index = active_pattern_index()
                matches = find_pattern_matches(
                    history, pattern_index, query_ticker=ticker, top_n=20
                )
            if matches.empty:
                st.info("The local pattern database has no usable matches yet.")
            else:
                summary = outcome_summary(matches)
                best = matches.iloc[0]
                st.markdown(
                    f"<div class='card'><div class='eyebrow'>Closest historical fingerprint</div>"
                    f"<div class='score'>{best['similarity']:.0f}%</div>"
                    f"<b>{match_quality(best['similarity'])} · {best['ticker']} ending {best['end_date']:%B %Y}</b>"
                    f"<div class='fine'>Searched {pattern_index.size:,} windows across {pattern_index.stock_count:,} stocks</div></div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(pattern_overlay_chart(matches, ticker), width="stretch")

                p1, p2, p3, p4 = st.columns(4)
                p1.metric("Historical matches positive", pct(summary["positive_rate"]))
                p2.metric("Reached +25% in 12m", pct(summary["plus_25_rate"]))
                p3.metric("Fell −20% in 12m", pct(summary["minus_20_rate"]))
                p4.metric("Median 12m outcome", pct(summary["median_12m"]))
                st.caption(
                    "These are similarity-weighted outcomes from the closest historical windows, not probabilities or a forecast."
                )

                display = matches.head(12)[
                    [
                        "ticker",
                        "end_date",
                        "similarity",
                        "path_correlation",
                        "dtw_similarity",
                        "context_similarity",
                        "volume_similarity",
                        "forward_3m",
                        "forward_6m",
                        "forward_12m",
                    ]
                ].copy()
                display.columns = [
                    "Ticker",
                    "Pattern ended",
                    "Overall match",
                    "Path correlation",
                    "Time-warp match",
                    "Market context",
                    "Volume match",
                    "Next 3m",
                    "Next 6m",
                    "Next 12m",
                ]
                for column in ["Next 3m", "Next 6m", "Next 12m"]:
                    display[column] *= 100
                st.dataframe(
                    display,
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "Overall match": st.column_config.ProgressColumn(
                            min_value=0, max_value=100, format="%.1f%%"
                        ),
                        "Path correlation": st.column_config.NumberColumn(
                            format="%.1f%%"
                        ),
                        "Time-warp match": st.column_config.NumberColumn(
                            format="%.1f%%"
                        ),
                        "Market context": st.column_config.NumberColumn(
                            format="%.1f%%"
                        ),
                        "Volume match": st.column_config.NumberColumn(format="%.1f%%"),
                        "Next 3m": st.column_config.NumberColumn(format="%.1f%%"),
                        "Next 6m": st.column_config.NumberColumn(format="%.1f%%"),
                        "Next 12m": st.column_config.NumberColumn(format="%.1f%%"),
                    },
                )
                st.subheader("What happened after the five closest matches")
                st.plotly_chart(future_paths_chart(matches), width="stretch")

                with st.expander("How exact is each match score?"):
                    st.markdown(
                        """
                        - **45% exact path correlation:** did the six-month chart bend and reverse in the same places?
                        - **25% dynamic time warping:** is it the same shape if one stock moved a little faster or slower?
                        - **20% market context:** were return, volatility, drawdown, and distance from the low comparable?
                        - **10% volume fingerprint:** did participation expand and contract similarly?

                        The engine caps clustered results from one ticker and requires non-overlapping eras. A 90%+ result is labeled “dead-on”; lower scores are shown honestly instead of being forced into a dramatic analogy.
                        """
                    )

with pattern_tab:
    st.subheader("Local pattern database")
    current_index = active_pattern_index()
    active_name = (
        "Bundled + custom index" if CUSTOM_INDEX.exists() else "Bundled starter index"
    )
    i1, i2, i3, i4 = st.columns(4)
    i1.metric("Active database", active_name)
    i2.metric("Historical windows", f"{current_index.size:,}")
    i3.metric("Stocks represented", f"{current_index.stock_count:,}")
    i4.metric("History begins", str(current_index.first_date.year))
    st.write(
        "Every stock contributes rolling 126-trading-day fingerprints. Monthly sampling creates many independent eras without filling the results with near-duplicate dates. Build a larger index locally when you want broader coverage."
    )

    st.subheader("Build or expand the matcher")
    build_source = st.radio(
        "Universe for the new local index",
        ["Aggressive discovery", "U.S. listed sample", "Paste tickers"],
        horizontal=True,
        key="pattern_source",
    )
    build_a, build_b = st.columns([2, 1])
    with build_a:
        pattern_tickers = st.text_area(
            "Tickers for pattern indexing",
            placeholder="MU, PLTR, NVDA, RKLB, TSM, 9988.HK",
            disabled=build_source != "Paste tickers",
        )
    with build_b:
        index_size = st.slider("Stocks to index", 50, 5000, 250, 50)
        index_period = st.selectbox("History per stock", ["10y", "max"], index=1)
        sampling = st.selectbox(
            "Historical window spacing",
            ["Monthly · balanced", "Biweekly · more matches", "Quarterly · fastest"],
        )
    build_button = st.button("Build expanded pattern database", type="primary")

    if build_button:
        if build_source == "Aggressive discovery":
            symbols = yahoo_discovery(min(index_size, 250))
        elif build_source == "U.S. listed sample":
            universe = us_listed_universe()
            symbols = universe.sample(
                min(index_size, len(universe)), random_state=20260824
            )["symbol"].tolist()
        else:
            symbols = [
                value.strip().upper()
                for value in pattern_tickers.replace("\n", ",").split(",")
                if value.strip()
            ][:index_size]
        if not symbols:
            st.error("Add at least one valid equity ticker.")
        else:
            progress_bar = st.progress(0, text="Downloading adjusted price histories…")

            def update_pattern_progress(done: int, total: int):
                progress_bar.progress(
                    done / total, text=f"Downloaded {done:,} of {total:,} tickers"
                )

            histories = download_histories(
                symbols, period=index_period, progress=update_pattern_progress
            )
            spacing = {
                "Monthly · balanced": 21,
                "Biweekly · more matches": 10,
                "Quarterly · fastest": 63,
            }[sampling]
            if histories:
                with st.spinner("Encoding millions of comparable shape features…"):
                    built_index = build_pattern_index(
                        histories, CUSTOM_INDEX, step=spacing
                    )
                cached_pattern_index.clear()
                progress_bar.progress(1.0, text="Pattern database ready")
                st.success(
                    f"Built {built_index.size:,} historical windows from {built_index.stock_count:,} stocks. Re-run a deep dive to use it."
                )
            else:
                st.error(
                    "No usable histories were downloaded. Try a smaller universe or retry later."
                )

    if CUSTOM_INDEX.exists():
        st.caption(
            f"Custom index saved locally at `{CUSTOM_INDEX}`. Delete that file to fall back to the bundled starter index."
        )

with methodology_tab:
    st.subheader("What the score actually rewards")
    st.markdown("""
    - **28% Growth engine:** revenue, earnings, and quarterly earnings growth.
    - **30% Buy-low setup:** closeness to the 52-week low, with stabilization and volume confirmation.
    - **17% Alive-business check:** margins, liquidity, leverage, and free cash flow.
    - **10% Momentum ignition:** early price acceleration, without rewarding a fully chased move.
    - **15% Sector opportunity:** sector ETF momentum plus company growth relative to the sector regime.
    - **Up to −45 points in penalties:** includes a hard buy-high tax beginning 35% above the 52-week low, plus shrinking revenue, cash burn, leverage, nano-cap, and falling-knife risks.

    Dividends receive **zero positive weight**. An unusually high yield can be a small distress/maturity penalty. A low price alone is never enough: the business must show signs of life.
    """)
    st.subheader("How to read the probabilities")
    st.write(
        "Bull, base, and bear percentages are transparent scenario weights derived from the total score and realized volatility. They are intentionally called heuristic estimates: they have not been calibrated on a complete point-in-time global fundamentals database, and they should not be treated as odds of profit."
    )
    st.subheader("Coverage and data limits")
    st.write(
        "The deep dive accepts global Yahoo Finance equity symbols. Discovery combines several stock-only Yahoo screens; the U.S. mode uses Nasdaq Trader’s official Nasdaq/NYSE/NYSE American/Arca/BATS/IEX directory. Global exchanges use suffixes such as `.HK`, `.L`, or `.AS`. Free data can be delayed, missing, restated, or rate-limited."
    )
    st.subheader("Pattern matching limits")
    st.write(
        "Pattern matches are empirical chart analogs, not claims that two companies share the same fundamentals. Expanding the local database improves coverage, but even a dead-on historical shape can resolve differently because news, valuation, dilution, interest rates, and the underlying business are different."
    )

st.divider()
st.caption(
    "Asymmetric Upside · Local-first, no-key prototype · Verify filings, liquidity, dilution, catalysts, and position sizing before risking capital."
)
