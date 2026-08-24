from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

from screener.analogs import comparison_points, find_analogs, load_catalog
from screener.data import (
    fetch_history,
    fetch_many,
    fetch_stock,
    sector_momentum,
    us_listed_universe,
    yahoo_discovery,
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


@st.cache_data(ttl=86_400, show_spinner=False)
def cached_analog_histories():
    return {
        ticker: fetch_history(ticker, "max")
        for ticker in load_catalog()["ticker"].unique()
    }


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


discover_tab, deep_tab, methodology_tab = st.tabs(
    ["⚡ Opportunity Radar", "◎ Stock Deep Dive", "◇ Model & Limits"]
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
            st.subheader("Historical pattern matches")
            st.caption(
                "Similarity uses price path, volatility, drawdown, low proximity, and volume regime as of each historical anchor. It does not claim the businesses are identical or use point-in-time fundamentals."
            )
            with st.spinner("Searching historical regimes…"):
                analogs = find_analogs(
                    price_features(history), cached_analog_histories()
                )
            if analogs.empty:
                st.info("Historical analog data is temporarily unavailable.")
            else:
                for analog_number, analog in enumerate(analogs.to_dict("records")):
                    similarities, differences = comparison_points(
                        price_features(history), analog
                    )
                    outcome = analog.get("forward_12m")
                    with st.expander(
                        f"{analog['similarity']:.0f}% pattern match · {ticker} now vs. {analog['label']}",
                        expanded=analog_number == 0,
                    ):
                        a, b = st.columns(2)
                        with a:
                            st.markdown("**Similarities**")
                            for point in similarities or [
                                "Broad regime match; no individual feature landed inside the strict tolerance."
                            ]:
                                st.write(f"• {point}")
                            st.markdown(f"**Historical setup:** {analog['story']}")
                        with b:
                            st.markdown("**Differences**")
                            for point in differences:
                                st.write(f"• {point}")
                            st.markdown(
                                f"**Structural risk then:** {analog['structural_risk']}"
                            )
                        st.metric(
                            "What happened over the following ~12 months",
                            pct(outcome),
                            help="Historical adjusted-price return, not a forecast.",
                        )

with methodology_tab:
    st.subheader("What the score actually rewards")
    st.markdown("""
    - **30% Growth engine:** revenue, earnings, and quarterly earnings growth.
    - **22% Buy-low setup:** closeness to the 52-week low, with stabilization and volume confirmation.
    - **18% Alive-business check:** margins, liquidity, leverage, and free cash flow.
    - **15% Momentum ignition:** one-, three-, six-, and twelve-month price behavior.
    - **15% Sector opportunity:** sector ETF momentum plus company growth relative to the sector regime.
    - **Up to −25 points in penalties:** shrinking revenue, cash burn with negative margins, extreme leverage, nano-cap risk, and unresolved falling-knife behavior.

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

st.divider()
st.caption(
    "Asymmetric Upside · Local-first, no-key prototype · Verify filings, liquidity, dilution, catalysts, and position sizing before risking capital."
)
