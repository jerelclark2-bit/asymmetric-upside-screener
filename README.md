# Asymmetric Upside

**Buy low, sell high. Find asymmetric upside.**

A local-first Streamlit stock screener for aggressive investors. It ranks equities by potential growth, business survival, proximity to 52-week lows, early momentum, and sector opportunity. Dividends do not improve the score.

> This is speculative research software, not personalized investment advice. Its scenario probabilities are heuristic model outputs, not calibrated odds, guarantees, or price targets.

## What it does

- Analyzes any global equity ticker supported by Yahoo Finance (including exchange suffixes such as `9988.HK` or `ASML.AS`).
- Discovers candidates through multiple stock-only Yahoo Finance screens.
- Can sample the official Nasdaq Trader directory covering Nasdaq and other U.S. exchanges.
- Accepts pasted tickers or a CSV universe for repeatable batch research.
- Rewards growth, 52-week-low proximity with stabilization, momentum ignition, and sector strength.
- Penalizes stocks that have run too far above their 52-week low, plus shrinking revenue, cash burn, leverage, nano-cap risk, and falling knives.
- Produces bull/base/bear scenario weights, pros and cons, a component breakdown, and downloadable results.
- Searches a rolling local pattern index using exact path correlation, dynamic time warping, market context, and volume fingerprints.
- Shows what happened 3, 6, and 12 months after each match and can expand the database to as many as 5,000 equities.
- Ships with 64,353 historical windows from 408 stocks so pattern search works immediately without a long first-time build.

## Run locally

Python 3.11 or newer is recommended. It has also been tested locally on Python 3.14.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On macOS/Linux, the included one-command launcher performs those steps for you:

```bash
./run_local.sh
```

Then open the local address printed by Streamlit, normally `http://localhost:8501`.

No API key is required. Copy `.env.example` to `.env` only if you want to provide a descriptive user agent for official data downloads.

## Put it on GitHub

```bash
git init
git add .
git commit -m "Initial asymmetric-upside screener"
git branch -M main
git remote add origin https://github.com/YOUR-NAME/YOUR-REPO.git
git push -u origin main
```

The `.gitignore` excludes local environments, secrets, and caches. The project can also be deployed on Streamlit Community Cloud by selecting `app.py` as the entry point.
GitHub Actions runs the tests and linter automatically on each push and pull request.

## Scoring model

| Component | Weight | Intent |
|---|---:|---|
| Growth engine | 28% | Revenue and earnings expansion |
| Buy-low setup | 30% | Near a 52-week low, but stabilizing |
| Alive-business check | 17% | Margins, liquidity, leverage, and cash generation |
| Momentum ignition | 10% | Early acceleration rather than a chased move |
| Sector opportunity | 15% | Hot sectors or company growth emerging from a depressed regime |

Risk penalties can subtract up to 45 points. The buy-high tax begins 35% above the 52-week low and becomes severe beyond 75%, 120%, and 200%. See [`screener/scoring.py`](screener/scoring.py) for the complete, editable formula.

## Important limitations

- Yahoo Finance is a convenient free source, not an exchange-grade feed. Data can be delayed, unavailable, or rate-limited.
- A free endpoint cannot responsibly fetch complete fundamentals for every global listing in one instant. The app therefore analyzes any ticker but ranks configurable candidate pools.
- Historical matches compare chart and trading-regime features. They do not imply business equivalence, and historical outcomes are not forecasts.
- Fundamentals returned today are not point-in-time values for historical analog dates.
- Before investing, independently verify filings, dilution, debt maturities, liquidity, management claims, catalysts, and loss tolerance.

## Tests

```bash
python -m pytest
```
