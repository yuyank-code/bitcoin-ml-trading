# Bitcoin ML Trading Research System

Research-first BTCUSDT machine-learning trading system built around Binance market data, with a free on-chain research layer.

## What is included

- Binance Spot OHLCV history with paginated downloads
- 1m/5m/15m/1h/4h/1d research intervals
- Strict expanding-window walk-forward evaluation
- No random time-series shuffling
- Technical, momentum, volatility, trend, and volume features
- Cost-aware trade filtering
- ATR-based stop/target logic
- Risk-based position sizing
- Maximum drawdown and daily-loss protection
- Gradient boosting, Random Forest, and Logistic Regression baselines
- Buy/sell probability output
- Optional recent Binance futures funding/open-interest context for live signals
- Free Coin Metrics Community API on-chain metrics
- Free Blockchain.com Charts API network metrics
- Point-in-time-safe one-day-lagged on-chain features
- Separate on-chain enriched walk-forward research pipeline
- Backtest predictions, trades, equity curve and JSON performance reports

## Task 2: free on-chain data

No paid Glassnode subscription is required. `onchain_data.py` discovers metrics available through Coin Metrics' Community API and combines them with selected public Blockchain.com charts. Coin Metrics documents its Community HTTP API as requiring no API key for community endpoints and provides a community rate limit; Blockchain.com documents its Charts API as a public interface to chart/statistics data.

Run the on-chain research pipeline:

```bash
python onchain_research.py
```

The pipeline downloads daily BTC market data, fetches available free/community on-chain data, aligns it by UTC day, applies a full-day lag before model use, creates changes/z-scores, and evaluates the enriched features using expanding-window walk-forward training. This is intentionally separate from the main intraday model until the on-chain features prove useful out of sample.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Download the historical Binance dataset:

```bash
python trading_bot.py --mode data
```

Run the walk-forward backtest:

```bash
python trading_bot.py --mode backtest --interval 1h
```

Generate a fresh real-time-style signal from the latest Binance candles:

```bash
python trading_bot.py --mode signal --interval 1h
```

Run the free on-chain enriched research:

```bash
python onchain_research.py
```

## Outputs

The `outputs/` directory contains market history, features, predictions, trades, equity curves, latest signals, and performance reports. Task 2 additionally creates `bitcoin_onchain_daily.csv`, `onchain_enriched_predictions.csv`, and `onchain_report.json`.

## Important limitation

This repository currently **does not place real-money orders**. The signal engine is for research/forward observation. Live execution should only be added after paper trading, exchange-specific execution tests, independent risk controls, secret management, and a long enough forward-validation period.
