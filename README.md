# Bitcoin ML Trading Research System

Research-first BTCUSDT machine-learning trading system built around Binance market data.

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
- Backtest predictions, trades, equity curve and JSON performance report
- Regression tests for data validation and look-ahead leakage

## Research principles

The model is not treated as an oracle. A prediction only becomes a trade when the expected edge is large enough to clear estimated fees, slippage, and a configurable safety margin. This is motivated by recent BTC walk-forward research showing that transaction costs and the forecast-to-trade conversion can dominate raw model performance.

The historical training model uses only spot-derived features so the long history is not silently truncated by Binance's limited recent derivatives-history endpoints. Funding and open interest are treated as additional real-time context rather than pretending that a decade of derivatives data exists.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Commands

Download the historical dataset:

```bash
python trading_bot.py --mode data
```

Run the walk-forward backtest:

```bash
python trading_bot.py --mode backtest --interval 1h
```

Try the other model baselines:

```bash
python trading_bot.py --mode backtest --model random_forest
python trading_bot.py --mode backtest --model logistic
```

Generate a fresh real-time-style signal from the latest Binance candles:

```bash
python trading_bot.py --mode signal --interval 1h
```

Run tests:

```bash
pytest -q
```

## Outputs

The `outputs/` directory contains the downloaded history, features, out-of-sample predictions, trades, equity curve, latest signal, configuration, and performance report.

## Important limitation

This repository currently **does not place real-money orders**. The signal engine is for research/forward observation. Live execution should only be added after paper trading, exchange-specific execution tests, independent risk controls, secret management, and a long enough forward-validation period.
