# Bitcoin ML Trading Research System

A research-first Bitcoin (BTC-USD) machine-learning trading pipeline.

## Safety

The default mode is historical backtesting. Paper mode is intentionally scaffolded and live trading is not implemented. A positive backtest is not evidence of future profitability.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

Download data only:

```bash
python trading_bot.py --mode data
```

Run the walk-forward backtest:

```bash
python trading_bot.py --mode backtest
```

Try another baseline model:

```bash
python trading_bot.py --mode backtest --model random_forest
python trading_bot.py --mode backtest --model logistic
```

Paper mode is currently blocked by design:

```bash
python trading_bot.py --mode paper
```

## Outputs

The program writes datasets, predictions, trades, performance summaries, yearly/monthly returns, and charts to `outputs/`.

The backtest uses expanding-window walk-forward training and accounts for fees, slippage, risk-based position sizing, stop loss/take profit, daily loss limits, drawdown limits, and a Buy & Hold benchmark.
