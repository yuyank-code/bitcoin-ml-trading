# Bitcoin ML Trading Research System

Research-first BTCUSDT machine-learning trading system built around Binance market data and multiple independent information layers.

## Data layers

1. Binance spot OHLCV and public market data
2. Free Coin Metrics Community + Blockchain.com on-chain data
3. Binance futures derivatives context
4. Free FRED macro/global-market data
5. Fear & Greed + modular crypto news sentiment
6. Binance market breadth + optional normalized ETF/Trends CSV inputs

## Authoritative research path

Use this sequence for final research/model selection:

```bash
python trading_bot.py --mode data
python derivatives_data.py
python sentiment_data.py
python macro_research.py
python onchain_research.py
python alternative_signals.py
python unified_research.py
python production_research.py
python validation.py
python live_readiness.py
```

`unified_research.py` creates causal technical features first, then joins other sources with conservative one-day lags. `production_research.py` is the authoritative leakage-safe model-selection/backtest path. `validation.py` checks probability quality and performance artifacts. `paper_trader.py` and `forward_monitor.py` provide paper-only forward evaluation.

## Leakage policy

- No random train/test shuffling.
- Features at timestamp `t` can only use information available by `t`.
- External sources are conservatively lagged before joining.
- Future returns/labels are evaluation targets only.
- The final decision rule uses model probability, stop/target geometry, and estimated transaction costs; it does **not** inspect realized future return.
- Macro data that has later revisions should ultimately be evaluated with point-in-time ALFRED vintages for release-sensitive research.

## Validation and stress testing

The system evaluates Brier score, log loss, accuracy, AUC, CAGR, drawdown, Sharpe, Sortino, win rate, profit factor, expectancy, turnover/cost sensitivity, and signal distributions. Feature-group ablation compares technical-only against each information layer and the combined candidate set.

## Paper trading

`paper_trader.py` is a local virtual portfolio only. It persists state, applies fees/slippage, and records virtual trades. `forward_monitor.py` repeatedly obtains the latest public Binance-derived signal and updates the virtual portfolio. Neither file has exchange credentials or live-order capability.

## Live execution status

**Disabled.** The project has no real-money execution path. `live_readiness.py` is a hard gate that remains false until sufficient forward paper evidence and independent execution/risk review exist. Binance authenticated order APIs require signed credentials; public market-data access is separate from trading authorization. Binance documents signed order-test/trading endpoints and separate market-data streams. 

## Important research limitation

Historical performance is not evidence of guaranteed future profit. Data coverage differs by source, derivatives history is limited for several statistics, and macro/news publication timing can create subtle revision or availability bias. The system therefore prefers conservative lags and rejects live execution until forward validation is complete.
