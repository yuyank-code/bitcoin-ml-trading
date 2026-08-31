# Bitcoin ML Trading Research System

Research-first BTCUSDT machine-learning trading system built around Binance market data and multiple independent information layers.

## Data layers

1. Binance spot OHLCV and market data
2. Free Coin Metrics Community + Blockchain.com on-chain data
3. Binance futures derivatives context
4. Free FRED macro/global-market data
5. Fear & Greed + modular crypto news sentiment
6. Binance market breadth + optional normalized ETF/Trends CSV inputs

## Reference research path

The reference model-selection path is:

```text
python trading_bot.py --mode data
python derivatives_data.py
python sentiment_data.py
python macro_research.py
python onchain_research.py
python alternative_signals.py
python unified_research.py
python production_research.py
python validation.py
```

`unified_research.py` first creates causal technical features and then joins other sources with conservative one-day lags. `production_research.py` performs expanding-window walk-forward evaluation and feature-group ablation.

### Leakage policy

- No random train/test shuffling.
- Features at timestamp `t` can only use information available by `t`.
- External sources are conservatively lagged before joining.
- Future returns/labels are evaluation targets only.
- The final decision rule uses model probability, stop/target geometry, and estimated transaction costs; it does **not** inspect the realized future return.
- Macro data that has later revisions should ultimately be evaluated with point-in-time ALFRED vintages for release-sensitive research.

## Validation

`validation.py` reports probability quality, AUC/Brier/log-loss, trade statistics, equity metrics, and signal distributions for generated prediction files. The repository also contains regression tests covering OHLCV validation, feature causality, namespacing, and the probability/cost-based decision layer.

## Paper trading

`paper_trader.py` is a local virtual portfolio only. It persists state, applies fees/slippage, and records paper trades. It does not contain exchange credentials and cannot place real orders.

Real-money execution is intentionally disabled. Binance's signed order APIs require authenticated credentials; the research project must first demonstrate stable forward paper performance and pass an independent execution/risk review.

## Important research limitations

A model can have good historical statistics and still fail in live markets. The system therefore treats accuracy as insufficient and evaluates drawdown, Sharpe/Sortino, turnover, fees, slippage, regime performance, and sensitivity to higher transaction costs. Data availability also differs by source: some Binance derivatives statistics have only recent history, while macro series can contain revisions.
