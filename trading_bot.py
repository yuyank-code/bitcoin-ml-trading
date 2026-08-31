from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
LOG = logging.getLogger("btc-ml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


@dataclass
class Config:
    symbol: str = "BTCUSDT"
    interval: str = "1h"
    initial_train_bars: int = 24 * 180
    retrain_every_bars: int = 24 * 7
    horizon_bars: int = 6
    starting_capital: float = 100_000.0
    risk_per_trade: float = 0.005
    max_position_pct: float = 0.25
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.03
    stop_atr_multiple: float = 2.0
    target_atr_multiple: float = 3.0
    fee_bps: float = 5.0
    slippage_bps: float = 2.0
    min_edge_after_costs: float = 0.0015
    probability_long: float = 0.56
    probability_short: float = 0.44
    long_only: bool = True
    model: str = "hist_gradient_boosting"
    seed: int = 42


FEATURES = [
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24", "ret_72", "ret_168",
    "vol_6", "vol_24", "vol_72", "atr_pct", "range_pct", "body_pct",
    "rsi14", "macd", "macd_signal", "macd_hist", "sma24_ratio", "sma72_ratio",
    "ema24_ratio", "ema168_ratio", "bb_pos", "bb_width", "volume_z",
    "volume_ratio", "trend_24_168", "drawdown_168", "drawdown_720",
    "oi_change_1", "oi_change_6", "oi_value_change_24", "funding_rate",
    "funding_24_mean", "funding_72_mean", "funding_z",
]


def binance_klines(symbol: str, interval: str, limit: int = 1000, start: int | None = None, end: int | None = None) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    if start is not None:
        params["startTime"] = start
    if end is not None:
        params["endTime"] = end
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    rows = r.json()
    if not rows:
        return pd.DataFrame()
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    d = pd.DataFrame(rows, columns=cols)
    d["Date"] = pd.to_datetime(d.open_time, unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
    return d[["Date", "Open", "High", "Low", "Close", "Volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]]


def download_spot_history(cfg: Config, years: int = 8) -> pd.DataFrame:
    """Download Binance spot history in API-safe chunks.

    Binance's kline endpoint has a maximum of 1000 rows per request, so history is paged.
    The earliest reliable Binance spot history is used; this is not a claim of coverage back to BTC's 2009 genesis.
    """
    end = int(time.time() * 1000)
    start = int((pd.Timestamp.utcnow() - pd.Timedelta(days=365.25 * years)).timestamp() * 1000)
    chunks: list[pd.DataFrame] = []
    cursor = start
    while cursor < end:
        d = binance_klines(cfg.symbol, cfg.interval, 1000, cursor, end)
        if d.empty:
            break
        chunks.append(d)
        last = int(d.Date.iloc[-1].timestamp() * 1000)
        if last <= cursor:
            break
        cursor = last + 1
        if len(chunks) % 20 == 0:
            LOG.info("Downloaded %d bars...", sum(len(x) for x in chunks))
    if not chunks:
        raise RuntimeError("Binance returned no historical candles")
    d = pd.concat(chunks, ignore_index=True).drop_duplicates("Date").sort_values("Date").reset_index(drop=True)
    validate_ohlcv(d)
    d.to_csv(OUT / "binance_btcusdt_history.csv", index=False)
    LOG.info("History: %s -> %s (%d bars)", d.Date.iloc[0], d.Date.iloc[-1], len(d))
    return d


def validate_ohlcv(d: pd.DataFrame) -> None:
    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if d.Date.duplicated().any():
        raise ValueError("Duplicate timestamps")
    if d[["Open", "High", "Low", "Close", "Volume"]].isna().any().any():
        raise ValueError("NaN market data")
    if (d[["Open", "High", "Low", "Close"]] <= 0).any().any() or (d.Volume < 0).any():
        raise ValueError("Invalid price/volume")
    if (d.High < d[["Open", "Close"]].max(axis=1)).any() or (d.Low > d[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Invalid OHLC relationship")


def add_derivatives_features(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Merge optional Binance futures features. Missing derivatives history never breaks spot research."""
    try:
        oi_url = "https://fapi.binance.com/futures/data/openInterestHist"
        oi = requests.get(oi_url, params={"symbol": cfg.symbol, "period": "1h", "limit": 500}, timeout=20)
        if oi.ok:
            o = pd.DataFrame(oi.json())
            if not o.empty:
                o["Date"] = pd.to_datetime(o.timestamp, unit="ms", utc=True)
                o["open_interest"] = pd.to_numeric(o.sumOpenInterest, errors="coerce")
                o["oi_value"] = pd.to_numeric(o.sumOpenInterestValue, errors="coerce")
                d = pd.merge_asof(d.sort_values("Date"), o[["Date", "open_interest", "oi_value"]].sort_values("Date"), on="Date", direction="backward")
    except Exception as e:
        LOG.warning("Open-interest feature unavailable: %s", e)

    try:
        fr_url = "https://fapi.binance.com/fapi/v1/fundingRate"
        fr = requests.get(fr_url, params={"symbol": cfg.symbol, "limit": 1000}, timeout=20)
        if fr.ok:
            f = pd.DataFrame(fr.json())
            if not f.empty:
                f["Date"] = pd.to_datetime(f.fundingTime, unit="ms", utc=True)
                f["funding_rate"] = pd.to_numeric(f.fundingRate, errors="coerce")
                d = pd.merge_asof(d.sort_values("Date"), f[["Date", "funding_rate"]].sort_values("Date"), on="Date", direction="backward")
    except Exception as e:
        LOG.warning("Funding feature unavailable: %s", e)
    return d


def make_features(raw: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = add_derivatives_features(raw.copy(), cfg)
    c, h, l, v, o = d.Close, d.High, d.Low, d.Volume, d.Open
    r = c.pct_change()
    for n in [1, 3, 6, 12, 24, 72, 168]:
        d[f"ret_{n}"] = c.pct_change(n)
    for n in [6, 24, 72]:
        d[f"vol_{n}"] = r.rolling(n).std()
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    d["atr_pct"] = tr.rolling(14).mean() / c
    d["range_pct"] = (h - l) / c
    d["body_pct"] = (c - o) / o
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    d["rsi14"] = 100 - 100 / (1 + rs)
    e12, e26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    d["macd"] = e12 - e26
    d["macd_signal"] = d.macd.ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d.macd - d.macd_signal
    for n in [24, 72]:
        d[f"sma{n}_ratio"] = c / c.rolling(n).mean() - 1
    for n in [24, 168]:
        d[f"ema{n}_ratio"] = c / c.ewm(span=n, adjust=False).mean() - 1
    mid, sd = c.rolling(24).mean(), c.rolling(24).std()
    d["bb_pos"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan)
    d["bb_width"] = 4 * sd / mid
    d["volume_z"] = (v - v.rolling(72).mean()) / v.rolling(72).std()
    d["volume_ratio"] = v / v.rolling(24).mean()
    d["trend_24_168"] = c.rolling(24).mean() / c.rolling(168).mean() - 1
    d["drawdown_168"] = c / c.rolling(168).max() - 1
    d["drawdown_720"] = c / c.rolling(720).max() - 1
    if "open_interest" not in d:
        d["open_interest"] = np.nan
        d["oi_value"] = np.nan
    if "funding_rate" not in d:
        d["funding_rate"] = 0.0
    d["oi_change_1"] = d.open_interest.pct_change()
    d["oi_change_6"] = d.open_interest.pct_change(6)
    d["oi_value_change_24"] = d.oi_value.pct_change(24)
    d["funding_24_mean"] = d.funding_rate.rolling(24, min_periods=1).mean()
    d["funding_72_mean"] = d.funding_rate.rolling(72, min_periods=1).mean()
    fr_std = d.funding_rate.rolling(72).std()
    d["funding_z"] = (d.funding_rate - d.funding_72_mean) / fr_std.replace(0, np.nan)
    d["future_return"] = c.shift(-cfg.horizon_bars) / c - 1
    d["label"] = (d.future_return > 0).astype(int)
    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna(subset=FEATURES + ["future_return", "label"]).reset_index(drop=True)
    return d


def model_pipeline(cfg: Config):
    if cfg.model == "random_forest":
        m = RandomForestClassifier(n_estimators=400, min_samples_leaf=12, max_features="sqrt", class_weight="balanced", random_state=cfg.seed, n_jobs=-1)
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", m)])
    if cfg.model == "logistic":
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", LogisticRegression(C=0.5, max_iter=3000, random_state=cfg.seed))])
    m = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=30, l2_regularization=1.0, random_state=cfg.seed)
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", m)])


def walk_forward(d: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = d.copy()
    out["prob_up"] = np.nan
    out["model_version"] = -1
    version = 0
    for start in range(cfg.initial_train_bars, len(out), cfg.retrain_every_bars):
        end = min(start + cfg.retrain_every_bars, len(out))
        train = out.iloc[:start]
        test = out.iloc[start:end]
        if train.label.nunique() < 2:
            continue
        model = model_pipeline(cfg)
        model.fit(train[FEATURES], train.label)
        out.loc[test.index, "prob_up"] = model.predict_proba(test[FEATURES])[:, 1]
        out.loc[test.index, "model_version"] = version
        version += 1
    return out.dropna(subset=["prob_up"]).reset_index(drop=True)


def trade_backtest(d: pd.DataFrame, cfg: Config):
    cash = cfg.starting_capital
    peak = cash
    trades = []
    equity = []
    daily_anchor = cash
    current_day = None
    halted = False
    consecutive_losses = 0

    for i in range(len(d) - cfg.horizon_bars):
        row = d.iloc[i]
        next_row = d.iloc[i + 1]
        day = row.Date.date()
        if day != current_day:
            current_day, daily_anchor, halted = day, cash, False
        drawdown = cash / max(peak, 1) - 1
        if drawdown <= -cfg.max_drawdown_pct or (daily_anchor - cash) / max(daily_anchor, 1) >= cfg.max_daily_loss_pct:
            halted = True
        p = float(row.prob_up)
        expected = float(row.future_return)
        cost_floor = 2 * (cfg.fee_bps + cfg.slippage_bps) / 10_000 + cfg.min_edge_after_costs
        if halted or expected <= cost_floor or p < cfg.probability_long:
            equity.append((row.Date, cash, drawdown))
            continue
        entry_raw = float(next_row.Open)
        entry = entry_raw * (1 + cfg.slippage_bps / 10_000)
        atr = float(row.atr_pct) * entry
        stop = entry - cfg.stop_atr_multiple * atr
        target = entry + cfg.target_atr_multiple * atr
        risk_cash = cash * cfg.risk_per_trade
        qty = min(risk_cash / max(entry - stop, entry * 1e-6), cash * cfg.max_position_pct / entry)
        if qty <= 0:
            continue
        hi, lo = float(next_row.High), float(next_row.Low)
        if lo <= stop:
            exit_raw, reason = stop, "stop"
        elif hi >= target:
            exit_raw, reason = target, "target"
        else:
            exit_raw, reason = float(next_row.Close), "horizon"
        exit_px = exit_raw * (1 - cfg.slippage_bps / 10_000)
        gross = qty * (exit_px - entry)
        fees = (qty * entry + qty * exit_px) * cfg.fee_bps / 10_000
        net = gross - fees
        cash += net
        peak = max(peak, cash)
        if net < 0:
            consecutive_losses += 1
        else:
            consecutive_losses = 0
        trades.append({"signal_time": row.Date, "entry_time": next_row.Date, "exit_time": next_row.Date, "prob_up": p, "expected_return": expected, "entry": entry, "stop": stop, "target": target, "exit": exit_px, "qty": qty, "gross_pnl": gross, "fees": fees, "net_pnl": net, "exit_reason": reason, "capital_after": cash})
        equity.append((next_row.Date, cash, cash / max(peak, 1) - 1))

    eq = pd.DataFrame(equity, columns=["Date", "capital", "drawdown"]).drop_duplicates("Date").sort_values("Date")
    tr = pd.DataFrame(trades)
    return eq, tr


def metrics(eq: pd.DataFrame, tr: pd.DataFrame, cfg: Config) -> dict:
    final = float(eq.capital.iloc[-1])
    ret = final / cfg.starting_capital - 1
    daily = eq.set_index("Date").capital.resample("1D").last().ffill().pct_change().dropna()
    vol = daily.std() * math.sqrt(365) if len(daily) > 1 else np.nan
    sharpe = daily.mean() / daily.std() * math.sqrt(365) if daily.std() > 0 else np.nan
    downside = daily[daily < 0].std()
    sortino = daily.mean() / downside * math.sqrt(365) if downside and downside > 0 else np.nan
    wins = int((tr.net_pnl > 0).sum()) if not tr.empty else 0
    losses = int((tr.net_pnl <= 0).sum()) if not tr.empty else 0
    gp = float(tr.loc[tr.net_pnl > 0, "net_pnl"].sum()) if wins else 0
    gl = float(-tr.loc[tr.net_pnl <= 0, "net_pnl"].sum()) if losses else 0
    return {"starting_capital": cfg.starting_capital, "final_capital": final, "return_pct": ret * 100, "max_drawdown_pct": float(eq.drawdown.min() * 100), "annualized_volatility_pct": float(vol * 100) if pd.notna(vol) else None, "sharpe": float(sharpe) if pd.notna(sharpe) else None, "sortino": float(sortino) if pd.notna(sortino) else None, "trades": len(tr), "wins": wins, "losses": losses, "win_rate_pct": wins / len(tr) * 100 if len(tr) else None, "profit_factor": gp / gl if gl else None, "expectancy": float(tr.net_pnl.mean()) if len(tr) else None}


def latest_signal(cfg: Config) -> dict:
    raw = binance_klines(cfg.symbol, cfg.interval, 1000)
    raw = add_derivatives_features(raw, cfg)
    feat = make_features(raw, cfg)
    if len(feat) < cfg.initial_train_bars + 5:
        raise RuntimeError("Not enough live history for model training")
    train = feat.iloc[:-1]
    current = feat.iloc[[-1]]
    model = model_pipeline(cfg)
    model.fit(train[FEATURES], train.label)
    p = float(model.predict_proba(current[FEATURES])[:, 1][0])
    atr = float(current.atr_pct.iloc[0]) * float(current.Close.iloc[0])
    expected = float(model.predict_proba(current[FEATURES])[:, 1][0] - 0.5) * 2 * atr / float(current.Close.iloc[0])
    signal = "HOLD"
    if p >= cfg.probability_long and expected > cfg.min_edge_after_costs:
        signal = "LONG"
    elif not cfg.long_only and p <= cfg.probability_short and expected < -cfg.min_edge_after_costs:
        signal = "SHORT"
    return {"timestamp": current.Date.iloc[0].isoformat(), "symbol": cfg.symbol, "close": float(current.Close.iloc[0]), "probability_up": p, "expected_return_proxy": expected, "atr_pct": float(current.atr_pct.iloc[0]), "signal": signal, "model": cfg.model}


def run_backtest(cfg: Config):
    raw = download_spot_history(cfg)
    features = make_features(raw, cfg)
    features.to_csv(OUT / "features.csv", index=False)
    pred = walk_forward(features, cfg)
    pred.to_csv(OUT / "predictions.csv", index=False)
    eq, tr = trade_backtest(pred, cfg)
    eq.to_csv(OUT / "equity_curve.csv", index=False)
    tr.to_csv(OUT / "trades.csv", index=False)
    summary = metrics(eq, tr, cfg)
    report = {"strategy": summary, "data_start": str(raw.Date.iloc[0]), "data_end": str(raw.Date.iloc[-1]), "bars": len(raw), "config": asdict(cfg)}
    (OUT / "performance_summary.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2))


def run_signal(cfg: Config):
    signal = latest_signal(cfg)
    (OUT / "latest_signal.json").write_text(json.dumps(signal, indent=2))
    print(json.dumps(signal, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["backtest", "signal"], default="backtest")
    p.add_argument("--model", choices=["hist_gradient_boosting", "random_forest", "logistic"], default="hist_gradient_boosting")
    p.add_argument("--interval", choices=["1m", "5m", "15m", "1h", "4h", "1d"], default="1h")
    args = p.parse_args()
    cfg = Config(model=args.model, interval=args.interval)
    if args.mode == "signal":
        run_signal(cfg)
    else:
        run_backtest(cfg)


if __name__ == "__main__":
    main()
