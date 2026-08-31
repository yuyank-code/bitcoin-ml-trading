"""Build the unified research dataset from locally downloaded source tables.

The builder uses backward as-of joins so a source observation is never pulled
from the future. It also keeps source columns namespaced and creates the
candidate dataset consumed by unified_engine.py.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def _load(path: Path, date_col: str = "Date") -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    if date_col not in d.columns:
        return pd.DataFrame()
    d[date_col] = pd.to_datetime(d[date_col], utc=True, errors="coerce")
    return d.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(date_col, keep="last")


def _prefix(d: pd.DataFrame, source: str) -> pd.DataFrame:
    d = d.copy()
    keep = [c for c in d.columns if c != "Date"]
    return d.rename(columns={c: f"{source}_{c}" for c in keep})


def build() -> pd.DataFrame:
    base = _load(OUT / "binance_btcusdt_history.csv")
    if base.empty:
        raise FileNotFoundError("Run trading_bot.py --mode data first")
    base = base[[c for c in base.columns if c in {"Date", "Open", "High", "Low", "Close", "Volume"}]].copy()

    sources = [
        ("onchain", [OUT / "bitcoin_onchain_daily.csv"]),
        ("derivatives", [OUT / "derivatives_features.csv", OUT / "binance_derivatives.csv"]),
        ("macro", [OUT / "macro_features.csv", OUT / "macro_data.csv"]),
        ("sentiment", [OUT / "sentiment_features.csv"]),
        ("breadth", [OUT / "alternative_signals.csv", OUT / "etf_flows.csv", OUT / "breadth_features.csv"]),
    ]
    result = base.sort_values("Date")
    for name, paths in sources:
        src = next((_load(p) for p in paths if p.exists()), pd.DataFrame())
        if src.empty:
            continue
        src = _prefix(src, name)
        # Conservative: source values are only available at or before the BTC bar.
        result = pd.merge_asof(result.sort_values("Date"), src.sort_values("Date"), on="Date", direction="backward")
    result.to_csv(OUT / "unified_dataset.csv", index=False)
    return result


if __name__ == "__main__":
    d = build()
    print(f"Unified rows: {len(d)}; columns: {len(d.columns)}")
