from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

LOG = logging.getLogger("btc-ml.onchain")
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

COINMETRICS = "https://community-api.coinmetrics.io/v4"
BLOCKCHAIN = "https://api.blockchain.info/charts"

# Conservative free/community set. We discover the catalog first and only request
# metrics actually available from the community endpoint.
PREFERRED_METRICS = [
    "AdrActCnt", "TxCnt", "TxTfrCnt", "FeeTotNtv", "FeeTotUSD",
    "SplyCur", "SplyAct1d", "SplyAct30d", "SplyAct90d", "SplyAct180d",
    "SplyAct1yr", "SplyAct2yr", "SplyAct3yr", "SplyAct5yr", "SplyAct7yr",
    "SplyAct10yr", "RevNtv", "RevUSD", "HashRate", "DiffMean",
]

BLOCKCHAIN_CHARTS = {
    "transactions_per_second": "transactions-per-second",
    "transaction_fees_usd": "transaction-fees-usd",
    "total_btc": "total-Bitcoins",
    "hash_rate": "hash-rate",
    "n_transactions": "n-transactions",
}


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, params=params or {}, timeout=60)
    r.raise_for_status()
    return r.json()


def discover_community_metrics() -> set[str]:
    """Return metric names exposed by the free Coin Metrics community catalog."""
    data = _get(f"{COINMETRICS}/catalog/assets", {"assets": "btc"})
    rows = data.get("data", [])
    if not rows:
        return set()
    metrics = rows[0].get("metrics", [])
    return {m.get("metric") for m in metrics if m.get("metric")}


def fetch_coinmetrics(start: str, end: str, metrics: Iterable[str] | None = None) -> pd.DataFrame:
    available = discover_community_metrics()
    requested = [m for m in (metrics or PREFERRED_METRICS) if m in available]
    if not requested:
        raise RuntimeError("No preferred Coin Metrics community metrics are currently available.")

    rows = []
    page_url = f"{COINMETRICS}/timeseries/asset-metrics"
    params = {
        "assets": "btc",
        "metrics": ",".join(requested),
        "start_time": start,
        "end_time": end,
        "frequency": "1d",
        "page_size": 1000,
    }
    while page_url:
        payload = _get(page_url, params)
        rows.extend(payload.get("data", []))
        page_url = payload.get("next_page_url")
        params = {}
    if not rows:
        return pd.DataFrame(columns=["Date"] + requested)
    d = pd.DataFrame(rows)
    d["Date"] = pd.to_datetime(d["time"], utc=True).dt.floor("D")
    keep = ["Date"] + [m for m in requested if m in d.columns]
    d = d[keep].drop_duplicates("Date").sort_values("Date")
    for c in d.columns[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.reset_index(drop=True)


def fetch_blockchain_chart(name: str, timespan: str = "all") -> pd.DataFrame:
    chart = BLOCKCHAIN_CHARTS[name]
    payload = _get(f"{BLOCKCHAIN}/{chart}", {"timespan": timespan, "format": "json"})
    values = payload.get("values", [])
    if not values:
        return pd.DataFrame(columns=["Date", name])
    d = pd.DataFrame(values)
    d["Date"] = pd.to_datetime(d["x"], unit="s", utc=True).dt.floor("D")
    d[name] = pd.to_numeric(d["y"], errors="coerce")
    return d[["Date", name]].drop_duplicates("Date").sort_values("Date").reset_index(drop=True)


def fetch_all(start: str, end: str) -> pd.DataFrame:
    """Build a daily on-chain table from free/community sources."""
    cm = fetch_coinmetrics(start, end)
    result = cm.copy()
    for name in BLOCKCHAIN_CHARTS:
        try:
            b = fetch_blockchain_chart(name)
            result = result.merge(b, on="Date", how="outer")
        except Exception as exc:
            LOG.warning("Blockchain.com chart %s unavailable: %s", name, exc)
    result = result.sort_values("Date").reset_index(drop=True)
    result.to_csv(OUT / "bitcoin_onchain_daily.csv", index=False)
    return result


def merge_daily_market_onchain(market: pd.DataFrame, onchain: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time-safe daily merge: only use an on-chain observation known by its date."""
    m = market.copy()
    m["Date"] = pd.to_datetime(m["Date"], utc=True).dt.floor("D")
    o = onchain.copy()
    o["Date"] = pd.to_datetime(o["Date"], utc=True).dt.floor("D")
    return pd.merge_asof(m.sort_values("Date"), o.sort_values("Date"), on="Date", direction="backward")
