from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

FNG_URL = "https://api.alternative.me/fng/"
NEWS_URL = "https://cryptocurrency.cv/api/bitcoin"


def fetch_fear_greed(limit: int = 0) -> pd.DataFrame:
    r = requests.get(FNG_URL, params={"limit": limit, "format": "json"}, timeout=30)
    r.raise_for_status()
    rows = r.json().get("data", [])
    if not rows:
        return pd.DataFrame(columns=["Date", "fear_greed", "fear_greed_class"])
    d = pd.DataFrame(rows)
    d["timestamp"] = pd.to_numeric(d["timestamp"], errors="coerce")
    d["Date"] = pd.to_datetime(d["timestamp"], unit="s", utc=True)
    d["fear_greed"] = pd.to_numeric(d["value"], errors="coerce")
    d["fear_greed_class"] = d["value_classification"].astype(str)
    return d[["Date", "fear_greed", "fear_greed_class"]].sort_values("Date").drop_duplicates("Date")


def _sentiment(text: str) -> float:
    """Small deterministic baseline lexicon; intentionally not presented as an LLM sentiment model."""
    text = re.sub(r"[^a-z0-9 ]", " ", str(text).lower())
    positive = {"bullish", "surge", "surges", "rally", "rallies", "gain", "gains", "rise", "rises", "record", "adoption", "inflow", "approve", "approved", "approval", "breakout", "optimistic"}
    negative = {"bearish", "crash", "crashes", "fall", "falls", "drop", "drops", "loss", "losses", "outflow", "hack", "hacked", "ban", "banned", "lawsuit", "liquidation", "panic", "selloff", "recession", "negative"}
    words = set(text.split())
    score = len(words & positive) - len(words & negative)
    return float(np.tanh(score / 3.0))


def fetch_bitcoin_news(limit: int = 100) -> pd.DataFrame:
    r = requests.get(NEWS_URL, params={"limit": limit}, timeout=30)
    r.raise_for_status()
    payload = r.json()
    rows = payload.get("articles", payload.get("data", payload if isinstance(payload, list) else []))
    out = []
    for x in rows:
        if not isinstance(x, dict):
            continue
        title = str(x.get("title", ""))
        published = x.get("published_at", x.get("publishedAt", x.get("pubDate", x.get("timestamp"))))
        if published is None:
            continue
        ts = pd.to_datetime(published, utc=True, errors="coerce")
        if pd.isna(ts):
            try:
                ts = pd.to_datetime(float(published), unit="s", utc=True)
            except Exception:
                continue
        score = _sentiment(title)
        out.append({"Date": ts, "news_title": title, "news_source": x.get("source", x.get("source_name", "unknown")), "news_sentiment": score})
    d = pd.DataFrame(out)
    if d.empty:
        return pd.DataFrame(columns=["Date", "news_title", "news_source", "news_sentiment"])
    return d.sort_values("Date").drop_duplicates(["Date", "news_title"])


def build_sentiment_features(fng: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    if fng.empty:
        return pd.DataFrame()
    d = fng[["Date", "fear_greed"]].copy().sort_values("Date")
    d["fear_greed_change_1d"] = d.fear_greed.diff()
    d["fear_greed_ma_7"] = d.fear_greed.rolling(7, min_periods=3).mean()
    d["fear_greed_ma_30"] = d.fear_greed.rolling(30, min_periods=7).mean()
    d["fear_greed_z_30"] = (d.fear_greed - d.fear_greed.rolling(30, min_periods=7).mean()) / d.fear_greed.rolling(30, min_periods=7).std()
    d["extreme_fear"] = (d.fear_greed <= 20).astype(int)
    d["extreme_greed"] = (d.fear_greed >= 80).astype(int)

    if not news.empty:
        n = news.set_index("Date").sort_index()
        daily = n.news_sentiment.resample("1D").agg(["mean", "count", "std"]).rename(columns={"mean":"news_sentiment_mean", "count":"news_count", "std":"news_sentiment_std"})
        daily["news_sentiment_change"] = daily.news_sentiment_mean.diff()
        daily["news_count_z_30"] = (daily.news_count - daily.news_count.rolling(30, min_periods=7).mean()) / daily.news_count.rolling(30, min_periods=7).std()
        daily = daily.reset_index().rename(columns={"Date":"news_date"})
        d["calendar_date"] = d.Date.dt.floor("D")
        d = d.merge(daily, left_on="calendar_date", right_on="news_date", how="left").drop(columns=["news_date", "calendar_date"])
    return d


def main() -> None:
    fng = fetch_fear_greed(0)
    news = fetch_bitcoin_news(200)
    fng.to_csv(OUT / "fear_greed_history.csv", index=False)
    news.to_csv(OUT / "bitcoin_news.csv", index=False)
    features = build_sentiment_features(fng, news)
    features.to_csv(OUT / "sentiment_features.csv", index=False)
    print(f"Fear & Greed rows: {len(fng)}")
    print(f"News rows: {len(news)}")
    print(f"Feature rows: {len(features)}")


if __name__ == "__main__":
    main()
