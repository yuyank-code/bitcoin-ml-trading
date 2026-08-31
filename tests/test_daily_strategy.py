import pandas as pd
from daily_conviction import build_conviction, select_daily_trades


def test_conviction_requires_group_agreement():
    d = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01T01:00Z", "2026-01-01T10:00Z"]),
        "prob_up": [0.80, 0.80], "prob_std": [0.02, 0.02],
        "technical_prob_up": [0.80, 0.80],
        "onchain_prob_up": [0.80, 0.20],
        "derivatives_prob_up": [0.80, 0.80],
        "macro_prob_up": [0.80, 0.80],
        "sentiment_prob_up": [0.80, 0.80],
        "breadth_prob_up": [0.80, 0.80],
    })
    out = build_conviction(d)
    assert out.loc[0, "direction"] == "LONG"
    assert out.loc[1, "conviction"] < out.loc[0, "conviction"]


def test_at_most_one_trade_per_day():
    d = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01T01:00Z", "2026-01-01T10:00Z", "2026-01-02T01:00Z"]),
        "prob_up": [0.90, 0.85, 0.90], "prob_std": [0.01, 0.01, 0.01],
        **{f"{g}_prob_up": [0.90, 0.85, 0.90] for g in ("technical","onchain","derivatives","macro","sentiment","breadth")}
    })
    out = select_daily_trades(d, min_conviction=70, min_groups=4)
    assert (out["day"].value_counts() <= 1).all()
    assert len(out) == 2
