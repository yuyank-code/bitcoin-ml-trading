"""Continuous mark-to-market execution ledger.

Consumes frozen OOS probabilities and market bars. Long-only, non-overlapping
positions. Cash is debited at entry including fees; open positions are marked
at each bar close; exit proceeds are credited after exit fees. Intrabar stop
and target collisions use a conservative deterministic rule.
"""

from __future__ import annotations

import pandas as pd


def run_continuous_mtm(
    d: pd.DataFrame,
    *,
    horizon_bars: int = 6,
    fee_bps: float = 5.0,
    slippage_bps: float = 2.0,
    prob_threshold: float = 0.56,
    stop_atr: float = 2.0,
    target_atr: float = 3.0,
    risk_fraction: float = 0.005,
    max_position: float = 0.25,
    initial_cash: float = 100_000.0,
):
    required = {"Date", "Open", "High", "Low", "Close", "atr_pct", "prob_up"}
    missing = required - set(d.columns)
    if missing:
        raise ValueError(f"missing execution columns: {sorted(missing)}")
    if horizon_bars < 1:
        raise ValueError("horizon_bars must be >= 1")

    x = d.sort_values("Date").reset_index(drop=True)
    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0

    cash = float(initial_cash)
    peak = cash
    equity_rows = []
    trades = []
    i = 0

    while i < len(x) - 1:
        signal = x.iloc[i]
        p = float(signal.prob_up)

        if p < prob_threshold:
            equity = cash
            peak = max(peak, equity)
            equity_rows.append((signal.Date, equity, cash, 0.0, 0.0, equity / peak - 1.0))
            i += 1
            continue

        entry_i = i + 1
        end_i = min(entry_i + horizon_bars - 1, len(x) - 1)
        entry_bar = x.iloc[entry_i]
        mid_entry = float(entry_bar.Open)
        entry_px = mid_entry * (1.0 + slip)

        atr_pct = float(signal.atr_pct)
        stop_dist = max(entry_px * atr_pct * stop_atr, entry_px * 0.002)
        target_dist = entry_px * atr_pct * target_atr
        stop_px = entry_px - stop_dist
        target_px = entry_px + target_dist

        qty = min(
            cash * risk_fraction / stop_dist,
            cash * max_position / entry_px,
        )
        if qty <= 0:
            i += 1
            continue

        entry_notional = qty * entry_px
        entry_fee = entry_notional * fee
        if entry_notional + entry_fee > cash:
            i += 1
            continue

        cash -= entry_notional + entry_fee

        # Mark the open position from the entry bar through the exit bar.
        exit_i = end_i
        reason = "horizon"
        ambiguous = False

        for j in range(entry_i, end_i + 1):
            bar = x.iloc[j]
            hi = float(bar.High)
            lo = float(bar.Low)
            hit_stop = lo <= stop_px
            hit_target = hi >= target_px

            if hit_stop and hit_target:
                exit_i = j
                reason = "stop_ambiguous"
                ambiguous = True
                break
            if hit_stop:
                exit_i = j
                reason = "stop"
                break
            if hit_target:
                exit_i = j
                reason = "target"
                break

            mark = float(bar.Close)
            mtm = cash + qty * mark
            peak = max(peak, mtm)
            equity_rows.append((
                bar.Date, mtm, cash, qty * mark, qty, mtm / peak - 1.0
            ))

        exit_bar = x.iloc[exit_i]
        if reason == "target":
            mid_exit = target_px
        elif reason.startswith("stop"):
            mid_exit = stop_px
        else:
            mid_exit = float(exit_bar.Close)

        exit_px = mid_exit * (1.0 - slip)
        exit_notional = qty * exit_px
        exit_fee = exit_notional * fee
        cash += exit_notional - exit_fee

        peak = max(peak, cash)
        equity_rows.append((
            exit_bar.Date, cash, cash, 0.0, 0.0, cash / peak - 1.0
        ))

        trades.append({
            "signal_time": signal.Date,
            "entry_time": entry_bar.Date,
            "exit_time": exit_bar.Date,
            "signal_index": i,
            "entry_index": entry_i,
            "exit_index": exit_i,
            "holding_bars": exit_i - entry_i + 1,
            "prob_up": p,
            "entry_mid": mid_entry,
            "entry": entry_px,
            "stop": stop_px,
            "target": target_px,
            "exit_mid": mid_exit,
            "exit": exit_px,
            "qty": qty,
            "entry_fee": entry_fee,
            "exit_fee": exit_fee,
            "total_fees": entry_fee + exit_fee,
            "net_pnl": cash - (cash + 0.0),
            "exit_reason": reason,
            "ambiguous_intrabar": ambiguous,
            "capital_after": cash,
        })

        i = exit_i + 1

    eq = pd.DataFrame(
        equity_rows,
        columns=["Date", "equity", "cash", "position_value", "qty", "drawdown"],
    )
    tr = pd.DataFrame(trades)

    # Recompute trade P&L from cash transitions rather than the placeholder
    # above; this keeps accounting auditable without reconstructing fills.
    if not tr.empty:
        tr["gross_pnl"] = (
            tr["qty"] * (tr["exit_mid"] - tr["entry_mid"])
        )
        tr["slippage_cost"] = (
            tr["qty"] * (tr["entry"] - tr["entry_mid"])
            + tr["qty"] * (tr["exit_mid"] - tr["exit"])
        )
        tr["net_pnl"] = tr["gross_pnl"] - tr["slippage_cost"] - tr["total_fees"]

    return eq, tr
