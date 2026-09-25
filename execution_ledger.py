"""Canonical non-overlapping event-time execution ledger.

The ledger consumes only information available at the signal timestamp and
executes from the next bar through the configured horizon. It is deliberately
separate from prediction/model code so execution semantics can be unit-tested.
"""

from __future__ import annotations
import pandas as pd


def run_execution_ledger(
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
    cash = float(initial_cash)
    peak = cash
    trades = []
    equity = []
    i = 0

    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0

    while i < len(x) - 1:
        r = x.iloc[i]
        p = float(r.prob_up)
        if p < prob_threshold:
            equity.append((r.Date, cash))
            i += 1
            continue

        entry_i = i + 1
        end_i = min(entry_i + horizon_bars - 1, len(x) - 1)
        entry_bar = x.iloc[entry_i]
        mid_entry = float(entry_bar.Open)
        entry = mid_entry * (1.0 + slip)

        atr = float(r.atr_pct)
        stop_dist = max(entry * atr * stop_atr, entry * 0.002)
        target_dist = entry * atr * target_atr
        stop = entry - stop_dist
        target = entry + target_dist

        qty = min(cash * risk_fraction / stop_dist, cash * max_position / entry)

        exit_i = end_i
        reason = "horizon"
        ambiguous = False
        for j in range(entry_i, end_i + 1):
            bar = x.iloc[j]
            hi, lo = float(bar.High), float(bar.Low)
            hit_stop = lo <= stop
            hit_target = hi >= target
            if hit_stop and hit_target:
                # Conservative deterministic OHLC rule.
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

        exit_bar = x.iloc[exit_i]
        mid_exit = target if reason == "target" else stop if reason.startswith("stop") else float(exit_bar.Close)
        exit_px = mid_exit * (1.0 - slip)

        gross_no_slippage = qty * (mid_exit - mid_entry)
        entry_slippage_cost = qty * (entry - mid_entry)
        exit_slippage_cost = qty * (mid_exit - exit_px)
        slippage_cost = entry_slippage_cost + exit_slippage_cost
        fees = (qty * entry + qty * exit_px) * fee
        net = gross_no_slippage - slippage_cost - fees

        cash += net
        peak = max(peak, cash)
        trades.append({
            "signal_time": r.Date,
            "entry_time": entry_bar.Date,
            "exit_time": exit_bar.Date,
            "signal_index": i,
            "entry_index": entry_i,
            "exit_index": exit_i,
            "holding_bars": exit_i - entry_i + 1,
            "prob_up": p,
            "entry_mid": mid_entry,
            "entry": entry,
            "stop": stop,
            "target": target,
            "exit_mid": mid_exit,
            "exit": exit_px,
            "qty": qty,
            "gross_pnl_no_slippage": gross_no_slippage,
            "entry_slippage_cost": entry_slippage_cost,
            "exit_slippage_cost": exit_slippage_cost,
            "slippage_cost": slippage_cost,
            "fees": fees,
            "net_pnl": net,
            "exit_reason": reason,
            "ambiguous_intrabar": ambiguous,
            "capital_after": cash,
        })
        equity.append((exit_bar.Date, cash))

        # No overlapping positions. A signal on the exit bar is eligible only
        # on the next loop iteration, so its entry cannot precede this exit.
        i = exit_i + 1

    eq = pd.DataFrame(equity, columns=["Date", "capital"])
    tr = pd.DataFrame(trades)
    return eq, tr
