# Research 122 — validation status and next hard gate

Date: 2026-09-23

## Meaningful update
A fresh audit of the live repository shows that the major unresolved issue is not model architecture. The open validation gates H114/H171/H173/H174 remain active, while the latest code commits confirm that point-in-time joins, causal derivative joins, nested policy-selection invariants, and seed/window robustness infrastructure have been added.

## Literature update
Bysik & Ślepaczuk (2026), arXiv:2606.00060, evaluate ~70,000 hourly BTC observations with 27 walk-forward folds. They report that naive sign-based strategies fail at 10 bps while cost-aware forecast filtering can restore selected configurations; bootstrap evidence does not establish formal dominance of XGBoost over neural alternatives.

Gort et al. (2022), arXiv:2209.05559, show that explicitly detecting/rejecting backtest overfitting can improve the reliability of crypto trading agents, reinforcing selection-aware validation rather than architecture search.

## Current repository state
Open gates include:
- #114: legacy backtest directly uses realized `future_return` at entry; all results from that path remain invalid.
- #115/#117: event-time purge/embargo and future-perturbation/null-workflow validation are not yet closed.
- #118: row-count purge is only safe under strict regular hourly cadence; event-time label-end separation is required.
- #116: switching-cost/frozen-cost stress is defined but not promotion evidence until temporal gates pass.
- #108: vintage-aware join helper exists, but authoritative integration remains an open gate.

## New hard gate
Before any candidate performance is promoted, require a single machine-checked event-time contract:
`max(train_label_end_time) < min(test_decision_time)` for every fold, plus strict timestamp uniqueness/cadence validation and an explicit decision/execution/label horizon alignment.

After this passes, run the full frozen workflow on null/placebo and injected-signal controls before interpreting real BTC alpha. Keep confirmation untouched.

## Verdict
No robust alpha claim. No candidate promotion. The most valuable next work is completing temporal/event-time validation and executing the falsification suite, not adding another model architecture.
