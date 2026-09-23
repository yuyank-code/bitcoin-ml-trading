# Validation Status — 2026-09-23

## Purpose
Record the current state of the BTC ML research pipeline without promoting any model performance result.

## Fresh literature
- Arian, Norouzi Mobarekeh & Seco, *Backtest overfitting in the machine learning era* (Knowledge-Based Systems, 2024): controlled experiments report CPCV as more resistant to backtest overfitting than conventional OOS methods, with lower PBO and stronger DSR statistics.
- Kim, *Beyond Accuracy: A Validation Framework for Machine Learning in Cryptocurrency Trading* (2026): reports substantial false-positive risk from prediction-only validation and emphasizes economic/cost gates and CPCV/PBO.
- Huang et al., *Research on Machine Learning High-Frequency Trading Strategies Under Transaction Cost* (2026): reports a large prediction-to-net-return gap and emphasizes explicit fees, spread and slippage.
- Huang et al., *Research on Machine Learning High-Frequency Trading Strategy of Cryptocurrency Based on Transaction Cost-Aware Filtering* (2026): reports that naive BTC/USDT strategies can collapse under 10 bps costs while cost-aware filtering materially changes turnover/economics. Treat as directional literature evidence, not as proof of our strategy.
- Bysik & Slepaczuk, *Machine Learning-Based Bitcoin Trading Under Transaction Costs* (2026): finds naive sign strategies can fail under 10 bps costs; cost-aware execution filtering can improve selected configurations, while formal model dominance remains fragile.
- Zhai, *The Prediction Paradox* (2026): reports a live-trading failure case where strong walk-forward prediction metrics did not translate into live profitability, highlighting execution/prediction mismatch and regime instability.

## Live repository audit
The current authoritative `production_research.py` now purges `HORIZON_BARS` rows before each test block and explicitly avoids using realized future returns as an entry feature/decision. Point-in-time vintage joins and causal derivative joins have regression tests in recent commits.

However, confirmation-grade model selection remains blocked by unresolved execution/validation contracts already tracked in the repository:

1. **Event-time embargo invariant (H171/H173):** the code does not yet expose a machine-checked assertion that `max(train.label_end_time) < min(test.signal_time)` for every fold. Purging by row count is not sufficient as the canonical audit artifact.
2. **Target/execution alignment (H80/H101/H139):** the model label is `Close[t+6]/Close[t]-1`, while the simulator enters at `Open[t+1]` and currently resolves a no-barrier trade at `t+1` close. The predictive horizon and monetized holding period are therefore not the same event.
3. **Portfolio state (H141):** the simulator does not yet provide a canonical open-position book for overlapping signals; sequential cash updates can misrepresent simultaneous exposure.
4. **Frozen-cost accounting (H155):** the frozen-cost replay path needs the same event-time equity convention as the canonical simulator before its CAGR/Sharpe/drawdown are interpreted.
5. **Intrabar ambiguity (H144):** stop-first handling is deterministic but not evidence of actual first-touch order when hourly OHLC touches both barriers.

## Research consequence
No Sharpe, CAGR, or feature-family ranking from the current production artifacts should be treated as confirmation-grade alpha evidence until these gates are closed and the full candidate family is regenerated under the frozen protocol.

## Next decisive sequence
1. Make signal/entry/event-end timestamps explicit.
2. Implement event-interval purge + predeclared embargo and fail-closed assertions.
3. Align the training target with the executable holding-period contract.
4. Move to a canonical portfolio state machine with explicit overlapping-position rules.
5. Validate frozen-cost replay and intrabar bounds.
6. Run development-only CPCV/PBO/DSR and full-pipeline null/placebo falsification.
7. Only then evaluate switching-cost/cost-aware execution policies on an untouched confirmation interval.

## Promotion rule
A model is not promoted merely for positive OOS Sharpe or AUC. Promotion requires causal temporal separation, correct event alignment, portfolio/execution integrity, realistic friction stress, selection-aware inference, null falsification, and untouched confirmation evidence.
