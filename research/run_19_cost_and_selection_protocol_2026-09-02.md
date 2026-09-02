# Run 19 — Cost-aware selection and robustness protocol

Date: 2026-09-02

## Findings

1. The production research runner has the 6-bar purge, but the supervised target is still `Close[t+6]/Close[t]-1` while execution enters at `Open[t+1]`. This target/execution mismatch remains a hard gate before final performance claims.
2. The repository now has a repaired validation/cost-stress harness, but no new profitability result is promoted from this run because executable-event labels are not yet frozen.
3. Recent literature reinforces two priorities: (a) multiple-testing correction with Deflated Sharpe Ratio / PBO-style analysis; (b) explicit transaction-cost-aware execution and turnover control.

## Testable hypotheses

H1: An executable-event label (entry at next open, predeclared exit horizon/stop/target convention) produces more stable OOS ranking than the current close-to-close label.

H2: A cost-aware abstention threshold improves net OOS performance mainly by reducing low-edge trades, and its benefit should persist across a predeclared cost grid rather than only at one calibrated cost.

H3: After accounting for the full model × feature-set × threshold trial count, any apparent winner will show materially lower evidence under DSR/PBO than its naive Sharpe suggests if selection overfit is substantial.

H4: Adding spread/impact stress will reduce performance monotonically; a robust candidate should remain viable under adverse but plausible execution assumptions rather than only fee+slippage point estimates.

## Required experiment before model ranking

1. Freeze executable event definition.
2. Build event start/end timestamps and purge training observations by interval overlap.
3. Add explicit post-test embargo.
4. Generate immutable OOS predictions without threshold selection.
5. Apply a fixed cost grid without changing predictions.
6. Select policies only on a development OOS segment.
7. Freeze the policy and evaluate once on an untouched final holdout.
8. Report naive Sharpe, DSR, PBO/CSCV diagnostics, turnover, trade count, and cost sensitivity.

## Literature anchors

- Bailey & López de Prado (2014), *The Deflated Sharpe Ratio*, Journal of Portfolio Management: DSR corrects performance inflation from multiple testing and non-normality.
- Arian, Norouzi Mobarekeh & Seco (2024), *Backtest overfitting in the machine learning era*, Knowledge-Based Systems: controlled experiments report stronger PBO/DSR behavior for CPCV than conventional walk-forward evaluation.
- Bysik & Ślepaczuk (2026), *Machine Learning-Based Bitcoin Trading Under Transaction Costs*: hourly BTC-USDT ML strategies can lose apparent performance under 10 bps costs; cost-aware forecast filtering can reduce turnover and recover selected strategies, without formal model-family dominance.
- Brauneis et al. (2022), *The anatomy of a fee change — evidence from cryptocurrency markets*: fee changes affect spreads, depth, transaction activity, and trade size, supporting explicit execution-cost sensitivity rather than a single static fee assumption.

## Verdict

No new trading edge is claimed. The next meaningful result must come from the executable-event-aligned pipeline and frozen OOS artifacts.
