# Research audit — 2026-09-15

## New literature evidence

1. **Santoni, Jouanne & Scullin (2026), MinervaScore.** The paper proposes an auditable post-selection robustness layer combining Deflated Sharpe Ratio (DSR), Probability of Backtest Overfitting (PBO), Superior Predictive Ability (SPA), Minimum Track Record Length (MTRL), and regime stability. Crucially, the authors report that the score should be treated as a validation/reporting layer, not as evidence that future returns are predictable. Source: https://arxiv.org/abs/2608.23808

2. **Kim (2026), VALID.** Across 340 crypto strategy variants, the paper reports substantial false-positive risk when predictive metrics are used for selection and large erosion of gross alpha from transaction costs. This reinforces complete trial accounting and economic gates. Source: https://doi.org/10.2139/ssrn.6508779

3. **Bysik & Ślepaczuk (2026).** In roughly 70,000 hourly BTC-USDT observations and 27 walk-forward folds, naive sign strategies fail under 10 bps while cost-aware filtering can rescue selected configurations; model-family superiority is not formally established. Source: https://arxiv.org/abs/2606.00060

4. **Pindza (2026), microstructure alpha.** With >3M minute observations across Binance spot/perpetual markets, weak microstructure information survives strict leakage controls, but flexible gradient boosting overfits and no strategy survives standard exchange fees. Source: https://doi.org/10.3389/fbloc.2026.1811716

5. **Zhai (2026), live-trading negative evidence.** A system with 0.85 walk-forward AUC produced about $0.12/day in live PnL, illustrating that strong predictive metrics can fail to translate into executable performance. Source: https://doi.org/10.2139/ssrn.6566940

## New validation gate: H106

Issue #54 adds a development-only post-selection robustness gate. The purpose is to prevent raw Sharpe/CAGR from being interpreted without accounting for the number of materially tried candidates, dependence from the six-bar target, and finite track record.

Required artifacts:
- complete immutable trial registry;
- raw OOS return vectors and candidate ranks;
- DSR/PBO/SPA (or predeclared equivalents);
- minimum track-record length and effective sample size;
- regime/path stability diagnostics.

The robustness layer must not be used to tune the model. It is a post-selection admissibility/reporting layer.

## Current pipeline status

The authoritative README still states that `production_research.py` is the leakage-safe model-selection/backtest path and that the final decision rule uses model probability, barrier geometry and estimated transaction costs. The documented leakage policy also prohibits random shuffling and requires timestamp-causal features. These policies remain appropriate. fileciteturn5file0

The latest repository commit remains the event-time/cost-replay audit. Its research note explicitly blocks economic model ranking until deterministic execution tests pass. fileciteturn4file0

Open prerequisites remain H80/H101/H102/H103/H94/H90. H104/H105 and the economic policy experiments remain downstream of those gates.

## Interpretation

The new evidence does not justify another architecture search. The highest-value research improvement is to make the candidate-selection history itself auditable and selection-adjusted. If a future candidate has an attractive raw Sharpe but fails PBO/DSR/SPA/MTRL or is unstable across development paths, the correct conclusion is that the evidence is insufficient—not to add more tuning.

No new performance claim is made in this audit.
