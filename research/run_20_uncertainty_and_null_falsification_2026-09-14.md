# Run 20 — uncertainty, disagreement, and null falsification (2026-09-14)

## Scope
Fresh literature audit and repository/CI audit. No economic model promotion was attempted because H80/H71/H90/H94 remain unresolved and the frozen OOS artifact bundle is not committed.

## Literature findings

### 1. Transaction costs remain the dominant prediction-to-trading bottleneck
Bysik & Ślepaczuk (2026), *Machine Learning-Based Bitcoin Trading Under Transaction Costs*, evaluates roughly 70,000 hourly BTC-USDT observations in 27 walk-forward folds. Naive sign strategies fail under 10 bps; a cost-aware execution filter reduces turnover and restores profitability in selected configurations. The authors do not establish formal model-family dominance with bootstrap evidence. This supports testing execution policy separately from predictive-model choice.

### 2. Forecast uncertainty is economically relevant
Liu, Luo, Wang & Zhang (2026), *Uncertainty-Adjusted Sorting for Asset Pricing with Machine Learning*, reports that uncertainty-adjusted prediction bounds can improve out-of-sample portfolio performance versus point-prediction sorting in equities, mainly through reduced volatility. Liao, Ma, Neuhierl & Schilling (CEPR DP20080, 2025) similarly develop uncertainty estimates for neural-network return forecasts and an uncertainty-averse investment framework.

Transfer to single-asset BTC is a hypothesis only. The project already has an ensemble dispersion variable (`model_disagreement`), so uncertainty can be tested without introducing another model architecture.

### 3. Disagreement itself may contain signal, but can proxy for price information
Bali, Kelly, Mörke & Rahman (2026), *Machine Forecast Disagreement*, finds a strong relationship between cross-model forecast disagreement and future equity returns. Chu, Shen & Zhu (2026), *Machine Forecast Disagreement in the Cryptocurrency Market*, finds a significantly negative cross-sectional relation between disagreement and future crypto returns and reports that past returns are major drivers of disagreement.

Implication for H96: do not assume low disagreement is always better. Test low/high/tail bins as predeclared arms and require incremental information beyond existing price-derived features.

### 4. Full-pipeline falsification is necessary
Nikolopoulos (2026), *Spurious Predictability in Financial Machine Learning*, argues that adaptive specification search can create significant walk-forward results under martingale-difference nulls and proposes falsifying the complete research workflow on synthetic zero-predictability environments and microstructure placebos.

This strengthens H73: the null test must include the selection process, not merely one fixed model.

## Repository audit

Current main remains commit `3e3321cd` (2026-09-13), with no newer implementation commit observed. The production engine still has the known H80/H94/H90 execution-timeline issues. The repository does contain a GitHub Actions test workflow (`.github/workflows/tests.yml`) that runs `pytest -q`, but there is no workflow run associated with the current HEAD through the available GitHub Actions interface. Therefore CI health is currently **unverified**, not passing.

Existing `test_research_pipeline.py` contains only three lightweight tests: external-column prefixing, ability to run `signal_backtest()` without a usable future-return field, and a high-probability trade smoke test. It does not yet encode H80/H71/H90/H94 invariants. This confirms H86 is still an important engineering gate.

## Research decision

Do not add a new model architecture this run.

The highest-value sequence remains:

1. H80 executable six-bar label/entry/horizon alignment.
2. H71 multi-bar exposure/overlap policy.
3. H94 immutable exit timestamps.
4. H90 full OOS-grid frozen cost repricing.
5. H86 executable invariants in CI.
6. H77 dependence-aware predictive comparison.
7. H73 full-pipeline null/permutation falsification.
8. H74 CPCV/PBO/DSR selection audit.
9. H96/H99 disagreement and uncertainty experiments.
10. Only then final-holdout promotion.

## Current conclusion

No model promoted. No new Sharpe/CAGR/alpha claim. The robust progress is methodological: current literature increasingly converges on uncertainty-aware decisions, explicit transaction-cost modeling, and full-pipeline falsification, while the repository audit confirms that execution-event correctness and CI invariants must be repaired before those hypotheses can be evaluated credibly.
