# Research audit — 2026-09-15: walk-forward window sensitivity

## New literature evidence

Mroziewicz & Ślepaczuk (2026), “A Novel Approach to Trading Strategy Parameter Optimization, using Double Out-of-Sample Data and Walk-Forward Techniques” (SSRN 6217160), studies intraday Bitcoin across six frequencies and many walk-forward window specifications. Their central methodological result is that strategy performance can be strongly dependent on the chosen training/testing window lengths and that a separate out-of-sample confirmation is important after window selection.

This does not establish profitability for the project's six-bar BTC strategy. It identifies another selection dimension that must be explicitly accounted for.

## New validation gate: H115

Issue #63 freezes walk-forward window specification as an explicit research dimension and requires double-OOS confirmation.

### Required artifacts
- a predeclared, small WFO window family;
- immutable trial IDs including train length, validation/test length and cadence;
- development-only window selection;
- a disjoint chronological confirmation segment;
- per-fold/path metrics and raw OOS vectors;
- dependence-aware uncertainty;
- unchanged execution horizon, overlap policy, cost/slippage model and trade ledger during confirmation.

### Falsification
If the economic ranking changes materially across reasonable predeclared windows, classify the candidate as window-sensitive. If the frozen candidate fails on the disjoint confirmation segment, classify it as a generalization failure rather than opening another unregistered window search.

## Broader literature signal

The new evidence complements three existing findings:

1. CPCV can reduce backtest-overfitting risk in controlled financial experiments, but its path/accounting implementation must itself be audited.
2. Recent crypto validation work finds predictive metrics can have substantial false-positive rates and transaction costs can consume most gross alpha.
3. Recent hourly BTC work shows cost-aware execution filtering can matter more than escalating model architecture, while model-family differences may not be statistically significant.

Together, these imply that the project's next meaningful economic comparison should not be a larger model search. The priority is to freeze every selection dimension, verify execution, and then test generalization on untouched chronological data.

## Current project status

Latest repository commit remains `025e3e8b9ec38b5bc2be41e739e61d9be8fc600a` (H106 post-selection robustness audit). Open execution/data gates include H101/H102/H103/H113/H114 and the related cost/economic gates. H115 is downstream of those gates.

No new performance claim is made in this audit.
