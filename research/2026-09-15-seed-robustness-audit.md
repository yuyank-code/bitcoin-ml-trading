# Seed and optimizer-path robustness audit — H116

## Date
2026-09-15

## Hypothesis
A candidate model's economic performance should not materially depend on random seed or optimizer initialization when data, timestamps, features, hyperparameters, WFO windows, and execution rules are frozen.

## Literature basis
A 2026 large-scale financial time-series benchmark evaluates models not only by risk-adjusted OOS performance but also by robustness to random seed and breakeven transaction costs. This is a useful control for the trading project because seed selection is itself a hidden model-selection axis.

The 2026 BTC literature also continues to show a prediction/economic disconnect: complex models can improve forecast metrics while realistic transaction costs eliminate the trading edge. Therefore seed robustness must be evaluated on executable net returns, not AUC alone.

## Test protocol
1. Freeze dataset and decision-time timestamp manifest.
2. Freeze feature set, WFO window specification, model hyperparameters, execution horizon and cost/slippage grid.
3. Declare a small seed set before inspecting results.
4. Fit each seed independently within each walk-forward training window.
5. Evaluate all seeds on identical OOS timestamps.
6. Preserve raw OOS predictions and net trade returns per seed and fold.
7. Report median, dispersion, worst fold, and seed-to-seed rank stability.
8. Repeat the comparison across the predeclared realistic cost/slippage grid.
9. Do not select a seed; a candidate must be robust across the declared seed set.
10. Keep the final chronological confirmation holdout untouched.

## Failure criteria
Classify the candidate as seed-sensitive if one seed/path drives the economic result, if fold-level conclusions are unstable, or if the result disappears under modest realistic cost/slippage stress.

## Promotion constraint
H116 is an audit gate. It cannot be used to choose the best seed or tune the model. All seed-level OOS vectors must enter the H106 post-selection trial accounting.

## Status
Protocol registered. No numerical model-performance claim is made until the event-time execution and cost-replay gates are executable and verified.
