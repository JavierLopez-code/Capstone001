# Systematic Multi-Asset Portfolio — Project Brief

**Student:** Javier Lopez · **Capstone project, ML/AI Professional Certificate**

## The question

Can a monthly-rebalanced, multi-asset ETF portfolio achieve better
**risk-adjusted returns** (Sharpe, Sortino, maximum drawdown) than simply
holding the S&P 500? The goal is not to beat the market in raw returns,
but to participate in long-term growth with lower volatility and
controlled drawdowns.

## The data

- 26 years of daily prices for 7 liquid ETFs: US equity (SPY), international
  developed (VEA), energy (XLE), utilities (XLU), medium- and short-term
  Treasuries (IEF, SHY), gold (GLD) — dividend-adjusted, free sources.
- Market indicators: VIX, VIX3M, Treasury yield-curve spread (FRED).

## Course techniques and how each one is used

1. **Time series analysis (ACF, Ljung-Box)** — diagnostic that justifies the
   core design decision. Result: weekly returns show statistically detectable
   but economically negligible autocorrelation (largest |acf| ≈ 0.07 on average, 0.082 at most; <1% of
   next-week variance, below transaction costs), while **squared** returns
   show overwhelming persistence (p ≈ 10⁻²⁴ to 10⁻¹²²). Conclusion: do not
   forecast return direction; estimate and track risk instead.

2. **K-means clustering — two uses.** (a) Clustering assets by co-movement
   validated the universe: the data recovered exactly the designed structure
   {equity} / {bonds} / {gold alone}, confirming diversification is real, not
   nominal. (b) Clustering months by market conditions (volatility, trailing
   return, cross-asset correlation) built a point-in-time **regime detector**
   (calm / normal / stress) that flagged every known crisis (2015-16, 2018Q4,
   COVID, 2022) with no future data — the 2015-16 slow-grind drawdown was
   missed by a first version anchored on the 2008 crash as permanent training
   history; diagnosed, then fixed by switching to a 5-year rolling training
   window (disclosed as an in-sample decision, confirmed by a dedicated
   backtest, not just by the diagnostic that found the problem).

3. **Regularized covariance estimation (Ledoit-Wolf shrinkage)** —
   conceptually Ridge regression applied to a covariance matrix: noisy
   pairwise correlations are shrunk toward their cross-sectional mean, with
   the shrinkage intensity self-calibrated from the data. This stabilizes the
   input to portfolio optimization. An EWMA covariance (λ = 0.94 / 0.97 / 0.99)
   ran in parallel as a competitor and tied Ledoit-Wolf in the backtest, so
   the simpler estimator stayed (rule: sophistication must beat its baseline).

4. **Portfolio construction (scipy optimization).** Minimum variance with
   caps and Risk Parity (equal risk contributions, SLSQP / L-BFGS-B) compete
   on the same covariance; expected returns are never an optimizer input.
   After optimization: a 12-1 month momentum tilt capped at ±20% of base
   weight, and regime-scaled exposure governed by **asymmetric hysteresis**
   (enter STRESS on one signal, leave after two). Risk Parity was chosen over
   minimum variance on drawdown and diversification, not on Sharpe (a
   bootstrap tie).

5. **Hidden Markov Model vs K-means (notebook 11).** A 3-state Gaussian HMM
   on the same weekly features, refit point-in-time and read through
   *filtered* probabilities only (the smoothed posterior would leak future
   data). Monthly agreement with K-means 64%; the final configuration run
   with each detector gives a Sharpe difference of +0.004, bootstrap CI
   [−0.04, +0.06] — a tie. Kept as a robustness check and a monitoring view,
   not adopted.

6. **Supervised classification with temporal cross-validation (notebook
   12).** Logistic regression and random forest predicting whether month
   T+1 is a STRESS month from the detector's three features, with
   `TimeSeriesSplit` (one-month gap enforcing the label-lag rule),
   pre-declared grids searched with `GridSearchCV`, `class_weight='balanced'`
   and recall on STRESS as the primary metric. Out-of-fold on 120 months
   neither beats the persistence baseline: recall 0.80 vs 0.70, bought with
   28 and 23 false alarms vs 6; balanced accuracy 0.76 and 0.79 vs 0.82. At a
   one-month horizon prediction collapses to description; not adopted.

7. **Walk-forward validation and bootstrap inference.** Every monthly decision
   uses only data available at that date (train/test discipline applied to
   time series). Final Sharpe differences are tested with a moving-block
   bootstrap (6-month blocks, 10,000 resamples), plus period splits and
   transaction-cost sensitivity (0 / 5 / 15 bps) — the honest headline: every
   difference between my system, a 60/40 benchmark and SPY is a
   **statistical tie**.

## Headline result (16-year backtest, transaction costs included)

The final configuration returns ~7.9%/yr with a worst drawdown of **−14.7%**
versus −33.7% for SPY and −21.3% for 60/40 — the best return-per-unit-of-
drawdown (Calmar) in the study — with Sharpe statistically indistinguishable
from both benchmarks. A key honest finding: several sophisticated layers
(adaptive covariance, tighter regime gating, an HMM detector, supervised
regime prediction) did **not** improve results and were removed, simplified
or left out based on evidence.

## What I want to discuss

Whether the negative findings (sophistication that didn't pay, documented
with bootstrap ties) are presented with the right weight in the final report
— I believe they are the most instructive part of the project.
