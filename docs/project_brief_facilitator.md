# Systematic Multi-Asset Portfolio — Project Brief

**Student:** Javier · **Capstone project, ML/AI Professional Certificate**

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
   input to portfolio optimization (minimum-variance and risk-parity weights).

4. **Walk-forward validation and bootstrap inference.** Every monthly decision
   uses only data available at that date (train/test discipline applied to
   time series). Final Sharpe differences are tested with a moving-block
   bootstrap (10,000 resamples) — the honest headline: most differences
   between my system, a 60/40 benchmark and SPY are **statistical ties**.

## Headline result (16-year backtest, transaction costs included)

The final configuration returns ~7.9%/yr with a worst drawdown of **−14.7%**
versus −33.7% for SPY and −21.3% for 60/40 — the best return-per-unit-of-
drawdown (Calmar) in the study — with Sharpe statistically indistinguishable
from both benchmarks. A key honest finding: several sophisticated layers
(adaptive covariance, tighter regime gating) did **not** improve results and
were removed or simplified based on evidence.

## What I want to discuss

Whether the negative findings (sophistication that didn't pay, documented
with bootstrap ties) are presented with the right weight in the final report
— I believe they are the most instructive part of the project.
