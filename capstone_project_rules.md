# Systematic Multi-Asset Portfolio — Capstone Project Rules

## What this project is

A monthly-rebalanced, multi-asset ETF portfolio system. Goal: better
**risk-adjusted returns** (Sharpe, Sortino, max drawdown) than holding SPY —
NOT higher raw returns. Participate in long-term growth with lower volatility
and controlled drawdowns.

Owner: Javier. This is a learning project with real-money intent: he must
understand every piece. Prefer clarity over cleverness.

## Universe (FIXED — do not change)

Optimizable (6): SPY, VEA, XLE, XLU, IEF, GLD
Anchor (not optimized): SHY — receives the capital freed when the system
reduces total exposure in response to the detected regime (see step 5).

Market indicators (watched, never traded): VIX, yield curve spread (T10Y2Y,
FRED), Cleveland Fed recession probability, CAPE (shillerdata.com).

## Architecture (monthly cycle)

1. Weekly returns, 3-year rolling window → sample covariance
2. Ledoit-Wolf shrinkage (later: EWMA, then Kalman if justified)
3. Weight optimization: MinVar-with-caps AND Risk Parity (they compete)
4. Momentum tilt (12-1 month), capped at ±20% of base weight
5. Regime detection (K-means on volatility/returns/correlation,
   asymmetric hysteresis) → scales total exposure. Exposure map is a
   TUNABLE under evaluation in step 10 (original: 100/80/50; softer
   variants under test); freed capital → SHY.
6. Final weights → monthly rebalance

## INVIOLABLE RULES — read before writing ANY code

1. **Point-in-time everything.** Every computed object (covariance, weights,
   regime label, prediction) uses ONLY data available at that date. Store
   each month's output as computed at that time; never overwrite with
   recomputed history.
2. **No lookahead, ever.** In backtests, a decision at month T may not touch
   any data after T. Regime labels: use only the CURRENT month's label from
   the model trained up to that month — never retrospective relabeling.
3. **Covariance uses WEEKLY returns.** A design choice, consistent with the
   weekly features of the regime detector and the monthly cadence; it was not
   compared against covariance from daily returns. (SPY–VEA correlation is
   practically the same at both frequencies in this data — 0.883 daily, 0.880
   weekly — because VEA is a US-listed ETF, so there is no close-time lag to
   correct.) Daily data is fine for monitoring and indicators; aggregate to
   weekly for covariance estimation.
4. **Never estimate expected returns as optimizer input.** We minimize risk
   (or equalize risk contributions). Momentum enters only as a capped tilt
   AFTER optimization, never inside it.
5. **Label lag (phase 2 classifier).** Labels are written with H=4 weeks of
   future data. When training at date T, samples in (T−H, T] have no label
   yet — exclude them. Labeling them from stored future data = lookahead.
6. **Simple beats sophisticated until proven otherwise.** Baselines first
   (sample cov → LW → EWMA before Kalman; K-means before HMM). Advanced
   versions must BEAT their baseline in the backtest to earn their place.
7. **Sanity checks are mandatory.** After computing correlations, verify:
   SPY–VEA high (~0.8), SPY–IEF near zero/negative, SHY near zero with
   everything. If not, suspect a data bug before anything else.

## Code style

- Jupyter notebooks, one per step, in `notebooks/`. Named `01_data.ipynb`,
  `02_exploration.ipynb`, etc.
- Plain pandas/numpy/sklearn/scipy. NO custom frameworks, NO class
  hierarchies, NO abstraction layers. Simple functions, clear names.
- Explain non-obvious steps in short markdown cells (Javier reads everything).
- Data to `data/` (raw and processed separate). Point-in-time outputs to
  `outputs/` as parquet/csv with a date column meaning "as computed at".
- Do NOT skip ahead. Implement only the step requested in the prompt.
  The step-by-step specs come separately.
  - Every notebook's FIRST cell is a markdown cell with the full spec/prompt
  for that step, verbatim, titled "# Step N — Specification".

## Step plan (specs arrive one at a time — do not pre-build)

1. Data download & clean returns (prices adjusted for dividends)
2. Exploration: ACF of returns vs volatility; K-means clustering of assets
3. Covariance: sample + Ledoit-Wolf (rolling 3y weekly)
4. Covariance: EWMA variant
5. Optimization: MinVar-with-caps vs Risk Parity
6. Regime v1: K-means on market conditions → exposure scaling. Training
   window REVISED (expanding -> 5-year rolling), after diagnosing that the
   expanding window was permanently anchored by 2008 and under-detected
   the slow 2015-16 drawdown. Adopted as the production model after a
   dedicated backtest comparison (notebooks/exp_regime_window_backtest.ipynb)
   — a disclosed in-sample decision, not a proven win: bootstrap on the
   Sharpe difference is a tie that leans favorable (never worse episode-by-
   episode, modestly higher full-period Sharpe). Same features, same k=3,
   same asymmetric hysteresis as v1 — only the training window changed.
7. Momentum tilt + full monthly pipeline assembly
8. Walk-forward backtest with transaction costs + crisis event studies
   (2000-02, 2008, 2020, 2022)
9. Statistical validation: bootstrap, sensitivity analysis
10. Aggressiveness rebalance: five pre-declared variants (IEF cap,
    softer/faster regime map, combined, growth risk-budget, regime
    switching). Kalman covariance upgrade DEFERRED with evidence: step 8/9
    showed a three-way covariance tie (no prize for a finer map).
    Conditions to revisit: prolonged bear-market era, detector false
    positives in live tracking, or an expanded universe.
11. DONE: HMM regime detector, built and compared against the K-means
    baseline (notebooks/11_hmm_regime.ipynb).
12. DONE: supervised model comparison (notebooks/12_supervised.ipynb):
    logistic regression and random forest predicting whether month T+1 is
    an official STRESS month from the detector's three features, with
    temporal cross-validation (TimeSeriesSplit, label-lag gap per rule 5)
    and pre-declared grid search. Verdict: neither beats the persistence
    baseline out-of-fold (recall 0.80 vs 0.70, bought with 23-28 false
    alarms vs 6; balanced accuracy 0.76-0.79 vs 0.82). Not adopted; the
    frozen configuration is unchanged.
13. Tested as a standalone exercise, not adopted: an intramonth
    (weekly-check) extraordinary-rebalance trigger
    (a side notebook, not included in this submission), reopening the idea originally
    suspended after step 9 (the monthly regime layer's cost was the
    closest-to-significant effect in this project; adding intramonth
    sophistication to the least-paying layer wasn't judged worth it).
    Result confirms rather than overturns that call: detection-lag savings
    are real but rare (5 trigger events in 16 years; COVID's documented
    "1-month lag" turned out to be a hysteresis-confirmation issue, already
    fixed, not a measurement-frequency one), and full-backtest performance
    is a tie leaning slightly negative. Revisit only under the same
    conditions as step 10's deferred upgrades.

## Expected historical behavior (for sanity checking results)

- Slow crises (2000-02): system should exit early via momentum + regime —
  minimal drawdown vs SPY.
- Announced crises (2008): months of prior signals; enters the acute phase
  already de-risked.
- Vertical crashes (2020): first ~-8% is unavoidable; system de-risks within
  days (phase 2) or at month-end (phase 1). Slow re-entry costs part of the
  rebound — this is known and accepted.
- IEF and GLD should show POSITIVE returns in equity crisis periods
  (except 2022, where bonds fell with equity — that's real, not a bug).
