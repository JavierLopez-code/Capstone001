# Systematic Multi-Asset Portfolio with Regime-Scaled Exposure

**Capstone Project — final submission.** Author: Javier Hernan Lopez.

## What this project is

A monthly-rebalanced portfolio of seven liquid ETFs, built to answer one question:

> Can a diversified, rules-based portfolio achieve better **risk-adjusted returns** (Sharpe, Sortino,
> maximum drawdown) than passively holding the S&P 500 — participating in long-term growth with lower
> volatility and controlled drawdowns?

The goal is explicitly *not* to beat SPY in raw return. Success was declared on risk-adjusted metrics
before any data was touched. Everything is **point-in-time**: each month's covariance, weights and regime
label use only data available at that date, and are stored as computed — never overwritten with recomputed
history. Without that discipline any backtest would be fiction.

## Headline result

Walk-forward backtest, 2010-08 to 2026-07, 5 bps one-way transaction costs
(`outputs/backtest_metrics.csv`, `outputs/aggressiveness_metrics.csv`):

| Track | CAGR % | Sharpe | MaxDD % | Calmar |
|---|---|---|---|---|
| 60/40 (SPY/IEF) | 9.8 | 0.878 | −21.3 | 0.460 |
| RP-LW base (no tilt, no regime) | 7.4 | 0.838 | −16.1 | 0.460 |
| SPY | 14.7 | 0.810 | −33.7 | 0.437 |
| RP-LW full chain | 6.2 | 0.809 | −13.0 | 0.479 |
| **Final configuration (V-AB)** | **7.9** | **0.808** | **−14.7** | **0.535** |

- **The drawdown objective was met.** Worst drawdown −14.7% against SPY's −33.7% and 60/40's −21.3%, and the
  highest Calmar (return per unit of drawdown) of any track in the study.
- **Efficiency is a statistical tie.** In a moving-block bootstrap every Sharpe difference between the system,
  SPY and 60/40 has a 95% confidence interval that spans zero. The honest headline is not "the system won" but
  "same efficiency, radically gentler drawdowns".
- **Where the edge is:** the depth of the hole, not the speed of the climb. After de-risking the system
  participates less in rebounds and, in several episodes, regained its own prior peak no faster than SPY.

Crisis episodes (full chain vs SPY vs 60/40; total return / worst drawdown inside the episode):

| Episode | RP-LW full chain | SPY | 60/40 |
|---|---|---|---|
| 2011 (Jul–Oct) | +3.9% / −3.9% | −4.4% / −18.4% | +1.2% / −8.4% |
| 2015–16 (Aug–Feb) | −0.6% / −4.8% | −7.0% / −12.8% | −2.2% / −6.3% |
| 2018Q4 (Oct–Dec) | −3.2% / −4.9% | −13.5% / −19.2% | −6.7% / −10.5% |
| 2020 COVID (Feb–Aug) | +0.2% / −10.7% | +4.8% / −33.7% | +6.8% / −18.9% |
| 2022 (Jan–Oct) | −9.3% / −13.0% | −17.7% / −24.5% | −17.1% / −20.9% |
| 2025 (Apr episode) | +2.5% / −7.2% | +10.8% / −12.1% | +7.0% / −7.4% |

## Folder structure

```
Capstone project.md        project rules: universe, architecture, inviolable rules, step plan
README.md                  this file
requirements.txt           pinned package versions
data/                      frozen data snapshot (daily prices, daily and weekly returns, indicators)
outputs/                   point-in-time outputs (parquet/csv; the date column means "as computed at")
docs/
  project_report_full.md   full report: design, evidence, conclusions
  project_brief_facilitator.md   one-page brief
notebooks/
  01_data.ipynb            download, cleaning and validation of dividend-adjusted prices
  02_exploration.ipynb     ACF diagnostics + K-means clustering of the assets
  03_covariance.ipynb      rolling 3-year weekly covariance: sample + Ledoit-Wolf shrinkage
  04_ewma.ipynb            EWMA covariance variants (λ = 0.94 / 0.97 / 0.99)
  05_optimization.ipynb    minimum variance with caps vs Risk Parity
  06_regime.ipynb          regime detector (K-means), hysteresis and the training-window revision
  07_pipeline.ipynb        momentum tilt + regime exposure: the full monthly pipeline
  08_backtest.ipynb        walk-forward backtest with costs + crisis event studies
  09_validation.ipynb      bootstrap, period splits, cost sensitivity
  10_aggressiveness.ipynb  five pre-declared variants; final configuration
  11_hmm_regime.ipynb      HMM regime detector built and compared against K-means
  exp_regime_window_backtest.ipynb   evidence for the training-window revision (see below)
```

Every notebook opens with a markdown cell holding its specification verbatim, and explains non-obvious
steps in short markdown cells.

## How to reproduce

```
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb
```

- Run order: `01` (optional, safe by default) → `02` → `03` → `04` → `05` → `06` → `07` → `08` → `09` → `10` →
  `11`, then `exp_regime_window_backtest`. Each reads what the previous ones saved to `outputs/`.
- `data/` is a frozen snapshot (prices through 2026-08-04; weekly returns through 2026-08-07). Run from it,
  every notebook reproduces the stored outputs exactly. The monthly schedules are bounded by the last real
  observation, not by today's date, so re-running later does not change the results.
- `01_data.ipynb` has a data switch, `REFRESH_DATA`, **off by default**. With `False` it loads the frozen
  snapshot, downloads nothing, writes nothing, and still runs every validation, so it is safe to run. Set it
  to `True` only to rebuild the snapshot from Yahoo Finance and FRED (needs internet): that **overwrites
  `data/`** with newer data and changes every downstream result in notebooks 02–11.

## The method in one page

Universe: SPY, VEA, XLE, XLU, IEF, GLD are optimized; SHY is the anchor that receives capital freed when the
system reduces exposure. VIX, the yield-curve spread, recession probability and CAPE are watched, never traded.

Monthly cycle, each step using only data available at that month-end:

1. **Weekly returns, 3-year rolling window → sample covariance.** Weekly, not daily: a design choice, consistent
   with the weekly regime features and the monthly cadence. It was not compared against daily-return covariance;
   SPY–VEA correlation is practically the same at both frequencies in this data (0.883 daily, 0.880 weekly)
   because VEA is a US-listed ETF.
2. **Ledoit-Wolf shrinkage** (EWMA runs in parallel as a competing estimator).
3. **Weight optimization:** minimum variance with caps, and Risk Parity (equal risk contributions). Expected
   returns are never an optimizer input — the ACF diagnostics show return direction is economically
   unpredictable (largest autocorrelation ≈ 0.07 on average, 0.08 at most) while volatility clusters strongly.
4. **Momentum tilt** (12-1 month), capped at ±20% of base weight, applied *after* optimization.
5. **Regime detection** (K-means, k=3, on realized volatility, trailing return and cross-asset correlation;
   asymmetric hysteresis) scales total exposure; the capital freed goes to SHY.
6. **Final weights → monthly rebalance.**

### The regime detector and how it evolved

- **v1:** K-means on three weekly features, expanding training window, symmetric rule (two consecutive
  signals to change regime). It ignored the February-2020 signal and flagged stress in only 3% of months.
- **Asymmetric hysteresis:** enter STRESS on one signal, leave it only after two. Reacting fast to danger and
  slowly to relief follows the asymmetry of harms.
- **Training window, expanding → 5-year rolling:** the 2008 crash permanently anchored the STRESS centroid
  at crisis-level volatility, so the slow 2015-16 drawdown was missed. A rolling window lets 2008 roll off.
  `exp_regime_window_backtest.ipynb` pushed both detectors through the full pipeline: Sharpe 0.775 → 0.809,
  bootstrap CI [−0.048, +0.094]. A **tie that leans favorable**, adopted as a disclosed in-sample decision.
  Official STRESS is now 14.1% of months (41% CALM, 45% NORMAL).

## Validation

Moving-block bootstrap (6-month blocks, 10,000 resamples, paired monthly returns; `outputs/validation_summary.csv`):

| Pair | Sharpe difference | 95% CI | Verdict |
|---|---|---|---|
| 60/40 − RP-LW base | 0.101 | [−0.23, +0.44] | tie |
| 60/40 − RP-LW full | 0.145 | [−0.18, +0.48] | tie |
| RP-LW base − full (cost of the regime layer) | 0.044 | [−0.06, +0.16] | tie |
| RP-LW full − SPY | −0.112 | [−0.50, +0.26] | tie |
| RP-LW full − RP-RAW full (shrinkage) | 0.033 | [−0.03, +0.10] | tie |

Other checks in `09_validation`: the ranking does not hold across the two halves of the sample (60/40 leads
in 2010–2017; SPY leads in 2018–2026), the system's Sharpe gap vs 60/40 reverses to positive in recent years,
and the ranking is nearly unchanged by transaction costs of 0, 5 and 15 bps (only adjacent swaps among mid-table tracks).

## Final configuration

Five variants were declared **before running** (`10_aggressiveness`) to attack the measured shortfall in calm
months, and tested through the same engine:

| Variant | CAGR % | Sharpe | MaxDD % | Calmar |
|---|---|---|---|---|
| V-A: IEF capped at 25% | 7.3 | 0.809 | −13.8 | 0.531 |
| V-B: softer exposure map 100/90/60 + 1-month stress exit | 6.6 | 0.811 | −14.0 | 0.476 |
| **V-AB: V-A + V-B (selected)** | **7.9** | **0.808** | **−14.7** | **0.535** |
| V-C: risk budget tilted to growth assets | 6.6 | 0.792 | −13.4 | 0.490 |
| V-D: hold 60/40 in calm months, defensive book otherwise | 6.3 | 0.736 | −21.4 | 0.295 |

V-AB won on Calmar. Calmar was the pre-declared objective metric, but it was **not bootstrap-tested** (maximum
drawdown is a single order statistic of one historical path), so "best Calmar" is a descriptive fact of this
sample. V-D was rejected by the data: it puts the whole book on the detector at every transition, made 16
switches with ~1,600% lifetime turnover and the highest cost drag, and in 2022 it held 60/40 into the
stock-and-bond selloff (−21.1% drawdown).

## HMM

`11_hmm_regime.ipynb` builds a point-in-time Gaussian HMM (3 states, filtered probabilities only — the
smoothed posterior would leak future data) and compares it with K-means: monthly agreement (64%), detection
lag at crisis starts, a week-by-week "detector race" chart, and a backtest of the final configuration with each
detector. Result: a **tie** — Sharpe difference +0.004, bootstrap CI [−0.04, +0.06]. Under rule 6 the HMM does
not replace the simpler baseline; it stays as a robustness check and as a monitoring view of soft
probabilities. One asymmetry is stated in the notebook: the HMM keeps its original design (expanding window),
while only the K-means side received the window revision.

## Methodological honesty

- **Five choices were adjusted after seeing results:** asymmetric hysteresis, the training window, the IEF
  cap, the softer exposure map and the faster stress exit. Each was economically pre-justified (none was
  grid-searched) and validated only to a tie. The configuration is now **frozen**; true validation is
  out-of-sample paper-tracking going forward.
- **No long bear market in the sample.** 2010–2026 contains only V-shaped recoveries, so the regime layer's
  core insurance value — protection through a multi-year grinding decline like 2000–2002 — is untested and
  untestable with this data. Its measured cost (a Sharpe tie) is the premium for that unobserved scenario;
  keeping it is a declared preference, not a statistical conclusion.
- Backtested returns are a windowed average from one of the most equity-favorable 16-year windows in modern
  history, not a promise.

## Relationship to the partial submission

The partial submission (initial report and EDA, steps 1–5) fixed the question, the data, the exploratory
findings, the covariance estimators and the optimization baseline. This final submission keeps all of that
unchanged and adds steps 6–11: the regime detector, momentum tilt, the full pipeline, the walk-forward backtest,
statistical validation, the aggressiveness variants and the HMM comparison. Every "next step" promised there was
delivered, with more crisis episodes than announced.

Re-checking the partial report against the final data, **five statements were made more precise. No result
changed**; these are corrections of wording:

| In the partial report | In this submission |
|---|---|
| "Weekly log returns" | Weekly **simple** returns (Friday-to-Friday price change); the code always used these |
| max \|acf\| ≈ 0.07 on returns, up to ≈ 0.26 on squared returns | 0.07 and 0.26 are **averages across assets** of each asset's largest autocorrelation; the maxima are 0.082 and 0.396 (XLU). The conclusion holds: 0.082² is under 1% of next-week variance |
| Weekly covariance chosen because daily returns understate SPY–VEA correlation (close-time lag) | Weekly is a **design choice**, not compared against daily. The lag effect is not present in this data: SPY–VEA correlation is 0.883 daily vs 0.880 weekly, because VEA is a US-listed ETF |
| SHY ≈ flat against all risk assets | SHY is **weakly** correlated with the risk assets (−0.19 to +0.22) and strongly with IEF (0.78, both Treasuries) |
| SPY–IEF ≈ −0.22 | −0.22 on the full sample; −0.21 on the common sample from 2007 (the one used for clustering) |

What was re-verified and holds as stated: SPY–VEA 0.88 and SHY–IEF 0.78 (weekly); IEF's 3.6% annualized
adjusted return; 193 monthly covariance matrices; Ledoit-Wolf shrinkage between 0.03 and 0.19 with its peak
(0.186) in March 2020; the asset clusters at k=2 and k=3; the SPY–IEF correlation turning positive in the EWMA
estimate in June 2022 and in the rolling window in December 2022; and Risk Parity's IEF weight falling from 46%
(2021) to 38.6% (2023) and 33% today.

## Conclusions

1. The declared objective was met: much shallower drawdowns than SPY and 60/40 while participating in growth.
2. Efficiency (Sharpe) is a statistical tie with both benchmarks; claims of superiority in either direction
   are unsupported at this sample size.
3. Sophistication had to earn its place, and mostly did not: covariance refinements tied three ways, the
   momentum tilt was neutral, the original regime map cost more than it returned and was softened, and the HMM
   tied K-means. The one change that was adopted — the rolling training window — was adopted as a disclosed
   judgment call, not a proven win.
4. The most reusable outputs are methodological: statistically detectable is not economically exploitable,
   point-in-time discipline (including label lag), asymmetric hysteresis, and negative findings documented
   with the same rigor as positive ones.

Full detail, including every step's evidence, is in `docs/project_report_full.md`.
