# A Systematic Multi-Asset Portfolio with Regime-Scaled Exposure
## Full project report — design, evidence, and conclusions

**Author:** Javier · 2026

---

## 1. Question and objective

Can a monthly-rebalanced portfolio of liquid ETFs achieve better
**risk-adjusted returns** than passively holding the S&P 500?

The objective was declared before any data was touched: **participate in
long-term market growth with materially lower volatility and controlled
drawdowns.** Success metrics: Sharpe ratio (comparison), Sortino
(validation), maximum drawdown and Calmar ratio (the objective itself).
Explicitly *not* an objective: beating SPY in raw return.

## 2. Design principles (fixed before implementation)

1. **Estimate only what is estimable.** Return direction is nearly
   unpredictable; risk (covariance, volatility) is estimable. The optimizer
   therefore never receives expected returns — it minimizes or budgets risk.
2. **Point-in-time everything.** Every monthly decision (covariance, weights,
   regime label) uses only data available at that date. Labels are written
   once and never retro-corrected ("diary rule").
3. **Simple first; sophistication must beat its baseline to stay.**
4. **Asymmetry of harms.** Protecting too early is cheap; protecting too
   late is expensive. This principle shaped the regime hysteresis.

## 3. Universe and data

Seven ETFs, three genuinely distinct behavior families (validated in §5):

| Role | Assets |
|---|---|
| Growth engine | SPY, VEA, XLE, XLU |
| Recession airbag / anchor | IEF (optimized) · SHY (outside optimization; receives capital freed by the regime layer) |
| Tail hedge | GLD |

Daily adjusted prices from 2000 (each asset from its own inception; VEA
2007). Weekly returns feed covariance estimation — a design choice, consistent with the
weekly regime features and the monthly cadence, not compared against daily-return
covariance (SPY–VEA correlation is 0.883 daily vs 0.880 weekly in this data: VEA is
a US-listed ETF, so there is no close-time lag to correct). Indicators: VIX,
VIX3M, T10Y2Y (FRED).

## 4. Architecture (monthly cycle)

```
weekly returns, 3y rolling window
  → sample covariance → Ledoit-Wolf shrinkage (EWMA variant in parallel)
  → weight optimization (Risk Parity / MinVar-with-caps; 6 assets)
  → momentum tilt (12-1, rank-based, capped ±20%)
  → regime exposure scaling (K-means detector; freed capital → SHY)
  → final weights → rebalance, 5 bps one-way costs
```

## 5. Evidence base, step by step

**Data validation.** Adjusted-price signature confirmed (IEF 3.6%
annualized — unadjusted data would show ~1%). Correlation sanity: SPY–VEA
0.88, SPY–IEF −0.22 on the full sample (−0.21 on the common sample from 2007; the
airbag), SHY weakly correlated with the risk assets (−0.19 to +0.22) and strongly
with IEF (0.78, both Treasuries), GLD an island.

**ACF diagnostics (the design's empirical foundation).** Weekly returns:
Ljung-Box significant in several assets but the largest |acf| is small (≈ 0.07 on
average across assets, 0.082 at most) — under 1% of next-week variance, below cost thresholds. *Statistically detectable ≠
economically exploitable.* Squared returns: p-values 10⁻²⁴ to 10⁻¹²²,
largest |acf| ≈ 0.26 on average, up to 0.40 (XLU) — volatility clusters strongly for 4–6 weeks. Direction:
unpredictable. Intensity: persistent. The system's entire shape follows
from this asymmetry.

**Clustering validation of the universe.** K-means on co-movement profiles
recovered, unsupervised, exactly the designed families: k=2 → {equity} vs
{IEF, SHY, GLD}; k=3 → {SPY, VEA, XLE, XLU} · {IEF, SHY} · {GLD alone}.
The dendrogram placed XLU as the most distant equity member (the defensive)
and GLD as a genuine third behavior — diversification is real, not nominal.

**Covariance estimation.** ~190 point-in-time monthly matrices. Ledoit-Wolf
shrinkage intensity self-calibrated in 0.03–0.19, doubling in March 2020
(chaotic sample → more shrinkage) — statistical humility as a formula. Key
market finding: the SPY–IEF correlation, negative for a decade, **flipped
positive in 2022** (the airbag failure year); the rolling estimation saw it,
and downstream, Risk Parity automatically cut IEF's weight from ~46% to ~33%
— a decision no human made. EWMA (λ=0.97) detected the flip ~6 months
earlier than the rolling window and eliminated the window's mechanical
"COVID exits the sample" step artifact of 2023.

**Regime detector.** K-means (k=3) on realized volatility, trailing return,
and average equity correlation; monthly refit, point-in-time labels only. A
symmetric 2-month confirmation rule initially destroyed its value (official
stress = 3.1% of months; the Feb-2020 signal was ignored). Diagnosis:
symmetric rules violate the asymmetry of harms. Fix: **asymmetric
hysteresis** — enter stress on 1 signal, exit on 2.

**Training window revised: expanding → 5-year rolling.** With the asymmetric
fix alone, stress still sat at only ~3% of months (expanding window), because
2008 never left the training set and permanently anchored the STRESS
centroid at crisis-level volatility — the detector caught every sharp
episode (2018Q4, 2020, 2022) but missed the slow 2015-16 drawdown entirely.
Diagnosed with a cheap, rejected control first (adding VIX as a 4th feature
changed nothing — the problem was the training window's reference scale, not
the feature set), then confirmed by testing rolling windows: an 8-year
window wasn't long enough to exclude 2008 by 2015-16 (no fair test); a
5-year window is, and does catch it (Oct-Dec 2015 reads NORMAL, Jan-Feb 2016
reads STRESS). Adopted after a dedicated walk-forward backtest confirmed the
effect holds for real portfolio performance, not just regime-frequency
diagnostics (§6-7) — disclosed as an in-sample decision, validated only to a
bootstrap tie that leans favorable. Result: official stress = 14.1% of
months (within the 5-30% target band), COVID protected from *February*
2020, 2018Q4/2022 captured, and now 2015-16 as well.

**Supervised comparison: can next month's regime be predicted, not just
described? (notebook 12).** Logistic regression (with a volatility ×
correlation interaction) and a random forest were trained to predict whether
month T+1 is an official STRESS month from the detector's three features at
month-end T, z-scored with expanding statistics. Temporal cross-validation
(`TimeSeriesSplit`, 5 folds of 24 months, one-month gap enforcing the
label-lag rule), pre-declared grids searched with `GridSearchCV`, recall on
STRESS as the primary metric and `class_weight='balanced'` for the 14%
minority class; grids, metrics and the decision rule were fixed in markdown
before any result. Out-of-fold on 120 held-out months (20 STRESS): persistence
("next month = this month") scores recall 0.70 / precision 0.70 / balanced
accuracy 0.82; the tuned logistic and random forest reach recall 0.80 but
with 28 and 23 false alarms against persistence's 6, and balanced accuracy
0.76 and 0.79. Persistence misses exactly the six episode onsets and
false-alarms exactly the six exits; the models catch two of the six onsets
(both inside the 2022 bear) and miss the four sudden ones — for February
2020 the logistic probability was 0.48. All sixteen hyperparameter
configurations sit within two points of CV recall: hyperparameters are not
what limits the task, the information content of three backward-looking
features at a one-month horizon is. Verdict under rule 6: **prediction
collapses to description**; no supervised model enters the pipeline and the
frozen configuration is unchanged. A side finding: in this window the average
official STRESS month was not a losing month for the diversified book — the
case for calling stress lives in the tail (10th-percentile STRESS month −3.6%
vs −2.1% otherwise), which is why recall was the right primary metric and
also why recall alone is not enough.

## 6. Backtest results (2010–2026, 5 bps costs, all point-in-time)

| Track | CAGR % | Sharpe | MaxDD % | Calmar |
|---|---|---|---|---|
| 60/40 | 9.8 | 0.878 | −21.3 | 0.460 |
| RP-LW base (no tilt/regime) | 7.4 | 0.838 | −16.1 | 0.460 |
| SPY | 14.7 | 0.810 | −33.7 | 0.437 |
| RP-LW full chain | 6.2 | 0.809 | −13.0 | 0.479 |
| **Final config (V-AB)** | **7.9** | **0.808** | **−14.7** | **0.535** |

**Crisis episodes (full chain vs SPY)**: 2011: +3.9% vs −4.4%; **2015-16 (now caught by the revised detector): −0.6% vs −7.0%** (recovery 4.6 months, vs 2.1 for SPY — the system still trades climb-speed for hole-depth even here); 2018Q4: −3.2% vs −13.5% (maxDD −4.9% vs −19.2%, trough-to-recovery 5.8 vs 3.6 months); COVID: max drawdown −10.7% vs −33.7%, recovery 4.5 vs 4.6 months; 2022: −9.3% vs −17.7% (maxDD −13.0% vs −24.5%, recovery 15.0 vs 14.0 months). The "controlled drawdowns" objective was met in every episode of the sample, now including the slow-grind case the detector originally missed. An honest nuance surfaced by a bug fix: the system's edge is the depth of the hole, not the speed of the climb — after de-risking it participates less in rebounds, and in several episodes it regained its own prior peak later than SPY despite falling much less. Earlier drafts reported implausibly short recovery times caused by a reference-peak bug in the episode windows, since corrected in notebook 08.

**Layer attribution.** Sharpe: base 0.838 →
+tilt 0.825 → +regime 0.809. The regime layer still trades some Sharpe for
drawdown control, but the gap narrowed sharply once the detector's training
window was fixed (previously 0.836 → 0.767, a 0.069 cost; now a 0.029 cost) —
the revised detector is doing its job more cheaply, not just more sensitively.

**Covariance ladder.** RAW 0.783 / LW 0.809 / EWMA 0.775 — a three-way tie.
With 6 liquid assets, estimation error was a small problem. Consequence:
Kalman filtering (planned upgrade) was **deferred with evidence** — no prize
available for a finer map. HMM regime detection was built and compared
against the K-means baseline (notebook 11); separately, the K-means
detector's own training window was revised (see §5) after a 2008-anchoring
blind spot was diagnosed.

## 7. Statistical validation (the step that rewrote the verdict)

Moving-block bootstrap (6-month blocks, 10,000 resamples) on paired monthly
returns:

| Pair | Sharpe diff | 95% CI | Verdict |
|---|---|---|---|
| 60/40 − RP-LW base | 0.101 | [−0.23, +0.44] | TIE |
| 60/40 − RP-LW full | 0.145 | [−0.18, +0.48] | TIE |
| base − full (regime cost) | 0.044 | [−0.06, +0.16] | TIE |
| full − SPY | −0.112 | [−0.50, +0.26] | TIE |
| V-AB − 60/40 | −0.120 | [−0.47, +0.21] | TIE |

**No efficiency difference in this study is statistically distinguishable
from sampling noise.** The honest headline is not "60/40 won" but "a
statistical tie in efficiency — with radically different drawdown profiles,
where the system is unambiguously gentler (observed fact, not estimate)."

Period dependence: the system's Sharpe disadvantage was concentrated in the
2010–2017 bull era (RP-LW full 1.061 vs 60/40's 1.577); in 2018–2026 the
simple system essentially matches 60/40 (0.725 vs 0.722) with little more
than half the drawdown (−12.1% vs −20.5%). Regime decomposition (revised
detector): the governing-regime split flipped from mostly-CALM to
mostly-non-CALM (67 CALM vs 125 NORMAL+STRESS months, inverted from the
expanding-window detector's 112/79) — the system's shortfall now lives in
a smaller, more genuinely-calm CALM bucket (7.5% vs SPY's 12.4% annualized),
while NON-CALM months are closer between the two (5.6% vs 16.0%) than
before, since fewer months get the "calm" label in the first place.

## 8. Final configuration (chosen by the data)

Five pre-declared aggressiveness variants were tested against the measured
CALM-months shortfall. Winner — **V-AB**: Risk Parity + Ledoit-Wolf, **IEF
capped at 25%**, momentum tilt ±20%, K-means regime with asymmetric
hysteresis, **softened exposure map 100/90/60** with 1-month stress exit.

Results: CAGR 7.9%, MaxDD −14.7%, **Calmar 0.535 — best in the entire study — Sharpe tie with all benchmarks.** Note: Calmar was the pre-declared objective metric and drives the selection, but unlike the Sharpe comparisons it was not bootstrap-tested; max drawdown is a single order statistic, so the Calmar ranking is a descriptive fact of this sample, not an established statistical difference. A regime-switching alternative (hold 60/40 in calm, defensive in stress) was tested at the owner's
proposal and rejected by the data — and more decisively so under the revised
(more reactive) detector: it inherits the full 60/40 drawdown in slow-grind
bears (−21.1% in 2022, vs 60/40's −20.9%) and now switches 16 times over the
sample (up from 11 under the old detector), with ~1,599% lifetime one-way
turnover and 10.0 bps/yr cost drag — both the highest of any variant by a
wide margin. Its bootstrap vs 60/40 remains the closest to significantly
*negative* in the project (sign-flip 6.6%).

## 9. Honest limitations

1. **The sample contains no long bear market.** Every crisis 2010–2026
   recovered in a V (2015-16 included, once the detector was fixed to see
   it). The regime layer's insurance value against a 2000–2002-type 30-month
   decline is untested and untestable with this data. Its measured cost
   (~0.044 Sharpe, itself a statistical tie, down from ~0.08 before the
   detector revision) is the premium for that unobserved scenario — keeping
   it is a declared preference, not a statistical conclusion.
2. **Five parameters were adjusted after seeing results** (asymmetric
   hysteresis; the regime detector's training window, expanding → 5-year
   rolling; IEF cap; softer exposure map; faster stress exit). Defense: few,
   each pre-justified on its own terms (not grid-searched) and declared
   explicitly before/alongside running, validated only to a tie or a
   tie-leaning-favorable — never a proven win. True validation is
   **out-of-sample paper-tracking going forward**; the configuration is now
   frozen — no further tuning on this sample. A sixth candidate (an
   intramonth weekly-check trigger, reopening a previously-suspended idea)
   was tested as a standalone exercise (a side notebook not included in this
   submission) and **not adopted** — a tie leaning slightly negative, on only
   5 trigger events in 16 years.
3. Backtested returns are a windowed average, not a promise: the ~7.9%
   CAGR came from the most equity-favorable 16 years in modern history.

## 10. Conclusions

1. The declared objective — growth participation with controlled drawdowns
   — **was met**: worst drawdown one-third of SPY's, recovery times comparable to the benchmarks (the edge is shallower holes, not faster climbs), best Calmar in the study.
2. Efficiency (Sharpe) is a **statistical tie** with both SPY and 60/40;
   claims of superiority in either direction are unsupported at this
   sample size.
3. **Sophistication must earn its place — sometimes it didn't, sometimes it
   did, and the difference was always measured, not assumed:** covariance
   refinements tied three ways; tilt was neutral; the original regime map
   charged more than it returned and was softened; a 4th feature (VIX) for
   the regime detector was tested and cleanly rejected; a revised training
   window (expanding → 5-year rolling) WAS adopted after a dedicated
   backtest, as a disclosed in-sample call, not a proven win; an HMM
   detector was built and compared against the K-means baseline; a
   supervised one-step-ahead classifier (logistic and random forest,
   temporal cross-validation, grid search) did not beat the persistence
   baseline out-of-fold; an intramonth trigger was tested and not adopted. Kalman remains deferred
   with evidence (no prize available for a finer covariance map).
4. The most valuable outputs are methodological: the
   significant-vs-exploitable distinction, point-in-time discipline
   (including label lag, now exercised in code in notebook 12), asymmetric hysteresis, and negative findings
   documented with the same rigor as positive ones — including when a
   candidate's rejection took two attempts to test fairly (an 8-year
   rolling window gave the 2008-anchoring hypothesis no real chance to be
   tested before concluding a 5-year window was needed).

*All results reproducible from notebooks 01–12 plus one standalone
experiment notebook (the regime training-window backtest); every figure and
table is point-in-time with transaction costs included.*
