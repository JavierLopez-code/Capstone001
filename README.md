### Systematic Multi-Asset Portfolio with Regime-Scaled Exposure

**Author:** Javier Hernan Lopez — Capstone Project, final submission (Berkeley ML/AI Professional Certificate).

#### Executive summary

A monthly-rebalanced portfolio of seven liquid ETFs, built to achieve better **risk-adjusted returns**
(Sharpe, Sortino, maximum drawdown) than passively holding the S&P 500 — not higher raw returns. Every
monthly decision is **point-in-time**: covariance, weights and regime label use only data available at that
date and are stored as computed, never overwritten with recomputed history.

In a walk-forward backtest from 2010-08 to 2026-07 with transaction costs, the final configuration had a worst
drawdown of **−14.7%** against **−33.7%** for SPY and **−21.3%** for a 60/40 portfolio, with the highest Calmar
ratio (return per unit of drawdown) of any track in the backtest and variant study (notebooks 08 and 10). Its
Sharpe ratio is a **statistical tie** with both benchmarks, tested directly: in a moving-block bootstrap every
Sharpe difference, including the final configuration against 60/40 and against SPY, has a 95% confidence
interval that spans zero. The honest headline is not "the system won" but "same efficiency, radically gentler drawdowns".

A second finding runs through the whole project: sophistication had to earn its place, and mostly did not.
Covariance refinements tied three ways, the momentum tilt was neutral, the original regime map cost more than
it returned and was softened, a Hidden Markov Model tied the K-means detector, and a supervised model
comparison (logistic regression and random forest, with temporal cross-validation and grid search) did not
beat the simple persistence baseline. Each negative result is documented with the same rigor as the positive
ones.

#### Rationale

Long-term investors are rarely hurt by average months; they are hurt by deep drawdowns, which force selling at
the bottom or abandoning a plan. A rules-based, diversified portfolio that participates in growth while
keeping drawdowns shallow is therefore worth more than its raw return suggests — but only if its backtest is
honest. Most published backtests fail quietly through lookahead: covariance estimated with future data, regime
labels relabeled with hindsight, parameters tuned on the period they are evaluated on. This project treats
point-in-time discipline, pre-declared metrics and bootstrap inference as the deliverable, not as formalities,
and it is a learning project with real-money intent: every piece had to be understood before it was kept.

#### Research Question

> Can a diversified, rules-based, monthly-rebalanced ETF portfolio achieve better **risk-adjusted returns**
> (Sharpe, Sortino, maximum drawdown) than passively holding the S&P 500 — participating in long-term growth
> with lower volatility and controlled drawdowns?

Success was declared on risk-adjusted metrics before any data was touched. The project rules — universe,
architecture, seven inviolable rules and the step plan — are in [capstone_project_rules.md](capstone_project_rules.md).

#### Data Sources

- **Seven ETFs, daily prices adjusted for dividends, from 2000 to 2026-08-04** (Yahoo Finance): SPY (US
  equity), VEA (international developed equity), XLE (energy), XLU (utilities), IEF (7–10 year Treasuries),
  SHY (1–3 year Treasuries), GLD (gold). SPY, VEA, XLE, XLU, IEF and GLD are optimized; SHY is the anchor that
  receives the capital freed when the system reduces exposure.
- **Market indicators, watched but never traded:** VIX and VIX3M (Yahoo Finance), the 10-year minus 2-year
  Treasury spread (FRED) and a recession-probability series. Two columns planned in the rules file, the Cleveland
  Fed recession probability and Shiller's CAPE, were **never populated**: their automated download failed and they
  remain documented all-NaN placeholders in `data/prices_daily.parquet`. The recession series actually loaded is
  FRED's RECPROUSM156N (Chauvet–Piger smoothed probability, monthly), a stand-in the code labels as such. None
  of these indicators enters any model; the regime features use the ETF returns only.
- **Frozen snapshot.** `data/` holds daily prices, daily returns and weekly (Friday-to-Friday, simple) returns
  through 2026-08-07 (the last weekly row is a partial week that closes on Tuesday 2026-08-04, the last daily
  observation). Every notebook runs from it and reproduces the stored outputs exactly. The data switch
  `REFRESH_DATA` in [01_data.ipynb](notebooks/01_data.ipynb) is **off by default**; with `False` the notebook
  loads the snapshot, downloads nothing, writes nothing and still runs every validation. Set it to `True` only
  to rebuild the snapshot (needs internet); that overwrites `data/` and changes every downstream result.
- **Sanity checks are mandatory** after every correlation computation (rule 7): SPY–VEA high (~0.8), SPY–IEF
  near zero or negative, SHY near zero with everything. In this data SPY–VEA is 0.88 and SHY–IEF 0.78 (weekly, full sample; 0.76 on the common sample from 2007).

#### Methodology

Monthly cycle, each step using only data available at that month-end:

1. **Weekly returns, 3-year rolling window → sample covariance.** Weekly rather than daily is a design choice,
   consistent with the weekly regime features and the monthly cadence; it was not compared against
   daily-return covariance. SPY–VEA correlation is practically the same at both frequencies in this data (0.883
   daily, 0.880 weekly) because VEA is a US-listed ETF.
2. **Ledoit-Wolf shrinkage** of the covariance matrix — conceptually Ridge regression applied to a covariance
   estimate, with the shrinkage intensity self-calibrated from the data (between 0.03 and 0.19 in this sample,
   peaking at 0.186 in March 2020). An EWMA estimator (λ = 0.94 / 0.97 / 0.99) runs in parallel as a competitor.
3. **Weight optimization:** minimum variance with caps, and Risk Parity (equal risk contributions). Expected
   returns are never an optimizer input — the ACF and Ljung-Box diagnostics show return direction is
   economically unpredictable (largest autocorrelation ≈ 0.07 on average, 0.08 at most) while volatility
   clusters strongly, so the system estimates and tracks risk instead of forecasting returns.
4. **Momentum tilt** (12-1 month), capped at ±20% of base weight, applied *after* optimization.
5. **Regime detection** by K-means (k = 3) on three weekly market-condition features — realized volatility,
   trailing return and cross-asset correlation — with **asymmetric hysteresis** (enter STRESS on one signal,
   leave it only after two) and a **5-year rolling training window**. The label (CALM / NORMAL / STRESS)
   scales total exposure; the capital freed goes to SHY. The detector's evolution is documented in
   [06_regime.ipynb](notebooks/06_regime.ipynb): a first version with an expanding window and a symmetric rule
   ignored the February-2020 signal and flagged stress in only 3% of months; the 2008 crash permanently
   anchored its STRESS centroid, so the slow 2015-16 drawdown was missed. The rolling window was adopted after a
   dedicated backtest ([exp_regime_window_backtest.ipynb](notebooks/exp_regime_window_backtest.ipynb)) as a
   disclosed in-sample decision: Sharpe 0.775 → 0.809, bootstrap CI [−0.048, +0.094], a tie that leans
   favorable. Official STRESS is 14.1% of months (41% CALM, 45% NORMAL). Two disclosures: the change was adopted on
   Sharpe and on catching 2015-16, while the pre-declared objective metric moved marginally the other way (Calmar
   0.484 → 0.479 at an identical −13.0% drawdown); and the detector's first label is 2013-01 while the backtest
   starts 2010-08, so for the first 30 months every regime-scaled track runs at the fixed NORMAL exposure (80%,
   or 90% under the softer map) — the 2011 episode in the crisis table reflects the covariance and weighting
   layers only, not the detector.
6. **Final weights → monthly rebalance.**
7. **Validation and challengers**, all under rule 6 ("simple beats sophisticated until proven otherwise"):
   - **Walk-forward backtest** with 5 bps one-way transaction costs and crisis event studies
     ([08_backtest.ipynb](notebooks/08_backtest.ipynb)). The event studies cover the sample's six episodes, 2011
     to 2025; the 2000-02 and 2008 crises named in the step plan fall before the backtest window, which starts in
     2010-08 because VEA begins in 2007 and the covariance needs three years of weekly history.
   - **Moving-block bootstrap** on Sharpe differences (6-month blocks, 10,000 resamples, paired monthly
     returns), period splits and cost sensitivity ([09_validation.ipynb](notebooks/09_validation.ipynb)).
   - **Five pre-declared aggressiveness variants** tested through the same engine, Calmar as the pre-declared
     objective ([10_aggressiveness.ipynb](notebooks/10_aggressiveness.ipynb)).
   - **Hidden Markov Model** regime detector (3 states, filtered probabilities only — the smoothed posterior
     would leak future data) compared against K-means ([11_hmm_regime.ipynb](notebooks/11_hmm_regime.ipynb)).
   - **Supervised model comparison** ([12_supervised.ipynb](notebooks/12_supervised.ipynb)): logistic
     regression and random forest predicting whether next month is an official STRESS month from the
     detector's three features, with temporal cross-validation (`TimeSeriesSplit`, a one-month gap enforcing
     the project's label-lag rule), pre-declared grids searched with `GridSearchCV`, recall on STRESS as the
     primary metric, and persistence and always-CALM as the baselines to beat.

Methodological commitments: metrics, grids and decision rules are declared in markdown before the cells that
use them; parameters were economically pre-justified rather than grid-searched wherever a design choice was
made; and every notebook opens with its specification verbatim and explains non-obvious steps in short
markdown cells.

#### Results

**Headline.** Walk-forward backtest, 2010-08 to 2026-07, 5 bps one-way transaction costs
(`outputs/backtest_metrics.csv`, `outputs/aggressiveness_metrics.csv`):

| Track | CAGR % | Sharpe | MaxDD % | Calmar |
|---|---|---|---|---|
| MinVar-LW full chain | 7.3 | 0.924 | −16.6 | 0.439 |
| 60/40 (SPY/IEF) | 9.8 | 0.878 | −21.3 | 0.460 |
| RP-LW base (no tilt, no regime) | 7.4 | 0.838 | −16.1 | 0.460 |
| SPY | 14.7 | 0.810 | −33.7 | 0.437 |
| RP-LW full chain | 6.2 | 0.809 | −13.0 | 0.479 |
| **Final configuration (V-AB)** | **7.9** | **0.808** | **−14.7** | **0.535** |

Sharpe in this table and in the variant table below is computed on **daily** returns. The bootstrap uses **monthly** returns, as its specification requires, and on that definition levels and ordering differ: SPY 0.950 and 60/40 0.983 sit above MinVar-LW 0.935 and RP-LW full 0.838. Every verdict is unaffected because every bootstrap comparison is a tie, but the two numbers are not interchangeable and the README now says which one each table uses.

- **The drawdown objective was met.** Worst drawdown −14.7% against SPY's −33.7% and 60/40's −21.3%, and the
  highest Calmar of any track in the backtest and variant study (notebooks 08 and 10). One caveat for precision:
  the HMM-detector version of the same configuration, built in notebook 11 as a robustness check and not adopted
  (Sharpe tie), posts a marginally higher Calmar, 0.551 against 0.535.
- **Efficiency is a statistical tie.** In the moving-block bootstrap every Sharpe difference between the system,
  SPY and 60/40 has a 95% confidence interval that spans zero. The final configuration was tested directly against
  both benchmarks in notebook 10: −0.120 against 60/40 with CI [−0.47, +0.21], and −0.087 against SPY with CI
  [−0.46, +0.25].
- **Where the edge is:** the depth of the hole, not the speed of the climb. After de-risking the system
  participates less in rebounds and, in several episodes, regained its own prior peak no faster than SPY.
- **Why Risk Parity and not minimum variance.** MinVar-LW full has the highest daily-return Sharpe in the Step 8 table (0.924; on monthly returns it is third,
  behind 60/40 and SPY)
  but was not selected, for three reasons that follow from the declared objective. Its drawdown is deeper (−16.6%
  against −13.0% for the Risk-Parity chain) and its Calmar is the lowest of the system's tracks (0.439), and the
  objective is drawdown control, not Sharpe. Its weights are a concentrated bet rather than a diversified book:
  IEF sits at its 35% cap in every month of the sample, gold averages 26%, and VEA and XLE are nearly absent
  (5% and 2%), with a risk budget dominated by a few assets (notebook 05), so a repeat of 2022, when bonds fell
  with equities, meets no offset. And the Sharpe gap is a statistical tie: the bootstrap pair added to notebook 09
  at the final review gives +0.097 with a 95% CI of [−0.08, +0.28].

**Statistical validation** (`outputs/validation_summary.csv`; Sharpe on monthly returns, see the note under the
headline table):

| Pair | Sharpe difference | 95% CI | Verdict |
|---|---|---|---|
| 60/40 − RP-LW base | 0.101 | [−0.23, +0.44] | tie |
| 60/40 − RP-LW full | 0.145 | [−0.18, +0.48] | tie |
| RP-LW base − full (cost of the regime layer) | 0.044 | [−0.06, +0.16] | tie |
| RP-LW full − SPY | −0.112 | [−0.50, +0.26] | tie |
| RP-LW full − RP-RAW full (shrinkage) | 0.033 | [−0.03, +0.10] | tie |
| MinVar-LW full − RP-LW full (minimum variance vs Risk Parity) | 0.097 | [−0.08, +0.28] | tie |

The ranking does not hold across the two halves of the sample (60/40 leads in 2010–2017; SPY leads in
2018–2026), the system's Sharpe gap vs 60/40 reverses to positive in recent years, and the ranking is nearly
unchanged by transaction costs of 0, 5 and 15 bps (only adjacent swaps among mid-table tracks).

**Final configuration.** Five variants were declared before running to attack the measured shortfall in calm
months:

| Variant | CAGR % | Sharpe | MaxDD % | Calmar |
|---|---|---|---|---|
| V-A: IEF capped at 25% | 7.3 | 0.809 | −13.8 | 0.531 |
| V-B: softer exposure map 100/90/60 + 1-month stress exit | 6.6 | 0.811 | −14.0 | 0.476 |
| **V-AB: V-A + V-B (selected)** | **7.9** | **0.808** | **−14.7** | **0.535** |
| V-C: risk budget tilted to growth assets | 6.6 | 0.792 | −13.4 | 0.490 |
| V-D: hold 60/40 in calm months, defensive book otherwise | 6.3 | 0.736 | −21.4 | 0.295 |

V-AB won on Calmar, the pre-declared objective metric. Calmar was **not bootstrap-tested** (maximum drawdown is
a single order statistic of one historical path), so "best Calmar" is a descriptive fact of this sample. V-D
was rejected by the data: it puts the whole book on the detector at every transition, made 16 switches with
~1,600% lifetime turnover and the highest cost drag, and in 2022 it held 60/40 into the stock-and-bond selloff
(−21.1% drawdown).

**Challengers that did not earn their place.**

- **HMM vs K-means:** monthly agreement 64%; a backtest of the final configuration with each detector gives a
  Sharpe difference of +0.004, bootstrap CI [−0.04, +0.06] — a tie. The HMM stays as a robustness check and a
  monitoring view of soft probabilities. One asymmetry is stated in the notebook: the HMM keeps its original
  expanding window, while only the K-means side received the window revision.
- **Supervised comparison (notebook 12):** out-of-fold on 120 held-out months, neither tuned model beats the
  persistence baseline — logistic regression and random forest reach STRESS recall 0.80 against 0.70, but buy
  it with 28 and 23 false alarms against 6, and their balanced accuracy (0.76 and 0.79) falls below
  persistence's 0.82; both catch only two of the six episode onsets, so at a one-month horizon prediction
  collapses to description. No supervised model enters the pipeline; the frozen configuration is unchanged.

**Methodological honesty.**

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

##### Crisis episodes (detail)

Full chain vs SPY vs 60/40; total return / worst drawdown inside the episode. The 2011 episode predates the
detector's first label (2013-01), so it reflects covariance and weighting only:

| Episode | RP-LW full chain | SPY | 60/40 |
|---|---|---|---|
| 2011 (Jul–Oct) | +3.9% / −3.9% | −4.4% / −18.4% | +1.2% / −8.4% |
| 2015–16 (Aug–Feb) | −0.6% / −4.8% | −7.0% / −12.8% | −2.2% / −6.3% |
| 2018Q4 (Oct–Dec) | −3.2% / −4.9% | −13.5% / −19.2% | −6.7% / −10.5% |
| 2020 COVID (Feb–Aug) | +0.2% / −10.7% | +4.8% / −33.7% | +6.8% / −18.9% |
| 2022 (Jan–Oct) | −9.3% / −13.0% | −17.7% / −24.5% | −17.1% / −20.9% |
| 2025 (Apr episode) | +2.5% / −7.2% | +10.8% / −12.1% | +7.0% / −7.4% |

##### Verification against the partial submission

The partial submission (initial report and EDA, steps 1–5) fixed the question, the data, the exploratory
findings, the covariance estimators and the optimization baseline. This final submission keeps all of that
unchanged and adds steps 6–12. Re-checking the partial report against the final data, **five statements were
made more precise. No result changed**; these are corrections of wording:

| In the partial report | In this submission |
|---|---|
| "Weekly log returns" | Weekly **simple** returns (Friday-to-Friday price change); the code always used these |
| max \|acf\| ≈ 0.07 on returns, up to ≈ 0.26 on squared returns | 0.07 and 0.26 are **averages across assets** of each asset's largest autocorrelation; the maxima are 0.082 and 0.396 (XLU). The conclusion holds: 0.082² is under 1% of next-week variance |
| Weekly covariance chosen because daily returns understate SPY–VEA correlation (close-time lag) | Weekly is a **design choice**, not compared against daily. The lag effect is not present in this data: SPY–VEA correlation is 0.883 daily vs 0.880 weekly, because VEA is a US-listed ETF |
| SHY ≈ flat against all risk assets | SHY is **weakly** correlated with the risk assets (−0.20 to +0.22 on the full weekly sample; −0.23 to +0.25 on the common sample from 2007) and strongly with IEF (0.78 full sample, 0.76 common sample; both Treasuries) |
| SPY–IEF ≈ −0.22 | −0.22 on the full sample; −0.21 on the common sample from 2007 (the one used for clustering) |

What was re-verified and holds as stated: SPY–VEA 0.88 and SHY–IEF 0.78 (weekly, full sample; 0.76 on the common
sample); IEF's 3.6% annualized
adjusted return; 193 monthly covariance matrices; Ledoit-Wolf shrinkage between 0.03 and 0.19 with its peak
(0.186) in March 2020; the asset clusters at k=2 and k=3; the SPY–IEF correlation turning positive in the EWMA
estimate in June 2022 and in the rolling window in December 2022; and Risk Parity's IEF weight falling from 46%
(2021) to 38.6% (2023) and 33% today.

#### Next steps

- **Out-of-sample paper-tracking** of the frozen configuration. The sample contains no multi-year bear market,
  so the regime layer's core insurance value can only be validated by living through one; the configuration
  must not be re-tuned on the way.
- **Deferred upgrades, with the conditions to revisit them stated in advance:** a Kalman-filter covariance
  (deferred because the covariance ladder tied three ways) and an intramonth extraordinary-rebalance trigger
  (tested as a side exercise, a tie leaning slightly negative). Revisit only under a prolonged bear-market
  era, detector false positives in live tracking, or an expanded universe.
- **If the one-step-ahead question is reopened,** the lever is features that *lead* the detector — credit
  spreads, the VIX term structure already in the data — not a better classifier on the same three
  backward-looking features.
- **Monitoring, not allocation:** the HMM's weekly filtered probabilities and the supervised out-of-fold
  probabilities are useful communication views of how stressed the market looks; neither should change
  allocations without earning it in a backtest.

#### Outline of project

Run order: `01` (optional, safe by default) → `02` → … → `12`, then `exp_regime_window_backtest`. Each
notebook reads what the previous ones saved to `outputs/`. Reproduce with
`pip install -r requirements.txt`, then from `notebooks/`:
`jupyter nbconvert --to notebook --execute --inplace <notebook>.ipynb`.

- [01_data.ipynb](notebooks/01_data.ipynb) — download, cleaning and validation of dividend-adjusted prices; frozen-snapshot switch
- [02_exploration.ipynb](notebooks/02_exploration.ipynb) — ACF and Ljung-Box diagnostics; K-means clustering of the assets
- [03_covariance.ipynb](notebooks/03_covariance.ipynb) — rolling 3-year weekly covariance: sample and Ledoit-Wolf shrinkage
- [04_ewma.ipynb](notebooks/04_ewma.ipynb) — EWMA covariance variants (λ = 0.94 / 0.97 / 0.99)
- [05_optimization.ipynb](notebooks/05_optimization.ipynb) — minimum variance with caps vs Risk Parity
- [06_regime.ipynb](notebooks/06_regime.ipynb) — K-means regime detector, hysteresis, the VIX and training-window experiments, the adopted revision
- [07_pipeline.ipynb](notebooks/07_pipeline.ipynb) — momentum tilt and regime exposure: the full monthly pipeline
- [08_backtest.ipynb](notebooks/08_backtest.ipynb) — walk-forward backtest with costs and crisis event studies
- [09_validation.ipynb](notebooks/09_validation.ipynb) — moving-block bootstrap, period splits, cost sensitivity
- [10_aggressiveness.ipynb](notebooks/10_aggressiveness.ipynb) — five pre-declared variants; the final configuration
- [11_hmm_regime.ipynb](notebooks/11_hmm_regime.ipynb) — HMM regime detector built and compared against K-means; weekly detector race
- [12_supervised.ipynb](notebooks/12_supervised.ipynb) — supervised comparison: logistic regression vs random forest vs persistence, temporal CV and grid search; negative result
- [exp_regime_window_backtest.ipynb](notebooks/exp_regime_window_backtest.ipynb) — backtest evidence for the training-window revision

Supporting files:

- [capstone_project_rules.md](capstone_project_rules.md) — project rules: universe, architecture, inviolable rules, step plan
- [docs/project_report_full.md](docs/project_report_full.md) — full report: design, evidence, conclusions
- [docs/project_brief_facilitator.md](docs/project_brief_facilitator.md) — one-page brief
- `data/` — frozen data snapshot; `outputs/` — point-in-time outputs (the date column means "as computed at");
  `requirements.txt` — pinned package versions

##### Contact and Further Information

Javier Hernan Lopez. Questions and comments are welcome through the issues of this repository. The full
evidence for every step is in [docs/project_report_full.md](docs/project_report_full.md).
