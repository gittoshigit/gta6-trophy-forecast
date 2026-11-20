# DESIGN ADDENDUM v2: response to DIAGNOSIS_V2 (Q1–Q5)

Scope: this addendum changes only what the diagnosis implicates. Anything it doesn't mention keeps the v1 DESIGN.md / `entry/main.py` behaviour: preprocessing, Models C/F/T, growth scenarios, convolution, render rails, auto‑trophy handling and the fallbacks. Python 3.13 stdlib only, deterministic, ≤ 8 s per case.

**Priority rule.** Only f00 pays. f00 always runs the R2 path. Its history runs from launch to 2026‑11‑21T11:30Z and contains the 04:00–04:30Z first refresh, so R0 can never trigger on f00. That gives two classes of change:

- **f00‑neutral (R0 only):** fix these freely toward practice‑shaped behaviour. They cannot touch f00.
- **f00‑relevant (R2 calibration, time staging):** every choice goes toward f00 robustness, even where the simulator or practice would prefer something else. Each such trade is marked **[TRADE]**.

Numbers marked *(hand)* were worked out by hand from the formulas below. They were not produced by running code. The implementer must reproduce them in test A1.

---

## 0. Corrections to the digest (read first)

1. **v1's n = 0 mass on p00 is ≈ 0.73, not ≈ 0.60** *(hand)*. The calculation:
   - age a = 03:00:00 − 23:50:49 = 3.153 h, so `mean = (8 + a)/2 = 5.58 h`;
   - shape 2, S₂(x) = (1+x)e^(−x);
   - p0 = S₂(4.653·0.358)/S₂(3.153·0.358) = 0.504/0.689 = 0.73.

   The conclusion is unchanged, only stronger: the median sits inside the persistence atom.
2. **The practice‑shaped flood scenario already exists in `tools/sim.py`.** `generate(..., practice=True)` puts the first refresh at U(4.2, 4.45) h, the checkpoint at 190 min, the target at 280 min, and g = 1 + U(4.2, 5.8), with new‑cohort non‑auto trophies calibrated to 0.032 per arrival. It is exposed as `--practice-scenarios N`. Earlier logs show v1 there: cov90 0.18–0.28, cov50 0.02–0.04, loss 0.88–1.00. So diagnosis item 4 needs **no new practice generator**. It needs a **new final‑shaped regime‑shift generator** (Q5 §S2), because nothing in the sim currently tests what Q3 is meant to fix.
3. **A defect the digest missed, and the biggest f00 risk (fixed in Q4‑1).** `optional_fit = not depth and work <= OPTIONAL_FIT_WORK_LIMIT`, with work = states × 55 × 24 ≤ 52,800. That means **the backtest and the second kernel are silently disabled whenever f00 has more than 40 fitted states** (`compress_states` only merges above 80). If Exophase refreshes about hourly, f00 has about 60 states, and the whole §7 calibration never runs. Separately, any `TimeoutError` raised in the main C path sends the **entire case to `safe_fallback`** (persistence‑like output). v1's measured max wall in the sim is 5.1 s (6.7 s in an earlier build) against a 6.0 s guard.

---

## Q1. R0 refresh clock (f00‑neutral)

Applies **only when `len(states) == 1`**. `renewal()` is unchanged for n ≥ 1, so R1 and R2 are untouched. Add a new function `r0_clock(states, checkpoint, target)` and call it from `forecast_case` in place of `renewal` in the R0 branch.

### 1a. Interval model

- Inter‑refresh interval I ~ Gamma(shape k = 4, mean μ). CV = 0.5 gives "modest dispersion".
- Discrete hyperprior on μ (hours), meaning "a few hours":

  | μ | 1.5 | 2 | 3 | 4 | 6 |
  |---|---|---|---|---|---|
  | π(μ) | 0.15 | 0.25 | 0.30 | 0.20 | 0.10 |

  Constants: `R0_SHAPE = 4`, `R0_MU = (1.5, 2, 3, 4, 6)`, `R0_PI = (.15, .25, .30, .20, .10)`.

### 1b. How state age enters

The single state is the **launch snapshot**. When it began is left‑censored: Exophase's previous refresh happened at an unknown time before the first read. So we observe the process from a random time point, and the right survival function is the **equilibrium (forward‑recurrence) survival**:

```
S_e,μ(x) = Q(k+1, y) − (y/k)·Q(k, y),     y = k·x/μ,   Q = regularized upper gamma (existing gamma_survival)
```

This is the same expression v1 already uses as `1 − cdf` in the stationary‑age branch.

With a = checkpoint − cur.start (hours) and H = target − checkpoint:

```
D        = Σ_μ π(μ)·S_e,μ(a)                     # evidence of the flat stretch
p0_ren   = Σ_μ π(μ)·S_e,μ(a+H) / D               # proper censored update, no plug-in mean
π̃(μ)     = π(μ)·S_e,μ(a) / D                     # posterior over μ
```

This replaces v1's plug‑in `mean = (8+a)/2`. The plug‑in treated the censored age as evidence of long intervals and then used a shape‑2 hazard.

### 1c. Selection term **[TRADE: practice‑shaped, f00‑neutral]**

An honest renewal prior cannot make a 90‑minute window near‑certain to contain a refresh. On p00, p0_ren ≈ 0.43 *(hand)*. The extra information is in the brief: the practice cases were **built** to straddle the launch refresh ("the practice rewards a forecast that EXPECTS a refresh"). A fully flat history only happens on launch night. Encode that as a fixed selection weight:

```
R0_KEEP = 0.30
p0 = R0_KEEP · p0_ren                                    # p00 ≈ 0.13, p01 ≈ 0.12 (hand)
ψ  = Σ_μ π̃(μ)·[1 − Q(k, k·(H/2)/μ)]                      # P(second refresh | ≥1), first-at-midpoint approximation
p2 = (1 − p0)·ψ                                          # p00: ψ ≈ 0.014 → p2 ≈ 0.013 (hand)
p1 = 1 − p0 − p2
```

There are no n ≥ 3 terms: they are lumped into n = 2. This is safe because R0 is unreachable on f00. It would be wrong for a genuine flat history mid‑game, and no paying case has one.

Return value: `[(cur.t, 0, p0), (target, 1, p1), (target, 2, p2)]`. Refresh timing inside the window doesn't matter in R0, which has no progression term.

---

## Q2. Upside mass: κ right tail tied to earliness

### 2a. R0: replace the lognormal‑κ draw with a count model (`r0()` rewritten)

The real failure was not only a too‑low κ tail. In a 638‑player pool, per‑trophy outcomes are **small integer counts**: C_i is 4–13 and the new earners ΔC_i average about 1.9. The ΔC_i = 0 floor (ratio 1/g ≈ 0.165) and the Poisson‑like spread above it are what pushed about 82% of trophies outside [q05, q95]. So model ΔC_i directly.

**Scenario grid.** Full 9 × 9 cross of `normal_grid(9)` values z_g and z_κ, each pair with weight 1/81, independent:

```
ln g   = ln 5   + 0.55·z_g      (floor g ≥ 1.2)       # replaces N(ln 4, 0.6) INSIDE r0 ONLY; jump_prior() unchanged (R1 untouched)
ln κ_c = ln .08 + 0.60·z_κ
P'     = max(P+1, round(g·P));   ΔP = P' − P;   for n = 2 use g·1.5 instead of g
```

**Earliness weights.** These redistribute the expected new‑cohort trophies, so they don't inflate the total. Over J = non‑auto, non‑platinum trophies with r > 0:

```
r̃_i  = max(r_i, 50/P)                                   # a zero-count trophy can still be earned
u_i  = clip((ln r̃_i − mean_J ln r) / max(0.1, pstdev_J ln r), −2.5, 2.5)
w_g  = sqrt((A_g/τ_g) / (A_0/τ_0)) from PRIORS → (1.000, 0.548, 0.400, 0.095) for groups (secret bronze, bronze, silver/gold, platinum)
w_i  = w_g(i) · exp(0.5·u_i)                             # R0_GAMMA = 0.5
Z    = Σ_{non-auto j} r̃_j·w_j / Σ_{non-auto j} r̃_j       # normalisation: Σ_i m_i equals κ_c·ΔP·Σ r̃/100
m_i  = κ_c · ΔP · (r̃_i/100) · w_i / Z
```

The two earliness signals:

- **Rank of the current rate within the case.** In a launch pool, higher shares mean trophies reached sooner.
- **Tier/secret group.** Secret bronzes are story missions. The group factor reuses the A/τ ratios already in `PRIORS`, square‑rooted to shrink them.

The §4 earn‑time proxies (τ_i) can't be used: they need at least 4 refreshes.

**Count distribution.** ΔC_i ~ NegBin(mean m_i, size φ = 1.0), i.e. geometric (`R0_NB_SIZE = 1.0`).

- If m_i ≤ 60, enumerate the pmf exactly:
  - P(0) = (φ/(φ+m))^φ
  - P(j+1) = P(j)·(j+φ)/(j+1)·m/(φ+m)
  - stop at cumulative ≥ 1 − 10⁻⁴ or j = 400; renormalise.
- If m_i > 60, use 5 Gauss–Hermite nodes (`GH`) on ln ΔC with mean ln m − v/2 and sd √v, where v = ln(1 + 1/m + 1/φ).

Each atom gives logit(100·min(P', C_i + ΔC)/P') with weight p_n·(1/81)·pmf. The n = 0 term is one atom at logit(r_i) with weight p0. Take quantiles with the existing `quantiles()`.

- Auto trophies are unchanged (render pins them).
- Platinum gets w_g = 0.095, so it stays near the dilution floor. That is correct: 4/3868 was 0.10.
- Cost: 55 × 81 × 2 × ≤ ~40 atoms ≈ 0.4 M operations. Well under 0.5 s and deterministic.

**Expected shape on p00** *(hand)*.

| Trophy | Group | C | m at the scenario median | Ratio band (r'/r) |
|---|---|---|---|---|
| Typical, 0.78% | secret bronze | 5 | ≈ 1.9 | 0.20 at ΔC=0, ≈ 0.24 median, ≈ 0.44 at ΔC's 95th pct (g = 5) |
| id 20, 2.04% | bronze | 13 | ≈ 13 | ≈ 0.40 median, ≈ 0.75 at ΔC's 95th pct |

The g spread widens the typical trophy's band to about [0.10, 0.45]. id 20 can exceed 1 only in the extreme tail. With p0 ≈ 0.13 > 0.05, q95 sits on the persistence atom (r) for most trophies, q50 and q75 sit in the jump region, and q05 sits below the ΔC = 0 floor.

### 2b. R1/R2: no hard caps at r (cleared, no change)

| Path | Check | Verdict |
|---|---|---|
| Model F (`flow_fit`) | `a0 = clip(κ·r/100, 1e-6, .995)` is a prior centre, not a cap; κ̂ is clipped to [0.005, 8]; per‑arrival a ≤ 0.995 can exceed r/100; b adds progression | rates can rise |
| `flow_path` | decay ((h_c)/(h))^0.3 damps, never caps | clear |
| Model C | A ∈ [0.002, 1], legacy floor via max(); rising rates come from G(·) growth | clear |
| Model T | unbounded slope (damped φ = 0.7) | clear |
| D (R1 only) | dilution + progression floor, intentionally ≤ r‑ish; weight 0.2; never used on f00 | clear for f00 |
| `pool_draws` `not count` branch | R2 clock count = max(1, …) ≥ 1, so the atom at r never appears in R2 | clear |

No change to R1/R2 for Q2.

---

## Q3. Regime‑aware calibration for R2 (f00‑relevant)

### 3a. Backtest origins (`calibrate()`)

Let ratio = target/last in raw hours since launch (as in v1). For f00, last ≈ 59.5 h and target ≈ 168.2 h, so ratio ≈ 2.83.

| Origin | Cut (hours since launch) | f00 value | Weight |
|---|---|---|---|
| A | last/ratio | ≈ 21.0 h | 0.45 |
| B | last/ratio^0.5 | ≈ 35.4 h | 0.30 |
| C (new) | max(12, states[3].start + 1e‑6), run only if cut_C ≤ cut_A − 3 | ≈ 12–15 h | 0.25 |

- Each origin still needs at least 4 states with start ≤ cut.
- Weights are renormalised over the origins that actually ran.
- Origin C is the longest log‑ratio forecast available: about 12 h → 59 h, i.e. ln(65/18) ≈ 1.28 against f00's ln(174/65.5) ≈ 0.98. Its training window is dominated by the early high‑growth stretch, so it is the in‑case **regime‑transition stress test**.

**Not adopted [TRADE]: origins inside the flood window** (cut < 4 states, i.e. before about 12 h). Those child forecasts run R0/R1, a different code path, and are graded on a ×6 launch flood that cannot recur at f00's P ≈ 10⁴–10⁵ scale. Their large errors would drive s* to the top of the grid and over‑widen f00 for a regime f00 doesn't contain. The flood is still **inside the training data** of every origin, and C exposes the post‑flood transition.

### 3b. s* → s_cal mapping

```
SCALE_GRID   = (0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2, 2.7, 3.2)        # adds 2.7, 3.2; drops 0.7
L_o(s)       = pinball(pooled_o, actual, auto, s)                     # as v1, per origin
s*           = argmin_s Σ_o w_o·L_o(s) / Σ_o w_o
s_hi         = max_o argmin_s L_o(s)                                   # worst-origin requirement
DEFAULT_SCALE = 1.25                                                   # was 1.15
s_cal        = clip(0.50·s* + 0.25·s_hi + 0.25·DEFAULT_SCALE, 1.00, 2.40)   # was clip(.5 s* + .5·1.15, .8, 2.0)
no origin ran → s_cal = DEFAULT_SCALE
```

- The v1 ceiling of 2.0 could never bind: the old maximum was (2.2 + 1.15)/2 = 1.675.
- The new reachable maximum is 0.5·3.2 + 0.25·3.2 + 0.31 = 2.71, clipped to 2.4.
- The floor of 1.0 means f00 is never narrower than the model's native width.

**[TRADE]** Under a Gaussian truth, too wide by 1.5× costs about +6.3% pinball and too narrow by 1/1.5 about +4.7% *(hand)*. Under bias or fat tails (what practice showed), narrow loses far more. The s_hi term and the 1.0 floor buy f00 robustness at an expected cost of about +0.01 sim loss.

**Component weights.** Same formula as v1 (exp(−3·ΔL), 50% shrink toward defaults, C‑veto at 1.2× T), now pooled over the 3 origins.

### 3c. Residual defaults

| Parameter | v1 | v2 | Where |
|---|---|---|---|
| σ_jump clamp | [0.03, 0.30] | **[0.06, 0.30]** | `posterior()`: MAP residual differences understate per‑refresh dispersion because the MAP fit absorbs it. With n_eff = 4 on f00, the floor adds ≥ 0.12 logit sd |
| σ_jump default | 0.10 | 0.10 | unchanged |
| σ_mod clamp | [0.04, 0.30] | unchanged | — |
| DEFAULT_SCALE | 1.15 | **1.25** | also used by the depth‑0 pre‑calibration rows |

Not adopted: a backtest‑estimated common median shift (bias correction). It would be learned at 24–47 h horizons and extrapolated to 108 h, so its sign error risk is too high for a single paid case.

---

## Q4. Other code paths that could reproduce the pathology on f00

| # | Path | Risk on f00 | Action |
|---|---|---|---|
| 4‑1 | `OPTIONAL_FIT_WORK_LIMIT` (≤ 40 states) | backtest and second kernel silently off if the real cadence is ≤ ~1.5 h, so s_cal = default | **Fix**, below |
| 4‑2 | `check_time` (6.0 s) inside choose_kernels / posterior / pool_draws / convolve / summarize, outside any try | a timeout sends the whole case to `safe_fallback` ≈ persistence (sim: 0.69 vs 0.28) | **Fix**: staged pipeline, below |
| 4‑3 | `compress_states` limit 80 | runtime ∝ states² | **Fix**: limit is a parameter, 48 at depth 0, 30 at depth 1 |
| 4‑4 | Peer cap for all‑zero trophies (`render`) | only trophies with zero counts in every state; none expected at P ≥ 3,868 | Clear |
| 4‑5 | Platinum ≤ min of others at every level | logically required | Clear |
| 4‑6 | Width rail max 6 logit (R1/R2) | e^6 odds, not binding at s_cal ≤ 2.4 | Clear |
| 4‑7 | Anchoring | `ANCHOR = 0` | Clear |
| 4‑8 | Auto trophy pinned [99.95, 99.99, 100, 100, 100] | detection needs ≥ 99.9 in all states with P ≥ 1000, so a drifting id 2 is modelled normally | Clear |
| 4‑9 | R2 clock (`renewal`, stationary age, n_eff ≤ 4) | timing is second‑order over 108 h | Clear |
| 4‑10 | Growth level shock 0.25 / β sd hypot(se, 0.3) | the common dilution error is priced, but not post‑checkpoint shifts | Covered by Q3 floor/s_hi; validated by sim S2 |

**4‑1/4‑2/4‑3 fix: staged pipeline in `forecast_case` (depth 0, rich branch).** Each stage runs in `try/except Exception`. A failure keeps the last completed `rows`; it never falls through to `safe_fallback`. `TIME_SCALE = 1.0` is a module constant that multiplies every threshold (used by the slow‑host test).

| Stage | Work | Gate (elapsed since case start) | Result on success |
|---|---|---|---|
| S1 | futures; `pool_draws(models=[])`; `predict` with weights {F: 0.6, T: 0.4}, scale DEFAULT_SCALE | always | rows_ready |
| S2 | `compress_states(limit=48)`; `choose_kernels` (12 combos, n_tau = 24; if fitted states × 55 × 24 > 63,360 then n_tau = 16); `posterior`; `pool_draws` (C, F, T); `predict` with the C/F/T defaults | start if < 2.5·TS | rows |
| S3 | second kernel (v1 logic) | start if < 3.5·TS | rows |
| S4 | `calibrate` (A, B, C; child `compress_states(limit=30)`, n_tau = 16, kernel fixed) | start if < 4.0·TS; each origin starts only if < 5.0·TS | rows re‑summarised with the calibrated weights and s_cal |

- Hard guard: `check_time(start, 6.3·TS)`.
- `OPTIONAL_FIT_WORK_LIMIT` is deleted. Gating is the deterministic work estimate plus these time gates.
- Determinism: on a normal host every gate passes, so output is byte‑identical. **[TRADE]** On a slow host, which stages ran depends on wall time. That is accepted: one f00 run counts, and a degraded stage beats `safe_fallback`.
- stderr diagnostic per case: `calib origins=<k> s*=<..> s_hi=<..> s_cal=<..> stage=<S1..S4>`. Output bytes are unaffected.

---

## Q5. Acceptance tests (all must pass before submission)

All runs are **alone** (no parallel jobs), `taskset -c 0`. "v1" means the current `entry/main.py` on the **same sim build and seeds**.

### A. Practice inputs (kit p00, p01; R0 path)

| ID | Test | Threshold |
|---|---|---|
| A1 | `r0_clock` masses | p00: p0 ∈ [0.08, 0.20], p2 ≤ 0.05; p01 the same. Log p0_ren too; expect ≈ 0.43 for p00 (hand value; mismatch > 0.05 = formula bug) |
| A2 | Validity / determinism | `tools/run_practice.py` passes: valid, 55 rows, byte‑identical across 2 runs, wall < 1.0 s each |
| A3 | Low side covers the ΔC = 0 floor (1/6.07 = 0.165) | ≥ 90% of non‑auto, non‑platinum trophies have q05 ≤ 0.165·r |
| A4 | Body in the jump region | median over trophies of q50/r ∈ [0.18, 0.35] |
| A5 | Upside exists without inflation | median q75/r ∈ [0.22, 0.55]; ≥ 90% of trophies have q95 ≥ 0.40·r; ≥ 2 trophies have q95 > 1.02·r |
| A6 | Aggregate sanity vs the known first refresh (non‑auto Σ = 10.54 pts) | Σ over non‑auto q50 ∈ [7, 17] percentage points |

### B. Sim, practice‑shaped (`--practice-scenarios 24`, seeds 20261120 and 7)

| ID | Threshold |
|---|---|
| B1 | cov90 ≥ 0.80 (diagnosis requirement) |
| B2 | cov50 ≥ 0.30 |
| B3 | mean loss ≤ 0.65 **and** ≤ 0.75 × v1 |

### C. Sim, final‑shaped (`--scenarios 30`, seeds 20261120 and 7, i.e. 60 cases)

| ID | Threshold |
|---|---|
| C1 | mean loss ≤ v1 + 0.02 (the v1 reference run on 6 cases was 0.276) |
| C2 | cov90 ∈ [0.88, **0.98**] **[TRADE]**: the diagnosis asked for ≤ 0.96, but v1 already sits at 0.967 and v2 deliberately widens for f00 robustness |
| C3 | cov50 ∈ [0.45, 0.65] |
| C4 | wins vs fitted‑curve ≥ 85% of cases; worst loss ratio vs curve ≤ 1.30 |
| C5 | calibration engaged (stderr `origins ≥ 2`) in ≥ 95% of cases, **including every case with > 40 states** (v1 = 0% there) |
| C6 | 0 occurrences of "safe fallback" / "last-resort" in stderr |

### S. Simulator additions (`tools/sim.py`)

- **S1.** No new practice generator: use `--practice-scenarios` (see §0‑2).
- **S2.** `--shift-scenarios N`: final‑shaped cases with post‑checkpoint regime shifts. Draw shift parameters from a **separate** `random.Random(seed*1000003 + index)` so the existing streams stay byte‑identical.
  - (i) Arrival multiplier M = exp(U(−0.7, 0.7)) on `mass_curve` increments for h ≥ CHECKPOINT/60 + U(0, 36).
  - (ii) With probability 0.5, a bulk ingest: a step of U(0.2, 0.6)·P(checkpoint) players added to `cumulative` at a uniformly chosen refresh after the checkpoint. These players enter as an arrival batch at that age and progress through the normal completion curves.
  - (iii) Refresh cadence × U(0.5, 2.0) for t ≥ CHECKPOINT/60, applied in `build_refreshes`.
  - `cumulative` must stay non‑decreasing.
- **S3.** `--dense`: force refresh mean 1.0 h, shape 6 (≈ 60 pre‑checkpoint states).
- **S4.** New `experiment_command` variants:
  - `"slow-host": {"TIME_SCALE": 0.4}`
  - `"v1-scale": {"DEFAULT_SCALE": 1.15}` (ablation)

### D. Regime shift (`--shift-scenarios 24`, seed 20261120)

| ID | Threshold |
|---|---|
| D1 | v2 mean loss ≤ v1 mean loss |
| D2 | v2 cov90 ≥ 0.85 |

### E. Timing and robustness

| ID | Test | Threshold |
|---|---|---|
| E1 | `--dense --scenarios 6` alone, taskset ‑c 0 | max wall ≤ 5.5 s; calibration engaged in all 6 |
| E2 | `--scenarios 12 --variant slow-host` | 0 safe fallbacks; loss ≤ full‑v2 loss + 0.05 |
| E3 | Fuzz (v1 §10 suite: nulls, one read, empty history, purges, 0%/100% trophies, length mismatch) | 100% valid outputs |
| E4 | Determinism: 3 final‑shaped sim cases run twice at TIME_SCALE = 1 | byte‑identical |

**Ship rule.** A, B, C, D and E all pass. If C1 fails only because of s_cal, the only permitted adjustments, in this order, are: (1) the `s_hi` weight 0.25 → 0.15 (the s* weight rises to 0.60); then (2) DEFAULT_SCALE 1.25 → 1.20. Keep the 1.0 floor and the σ_jump floor 0.06.

---

## Change summary for the implementer

| Location | Change |
|---|---|
| constants | `DEFAULT_SCALE = 1.25`; add `SCALE_GRID`, `SCALE_CLAMP = (1.0, 2.4)`, `JUMP_FLOOR = .06`, `TIME_SCALE = 1.0`, `R0_SHAPE`, `R0_MU`, `R0_PI`, `R0_KEEP = .30`, `R0_LOG_G = (ln 5, .55)`, `R0_LOG_KAPPA = (ln .08, .60)`, `R0_GAMMA = .5`, `R0_NB_SIZE = 1.0`; delete `OPTIONAL_FIT_WORK_LIMIT` |
| `r0_clock` (new) | Q1 (equilibrium survival, μ‑mixture, selection term, n ∈ {0, 1, 2}) |
| `r0` | Q2a count model (replaces lognormal κ draws) |
| `renewal`, `jump_prior`, `scenarios`, Models C/F/T, `convolve`, `render` | **unchanged** |
| `posterior` | σ_jump clamp lower bound 0.03 → 0.06 |
| `compress_states` | `limit` parameter (48 at depth 0, 30 at depth 1) |
| `calibrate` | origins A/B/C with weights 0.45/0.30/0.25, extended grid, s_hi, new mapping and clamp, per‑origin time gate |
| `forecast_case` | staged S1–S4 with per‑stage try/except, work‑based n_tau, stderr diagnostic |
| `tools/sim.py` | `--shift-scenarios`, `--dense`, `slow-host` / `v1-scale` variants, report the calibration‑engaged fraction |
