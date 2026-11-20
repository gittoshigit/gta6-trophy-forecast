# DESIGN.md: GTA VI trophy rarity forecaster (`entry/main.py`)

Bounty: `gta6-trophy-rarity-forecast-01`. Runtime: Python 3.13, standard library only, deterministic, no network, 8 s / 1 CPU / 1 GB per case.
Only the final case decides the prize. It has one case, f00: checkpoint 2026‑11‑21T11:30Z, target 2026‑11‑26T00:00Z, horizon 6,510 min. The practice cases p00 and p01 (90 min, straddling the first refresh) are a check that the output is valid and that the program copes when no refresh has been seen yet. They are not a tuning target.

---

## 0. Facts from the data that drive the design

1. **Rates are integer counts divided by players tracked.** In the frozen launch‑night read (P = 638), every rate is k/638: 0.63 = 4/638, 0.78 = 5, 0.94 = 6, 1.10 = 7, 1.25 = 8, 1.57 = 10, 2.04 = 13, and 100.00 = 638. Σ rates = 147.78 = 943/638 × 100, where 943 is `trophies_earned`.
   - BRIEF.md says "~144.8". The actual figure is 147.78.
   - So `trophies_earned` E = Σ_i C_i (the earner counts), and **C_i = r_i·P/100 can be recovered**. It is exact while P < 20,000 (the 2‑decimal rounding is worth < 0.5 player) and accurate to about ±P/20,000 above that.
2. **Trophy id 2 ("Just Like Paradise") is at 100% and stays there.** P behaves like "tracked players holding ≥1 GTA VI trophy", and id 2 is the trophy almost everyone gets first.
   - First refresh: P 638 → 3,868 and E 943 → 4,276. If id 2 stays at 3,868 earners, the other 54 trophies went from 305 to 408 earners. So the **3,230 new players brought only ~103 non‑auto trophies** (0.032 per player, against 0.478 per player in the launch pool).
   - The rate ratio r'/r ≈ 1/g + κ(1−1/g), with g = 6.06 and κ ≈ 0.067, gives ≈ 0.22, a logit drop of ≈ 1.5. This matches the official persistence loss of 1.51.
3. **The launch pool is anomalous.** 4 of 638 players hold the platinum on launch night. Treat this as a fixed legacy cohort whose counts are a floor that gets diluted as P grows. It matters for the rarest trophies, e.g. platinum: 4/100k = 0.004%, which clamps to 0.01.
4. **The page is piecewise constant.** The final history mixes a 30‑min read cadence (launch night) with a 10‑min cadence (in‑cluster recorder), so refresh times are known to within ±5 min for most of the history.
5. **What the answer actually is:** the page state at the last Exophase refresh before roughly T (the first read in [T, T+60 min)). With a horizon of ~108 h against refresh intervals of a few hours, **refresh timing is a small second‑order effect in the final. The main problem is extrapolating players‑tracked growth and per‑trophy progression from day 2.5 to day 7.**
6. **The bar:** the house LLM forecast is made at 09:30Z with nearly the same data. Its form is a median plus a symmetric logit spread. We beat it with structure (counts, dilution, cohort progression, a calibrated asymmetric spread), not by matching it.

---

## 1. Program structure

```
main():
  try:   out = forecast_case(case)                 # full model
  except: out = safe_fallback(case)                # persistence + widened spread (never fails)
  validate_and_repair(out)  -> write <id>.out
```

- `forecast_case(case, depth=0)` is pure. It is called again on a truncated history for the in‑case backtest (§7), with `depth=1`. At depth 1 there is no nested backtest and default weights are used.
- **Time guard:** read `time.monotonic()` at start. If more than 5.0 s have elapsed, skip the backtest and the second (k, a0) combination. If more than 6.5 s, emit what is ready, or `safe_fallback`.
- **Determinism:** no `hash()` (PYTHONHASHSEED). The seed is `int(hashlib.sha256(case_id).hexdigest()[:8],16)`, feeding `random.Random(seed)`. Better still, use stratified draws (a fixed grid of normal quantiles), so seeds barely matter.
- **Output:** 4 decimals, half‑up rounding as in the starter's `round4`. After rounding, re‑run the monotone fix, clip to [0, 100], check every value is finite, keep the ids in order and one row per trophy, and stay under 65,536 bytes (~5 KB in practice).

---

## 2. Preprocessing: from reads to states

1. Parse the reads and sort them by `at` (they should already be sorted).
2. **Fill nulls.** A null rate means that read lacked the trophy. Carry forward the last non‑null rate of the same trophy. If the read has no earlier non‑null, carry backward. If the trophy is null everywhere, mark it `unknown` (handled in §9).
3. **Build state signatures** of (P, E, rates tuple). Consecutive reads with the same signature are one **state**. A change in any component starts a new state (a **refresh**).
   - Refresh time estimate: t̂_k = midpoint of (last read of old state, first read of new state). Bracket half‑width w_k = half that gap.
   - A state that appears for a single read and then reverts to the previous signature is treated as a glitch. Drop it if it is the only such occurrence. Otherwise keep it, since Exophase can genuinely flip.
4. **Counts:** C_{k,i} = r_{k,i}·P_k/100, kept as float (do not round when P ≥ 20,000).
   - Diagnostic: |Σ_i C_{k,i} − E_k| ≤ 0.5 + 0.0001·P_k. If this fails, the state still counts, but its observation weight is halved (§5).
5. **Purges.** If P_k < P_{k−1}, mark the state `purge`.
   - For growth fitting, set ΔP = 0 and add the deficit to a shrinking term (§3).
   - Rates are taken as observed. Per‑trophy flows across that interval are excluded from the flow model.
6. Outputs of this step:
   - the state list S = [(t̂_k, w_k, P_k, E_k, r_{k,·}, C_{k,·})];
   - first‑read time t_first, last‑read time t_last, and the current state's start time t_cur (its first read; censored: it began no later than that read and no earlier than the previous state's last read);
   - the launch anchor L = t_first (2026‑11‑18T23:50Z in every real case). Time variable: h(t) = hours since L, plus h0 = 6 h for log transforms.

---

## 3. Refresh process and players‑tracked growth (Q1, Q3)

### 3a. Refresh timing: a renewal process

- Inter‑refresh intervals I_j: the differences of successive t̂_k. Also keep the censored current age a_cur = t_checkpoint − t_cur (lower bound).
- **Prior:** gamma intervals with mean μ0 = 4 h and shape k0 = 2.
  - Launch night shows one interval of at least 4.2 h (23:48Z → 04:00–04:30Z), then a quick unchanged 05:00Z read.
- **Estimation:**
  - μ̂ = (n0·μ0 + Σ I_j + a_cur) / (n0 + n_I), with n0 = 2 pseudo‑intervals. The censored current age adds exposure but no event.
  - If n_I ≥ 4: shape k̂ = clamp(1/CV², 1, 8) using the recency‑weighted interval CV, shrunk as k = (2·k0 + n_I·k̂)/(2 + n_I).
  - If n_I ≥ 6 and CV < 0.25, the process is near‑periodic. Then also estimate the phase and period by least squares on the refresh times, and use a "periodic + jitter" age distribution (below).
- **Age of the answering state.** Let A = T* − t_lastrefresh(T*), with T* = T + 0 (the 00:00 slot is the answering read in practice).
  - Final case: the horizon is far beyond many μ, so use the stationary renewal age. The density is S_I(a)/μ, where S_I is the gamma survival function. Sample it by inverse CDF on a 64‑point grid.
  - Periodic case: A = (T* − phase) mod period ± N(0, jitter).
  - Effective target time: **T_eff = T* − A**. Use 7 stratified A‑quantiles (0.07, 0.21, …, 0.93) in the scenario draws (§6).
- **Short horizons** (practice; anything with H < 3·μ):
  - p_ref = probability of ≥1 refresh in (t_checkpoint, T*], given the current state already has age ≥ a_cur: p_ref = 1 − S_I(a_cur + H)/S_I(a_cur).
  - Also compute the conditional distribution of the number of refreshes. Truncate at 3; it hardly matters at H = 90 min.
  - Worked p00 example (μ = 4 h, k = 2, a_cur ≥ 3.17 h, H = 1.5 h): p_ref ≈ 0.39. p01 is somewhat higher.
  - These priors are not tuned to make p_ref ≈ 1. Practice pays nothing, and an overconfident refresh prior would hurt in the final's thin edge cases.

### 3b. Players‑tracked growth

Arrival intensity per hour over each refresh interval: λ_k = ΔP_k / (t̂_k − t̂_{k−1}). Use only intervals after the first state, and exclude purges.

**Model:** ln λ(t) = α + β·ln h(t) (power law in time since launch). A pool that matures shows β < 0. An early ramp can show β > 0.

- **Fit:** weighted least squares. Weight = interval length × recency weight exp(−(t_last − t)/τ_rec), with τ_rec = 36 h. This must be ≥ 24 h so the diurnal cycle averages out.
- **Prior:** β ~ N(−0.6, 0.5²), as a ridge penalty equal to 3 pseudo‑observations. Clamp β to [−2.5, 1.0].
- **Level anchor:** after fitting β, re‑solve α so the model's predicted ΔP over the last 24 h of history equals the observed ΔP. This anchors the level where it matters most.
- **Forecast:** P(t) = P_cur + ∫_{t_cur}^{t} λ(u) du in closed form: for β ≠ −1 it is e^α·(h^{β+1}−h_c^{β+1})/(β+1), and the log form at β = −1.
  - Discretize future arrivals into 12 log‑spaced chunks between t_cur and T_eff for the cohort model.
- **Uncertainty:** draw β^(s) ~ N(β̂, se_β² + 0.3²) and a level shock ln α‑shift ~ N(0, 0.25²). The 0.25 covers the weekend of Nov 21–22, a possible US Thanksgiving‑eve effect and model error.
  - Use stratified draws, correlated across all trophies within a scenario. The scoring is marginal per item, so this correlation only matters through the marginal spreads, which is correct.
- **Purges:** a net negative ΔP is kept as a separate rate of "purge per hour" (default 0). It is applied in expectation only if purges are seen more than once.

### 3c. History too thin for growth fitting

Fewer than 2 positive‑ΔP intervals (practice: 0).

- Jump prior for the next refresh, on its growth factor g = P'/P: ln g ~ N(ln 4, 0.6²) when the pool is small (P < 5,000) and the game is under 24 h old. Otherwise ln g ~ N(ln(1 + 2·μ/(h+h0)), 0.5²), which is a generic power‑law decay at β = −1 per interval.
- With one observed jump g1: ln g ~ N(ln g1 + β_prior·ln(h_next/h_prev), 0.5²).

---

## 4. Per‑trophy dynamics across refreshes (Q2, Q4)

A refresh's implied marginal earn rate is q_{k,i} = ΔC_{k,i}/ΔP_k. It mixes new‑cohort dilution with old‑cohort progression; once growth slows it can even exceed 1. So q is **not** modelled directly in rich histories. Instead a **cohort‑convolution model** separates the two effects with few parameters. A simpler flow model and a trend model act as companions and as fallbacks.

### 4a. Model C: cohort convolution (primary when ≥ 4 refreshes)

The tracked population is:
- the **legacy cohort**: the first state, P0 players with counts c0_i;
- plus **arrival batches**: ΔP_s arriving at times t_s (refresh midpoints; arrivals are assumed spread over the preceding interval, approximated by the midpoint of that interval).

A tracked player who arrived at s has earned trophy i by time t with probability

```
F_i(t - s) = A_i · G( (t - s + a0) / τ_i ),     G(x) = 1 − exp(−x^k)    (Weibull ramp)
```

- A_i ∈ [0, 1]: the eventual share of players who earn trophy i.
- τ_i: its time scale in hours of calendar age since tracking.
- k: global shape. a0: global head start (players are often a little way in when Exophase first tracks them).

Then:

```
C_i(t) = max(c0_i, P0·A_i·G((t − L + a0)/τ_i))  +  A_i · Σ_s ΔP_s · G((t − t_s + a0)/τ_i)
r_i(t) = 100 · C_i(t) / P(t)
```

Given (k, a0, τ_i), C_i is linear in A_i apart from the max on the tiny legacy term. Define K_τ(t) = Σ_s ΔP_s G(·) for the arrival part. It is shared by all trophies and **precomputed once per (k, a0, τ)** over the state times.

**Grids:**
- k ∈ {1, 1.5, 2, 3}
- a0 ∈ {0, 3, 12} h
- τ: 24 log‑spaced values from 0.5 h to 3,000 h
- A: 25 values, logit‑spaced from 0.002 to 0.998, plus 1.0

**Fitting:**
1. **Global (k, a0).** For each of the 12 combinations and each trophy, profile A_i in closed form per τ (weighted least squares in rate space, clipped to [0, 1]) and take the minimum over τ. Sum the penalized objective (defined below) over the 30 trophies with the largest counts, which carry the most information. Keep the best two combinations, with weights ∝ exp(−½·ΔSSE_pooled / σ̄²) tempered by 1/4, then normalized.
2. **Per‑trophy posterior** on the (A, τ) grid, for each kept combination:
   - log w = −½ Σ_k ω_k (logit r̂_{k,i} − logit r_{k,i}(A, τ))² / σ_i² − ½ penalty
   - Observation weight ω_k: state duration factor min(1, dur/1 h) × recency exp(−(t_last − t_k)/72 h), halved for states that fail the count check.
   - σ_i² = σ_round,i² + σ_mod². σ_round = 0.0029/(r̂(1−r̂/100)) in logit units (rounding to 0.01 points). σ_mod = 0.08, then re‑estimated as the pooled residual sd of the MAP fits, clamped to [0.04, 0.3].
   - Penalty: (logit A − m_A,g)²/1.5² + (ln τ − m_τ,g)²/1.2². Group g is set by tier and secret flag:

     | group | A prior median | τ prior median |
     |---|---|---|
     | secret bronze (story missions) | 0.6 | 72 h |
     | non‑secret bronze | 0.3 | 120 h |
     | silver / gold | 0.2 | 150 h |
     | platinum | 0.03 | 400 h |

   - **Empirical‑Bayes update of the group medians:** after a first pass, set m_A,g and m_τ,g to the weighted mean of the group's posterior means, shrunk 50% toward the table above. Then run a second pass.
3. **Forecast** for each grid point with posterior weight > 1e‑4 (normally 20–80 points per trophy):
   - r_i(T_eff) = 100·[legacy_i(T_eff) + A·(K_τ(t_cur) + K_τ^future(T_eff))]/P^(s)(T_eff), for each growth/age scenario s.
   - Point mixture: each (scenario, grid point) pair carries weight w_s·w_grid.

**Q4, progression vs dilution.** Old‑cohort progression enters through G( ) continuing to rise for past arrivals. Dilution enters because recent arrivals sit on the steep, low part of G. Nothing needs to be estimated "per refresh". The legacy floor c0_i covers the launch‑night anomaly, including the 4 platinum holders. A separate per‑trophy saturating progression model on top would not be identifiable from aggregate counts, so **do not add one**. Its job is done by (A_i, τ_i, k, a0).

**Cost:**
- Precomputing K: 4·3·24 grid combinations × S² (S ≤ ~80 states) ≈ 1.8 M exp calls. That is about 1 s and can be cut by caching G per arrival‑state pair.
- Posterior: 55 trophies × 2 combinations × 650 grid points × S ≈ 5.7 M multiply‑adds, about 2 s.
- If S > 80, merge adjacent states older than 24 h pairwise. They carry little information individually.

### 4b. Model F: flow / marginal earn rate (primary for 1–3 refreshes; a companion otherwise)

Per trophy, regress counts over refresh intervals with non‑negative coefficients:

```
ΔC_{k,i} = a_i·ΔP_k + b_i·(P_{k−1} − C_{k−1,i})·Δt_k
```

- a_i: new‑cohort earn rate per arrival.
- b_i: hazard per hour for already‑tracked non‑earners.
- Use the recency weights above. Exclude purge intervals and the legacy‑only first state.
- With 1 refresh only a_i is identified, so b_i comes from the prior (see the priors list).

Priors (ridge on logit a and log b):
- logit a_i ~ N(logit(κ·r_i/100), 1.0²) with κ ~ 0.1. From the first refresh, κ ≈ 0.067, but most of that refresh's arrivals were minutes old. After the first refresh, re‑estimate a pooled κ̂ = Σ_i ΔC_i / Σ_i (r_i/100·ΔP) excluding auto trophies, and use κ̂ as the prior centre.
- b_i: log b ~ N(ln(0.002/h · r_i/r_med), 1.0²). This is a weak progression prior.

Forecast:
- C_i(T_eff) = C_i,cur + a_i·ΔP^(s) + b_i·∫(P − C_i)dt.
- The integral uses the scenario growth path, with a_i and b_i decayed by a factor (h_cur/h)^0.3 so that extrapolation over 4.5 days cannot run away.
- Then r_i = 100·C_i/P^(s), capped at 100.

### 4c. Model T: damped log‑time trend (companion; house‑curve‑like but on states)

- Per trophy, weighted least squares of logit r̂_{k,i} on ln h(t̂_k) over the states of the last 36 h. Include the state current at the start of that window.
- Damped slope: b_damp = φ·b with φ = 0.7.
- Forecast: logit r_i(T_eff) = logit r_cur + b_damp·(ln h(T_eff) − ln h(t_cur)).
- Uncertainty: residual sd ⊕ se(b)·Δln h ⊕ 0.3·|b|·Δln h.

### 4d. Model 0: no refresh observed (practice; single‑state histories)

Build a mixture over the number of refreshes n:
- n = 0 with probability 1 − p_ref (§3a): a point mass at r_cur.
- n ≥ 1 with probability p_ref. For n = 1 (n = 2 adds a second compounding step, with weight from the renewal count distribution):
  - r' = 100·(C + q·(P'−P))/P', where P' = g·P (g from §3c).
  - New‑cohort rate q_i = κ·r_i/100 with ln κ ~ N(ln 0.1, 0.8²), drawn per trophy around a common κ draw (common‑draw sd 0.6, idiosyncratic 0.5).
  - Auto trophies (§9): q = 1.
  - The common κ and g draws are shared across trophies within a scenario.

This expects a jump only through p_ref. With p_ref ≈ 0.4, the q05 and q25 quantiles move into the jump region (rates ~0.2–0.35× their current value), and the median stays near persistence. The expected practice loss is well below 1.51, though not near 0. That is acceptable.

---

## 5. Model regimes and blending

| Regime | Condition (on state list) | Components (default weights) |
|---|---|---|
| R0 | 0 refreshes | Model 0 only |
| R1 | 1–3 refreshes, or history span < 12 h | F 0.6, T 0.2, persistence‑with‑dilution 0.2 |
| R2 | ≥ 4 refreshes and span ≥ 12 h | C 0.5, F 0.3, T 0.2 |

- "Persistence‑with‑dilution" means r_cur scaled by a pure dilution P_cur/P^(s)(T_eff), plus the progression floor.
- **Blending is a linear pool:** allocate scenario mass to components in proportion to their weights, then take quantiles of the pooled sample. Pooling widens the spread where the models disagree, which is the desired behaviour.
- In R2 the weights come from the in‑case backtest (§7): w_m ∝ default_m·exp(−3·(L_m − min L)), shrunk 50% toward the defaults.
- **Aggregate anchoring (R1/R2):** Σ_i r_i = 100·E/P, so trophies per player m(t) = E/P is a low‑noise series.
  1. Forecast m(T_eff) with Model T applied to ln m.
  2. Per scenario, compute the factor f = m_forecast / Σ_i r_i^(s) over the non‑auto trophies.
  3. Multiply every non‑auto trophy's odds by f^0.5: logit += 0.5·ln f.

  This shrinks the common progression/dilution error toward the aggregate trend. Ship it only if the synthetic test (§10) shows a gain; it is controlled by the flag `ANCHOR=0.5`.

---

## 6. Quantile construction on the logit scale (Q3, Q5)

For each trophy:

1. **Draws.** Build N ≈ 2,000 weighted draws of logit r_i(T_eff) from:
   - 7 age‑quantile draws of A (refresh timing; §3a);
   - 9 stratified growth draws (β, level shock; §3b);
   - posterior grid points (Model C), or 15 stratified parameter draws for F and T (parameter uncertainty);
   - the component mixture (§5).
2. **Residual noise.** Convolve each draw with Gaussian noise on the logit scale, of sd

   σ_res,i = s_cal · sqrt( σ_mod² + σ_round,i(T)² + σ_jump² · n_eff )

   - σ_round at the forecast rate covers the 2‑decimal rounding of the answer.
   - σ_jump = 0.10 per refresh is the idiosyncratic per‑refresh dispersion. Estimate it from the cross‑trophy residuals of refresh‑to‑refresh logit changes after removing Model C's prediction, clamped to [0.03, 0.3].
   - n_eff = min(number of future refreshes, 4). Idiosyncratic jumps partly average out within the level model.
   - Do the convolution analytically: for each draw, put mass at 5 Gauss–Hermite nodes.
3. **Quantiles.** Sort the weighted draws and read the weighted quantiles at τ ∈ {0.05, 0.25, 0.5, 0.75, 0.95}, interpolating linearly in logit.
4. **Calibration scale s_cal.** Default 1.15 (the horizon is long and the models are misspecified). In R2, from the backtest: s* = argmin over {0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8, 2.2} of the backtest pinball, after re‑scaling the predictive deviations about each item's median. Then s_cal = clamp(0.5·s* + 0.5·1.15, 0.8, 2.0).
5. **Width rails** (logit units, q95 − q05):
   - At least 0.25. This guards against rounding and the 0.01 clamp.
   - At most 6.0, except in R0, where the bimodal mixture can be wider.
6. **Logical rails, applied to the quantile vectors:**
   - **Platinum:** q_plat,τ ≤ min_{i ≠ plat, non‑auto} q_i,τ at every level, since the platinum requires every other trophy.
   - **Auto trophies:** see §9.
   - **No rate below the legacy floor:** q_i,τ ≥ 100·c0_i/P_max,τ is not enforced. The floor is already inside the model, and the 0.01 clamp makes this moot.
7. **Output:** convert back with inv_logit × 100, clip to [0, 100], enforce non‑decreasing order, round to 4 decimals.
   - Values below 0.01 cost nothing extra, because the score clamps them. Output them clipped to ≥ 0.005 for readability.

**Shrinkage toward persistence/trend where data is thin** comes from three places:
- the priors in C and F (group medians, A/τ penalties, κ, b);
- the regime table;
- the damping in T and F.

In addition, any trophy whose effective sample is small (max_k C_{k,i} < 5) gets its C/F medians blended 30% toward Model T (fixed weights, no backtest) and its σ_mod doubled.

---

## 7. In‑case matched‑ratio backtest (calibration at runtime, R2 only)

The final forecast covers h_cur ≈ 59.7 h → h(T_eff) ≈ 168 h. That is a log‑time ratio ρ = h(T_eff)/h(t_last) ≈ 2.8.

1. **Backtest A.** Cut the history at t_b = L + (t_last − L)/ρ (≈ 21 h, i.e. the evening of Nov 19). Forecast the state at t_last from the truncated history with `forecast_case(depth=1)`, taking the target as t_last.
2. **Backtest B.** The same with ratio √ρ, for a shorter forecast with more training data.
3. **Requirement.** Each cut must keep ≥ 3 refreshes; otherwise skip that backtest.
4. **Score.** The exact contest pinball on the observed state, over all non‑auto trophies, per component and for the pooled model. Average A and B with weights 0.6 / 0.4.
5. **Use:** the component weights (§5) and s_cal (§6.4).
   - A **sanity veto:** if the pooled model's backtest loss exceeds Model T's backtest loss by more than 20%, set the C weight to 0.2 and redistribute the remainder to T and F. This guards against Model C misbehaving on real data.

The backtest runs the depth‑1 path with a reduced grid (k fixed at the depth‑0 choice, τ grid of 16), so it costs about 1.5 s.

---

## 8. Parameters: initial values and how each is set at runtime

| Parameter | Initial / prior | Runtime estimation |
|---|---|---|
| μ_int (mean refresh interval) | 4 h, 2 pseudo‑intervals | Pooled with observed intervals plus censored current age (§3a) |
| k_int (interval shape) | 2 | 1/CV² of intervals, shrunk, clamp [1, 8], when n_I ≥ 4 |
| period / phase | none | LS on refresh times when n_I ≥ 6 and CV < 0.25 |
| β (arrival power) | N(−0.6, 0.5²) | Weighted LS of ln λ on ln h, ridge 3 pseudo‑obs, clamp [−2.5, 1] |
| α (arrival level) | n/a | Matched to last‑24 h ΔP |
| level shock sd | 0.25 | Fixed (scenario spread) |
| β extra sd | 0.3 | Fixed (plus se_β) |
| g prior (no growth data) | ln g ~ N(ln 4, 0.6²) | Updated by an observed jump (§3c) |
| κ (new‑cohort earn multiplier) | median 0.1, ln sd 0.8 | Pooled from observed refreshes (§4b) |
| k (Weibull shape) | grid {1, 1.5, 2, 3} | Pooled profile likelihood, best 2 kept |
| a0 (head start) | grid {0, 3, 12} h | Same |
| A_i, τ_i | group priors (§4a) | Grid posterior per trophy; group medians by empirical Bayes |
| b_i (old‑cohort hazard) | 0.002/h × r_i/r_med | NNLS in Model F |
| σ_mod | 0.08 | Pooled residual sd, clamp [0.04, 0.3] |
| σ_jump | 0.10 | Cross‑trophy refresh residuals, clamp [0.03, 0.3] |
| φ (trend damping) | 0.7 | Fixed (validated in §10) |
| τ_rec (recency) | 36 h for growth, 72 h for C/F | Fixed |
| component weights | per regime (§5) | Backtest, 50% shrink |
| s_cal | 1.15 | Backtest, 50% shrink, clamp [0.8, 2] |
| ANCHOR | 0.5 | Fixed; switched off if §10 shows no gain |

---

## 9. Edge cases (Q6)

- **Auto trophy (rate 100.00).** Detection: r = 100.00 in the current state and ≥ 99.90 in all states with P ≥ 1,000; in the practice case, simply r = 100 in all states.
  - Output [99.95, 99.99, 100, 100, 100]. The score clamps at 99.99 (logit 9.21), so q25 and above score 0 when the answer is 100.00.
  - If the current value is 100 but an earlier state with P ≥ 1,000 was below 99.9, model the trophy normally on the 1 − r side. (The logit is symmetric, so all models apply unchanged.)
  - In Model 0 and Model F the auto trophy's q is 1.
- **Rates at the 0.01 clamp or 0.00.**
  - Counts handle these naturally: C = 0 or 1.
  - Forecasts below 0.01 clip harmlessly.
  - Keep q95 ≥ 0.01, plus the minimum logit width of §6.5 applied about the median, so the upside where a trophy starts being earned is covered.
  - A trophy at 0.00 with C = 0 in all states: Model C's posterior is driven by the prior, which is good. Additionally cap its median at the 30th percentile of the group's current rates.
- **Null rates.** Carry forward or backward (§2). If the current state has a null for trophy i, use the last observed value and widen σ_mod for that trophy by ×1.5. If the trophy is null everywhere, output the group's median forecast with group‑wide quantiles. Never emit fewer than 55 rows.
- **Single read / single state.** Regime R0. With zero reads, the history is empty, which the spec does not allow; emit the starter‑style fallback with r = group prior median if needed.
- **Rising rates.** Early‑game trophies can outrun dilution. Models C and F allow this (new‑cohort earn rate q > r, plus progression). T follows its trend. No constraint forces rates down.
- **players_tracked falls (purges).** Handled in §2 and §3b. Rates are taken as observed. If more than 2 purges occur, widen the level shock to 0.35.
- **E inconsistent with Σ counts.** Down‑weight (§2). If it is inconsistent in every state, bypass aggregate anchoring.
- **Duplicate or out‑of‑order reads.** Sort by `at`. For identical `at`, keep the last.
- **Trophy‑list mismatch** (rates length ≠ trophies length). Pad with null or truncate, logging to stderr only.
- **Horizon ≤ 0 or target before the last read.** Output persistence with the minimum width.
- **Levels differ from [0.05, 0.25, 0.5, 0.75, 0.95].** Compute quantiles at the requested levels anyway; the code is generic.
- **Numerics:** clamp before every logit, catch overflow in exp with min/max at ±50, and use float('inf')‑free comparisons. In the final validation, any non‑finite value causes the row to be replaced by the fallback row for that trophy.

---

## 10. Validation plan (Q7)

### 10a. Synthetic simulator (`tools/sim.py`, stdlib only, shares no code with main.py)

**Generator.** Each draw samples:
1. **Launch:** legacy pool P0 ~ U(300, 1,500), with counts copied from the real launch read and scaled. Platinum holders 0–6.
2. **Arrivals:** intensity power law with β ~ U(−1.5, 0.3), plus a diurnal modulation (amplitude 0.3, peaking around 02:00Z), a weekend bump of ×U(1.0, 1.4) on Sat/Sun, and P(T) between 30k and 300k.
   - Misspecification variants: lognormal‑in‑time arrivals; a mid‑history step from a news spike.
3. **Players and progression:** each arrival gets a head start ~ Exp(mean U(0, 10 h)) and an activity multiplier ~ lognormal(0, 0.5), which makes progression heterogeneous.
   - The model's kernel is Weibull. To test misspecification, generate with gamma or lognormal kernels.
   - Trophies: a story chain of ~25 secret bronzes with increasing τ (2 h → 200 h) and A falling from 0.95 to 0.4; side trophies with A ~ Beta(1, 3); the platinum requires all.
4. **Exophase refreshes:** gamma intervals with mean ~ U(1, 8) h and shape ~ U(1, 6); some cases with a regime change (the interval halves at day 2). Purges with probability 0.1 per case.
5. **House recorder:** reads every 30 min until 2026‑11‑19T12:00Z, then every 10 min; 2‑decimal rounding; 1% of reads dropped; occasional null trophies.
6. **Case cutting:** the same shape as f00 (checkpoint L + 59.7 h, target L + 168 h, answer = first read ≥ T), plus practice‑shaped cases (90 min) with no refresh in history.

The generator's free parameters are fitted so that the first refresh reproduces the observed one (g ≈ 6, mean logit drop ≈ 1.5) within its prior range.

**Contenders**, scored with the exact contest loss:
- house persistence (exact formula);
- house curve: the exact recipe, last 24 h of reads, ln(1 + hours since 2026‑11‑19T00:00Z), RW spread ⊕ residual;
- a proxy for the house LLM: "oracle‑median‑with‑noise". Its median is the true value perturbed by N(0, 0.6²) in logit, with s = 0.6. This is only an orientation bar, since the real LLM cannot be simulated;
- our candidates: R2 full; C only; F only; T only; with and without anchoring; with and without the backtest; φ ∈ {0.5, 0.7, 0.9}.

**Ship criteria** over ≥ 300 final‑shaped simulations:
- mean loss below the house curve's;
- per‑simulation win rate ≥ 70%;
- no simulation worse than 1.5× the house curve's loss;
- coverage_90 within [0.80, 0.95];
- 50% coverage within [0.40, 0.60];
- 100% valid outputs, including fuzzed inputs (nulls, empty or one‑read histories, purges, 100% and 0% trophies);
- maximum runtime < 5 s on a 1‑CPU‑throttled run (`taskset -c 0`).

Decisions the simulator settles: ANCHOR on or off, φ, the default component weights, the s_cal default, and whether to keep the second (k, a0) combination.

### 10b. Real‑data checks

- **Launch‑night replay:** feed the practice inputs. Check that the p00 and p01 outputs are valid, and that the R0 mixture puts q05 near 0.2–0.35× the current rate for typical trophies.
- **First refresh replay:** construct a synthetic case from the 04:30Z state onward (using only the values given in the brief). Check that Model F's κ̂ ≈ 0.067 and g ≈ 6.06 are recovered.
- **Optional, needs the user's OK because it is outward‑facing:** record the public Exophase page ourselves from now to the deadline at a low rate (≤ 1 read per 30 min) to get the real refresh cadence and growth. Then run §7's backtest offline on real states before choosing the final defaults. Exophase cannot be back‑fetched, so this is the only real post‑launch calibration data available before the deadline.

### 10c. Submission budget (≤ 20 total, ≤ 10 per day; Force_Last)

1. **Submission #1, early:** confirms valid output, runtime, and the bundling pipeline. Read its practice loss only as a sanity check (expected noticeably < 1.51).
2. **At most 2–3 more** for real changes.
   - Never submit an untested build. Every submission passes the full simulator and fuzz suite first.
   - Practice loss is not a tuning signal: it is one event, and it only tests R0.
3. **Final submission by about 2026‑11‑21T09:00Z,** leaving more than 2 h of margin before the 11:30Z deadline. Under Force_Last a late broken build would be fatal, so keep the last known‑good build ready to re‑submit.

---

## 11. Answers to the BRIEF questions, in brief

1. **Refresh detection and process.** A refresh is any change of the (P, E, rates) signature between reads; its time is the bracket midpoint. The count of refreshes in the horizon comes from a gamma renewal process with censored current age; it is periodic if the intervals are regular. P growth follows a power‑law arrival intensity in time since launch (β < 0 as the pool matures), anchored to the last 24 h, with scenario draws.
2. **q_i estimation.** In rich histories, Model C's cohort kernel replaces q_i: it is A_i·G over arrival ages. With 1–3 refreshes, Model F uses NNLS of ΔC on ΔP and P·Δt. With no refresh, q_i = κ·r_i, where κ ~ lognormal(median 0.1); auto trophies have q = 1. All of these shrink to tier/secret group priors.
3. **Refresh timing.** The forecast is evaluated at T_eff = T − A, with A the renewal age of the answering state, mixed over 7 stratified ages. In practice‑length horizons, a mixture of no‑refresh and refresh outcomes weighted by p_ref.
4. **Progression.** It is modelled inside the cohort kernel (old arrivals keep climbing G), not as a separate per‑trophy saturating process, which is unidentifiable from aggregates. Model F's b_i is the simpler stand‑in for thin histories.
5. **Quantiles.** A weighted linear pool on the logit scale over (a) refresh timing and growth scenarios, (b) σ_jump and σ_mod residuals, and (c) grid posteriors and parameter draws. The spread is scaled by the backtest‑calibrated s_cal, with width rails and logical rails (platinum ≤ the others; auto trophy pinned at 100).
6. **Edge cases:** see §9.
7. **Validation:** the §10 simulator against both house baselines plus an LLM‑proxy, ship criteria, real‑data replays, and a conservative submission budget.
