# V2 diagnosis digest (prepared by Sitri, 2026-11-20)

## What happened
Submission #1 practice grade (server, cases p00+p01): **loss 0.9508** (house persistence
baseline 1.51 — we beat it), typical_miss **0.63 pp**, **coverage_90 = 0.1818** (only ~18% of
trophies inside [q05,q95]; field median among entries beating 1.51 is ~0.96), valid, run 3,150 ms.
Rank 4,164 / 5,883. The prize is decided ONLY by the final case f00 (program is re-run by the
house on f00: history = launch → 2026-11-21T11:30Z, horizon 6,510 min, answer = read at
2026-11-26T00:00Z). Practice is a smoke test, but it exposed real defects that can also bite on f00.

## Practice-case facts (from the kit inputs + the known first refresh)
- p00/p01 histories are FLAT: 7–8 reads, players_tracked = 638, trophies_earned = 943 the whole
  time (state age at checkpoint ≥ 3 h 10 m; last change before 2026-11-18T23:50Z).
- Rates: one trophy at 100% (auto), the rest 0.63–2.04%. Mean non-auto rate ≈ 0.88%.
- The answer read (first refresh, 04:00–04:30Z): players 638 → **3,868 (g = 6.07)**,
  trophies_earned 943 → 4,276 (×4.54). Mean non-auto rate fell to ≈ 0.196% → **mean rate ratio
  ≈ 0.22**, implying a mean new-cohort relative earn rate κ ≈ 0.066.

## Our v1 output on p00 (per trophy, vs current rate r)
- Typical row: q = [0.24r, 0.65r, 0.94r, 1.00r, 1.00r]. Median logit width ≈ 1.49.
- q75 = q95 = r exactly for nearly every trophy: **no scenario mass above the current rate**
  (in R0, r′ < r whenever κ < 1, and the n = 0 point mass sits at r).
- The pooled median sits at 0.94r: the **n = 0 (no-refresh) point mass ≈ 60%** dominates the
  median and upper quantiles, even though a refresh was near-certain (state age > 3 h, refresh
  cadence "every few hours", 90-min horizon, and the case construction guarantees a refreshed
  answer read in [target, target+60 m)).
- Truth (mean ratio 0.22) landed at/below our q05 (0.24r) for the average trophy; per-trophy κ
  dispersion (κ→0 gives ratio 1/g ≈ 0.165; fast early trophies can have κ > 1 → ratio > 1)
  pushed ~82% of trophies outside [q05, q95] on both sides.

## Code facts (entry/main.py, DESIGN.md)
- R0 path: `r0()` uses `jump_prior()`: for cur.p < 5000 and cur.t < 24 h → ln g ~ N(ln 4, 0.6).
  κ = 0.1·exp(0.6·common + 0.5·idio). Refresh-count clock comes from §3a renewal model; on a
  flat history it put ≈ 60% mass on n = 0 over a 90-min horizon.
- §6: residual σ_res = s_cal·sqrt(σ_mod² + σ_round² + σ_jump²·n_eff); s_cal default 1.15,
  R2 backtest-selected s* clamped so s_cal ∈ [0.8, 2.0]; σ_jump estimated, clamped [0.03, 0.3],
  default 0.10.
- §7 in-case backtest (R2 only) selects component weights and s_cal — but if its origins are
  concentrated in the recent (post-flood, stable) regime, it cannot price regime-shift risk over
  the 6,510-min extrapolation.
- Synthetic-final coverage of v1 was 97% (slightly wide) with loss 0.276 vs persistence 0.688 /
  fitted curve 0.389 — but the simulator shares the design's regime assumptions; the real
  practice case shows real-world dispersion is larger when the regime shifts.

## What v2 must achieve
1. **R0 hazard fix**: refresh probability over the horizon must account for state age and a
   realistic refresh-interval prior (intervals of a few hours, modest dispersion), so a stale
   flat history does not leave ~60% mass at persistence for the median. (Also helps any
   short-history final-like edge case; must not distort R1/R2.)
2. **Upside mass**: allow κ > 1 for fast/early trophies in R0 (and check R1/R2 paths for
   equivalent hard caps at the current rate) so q75/q95 can exceed r where progression can
   outrun dilution.
3. **Regime-aware calibration for R2 (the money case)**: backtest origins must span the whole
   history including the launch/flood window, so s_cal and component weights price regime
   shifts; consider raising the s_cal clamp ceiling and/or the default (1.15) and σ_jump floor,
   validated on the simulator (final-shaped loss must stay ≲ v1 + 0.02 and coverage within
   [0.88, 0.96]).
4. **Simulator**: add a practice-shaped flood scenario (flat pre-history at small P, then a
   ×6 player jump at the answer read) to tools/sim.py; v2 must reach coverage ≥ 0.8 there
   without wrecking final-shaped results.
5. Keep: stdlib-only, deterministic, ≤ 8 s/case on the dense stress case, all existing
   robustness fallbacks, byte-identical repeat runs on practice inputs.

## Questions for the design addendum
- Q1: Exact form of the R0 refresh hazard (prior on inter-refresh interval, how state age
  enters, how the clock mass over n ∈ {0,1,2} is computed) with parameter values.
- Q2: How to give κ a right tail above 1 tied to trophy "earliness" (what signal? current
  rate rank? tier? earn-time proxy from §4?) without inflating all trophies.
- Q3: Backtest origin scheme for §7 (which origins, how weighted, how s* maps to s_cal,
  new clamp range) and any change to defaults (s_cal, σ_jump floor/default).
- Q4: Anything else in the current code paths (peer caps, anchoring, width rails) that would
  reproduce the practice pathology on f00-shaped inputs; fix or explicitly clear each.
- Q5: Acceptance tests with numeric thresholds for the implementer (sim scenarios, practice
  outputs' qualitative shape, stress timing).
