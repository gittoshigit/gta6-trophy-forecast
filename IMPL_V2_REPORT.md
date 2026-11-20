# V2 implementation and acceptance report

## File changes

- `entry/main.py` — Reviewed the interrupted implementation against the addendum. Corrected the calibration child’s rich-path eligibility so an origin with four states can use the fixed-kernel model, including origin C when its last retained read precedes the 12-hour cut.
- `main.py` — Synchronized with `entry/main.py`.
- `tools/sim.py` — Added final-shaped shift scenarios, dense cadence, the `slow-host` and `v1-scale` variants, and per-model calibration engagement and maximum wall time. Shift parameters use a per-case random stream seeded with `seed * 1000003 + index`. Shifted arrivals enter the simulator’s existing completion batches.
- `IMPL_V2_REPORT.md` — Records the acceptance results.

## Test setup

Acceptance commands ran serially under `taskset -c 0` using Python 3.13.13. The v1 entry was extracted with `git show HEAD:main.py` and evaluated by the same simulator and seed. Simulator elapsed time includes scenario generation, baselines, and subprocess startup. Case wall times measure only the forecast subprocess. For B and D, the simulator’s default six final cases also ran before the requested cohorts. Test results for B and D below refer to their requested cohorts.

## A. Official practice cases

| ID | p00 | p01 | Status |
| --- | --- | --- | --- |
| A1 | p0 0.12741173, p1 0.86001777, p2 0.01257050, p0_ren 0.42470578 | p0 0.12893714, p1 0.86066991, p2 0.01039294, p0_ren 0.42979048 | Pass |
| A2 | 55 valid rows, byte-identical (2,777 bytes), 0.435 s and 0.413 s | 55 valid rows, byte-identical (2,763 bytes), 0.418 s and 0.415 s | Pass |
| A3 | 53/53 at q05 ≤ 0.165r | 53/53 at q05 ≤ 0.165r | Pass |
| A4 | Median q50/r 0.27782051 | Median q50/r 0.27820513 | Pass |
| A5 | Median q75/r 0.48063830, 55/55 with q95 ≥ 0.40r, 0 with q95 > 1.02r | Median q75/r 0.48063830, 55/55 with q95 ≥ 0.40r, 0 with q95 > 1.02r | Fail |
| A6 | Non-auto sum q50 14.0064 pp | Non-auto sum q50 14.0196 pp | Pass |

A3 excludes the auto and platinum trophies. A4 and A5 above use all 55 trophies. For non-auto trophies only, the q50/r medians were 0.27681935 and 0.27701166, the q75/r medians were 0.46739362 and 0.46744681, and 54/54 in each case met q95 ≥ 0.40r. The requirement of at least two trophies with q95 > 1.02r failed for both inputs. The additional practice-shape subprocesses took 0.4274 s and 0.4318 s.

## B. Practice-shaped simulator

24 practice cases per seed and build.

| Seed | Build | Mean loss | Cov90 | Cov50 | Max wall (practice cohort) | Elapsed (whole command) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20261120 | v2 | 0.229275 | 0.992 | 0.555 | 0.684 s | 50.309 s |
| 20261120 | v1 | 0.920316 | 0.261 | 0.020 | 0.720 s | 45.805 s |
| 7 | v2 | 0.237193 | 0.995 | 0.455 | 0.672 s | 52.562 s |
| 7 | v1 | 0.949900 | 0.248 | 0.018 | 0.668 s | 49.210 s |

| ID | Result | Status |
| --- | --- | --- |
| B1 | Cov90 0.992 and 0.995 (minimum 0.80) | Pass |
| B2 | Cov50 0.555 and 0.455 (minimum 0.30) | Pass |
| B3 | Loss 0.229275 and 0.237193, below 0.65 and below 75% of v1 on both seeds (0.690237 and 0.712425) | Pass |

## C. Final-shaped simulator

| Seed | Build | Mean loss | Cov90 | Cov50 | Wins vs curve | Worst loss ratio | Calibrated | Max wall | Elapsed |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 20261120 | v2 | 0.259707 | 0.983 | 0.694 | 27/30 | 1.104 | 30/30 | 6.470 s | 164.811 s |
| 20261120 | v1 | 0.250420 | 0.979 | 0.658 | 29/30 | 1.093 | 0/30 reported | 5.386 s | 128.375 s |
| 7 | v2 | 0.233027 | 0.987 | 0.778 | 30/30 | 0.966 | 27/30 | 6.094 s | 160.977 s |
| 7 | v1 | 0.224336 | 0.994 | 0.764 | 30/30 | 0.905 | 0/30 reported | 5.457 s | 126.295 s |

The v1 entry has no calibration diagnostic. Pooled metrics give each run equal weight because both have the same number of cases and trophies. Coverage is based on the simulator's displayed precision.

| ID | Result | Status |
| --- | --- | --- |
| C1 | Pooled loss 0.246367 against v1 0.237378, a +0.008989 increase (limit +0.02) | Pass |
| C2 | Pooled cov90 0.985 (ceiling 0.98) | Fail |
| C3 | Pooled cov50 0.736 (ceiling 0.65) | Fail |
| C4 | 57/60 wins (95.0%), worst loss ratio 1.104 | Pass |
| C5 | 57/60 engaged (95.0%) overall and 5/5 with more than 40 states | Pass |
| C6 | 0 occurrences of “safe fallback” or “last-resort” | Pass |

The three C5 misses were seed 7 cases f005, f015, and f021, with 9, 9, and 8 states respectively. The five cases with more than 40 states had 45, 46, 56, 53, and 54 states.

## D. Regime shift

24 shifted cases per build, seed 20261120.

| Build | Mean loss | Cov90 | Cov50 | Max wall (shift cohort) | Elapsed (whole command) |
| --- | ---: | ---: | ---: | ---: | ---: |
| v2 | 0.270380 | 0.993 | 0.786 | 6.269 s | 166.049 s |
| v1 | 0.264779 | 0.993 | 0.743 | 5.389 s | 132.781 s |

| ID | Result | Status |
| --- | --- | --- |
| D1 | v2 loss exceeded v1 by 0.005601 | Fail |
| D2 | v2 cov90 0.993 (minimum 0.85) | Pass |

## E. Timing and robustness

| ID | Result | Status |
| --- | --- | --- |
| E1 | Dense calibration engaged in 6/6. Maximum wall 6.095 s against 5.5 s. Command elapsed 37.149 s. | Fail |
| E2 | 0 fallback or last-resort messages in the 12 slow-host cases. Slow-host loss 0.279761 against full-v2 0.261678 (+0.018083, limit +0.05). Maximum wall 2.761 s slow-host and 6.406 s full v2. Command elapsed 95.131 s. | Pass |
| E3 | 7/7 valid (100%) across nulls, one read, empty history, purge, 0%/100% extremes, and short and long rate vectors. | Pass |
| E4 | 3/3 final-shaped cases byte-identical across two runs at TIME_SCALE = 1. | Pass |

Dense state counts were 66, 62, 65, 68, 57, and 58. Fuzz walls in table order were 4.2729, 0.4125, 0.1206, 5.7718, 4.3272, 4.4904, and 4.5870 s. E4 repeat walls were 4.5762/4.6220, 4.5118/4.5290, and 4.1466/4.1942 s. The isolated helper for practice shape, E3, E4, state counts, and RNG verification took 93.124 s.

## Ship decision

**Acceptance failed** on A5, C2, C3, D1, and E1. C1 passed, so the ship rule’s scale ladder did not apply. Both entry copies are byte-identical.

For the original simulator streams, eight sampled worlds across both seeds matched the original simulator byte for byte, and the random states and oracle draws matched. The final A2 command elapsed 1.841 s. Raw acceptance logs are in `/tmp/trophy-v2-accept/`. Token usage was unavailable.
