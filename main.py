"""Deterministic Exophase refresh and trophy progression forecast."""
import bisect
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

LEVELS = (.05, .25, .5, .75, .95)
NORMAL = statistics.NormalDist()
PRIORS = ((.6, 72), (.3, 120), (.2, 150), (.03, 400))
# Enabled for offline ablations. The synthetic smoke run favored no anchoring.
ANCHOR = 0.
PHI = .7
DEFAULT_SCALE = 1.25
SCALE_GRID = (.85, 1., 1.15, 1.3, 1.5, 1.8, 2.2, 2.7, 3.2)
SCALE_CLAMP = (1., 2.4)
JUMP_FLOOR = .06
TIME_SCALE = 1.0
R0_SHAPE = 4
R0_MU = (1.5, 2, 3, 4, 6)
R0_PI = (.15, .25, .30, .20, .10)
R0_KEEP = .30
R0_LOG_G = (math.log(5), .55)
R0_LOG_KAPPA = (math.log(.08), .60)
R0_GAMMA = .5
R0_NB_SIZE = 1.0
DO_BACKTEST = True
SECOND_KERNEL = True
ONLY_COMPONENT = None
GH = ((-2.856970, .011257), (-1.355626, .222076), (0, .533334),
      (1.355626, .222076), (2.856970, .011257))


def clip(x, lo, hi):
    return min(hi, max(lo, x))


def logit(rate):
    if rate <= .01:
        return -9.21024036697585
    if rate >= 99.99:
        return 9.210240366975338
    return math.log(rate / (100 - rate))


def inverse(x):
    if x < -50:
        x = -50
    elif x > 50:
        x = 50
    return 100 / (1 + math.exp(-x))


def normal_grid(n):
    return [NORMAL.inv_cdf((i + .5) / n) for i in range(n)]


def stamp(value):
    return datetime.fromisoformat(value).timestamp() / 3600


def hours(t):
    return math.log(max(6, t + 6))


def group(trophy):
    tier = str(trophy.get("tier", "")).lower()
    if tier == "platinum":
        return 3
    if tier in ("gold", "silver"):
        return 2
    return 0 if trophy.get("secret") else 1


def round4(x):
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= .5:
        whole += 1
    return whole / 10000 if whole % 10000 else whole // 10000


def safe_fallback(case, minimum=False):
    trophies = case["trophies"]
    latest = [None] * len(trophies)
    reads = sorted(case.get("history") or [], key=lambda read: read.get("at", ""))
    for read in reversed(reads):
        for i, rate in enumerate((read.get("rates") or [])[:len(trophies)]):
            if latest[i] is None and isinstance(rate, (int, float)) and math.isfinite(rate):
                latest[i] = rate
        if None not in latest:
            break
    spread = .25 / 3.29 if minimum else .35 + .65 * math.sqrt(max(0, case.get("horizon_minutes", 0)) / 1440)
    z = [NORMAL.inv_cdf(clip(q, .00001, .99999)) for q in case.get("levels", LEVELS)]
    return [[inverse(logit(rate if rate is not None else 100 * PRIORS[group(t)][0]) + spread * x)
             for x in z] for t, rate in zip(trophies, latest)]


@dataclass
class State:
    t: float
    bracket: float
    start: float
    end: float
    p: int
    earned: int
    rates: tuple
    counts: tuple
    consistent: bool


def preprocess(case):
    reads = sorted({stamp(r["at"]): r for r in case["history"]}.items())
    launch = reads[0][0]
    n = len(case["trophies"])
    vectors = []
    for _, read in reads:
        rates = read.get("rates") or []
        if len(rates) != n:
            print("rate vector length differs from trophy list", file=sys.stderr)
        vectors.append((rates + [None] * n)[:n])
    missing = [value is None for value in vectors[-1]]
    unknown = []
    for i, trophy in enumerate(case["trophies"]):
        first = next((v[i] for v in vectors if v[i] is not None), None)
        unknown.append(first is None)
        rate = 100 * PRIORS[group(trophy)][0] if first is None else first
        for v in vectors:
            if v[i] is not None:
                rate = v[i]
            v[i] = rate
    runs = []
    for (at, read), vector in zip(reads, vectors):
        t = at - launch
        signature = (max(1, read["players_tracked"]), read["trophies_earned"], tuple(vector))
        if runs and runs[-1][0] == signature:
            runs[-1][2:] = [t, runs[-1][3] + 1]
        else:
            runs.append([signature, t, t, 1])
    glitches = [i for i in range(1, len(runs) - 1)
                if runs[i][3] == 1 and runs[i-1][0] == runs[i+1][0]]
    if len(glitches) == 1:
        i = glitches[0]
        runs[i-1][2] = runs[i+1][2]
        del runs[i:i+2]
    states = []
    for (p, earned, rates), start, end, _ in runs:
        bracket = (start - states[-1].end) / 2 if states else 0
        raw = [r * p / 100 for r in rates]
        counts = tuple(float(round(c)) if p < 20000 else c for c in raw)
        states.append(State(start - bracket, bracket, start, end, p, earned, rates,
                            counts, abs(sum(raw) - earned) <= .5 + .0001 * p))
    return states, launch, reads[-1][0] - launch, unknown, missing


def wls(rows, ridge=0, prior=0):
    if not rows:
        return 0, prior, 0, 0
    total = sum(w for _, _, w in rows)
    mx = sum(x*w for x, _, w in rows) / total
    my = sum(y*w for _, y, w in rows) / total
    xx = sum(w*(x-mx)**2 for x, _, w in rows)
    xy = sum(w*(x-mx)*(y-my) for x, y, w in rows)
    slope = (xy + ridge*prior) / max(1e-12, xx + ridge)
    intercept = my - slope*mx
    sd = math.sqrt(sum(w*(y-intercept-slope*x)**2 for x, y, w in rows) / total)
    se = sd / math.sqrt(max(1e-12, (xx + ridge) * len(rows) / total))
    return intercept, slope, sd, se


def gamma_survival(shape, x):
    """Regularized upper incomplete gamma, using the two standard expansions."""
    if x <= 0:
        return 1.
    factor = math.exp(shape * math.log(x) - x - math.lgamma(shape))
    if x < shape + 1:
        term = total = 1 / shape
        for n in range(1, 120):
            term *= x / (shape + n)
            total += term
            if term < total * 1e-13:
                break
        return clip(1 - factor * total, 0, 1)
    b = x + 1 - shape
    c, d = 1e30, 1 / b
    result = d
    for n in range(1, 120):
        a = n * (shape - n)
        b += 2
        d = 1 / max(1e-30, a*d + b)
        c = max(1e-30, b + a/c)
        change = c*d
        result *= change
        if abs(change - 1) < 1e-13:
            break
    return clip(factor * result, 0, 1)


def renewal(states, checkpoint, target):
    cur = states[-1]
    age = max(0, checkpoint - cur.start)
    intervals = [b.t-a.t for a, b in zip(states, states[1:])]
    n = len(intervals)
    mean = clip((8 + sum(intervals) + age) / (2+n), .15, 96)
    shape = 2
    periodic = None
    if n >= 4:
        weights = [math.exp((s.t-cur.t)/36) for s in states[1:]]
        avg = sum(w*x for w, x in zip(weights, intervals)) / sum(weights)
        cv2 = sum(w*(x/avg-1)**2 for w, x in zip(weights, intervals)) / sum(weights)
        shape = (4 + n*clip(1/max(cv2, 1e-9), 1, 8))/(2+n)
        if n >= 6 and cv2 < .25**2:
            periodic = wls([(i, s.t, 1) for i, s in enumerate(states[1:], 1)])
    rate = shape / mean
    horizon = max(0, target-checkpoint)
    if horizon < 3*mean or n == 0:
        survival = max(1e-300, gamma_survival(shape, age*rate))
        p0 = clip(gamma_survival(shape, (age+horizon)*rate)/survival, 0, 1)
        # Quadrature over the residual first interval, then sums of gamma intervals.
        refreshes = []
        divisions = 48
        for i in range(divisions):
            lo, hi = horizon*i/divisions, horizon*(i+1)/divisions
            mass = (gamma_survival(shape, (age+lo)*rate) -
                    gamma_survival(shape, (age+hi)*rate)) / survival
            first = (lo+hi)/2
            left = horizon-first
            one = gamma_survival(shape, left*rate)
            two_or_less = gamma_survival(2*shape, left*rate)
            for count, p, t in ((1, one, checkpoint+first),
                                (2, two_or_less-one, checkpoint+first+left*.55),
                                (3, 1-two_or_less, checkpoint+first+left*.8)):
                if mass*p > 1e-10:
                    refreshes.append((t, count, mass*p))
        result = [(cur.t, 0, p0)] if p0 > 0 else []
        for count in (1, 2, 3):
            items = [(t, w) for t, k, w in refreshes if k == count]
            mass = sum(w for _, w in items)
            if mass:
                for t in quantiles(items, [(j+.5)/7 for j in range(7)]):
                    result.append((t, count, mass/7))
        return mean, result
    if periodic:
        phase, period, jitter, _ = periodic
        ages = [((target-phase) % period - z*jitter) % period for z in normal_grid(7)]
    else:
        grid = [i*mean/8 for i in range(65)]
        # Integral of the survival function from zero to each grid point.
        cdf = [1 - gamma_survival(shape+1, x*rate) +
               x/mean*gamma_survival(shape, x*rate) for x in grid]
        ages = []
        for j in range(7):
            q = (j+.5)/7
            i = min(64, max(1, bisect.bisect_left(cdf, q)))
            frac = (q-cdf[i-1])/max(1e-12, cdf[i]-cdf[i-1])
            ages.append(grid[i-1] + frac*(grid[i]-grid[i-1]))
    return mean, [(max(cur.t, target-a), max(1, round((target-a-cur.t)/mean)), 1/7)
                  for a in ages]


def quantiles(draws, levels):
    draws = sorted(draws)
    positions = []
    cumulative = 0.
    for _, w in draws:
        positions.append(cumulative + w/2)
        cumulative += w
    values = []
    for level in levels:
        level *= cumulative
        j = bisect.bisect_left(positions, level)
        if j == 0 or j == len(draws):
            values.append(draws[0 if j == 0 else -1][0])
        else:
            fraction = (level-positions[j-1]) / (positions[j]-positions[j-1])
            values.append(draws[j-1][0] + fraction*(draws[j][0]-draws[j-1][0]))
    return values


def intervals(states):
    return [(a, b, b.t-a.t, b.p-a.p) for a, b in zip(states, states[1:])]


def integral(beta, a, b):
    exponent = beta+1
    if abs(exponent) < 1e-8:
        return math.log((b+6)/(a+6))
    return (a+6)**exponent * math.expm1(exponent*math.log((b+6)/(a+6))) / exponent


def jump_prior(cur, positive, mean, t):
    if positive:
        a, b, _, _ = positive[-1]
        return math.log(b.p/a.p) - .6*(hours(t)-hours(b.t)), .5
    if cur.p < 5000 and cur.t < 24:
        return math.log(4), .6
    return math.log1p(2*mean/(cur.t+6)), .5


def scenarios(states, clock, mean):
    cur = states[-1]
    positive = [row for row in intervals(states) if row[-1] > 0]
    purges = [row for row in intervals(states) if row[-1] < 0]
    purge = -sum(row[-1] for row in purges)/max(1, cur.t) if len(purges) > 1 else 0
    rows = [(hours((a.t+b.t)/2), math.log(dp/dt), dt*math.exp((b.t-cur.t)/36))
            for a,b,dt,dp in positive]
    _, beta, _, se = wls(rows, 3, -.6)
    beta = clip(beta, -2.5, 1)
    level_sd = .35 if len(purges) > 2 else .25
    start = max(0, cur.t-24)
    arrivals = sum(dp*max(0, b.t-max(start,a.t))/dt for a,b,dt,dp in positive)
    if not arrivals:
        start = positive[0][0].t if positive else 0
        arrivals = sum(row[-1] for row in positive)
    z = normal_grid(9)
    for t, count, mass in clock:
        if not count:
            yield (cur.t, cur.p, [], 0, mass)
            continue
        for j, shock in enumerate(z):
            slope = clip(beta + shock*math.hypot(se, .3), -2.5, 1)
            shift = level_sd*z[(4*j+2)%9]
            if len(positive) >= 2:
                total = arrivals * math.exp(shift) * integral(slope, cur.t, t) / max(1e-9, integral(slope, start, cur.t))
            else:
                center, scale = jump_prior(cur, positive, mean, t)
                total = cur.p*math.expm1(clip((center+scale*shock)*min(count, 12), 0, 7))
            edges = [(cur.t+6)*math.exp((hours(t)-hours(cur.t))*k/12)-6 for k in range(13)]
            duration = max(0, t-cur.t)
            denominator = max(1e-12, integral(slope, cur.t, t))
            batches = [((a+b)/2, total*integral(slope,a,b)/denominator, b-a)
                       for a,b in zip(edges, edges[1:])]
            yield (t, max(1, cur.p+total-purge*duration), batches, count, mass/9)


def check_time(start, limit=6.3):
    if time.monotonic()-start >= limit*TIME_SCALE:
        raise TimeoutError("forecast time guard")


def r0_clock(states, checkpoint, target):
    cur = states[-1]
    age = max(0., checkpoint-cur.start)
    horizon = max(0., target-checkpoint)
    shape = R0_SHAPE

    def equilibrium(x):
        return gamma_survival(shape+1, x) - x/shape*gamma_survival(shape, x)

    rates = [shape/mu for mu in R0_MU]
    posterior = [prior*equilibrium(age*rate) for prior,rate in zip(R0_PI,rates)]
    evidence = sum(posterior)
    p0 = R0_KEEP*sum(prior*equilibrium((age+horizon)*rate)
                     for prior,rate in zip(R0_PI,rates))/evidence
    second = sum(weight*(1-gamma_survival(shape, rate*horizon/2))
                 for weight,rate in zip(posterior,rates))/evidence
    p2 = (1-p0)*second
    return [(cur.t, 0, p0), (target, 1, 1-p0-p2), (target, 2, p2)]


def r0(states, clock, groups, auto, levels):
    cur = states[-1]
    grid = normal_grid(9)
    growth = [max(1.2, math.exp(R0_LOG_G[0]+R0_LOG_G[1]*z)) for z in grid]
    kappas = [math.exp(R0_LOG_KAPPA[0]+R0_LOG_KAPPA[1]*z) for z in grid]

    logs = [math.log(r) for r,g,skip in zip(cur.rates,groups,auto) if r > 0 and g != 3 and not skip]
    center = statistics.mean(logs) if logs else 0.
    spread = max(.1, statistics.pstdev(logs)) if logs else .1
    group_weights = [math.sqrt((a/tau)/(PRIORS[0][0]/PRIORS[0][1])) for a,tau in PRIORS]
    rates = [max(r, 50/cur.p) for r in cur.rates]
    weighted = [r*group_weights[g]*math.exp(R0_GAMMA*clip((math.log(r)-center)/spread, -2.5, 2.5))
                for r,g in zip(rates,groups)]
    total = sum(r for r,skip in zip(rates,auto) if not skip)
    norm = sum(r for r,skip in zip(weighted,auto) if not skip)/total if total else 1.
    scenarios = [(max(cur.p+1, round(g*cur.p*(1.5 if count == 2 else 1))), mass/81)
                 for _,count,mass in clock if count and mass > 0 for g in growth]
    p0 = clock[0][2]
    size = R0_NB_SIZE
    result = []
    for r,c,wr,skip in zip(cur.rates,cur.counts,weighted,auto):
        if skip:
            result.append([logit(r)]*len(levels))
            continue
        draws = [(logit(r), p0)] if p0 > 0 else []
        for players,mass in scenarios:
            increment = (players-cur.p)*(wr/100)/norm
            for kappa in kappas:
                mean = kappa*increment
                if mean > 60:
                    variance = math.log1p(1/mean+1/size)
                    center = math.log(mean)-variance/2
                    spread = math.sqrt(variance)
                    draws.extend((logit(100*min(players, c+math.exp(center+spread*z))/players),
                                  mass*w) for z,w in GH)
                    continue
                prob = (size/(size+mean))**size
                decay = mean/(size+mean)
                atoms = [(logit(100*min(players, c)/players), prob)]
                cumulative = prob
                for j in range(1,401):
                    if cumulative >= 1-1e-4:
                        break
                    prob *= (j-1+size)/j*decay
                    atoms.append((logit(100*min(players, c+j)/players), prob))
                    cumulative += prob
                draws.extend((x, w*(mass/cumulative)) for x,w in atoms)
        result.append(quantiles(draws, levels))
    return result


def trend(states, values):
    first = bisect.bisect_right([s.t for s in states], states[-1].t-36)-1
    return wls([(hours(s.t), y, math.exp((s.t-states[-1].t)/72))
                for s,y in list(zip(states, values))[max(0, first):]])


def flow_fit(states, auto):
    cur = states[-1]
    flows = [row for row in intervals(states) if row[-1] >= 0]
    numerator = sum(b.counts[i]-a.counts[i] for a,b,_,dp in flows if dp
                    for i in range(len(auto)) if not auto[i])
    denominator = sum(a.rates[i]*dp/100 for a,b,_,dp in flows
                      for i in range(len(auto)) if not auto[i])
    kappa = clip(numerator/denominator, .005, 8) if denominator else .1
    median = max(.01, statistics.median(cur.rates))
    result = []
    for i, rate in enumerate(cur.rates):
        a0 = clip(kappa*rate/100, .000001, .995)
        b0 = clip(.002*rate/median, .000001, .2)
        x, y = math.log(a0/(1-a0)), math.log(b0)
        observations = [(dp, max(0,a.p-a.counts[i])*dt, b.counts[i]-a.counts[i],
                         math.exp((b.t-cur.t)/72) / (max(2, math.sqrt(max(1,b.counts[i])))**2))
                        for a,b,dt,dp in flows]
        wx = wy = 0
        for _ in range(10):
            aa, bb = inverse(x)/100, math.exp(y)
            ux, uy = aa*(1-aa), bb
            gx, gy, hxx, hxy, hyy = x-math.log(a0/(1-a0)), y-math.log(b0), 1., 0., 1.
            for u,v,d,w in observations:
                error = u*aa+v*bb-d
                dx, dy = u*ux, v*uy if len(observations)>1 else 0
                gx += w*error*dx
                gy += w*error*dy
                hxx += w*dx*dx
                hxy += w*dx*dy
                hyy += w*dy*dy
            det = max(1, hxx*hyy-hxy*hxy)
            x -= clip((gx*hyy-gy*hxy)/det, -1, 1)
            y = clip(y-clip((gy*hxx-gx*hxy)/det, -1, 1), -20, 0)
            wx, wy = math.sqrt(hyy/det), math.sqrt(hxx/det)
        result.append((x, y, wx, wy))
    return result


def flow_path(cur, i, scenario, a, b):
    _, p, batches, _, _ = scenario
    count, population = cur.counts[i], cur.p
    for t, dp, dt in batches:
        decay = ((cur.t+6)/(t+6))**.3
        progression = max(0, population+dp/2-count) * -math.expm1(-b*decay*dt)
        count += a*decay*dp + progression
        population += dp
    return logit(100*count/p)


def kernel_rows(states, taus, shape, head, start):
    rows = []
    batches = []
    for a,b,_,dp in intervals(states):
        if dp > 0:
            batches.append(((a.t+b.t)/2, dp))
    for tau in taus:
        check_time(start)
        per_state = []
        for s in states:
            arrivals = sum(dp*-math.expm1(-((s.t-at+head)/tau)**shape)
                           for at,dp in batches if at <= s.t)
            legacy = states[0].p*-math.expm1(-((s.t+head)/tau)**shape)
            per_state.append((100*legacy/s.p, 100*arrivals/s.p))
        rows.append(per_state)
    return rows, batches


def obs_weights(states, last):
    return [max(.05, min(1, (states[j+1].t if j+1<len(states) else last)-s.t))
            * math.exp((s.t-last)/72) * (1 if s.consistent else .5)
            for j,s in enumerate(states)]


def choose_kernels(states, weights, groups, auto, taus, start, fixed):
    dominant = sorted((i for i in range(len(auto)) if not auto[i]),
                      key=lambda i: states[-1].counts[i], reverse=True)[:30]
    fits = []
    choices = [fixed] if fixed else [(k,a) for k in (1,1.5,2,3) for a in (0,3,12)]
    for shape,head in choices:
        check_time(start)
        rows,batches = kernel_rows(states, taus, shape, head, start)
        designs = [[u+v for u,v in row] for row in rows]
        profile_weights = [[w*x for w,x in zip(weights,row)] for row in designs]
        denominators = [max(1e-12,sum(w*x*x for w,x in zip(weights,row))) for row in designs]
        score = 0.
        for i in dominant:
            pa, pt = PRIORS[groups[i]]
            floor = [100*states[0].counts[i]/s.p for s in states]
            rate = [s.rates[i] for s in states]
            adjusted = [max(0,y-l) for y,l in zip(rate,floor)]
            best = math.inf
            for j,(tau,row) in enumerate(zip(taus,rows)):
                # Profile with the observed legacy floor held fixed, then
                # score the candidate using the full max-legacy model.
                amp = clip(sum(x*y for x,y in zip(profile_weights[j], adjusted)) /
                           denominators[j], .00001, 1)
                error = sum(w*(max(l,amp*u)+amp*v-y)**2 for w,l,(u,v),y in zip(weights,floor,row,rate))
                prior = ((logit(100*amp)-logit(100*pa))/1.5)**2+(math.log(tau/pt)/1.2)**2
                best = min(best, error + .08**2*prior)
            score += best
        fits.append((score,shape,head,rows,batches))
    fits.sort(key=lambda fit: fit[:3])
    return fits[:2], len(dominant)*sum(weights)


def posterior(states, fit, taus, weights, groups, auto, start):
    _,shape,head,rows,batches = fit
    alphas = [inverse(logit(.2)+j*(logit(99.8)-logit(.2))/24)/100 for j in range(25)] + [1.]
    params = [(j,a,logit(100*a),math.log(tau))
              for j,tau in enumerate(taus) for a in alphas]
    base = [(logit(100*a), math.log(t)) for a,t in PRIORS]
    cache = {}
    map_errors = []
    jump_errors = []
    means = [[] for _ in PRIORS]
    for i,g in enumerate(groups):
        check_time(start)
        if auto[i]:
            continue
        floor = [100*states[0].counts[i]/s.p for s in states]
        observed = [logit(s.rates[i]) for s in states]
        rms = [round_noise(s.rates[i]) for s in states]
        residuals = []
        wa = [w/(.08**2+r*r) for w,r in zip(weights,rms)]
        for j,a,_,_ in params:
            residuals.append(tuple(logit(max(l,a*u)+a*v)-obs
                                   for l,(u,v),obs in zip(floor,rows[j],observed)))
        scores = [sum(w*d*d for w,d in zip(wa,errors)) for errors in residuals]
        probabilities, best = posterior_weights(params,scores,base[g])
        means[g].append((sum(w*p[2] for w,p in zip(probabilities,params)),
                         sum(w*p[3] for w,p in zip(probabilities,params))))
        errors = residuals[best]
        map_errors.extend(errors)
        jump_errors.extend(b-a for a,b in zip(errors,errors[1:]))
        cache[i] = (residuals,rms)
    sigma = clip(statistics.pstdev(map_errors) if map_errors else .08,.04,.3)
    jump = clip(statistics.pstdev(jump_errors) if jump_errors else .10,JUMP_FLOOR,.3)
    pooled = [(tuple((v+statistics.mean(col))/2 for v,col in zip(prior,zip(*items)))
               if items else prior) for prior,items in zip(base,means)]
    points = {}
    for i,(residuals,rms) in cache.items():
        check_time(start)
        wa = [w/(sigma**2+r*r) for w,r in zip(weights,rms)]
        scores = [sum(w*d*d for w,d in zip(wa,errors)) for errors in residuals]
        probabilities,_ = posterior_weights(params,scores,pooled[groups[i]])
        retained = [(j,a,w) for (j,a,_,_),w in zip(params,probabilities) if w > 1e-4]
        points[i] = stratify_points(retained)
    return points, sigma, jump


def stratify_points(points, size=24):
    if len(points) <= size:
        return points
    cumulative = []
    total = 0.
    for _,_,weight in points:
        total += weight
        cumulative.append(total)
    selected = {}
    for k in range(size):
        point = bisect.bisect_left(cumulative,total*(k+.5)/size)
        selected[point] = selected.get(point,0)+1/size
    return [(points[j][0],points[j][1],weight) for j,weight in selected.items()]


def posterior_weights(params, scores, prior):
    pa, pt = prior
    losses = [s+((la-pa)/1.5)**2+((lt-pt)/1.2)**2
              for (_,_,la,lt),s in zip(params,scores)]
    best = min(range(len(losses)),key=losses.__getitem__)
    values = [math.exp(-.5*(loss-losses[best])) for loss in losses]
    total = sum(values)
    return [v/total for v in values], best


def round_noise(r):
    r = clip(r,.01,99.99)
    return .0029/(r*(1-r/100))


def cohort_future(taus, fit, scenario, p0):
    _,shape,head,_,past = fit
    t, p, future, _, _ = scenario
    batches = past + [(at,dp) for at,dp,_ in future]
    return [(100*p0/p*-math.expm1(-((t+head)/tau)**shape),
             100/p*sum(dp*-math.expm1(-((max(0,t-at)+head)/tau)**shape)
                       for at,dp in batches if at <= t))
            for tau in taus]


def compress_states(states, last, limit):
    while len(states) > limit:
        old, new = [], []
        for s in states[1:-1]:
            (old if s.t < last-24 else new).append(s)
        if len(old) < 2:
            states = [states[0]] + states[1:-1:2] + [states[-1]]
        else:
            states = [states[0]] + old[1::2] + new + [states[-1]]
    return states


def pool_draws(states, futures, groups, auto, models, taus, start, last):
    cur = states[-1]
    flow = flow_fit(states, auto)
    trends = [trend(states, [logit(s.rates[i]) for s in states]) for i in range(len(auto))]
    z = normal_grid(15)
    output = {name: [[] for _ in auto] for name in ("F", "T", "C" if models else "D")}
    anchors = [s for s in states if s.consistent]
    anchor = None
    if ANCHOR and anchors:
        remaining = [max(.0001, s.earned/s.p-sum(s.rates[i]/100 for i,v in enumerate(auto) if v))
                     for s in anchors]
        anchor = (remaining[-1], trend(anchors, [math.log(v) for v in remaining])[1])
    size = [max(s.counts[i] for s in states) for i in range(len(auto))]
    for j, future in enumerate(futures):
        check_time(start)
        t, p, _, count, mass = future
        delta = hours(t)-hours(cur.t)
        kernels = [cohort_future(taus, fit, future, states[0].p) for fit,_,_ in models]
        samples = {name: [[] for _ in auto] for name in output}
        for i in range(len(auto)):
            base = logit(cur.rates[i])
            if auto[i] or not count:
                for rows in samples.values():
                    rows[i] = [(base, 1)]
                continue
            _, slope, sd, se = trends[i]
            trend_mid = base + PHI*slope*delta
            trend_sd = math.sqrt(sd**2+(se*delta)**2+(.3*slope*delta)**2)
            rare = size[i] < 5
            parameter_ids = [(j+offset)%15 for offset in (0,5,10)]
            a,b,sa,sb = flow[i]
            for u in parameter_ids:
                f = flow_path(cur,i,future,inverse(a+sa*z[u])/100,
                              math.exp(clip(b+sb*z[(7*u+2)%15],-20,0)))
                samples["F"][i].append((.7*f+.3*trend_mid if rare else f, 1/3))
                samples["T"][i].append((trend_mid+trend_sd*z[(4*u+1)%15], 1/3))
            if models:
                for (fit,posterior_rows,weight), values in zip(models,kernels):
                    floor = 100*states[0].counts[i]/p
                    points = posterior_rows[i]
                    normalizer = sum(w for _,_,w in points)
                    for idx, amp, w in points:
                        legacy, arrivals = values[idx]
                        value = logit(max(floor, amp*legacy)+amp*arrivals)
                        samples["C"][i].append((.7*value+.3*trend_mid if rare else value,
                                                weight*w/normalizer))
            else:
                hazard = math.exp(b)
                floor = (cur.p-cur.counts[i])*-math.expm1(-hazard*max(0,t-cur.t))
                samples["D"][i] = [(logit(100*(cur.counts[i]+floor)/p),1)]
        for name, rows in samples.items():
            shift = 0
            if anchor and count:
                total = sum(inverse(x)*w/100 for i,row in enumerate(rows) if not auto[i] for x,w in row)
                expected = anchor[0]*math.exp(clip(PHI*anchor[1]*delta,-10,10))
                shift = ANCHOR*math.log(expected/max(.0001,total))
            for i,row in enumerate(rows):
                output[name][i].extend((x+shift,count,mass*w) for x,w in row)
    return output


def convolve(draws, states, sigma, jump, missing, auto, start):
    result = []
    for i,row in enumerate(draws):
        check_time(start)
        if auto[i]:
            result.append([(logit(100),1)])
            continue
        rare = max(s.counts[i] for s in states) < 5
        spread = sigma * (2 if rare else 1) * (1.5 if missing[i] else 1)
        cloud = []
        variance = spread**2
        jump_variance = [jump**2*n for n in range(5)]
        for x,n,w in row:
            if n == 0:
                cloud.append((x,w))
                continue
            rounding = .000058*(1+math.cosh(clip(x,-9.21024,9.21024)))
            sd = math.sqrt(variance+rounding*rounding+jump_variance[min(n,4)])
            cloud.extend((x+sd*node,w*mass) for node,mass in GH)
        result.append(cloud)
    return result


def summarize(components, weights, levels, start, scale=1.):
    result = []
    for i in range(len(next(iter(components.values())))):
        check_time(start)
        row = [(x,w*weights[name]) for name in weights for x,w in components[name][i]]
        if scale == 1:
            result.append(quantiles(row,levels))
        else:
            *values, median = quantiles(row, (*levels, .5))
            result.append([median+scale*(x-median) for x in values])
    return result


def predict(components, weights, states, sigma, jump, missing, auto, scale, levels, start):
    clouds = {name:convolve(components[name],states,sigma,jump,missing,auto,start)
              for name in weights}
    return summarize(clouds,weights,levels,start,scale), clouds


def pinball(rows, actual, auto, factor=1):
    terms = []
    for row,y,skip in zip(rows,actual,auto):
        if skip:
            continue
        mid = row[2]
        for q,p in zip(row,LEVELS):
            d = logit(y)-logit(inverse(mid+factor*(q-mid)))
            terms.append(2*max(p*d,(p-1)*d))
    return statistics.mean(terms) if terms else 0


def calibrate(case, states, launch, last, target, auto, default, fixed, start):
    ratio = max(1.001,target/max(1,last))
    totals = dict.fromkeys(default,0.)
    scales = dict.fromkeys(SCALE_GRID,0.)
    pooled_loss = weight_sum = 0.
    best_scales = []
    cut_a = last/ratio
    origins = [(cut_a,.45),(last/ratio**.5,.30)]
    cut_c = max(12,states[3].start+1e-6)
    if cut_c <= cut_a-3:
        origins.append((cut_c,.25))
    for cut, weight in origins:
        if time.monotonic()-start >= 5.0*TIME_SCALE:
            break
        if sum(s.start <= cut for s in states) < 4:
            continue
        history = [r for r in case["history"] if stamp(r["at"])-launch <= cut]
        child = dict(case, history=history, at=iso(launch+cut), target=iso(launch+last),
                     horizon_minutes=(last-cut)*60, levels=LEVELS)
        try:
            check_time(start)
            result = forecast_case(child,depth=1,start=start,fixed=fixed)
        except Exception:
            continue
        if not isinstance(result, tuple):
            continue
        pooled, separate = result
        actual = states[-1].rates
        if not separate:
            continue
        for name in totals:
            row = separate.get(name, pooled)
            totals[name] += weight*pinball(row,actual,auto)
        pooled_loss += weight*pinball(pooled,actual,auto)
        losses = {s:pinball(pooled,actual,auto,s) for s in scales}
        best_scales.append(min(losses,key=losses.get))
        for s,loss in losses.items():
            scales[s] += weight*loss
        weight_sum += weight
    if not weight_sum:
        return default, DEFAULT_SCALE, 0, None, None
    unscaled = {name:w*math.exp(-3*(totals[name]-min(totals.values()))/weight_sum)
                for name,w in default.items()}
    total = sum(unscaled.values())
    weights = {name:(w/total+default[name])/2 for name,w in unscaled.items()}
    if pooled_loss > 1.2*totals["T"] and "C" in weights:
        for name in ("F","T"):
            weights[name] *= .8/(1-weights["C"])
        weights["C"] = .2
    best = min(scales,key=scales.get)
    high = max(best_scales)
    scale = clip(.50*best+.25*high+.25*DEFAULT_SCALE,*SCALE_CLAMP)
    return weights, scale, len(best_scales), best, high


def iso(hour):
    from datetime import UTC
    return datetime.fromtimestamp(hour*3600, UTC).isoformat()


def render(rows, states, groups, auto, unknown, levels, is_r0):
    mid_index = min(range(len(levels)), key=lambda j: abs(levels[j]-.5))
    known = {}
    for i,g in enumerate(groups):
        if not unknown[i] and not auto[i]:
            known.setdefault(g,[]).append(i)
    result = []
    for i,row in enumerate(rows):
        if auto[i]:
            result.append([99.95 if p <= .05 else 99.99 if p <= .25 else 100 for p in levels])
            continue
        if unknown[i]:
            peers = known.get(groups[i], [])
            if peers:
                row = [quantiles([(rows[j][k],1) for j in peers],[p])[0]
                       for k,p in enumerate(levels)]
        median = row[mid_index]
        if not unknown[i] and all(s.counts[i] == 0 for s in states):
            peers = known.get(groups[i], [])
            cap = quantiles([(logit(states[-1].rates[j]),1) for j in peers],[.3])[0]
            if median > cap:
                row = [x+cap-median for x in row]
                median = cap
        width = row[-1]-row[0]
        if len(row)>1 and width < .25:
            row = [x+(p-.5)*(.25-width)/max(.01,levels[-1]-levels[0])
                   for p,x in zip(levels,row)]
        elif width > 6 and not is_r0:
            row = [median+(x-median)*6/width for x in row]
        result.append([max(.005 if j < len(row)-1 else .01,inverse(x)) for j,x in enumerate(row)])
    other = [row for i,row in enumerate(result) if groups[i] != 3 and not auto[i]]
    if other:
        caps = [min(col) for col in zip(*other)]
        for i,g in enumerate(groups):
            if g == 3 and not auto[i]:
                result[i] = [min(x,y) for x,y in zip(result[i],caps)]
    return result


def fit_primary(states, last, groups, auto, start, fixed, depth):
    fitted = compress_states(states,last,30 if depth else 48)
    n = 16 if depth or len(fitted)*len(groups)*24 > 63360 else 24
    taus = [.5*math.exp(math.log(6000)*j/(n-1)) for j in range(n)]
    weights = obs_weights(fitted,last)
    fits, profile_weight = choose_kernels(fitted,weights,groups,auto,taus,start,fixed)
    points, sigma, jump = posterior(fitted,fits[0],taus,weights,groups,auto,start)
    return fitted, taus, weights, fits, profile_weight, points, sigma, jump


def forecast_case(case, depth=0, start=None, fixed=None):
    diagnostic = dict(origins=0, best=None, high=None, scale=DEFAULT_SCALE, stage="S1")
    try:
        return forecast_rows(case,depth,start,fixed,diagnostic)
    finally:
        if not depth:
            def value(key):
                number = diagnostic[key]
                return "-" if number is None else f"{number:.4f}"
            print(f"calib origins={diagnostic['origins']} s*={value('best')} "
                  f"s_hi={value('high')} s_cal={value('scale')} stage={diagnostic['stage']}",
                  file=sys.stderr)


def forecast_rows(case, depth, start, fixed, diagnostic):
    start = time.monotonic() if start is None else start
    if not case.get("history") or case["horizon_minutes"] <= 0:
        return safe_fallback(case, minimum=bool(case.get("history")))
    states, launch, last, unknown, missing = preprocess(case)
    target = stamp(case["target"])-launch
    checkpoint = stamp(case["at"])-launch
    if target <= last:
        return safe_fallback(case, minimum=True)
    levels = case.get("levels", LEVELS)
    groups = [group(t) for t in case["trophies"]]
    auto = [not unknown[i] and r == 100 and
            all(s.rates[i] >= 99.9 for s in states if s.p >= 1000)
            for i,r in enumerate(states[-1].rates)]
    if len(states) == 1:
        rows = r0(states,r0_clock(states,checkpoint,target),groups,auto,levels)
        return render(rows,states,groups,auto,unknown,levels,True)
    mean, clock = renewal(states, checkpoint, target)
    rich = len(states) >= 4 if depth else len(states) >= 5 and last >= 12
    scale = 1. if depth else DEFAULT_SCALE
    if not rich:
        default = {"F":.6,"T":.2,"D":.2}
        if ONLY_COMPONENT in default and not depth:
            default = {ONLY_COMPONENT:1.}
        futures = list(scenarios(states,clock,mean))
        components = pool_draws(states,futures,groups,auto,[],[],start,last)
        rows, clouds = predict(components,default,states,.08,.10,missing,auto,scale,levels,start)
        if depth:
            return rows,{name:summarize(clouds,{name:1.},levels,start) for name in clouds}
        return render(rows,states,groups,auto,unknown,levels,False)

    default = {"C":.5,"F":.3,"T":.2}
    if ONLY_COMPONENT in default and not depth:
        default = {ONLY_COMPONENT:1.}
    if depth:
        fitted, taus, _, fits, _, points, sigma, jump = fit_primary(
            states,last,groups,auto,start,fixed,depth)
        futures = list(scenarios(states,clock,mean))
        components = pool_draws(fitted,futures,groups,auto,[(fits[0],points,1.)],taus,start,last)
        rows, clouds = predict(components,default,states,sigma,jump,missing,auto,scale,levels,start)
        return rows,{name:summarize(clouds,{name:1.},levels,start) for name in clouds}

    rows = futures = None
    try:
        futures = list(scenarios(states,clock,mean))
        weights_ft = {ONLY_COMPONENT:1.} if ONLY_COMPONENT in ("F","T") else {"F":.6,"T":.4}
        components = pool_draws(states,futures,groups,auto,[],[],start,last)
        rows, _ = predict(components,weights_ft,states,.08,.10,missing,auto,scale,levels,start)
    except Exception:
        pass

    if futures is not None and time.monotonic()-start < 2.5*TIME_SCALE:
        try:
            fitted, taus, weights, fits, profile_weight, points, sigma, jump = fit_primary(
                states,last,groups,auto,start,fixed,depth)
            fit = fits[0]
            components = pool_draws(fitted,futures,groups,auto,[(fit,points,1.)],taus,start,last)
            rows, clouds = predict(components,default,states,sigma,jump,missing,auto,scale,levels,start)
            diagnostic["stage"] = "S2"
        except Exception:
            pass
    if diagnostic["stage"] == "S1":
        if rows is None:
            return safe_fallback(case)
        return render(rows,states,groups,auto,unknown,levels,False)

    if SECOND_KERNEL and time.monotonic()-start < 3.5*TIME_SCALE:
        try:
            if len(fits) > 1 and profile_weight > 0:
                other = fits[1]
                # Profile scores and variance are in percent squared.
                rate_variance = max(.0001,fit[0]/profile_weight)
                relative = math.exp(-.125*(other[0]-fit[0])/rate_variance)
                if relative >= 1e-4:
                    points2,sigma2,jump2 = posterior(fitted,other,taus,weights,groups,auto,start)
                    norm = 1+relative
                    mixture = [(fit,points,1/norm),(other,points2,relative/norm)]
                    extra = pool_draws(fitted,futures,groups,auto,mixture,taus,start,last)
                    rows, clouds = predict(extra,default,states,(sigma+relative*sigma2)/norm,
                                           (jump+relative*jump2)/norm,missing,auto,scale,levels,start)
                    diagnostic["stage"] = "S3"
        except Exception:
            pass

    if DO_BACKTEST and ONLY_COMPONENT is None and time.monotonic()-start < 4.0*TIME_SCALE:
        try:
            calibrated, scale, origins, best, high = calibrate(
                case,states,launch,last,target,auto,default,fit[1:3],start)
            if origins:
                rows = summarize(clouds,calibrated,levels,start,scale)
                diagnostic.update(origins=origins,best=best,high=high,scale=scale,stage="S4")
        except Exception:
            pass
    return render(rows,states,groups,auto,unknown,levels,False)


def validate_and_repair(case, predictions):
    levels = case.get("levels", LEVELS)
    if len(levels) != len(LEVELS):
        raise ValueError("expected five quantile levels")
    fallback = None
    trophies = []
    for i,t in enumerate(case["trophies"]):
        row = predictions[i] if i < len(predictions) else []
        if len(row) != len(levels) or any(not math.isfinite(x) for x in row):
            fallback = fallback or safe_fallback(case)
            row = fallback[i]
        rounded = [round4(clip(x,0,100)) for x in row]
        for j in range(1,len(rounded)):
            rounded[j] = max(rounded[j-1],rounded[j])
        trophies.append({"id":t["id"],"q":rounded})
    return {"trophies":trophies}


def main():
    input_dir = Path(os.environ.get("MAKEFUN_INPUT_DIR","/input"))
    output_dir = Path(os.environ.get("MAKEFUN_OUTPUT_DIR","/output"))
    output_dir.mkdir(parents=True,exist_ok=True)
    for path in sorted(input_dir.glob("*.in")):
        start = time.monotonic()
        case = None
        destination = output_dir / f"{path.stem}.out"
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            try:
                predictions = forecast_case(case,start=start)
            except Exception as error:
                print(f"{path.name}: safe fallback ({error})",file=sys.stderr)
                predictions = safe_fallback(case)
            output = validate_and_repair(case,predictions)
            payload = json.dumps(output,separators=(",",":"),allow_nan=False)+"\n"
            if len(payload.encode("utf-8")) > 65536:
                raise ValueError("forecast exceeds output size limit")
            destination.write_text(payload,encoding="utf-8")
        except Exception as error:
            print(f"{path.name}: last-resort output ({error})",file=sys.stderr)
            try:
                trophies = case.get("trophies") if isinstance(case,dict) else None
                ids = dict.fromkeys(t["id"] for t in trophies if isinstance(t,dict)
                                    and type(t.get("id")) is int) if isinstance(trophies,list) else {}
                output = {"trophies":[{"id":ident,"q":[.5]*len(LEVELS)} for ident in ids]}
                payload = json.dumps(output,separators=(",",":"),allow_nan=False)+"\n"
                destination.write_text(payload,encoding="utf-8")
            except Exception as write_error:
                print(f"{path.name}: could not write last-resort output ({write_error})",file=sys.stderr)


if __name__ == "__main__":
    main()
