#!/usr/bin/env python3
"""Forecast GTA VI trophy earn rates on Exophase at a case's target.

Model, per trophy, in two parts (earners over players):
  * P(t): players tracked.  Fit ln P against ln(1 + hours since release);
    when the history is flat (no refresh seen yet) fall back to a prior for
    early launch growth: the tracked base multiplies by ~3 per refresh
    interval (~5 h), because launch-night joiners sync in waves.
  * E(t): tracked players who earned the trophy.  Fit ln E against the same
    log-time axis, shrink the slope toward the cross-trophy median slope,
    anchor the line at the last observed point, and enforce E never falls
    (earners are cumulative).  When E is flat, new joiners are assumed to
    have earned almost nothing yet except the common trophies: their earn
    rate for a trophy is a small prior growing with the trophy's own rate.

The forecast rate is 100 * E(target) / P(target), capped by a per-tier
ceiling (platinum/gold/silver/bronze).  Spreads are asymmetric on the logit
scale: the lower side is wider, because an Exophase refresh before the
target mostly dilutes rates (a wave of fresh players), while the upper side
only reflects ordinary trend uncertainty.  No randomness anywhere, so two
runs are byte-identical.  Standard library only, offline.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Per-tier ceilings on the week-one rate (percent).  A platinum needs every
# trophy, so almost nobody holds it after a week; an early bronze can be
# near-universal.
TIER_CAP = {"platinum": 10.0, "gold": 40.0, "silver": 65.0, "bronze": 97.0}

# Refresh prior, used only while the history shows no refresh at all:
# a refresh roughly triples the tracked base early in launch week, and
# refreshes arrive about every REFRESH_HOURS hours.
REFRESH_GROWTH = 3.0
REFRESH_HOURS = 5.0
# Growth of the tracked base across one refresh in the mixture branch:
# the first refresh of launch night grew it ~6x (638 -> 3,868 players),
# the backlog of launch-night syncs arriving at once.
MIXTURE_GROWTH = 6.0


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def logit_pct(x):
    p = clamp(x, 0.01, 99.99) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(l):
    if l >= 0:
        return 100.0 / (1.0 + math.exp(-l))
    e = math.exp(l)
    return 100.0 * e / (1.0 + e)


def parse_time(text):
    """ISO-8601 UTC timestamp -> epoch seconds (float)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def ols(xs, ys):
    """Least squares y ~ a + b x.  Returns (a, b, resid_sd, se_b, x_mean)."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    dof = max(n - 2, 1)
    sd = math.sqrt(sum(r * r for r in resid) / dof)
    se_b = sd / math.sqrt(sxx) if sxx > 0 else 0.0
    return (a, b, sd, se_b, mx)


def distinct_points(history, index):
    """Collapse consecutive reads with identical (players, rate) into one
    point per refresh; returns list of (hours_since_release, players, rate).
    Exophase only changes the page at a refresh, so repeats carry no signal.
    """
    pts = []
    last = None
    for read in history:
        rates = read.get("rates") or []
        if index >= len(rates) or rates[index] is None:
            continue
        key = (read.get("players_tracked"), rates[index])
        if key == last:
            continue
        last = key
        pts.append((read["_t"], read.get("players_tracked") or 0,
                    float(rates[index])))
    return pts


def mixture_quantiles(r_last, r_refresh, p_refresh):
    """Quantiles of a two-branch predictive distribution on the logit scale:
    with probability 1-p no refresh lands before the target (rate stays near
    r_last); with probability p at least one does (rate drops toward the
    diluted r_refresh, because a wave of fresh players joins).  Inverting
    the mixture CDF by bisection gives the honest quantiles -- e.g. when
    p is ~0.3 the 5% and 25% quantiles sit in the refresh branch while the
    median stays with the no-refresh branch.  Pure arithmetic, deterministic.
    """
    m_n = logit_pct(r_last)
    m_r = logit_pct(r_refresh)
    s_n, s_r = 0.35, 0.55
    lo, hi = -12.0, 12.0

    def cdf(x):
        return ((1.0 - p_refresh) * 0.5
                * (1.0 + math.erf((x - m_n) / (s_n * math.sqrt(2.0))))
                + p_refresh * 0.5
                * (1.0 + math.erf((x - m_r) / (s_r * math.sqrt(2.0)))))

    qs = []
    for tau in LEVELS:
        a, b = lo, hi
        for _ in range(80):
            mid = 0.5 * (a + b)
            if cdf(mid) < tau:
                a = mid
            else:
                b = mid
        qs.append(round(clamp(inv_logit_pct(0.5 * (a + b)), 0.01, 99.99), 4))
    for i in range(1, 5):
        if qs[i] < qs[i - 1]:
            qs[i] = qs[i - 1]
    return qs


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_h = case.get("horizon_minutes", 0) / 60.0

    # Time axis: hours since the release day began (game.release_date).
    rel = (case.get("game") or {}).get("release_date") or "2026-11-19"
    try:
        t0 = datetime.fromisoformat(rel).replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        t0 = parse_time(history[0]["at"]) if history else 0.0
    for read in history:
        read["_t"] = max(0.0, (parse_time(read["at"]) - t0) / 3600.0)
    target_t = max(0.0, (parse_time(case["target"]) - t0) / 3600.0)
    x_target = math.log1p(target_t)

    def x_of(t):
        return math.log1p(max(0.0, t))

    # ---- Players tracked: one series shared by every trophy. ----
    p_pts = []
    last_p = None
    for read in history:
        p = read.get("players_tracked") or 0
        if p != last_p:
            last_p = p
            p_pts.append((read["_t"], float(p)))
    p_last = p_pts[-1][1] if p_pts else 0.0

    # Refresh interval from the gaps between distinct reads; prior 5 h.
    if len(p_pts) >= 3:
        gaps = sorted(p_pts[i + 1][0] - p_pts[i][0]
                      for i in range(len(p_pts) - 1))
        t_refresh = clamp(gaps[len(gaps) // 2], 1.0, 48.0)
    else:
        t_refresh = REFRESH_HOURS
    # Chance at least one refresh lands before the target.
    p_refresh = 1.0 - math.exp(-horizon_h / t_refresh) if horizon_h > 0 else 0.0

    p_fit = None
    if len(p_pts) >= 3:
        p_fit = ols([x_of(t) for t, _ in p_pts],
                    [math.log(max(p, 1.0)) for _, p in p_pts])
    if p_fit is not None:
        _a, b_p, _sd, _se, _mx = p_fit
        b_p = clamp(b_p, 0.0, 2.5)
        # Anchor at the last observed point so the level is exact.
        a_p = math.log(max(p_last, 1.0)) - b_p * x_of(p_pts[-1][0])
        p_target = math.exp(a_p + b_p * x_target)
        p_target = clamp(p_target, p_last, p_last * 50.0)
        p_flat = False
    else:
        # No movement seen yet: prior growth over the expected refreshes.
        p_target = p_last * (REFRESH_GROWTH ** (horizon_h / t_refresh))
        p_target = clamp(p_target, p_last, p_last * 50.0)
        p_flat = True

    # ---- Per-trophy earner fits, with cross-trophy slope shrinkage. ----
    raw_slopes = []
    per_trophy = []
    for idx, trophy in enumerate(trophies):
        pts = distinct_points(history, idx)
        if not pts:
            per_trophy.append(None)
            continue
        r_last = pts[-1][2]
        e_last = max(r_last * p_last / 100.0, 0.0)
        fit = None
        if len(pts) >= 3:
            fit = ols([x_of(t) for t, _, _ in pts],
                      [math.log(max(r * p / 100.0, 0.5))
                       for t, p, r in pts])
        if fit is not None:
            raw_slopes.append(clamp(fit[1], -0.5, 4.0))
        per_trophy.append((pts, r_last, e_last, fit))

    if raw_slopes:
        raw_slopes.sort()
        beta_med = raw_slopes[len(raw_slopes) // 2]
    else:
        beta_med = 0.0

    rows = []
    for idx, trophy in enumerate(trophies):
        info = per_trophy[idx]
        tier = (trophy.get("tier") or "bronze").lower()
        cap = TIER_CAP.get(tier, 97.0)
        if info is None:
            rows.append({"id": trophy["id"], "q": [0.01] * 5})
            continue
        pts, r_last, e_last, fit = info
        n_pts = len(pts)
        x_last = x_of(pts[-1][0])

        if r_last >= 99.99:
            # Everybody holds it; a refresh cannot dilute it below ~everyone.
            rows.append({"id": trophy["id"],
                         "q": [99.9, 99.95, 99.99, 99.99, 99.99]})
            continue

        # New joiners' earn rate for this trophy (prior): near zero for rare
        # trophies, near one for the near-universal ones -- launch-night
        # joiners have earned almost nothing yet except the first trophies.
        f_new = min(1.0, 5.0 * (r_last / 100.0) ** 2)

        if p_flat:
            # No refresh seen yet: the forecast is a mixture over whether
            # one lands before the target.  The refresh branch dilutes the
            # rate by the player-base growth, with new joiners earning
            # almost nothing (f_new above).
            p_r = p_last * MIXTURE_GROWTH
            e_r = e_last + f_new * max(p_r - p_last, 0.0)
            r_refresh = clamp(100.0 * e_r / p_r if p_r > 0 else r_last,
                              0.01, min(cap, 99.99))
            rows.append({"id": trophy["id"],
                         "q": mixture_quantiles(r_last, r_refresh,
                                                p_refresh)})
            continue

        if fit is not None:
            _a, b_raw, sd, se_b, _mx = fit
            b_raw = clamp(b_raw, -0.5, 4.0)
            w = n_pts / (n_pts + 3.0)
            b = w * b_raw + (1.0 - w) * beta_med
            a = math.log(max(e_last, 0.5)) - b * x_last
            e_target = math.exp(a + b * x_target)
            sigma_fit = sd
            sigma_slope = se_b * abs(x_target - x_last)
        else:
            # Too few distinct points to fit: earners only grow by what
            # the new joiners earn.
            e_target = e_last + f_new * max(p_target - p_last, 0.0)
            sigma_fit = 0.25
            sigma_slope = 0.35

        # Earners never fall, and the rate cannot exceed the tier ceiling.
        e_target = max(e_target, e_last)
        rate = 100.0 * e_target / p_target if p_target > 0 else r_last
        rate = clamp(rate, 0.01, min(cap, 99.99))

        # ---- Spreads on the logit scale (asymmetric: refreshes dilute). ----
        sigma_refresh = p_refresh * 1.0
        sigma_down = math.sqrt(sigma_fit ** 2 + sigma_slope ** 2
                               + sigma_refresh ** 2 + 0.30 ** 2)
        sigma_up = math.sqrt(sigma_fit ** 2 + sigma_slope ** 2 + 0.30 ** 2)
        sigma_down = clamp(sigma_down, 0.30, 1.80)
        sigma_up = clamp(sigma_up, 0.30, 1.80)

        m = logit_pct(rate)
        qs = []
        for z in Z:
            s = sigma_down if z < 0 else sigma_up
            qs.append(round(clamp(inv_logit_pct(m + z * s), 0.01, 99.99), 4))
        # Enforce non-decreasing (rounding can only tie, never invert).
        for i in range(1, 5):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy["id"], "q": qs})

    return {"trophies": rows}


def main(argv):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    if not names:
        print("no .in files in %s" % in_dir, file=sys.stderr)
        return 1
    for path in names:
        with open(path, "r", encoding="utf-8") as fh:
            case = json.load(fh)
        cid = case.get("id") or os.path.splitext(
            os.path.basename(path))[0]
        out = forecast_case(case)
        data = json.dumps(out, separators=(",", ":")).encode("utf-8")
        if len(data) > 65536:
            print("output too large for %s" % cid, file=sys.stderr)
            return 1
        with open(os.path.join(out_dir, cid + ".out"), "wb") as fh:
            fh.write(data)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
