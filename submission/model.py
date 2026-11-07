"""PAIEC predictor: IRT subject prior + hierarchical Bayesian test-time update.

For a held-out benchmark b, the log-odds that subject s answers item i is

    eta(s, i) = c_b + lam * theta_s + u_{s,b} + v_i

* theta_s   subject ability fitted offline on measurement-db (prior.json), looked up
            by the subject's visible attributes; unknown subjects fall back to the
            provider mean or 0 with extra variance.
* c_b       benchmark intercept, prior N(m0, sc^2) estimated across training benchmarks.
* u_{s,b}   subject-by-benchmark deviation, prior N(0, su^2).
* v_i       item effect, prior N(0, sv^2); identified only when the same item was
            revealed for another subject.

At test time all revealed labels on the same anonymous benchmark_id (any subject)
are pooled; the MAP of (c_b, u, v) is found by coordinate-wise Newton steps and the
prediction integrates over the Laplace variance with the probit approximation
sigmoid(mu / sqrt(1 + pi * var / 8)). The model uses only the standard library,
makes no network calls and keeps no state other than a cache of fits keyed by the
evidence it was given.
"""

import hashlib
import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_PRIOR = None
_FIT_CACHE = {}
_CACHE_MAX = 256
_CLIP = (0.01, 0.99)


def _load_prior():
    global _PRIOR
    if _PRIOR is None:
        with open(os.path.join(_HERE, "prior.json"), encoding="utf-8") as fh:
            _PRIOR = json.load(fh)
    return _PRIOR


def _sigmoid(x):
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


def _norm(value):
    return " ".join(str(value or "").strip().lower().split())


def subject_keys(subject):
    """Lookup keys from most to least specific (must match tools/fit_prior.py)."""
    name = _norm(subject.get("normalized_name"))
    harness = _norm(subject.get("harness"))
    effort = _norm(subject.get("reasoning_effort"))
    return [
        "full|" + "|".join([name, harness, _norm(subject.get("harness_version")), effort]),
        "nhe|" + "|".join([name, harness, effort]),
        "ne|" + "|".join([name, effort]),
        "n|" + name,
    ]


def subject_prior(subject):
    """Return (prior mean, prior variance) of lam*theta_s for a subject."""
    prior = _load_prior()
    table = prior["theta"]
    for key in subject_keys(subject):
        if key in table:
            theta, var = table[key]
            return prior["lam"] * theta, prior["lam"] ** 2 * var
    provider = _norm(subject.get("provider"))
    if provider and provider in prior["provider"]:
        theta, var = prior["provider"][provider]
        return prior["lam"] * theta, prior["lam"] ** 2 * var + prior["unknown_extra_var"]
    return 0.0, prior["unknown_var"]


def _subject_id(subject):
    return json.dumps([subject.get(k, "") for k in sorted(subject)], ensure_ascii=False)


def _item_id(item):
    text = "\x1f".join([str(item.get("item_content", "")), str(item.get("item_features", "")),
                        str(item.get("interactors", ""))])
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _fit(benchmark, rows):
    """MAP + diagonal Laplace variances of c, u_s, v_i from rows on one benchmark.

    rows: list of (subject_id, item_id, offset, label) where offset = lam*theta_s.
    """
    prior = _load_prior()
    m0, vc, vu, vv = prior["m0"], prior["sc"] ** 2, prior["su"] ** 2, prior["sv"] ** 2
    subjects = sorted({r[0] for r in rows})
    items = sorted({r[1] for r in rows})
    c = m0
    u = {s: 0.0 for s in subjects}
    v = {i: 0.0 for i in items}
    by_s = {s: [] for s in subjects}
    by_i = {i: [] for i in items}
    for idx, r in enumerate(rows):
        by_s[r[0]].append(idx)
        by_i[r[1]].append(idx)

    def newton(value, prior_mean, prior_var, idxs, extra):
        grad = -(value - prior_mean) / prior_var
        hess = 1.0 / prior_var
        for k in idxs:
            s_id, i_id, off, y = rows[k]
            p = _sigmoid(extra(k) + value)
            grad += y - p
            hess += p * (1.0 - p)
        return value + max(-2.0, min(2.0, grad / hess)), hess

    hc = 1.0 / vc
    hu = {s: 1.0 / vu for s in subjects}
    hv = {i: 1.0 / vv for i in items}
    for _ in range(25):
        old = c
        c, hc = newton(c, m0, vc, range(len(rows)),
                       lambda k: rows[k][2] + u[rows[k][0]] + v[rows[k][1]])
        for s in subjects:
            u[s], hu[s] = newton(u[s], 0.0, vu, by_s[s],
                                 lambda k: rows[k][2] + c + v[rows[k][1]])
        for i in items:
            v[i], hv[i] = newton(v[i], 0.0, vv, by_i[i],
                                 lambda k: rows[k][2] + c + u[rows[k][0]])
        if abs(c - old) < 1e-4:
            break
    return {"c": (c, 1.0 / hc), "u": {s: (u[s], 1.0 / hu[s]) for s in subjects},
            "v": {i: (v[i], 1.0 / hv[i]) for i in items}}


def _evidence_rows(benchmark, labeled):
    rows = []
    for (obs_subject, obs_item), label in labeled or []:
        if obs_item.get("benchmark_id", "") != benchmark:
            continue
        try:
            y = float(label)
        except (TypeError, ValueError):
            continue
        if y not in (0.0, 1.0):
            continue
        off, _ = subject_prior(obs_subject)
        rows.append((_subject_id(obs_subject), _item_id(obs_item), off, y))
    return rows


_LAST = (None, {})


def _rows_for(benchmark, labeled):
    """Evidence rows and cache key; memoised on the identity of the labeled list.

    A reference to the list is kept (so its id cannot be reused while memoised) and
    the (list, memo) pair is swapped in one assignment, so concurrent threads never
    read rows that belong to a different list.
    """
    global _LAST
    if not benchmark or not labeled:
        return [], None
    last = _LAST
    if last[0] is not labeled:
        last = (labeled, {})
        _LAST = last
    hit = last[1].get(benchmark)
    if hit is None:
        rows = _evidence_rows(benchmark, labeled)
        key = hashlib.sha1(json.dumps([benchmark, sorted(rows)]).encode("utf-8")).hexdigest()
        hit = (rows, key)
        last[1][benchmark] = hit
    return hit


def predict(input, labeled=None):
    """Return P(response = 1) for input = [subject, item]."""
    subject, item = input[0], input[1]
    prior = _load_prior()
    off, var_theta = subject_prior(subject)
    benchmark = item.get("benchmark_id", "")
    rows, key = _rows_for(benchmark, labeled)
    if not rows:
        mu = prior["m0"] + off
        var = prior["sc"] ** 2 + prior["su"] ** 2 + prior["sv"] ** 2 + var_theta
    else:
        fit = _FIT_CACHE.get(key)
        if fit is None:
            fit = _fit(benchmark, rows)
            if len(_FIT_CACHE) >= _CACHE_MAX:
                _FIT_CACHE.clear()
            _FIT_CACHE[key] = fit
        c, vc = fit["c"]
        u, vu = fit["u"].get(_subject_id(subject), (0.0, prior["su"] ** 2))
        v, vv = fit["v"].get(_item_id(item), (0.0, prior["sv"] ** 2))
        mu = c + off + u + v
        # theta uncertainty is absorbed by u once the subject has labels.
        var = vc + vu + vv + (var_theta if _subject_id(subject) not in fit["u"] else 0.0)
    p = _sigmoid(mu / math.sqrt(1.0 + math.pi * var / 8.0))
    return float(min(_CLIP[1], max(_CLIP[0], p)))
