"""Fit the offline prior (submission/prior.json) from the aggregated measurement-db tables.

Two-way additive model on per-(subject, benchmark) log-odds accuracy:
    a_sb = c_b + theta_s + e_sb
fitted by alternating ridge least squares. Hyperparameters for the test-time model:
    m0, sc   mean / sd of benchmark intercepts c_b
    lam, su  slope and residual sd when a held-out benchmark's centred a_sb is regressed
             on leave-one-benchmark-out theta_s (sampling noise removed)
    sv       within-benchmark item-effect sd (sampling noise removed)
Usage: python tools/fit_prior.py --data data/ --out submission/prior.json [--exclude-benchmarks a,b]
"""
import argparse, json, re
import numpy as np
import pandas as pd

MIN_ITEMS = 20


def norm(x):
    return " ".join(str(x or "").strip().lower().split())


def keys(row):
    n, h, hv, e = norm(row.normalized_name), norm(row.harness), norm(row.harness_version), norm(row.reasoning_effort)
    return {"full": f"full|{n}|{h}|{hv}|{e}", "nhe": f"nhe|{n}|{h}|{e}", "ne": f"ne|{n}|{e}", "n": f"n|{n}"}


def load(data, exclude=()):
    sb = pd.read_parquet(f"{data}/sb.parquet")
    sb = sb[(sb.n_items >= MIN_ITEMS) & (sb.normalized_name.str.len() > 0)]
    sb = sb[~sb.benchmark.isin(exclude)].copy()
    sb["key"] = [keys(r)["full"] for r in sb.itertuples()]
    # Merge duplicate subject rows that share all visible attributes.
    sb = sb.groupby(["benchmark", "key"], as_index=False).agg(
        k=("k", "sum"), n=("n", "sum"), normalized_name=("normalized_name", "first"),
        harness=("harness", "first"), harness_version=("harness_version", "first"),
        reasoning_effort=("reasoning_effort", "first"), provider=("provider", "first"))
    p = (sb.k + 0.5) / (sb.n + 1.0)
    sb["a"] = np.log(p / (1 - p))
    sb["noise"] = 1.0 / (sb.n * p * (1 - p))
    nb = sb.groupby("benchmark").key.transform("count")
    return sb[nb >= 3].reset_index(drop=True)


def als(sb, tau2=1.0, iters=50):
    b_idx, b_names = pd.factorize(sb.benchmark)
    s_idx, s_names = pd.factorize(sb.key)
    a = sb.a.values
    theta = np.zeros(len(s_names))
    c = np.zeros(len(b_names))
    for _ in range(iters):
        c = np.bincount(b_idx, a - theta[s_idx]) / np.bincount(b_idx)
        r = a - c[b_idx]
        s2 = max(np.var(r - theta[s_idx]), 0.05)
        theta = np.bincount(s_idx, r, len(s_names)) / (np.bincount(s_idx, minlength=len(s_names)) + s2 / tau2)
        theta -= theta.mean()
    cnt = np.bincount(s_idx, minlength=len(s_names))
    resid = a - c[b_idx] - theta[s_idx]
    s2 = float(np.var(resid))
    var = s2 / (cnt + s2 / tau2)
    return dict(zip(b_names, c)), dict(zip(s_names, theta)), dict(zip(s_names, var)), s2


def item_sd(data, benchmarks):
    it = pd.read_parquet(f"{data}/items.parquet", columns=["benchmark", "item_id", "k", "n"])
    it = it[it.benchmark.isin(benchmarks)]
    out = []
    for b, g in it.groupby("benchmark"):
        agg = g.groupby("item_id")[["k", "n"]].sum()
        if len(agg) < 20 or agg.n.median() < 5:
            continue
        p = (agg.k + 0.5) / (agg.n + 1.0)
        lo = np.log(p / (1 - p))
        noise = (1.0 / (agg.n * p * (1 - p))).mean()
        out.append(max(lo.var() - noise, 0.0))
    return float(np.sqrt(np.median(out))), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-benchmarks", default="")
    ap.add_argument("--tau2", type=float, default=1.0)
    args = ap.parse_args()
    exclude = [x for x in args.exclude_benchmarks.split(",") if x]
    sb = load(args.data, exclude)
    c, theta, tvar, s2 = als(sb, args.tau2)
    cv = np.array(list(c.values()))

    # Leave-one-benchmark-out: how well does theta (fitted without b) predict b?
    xs, ys, noise = [], [], []
    for b in sb.benchmark.unique():
        rest = sb[sb.benchmark != b]
        _, th_b, _, _ = als(rest, args.tau2, iters=25)
        g = sb[sb.benchmark == b]
        known = g.key.isin(th_b.keys())
        if known.sum() < 3:
            continue
        g = g[known]
        x = np.array([th_b[k] for k in g.key])
        y = g.a.values
        xs.append(x - x.mean()); ys.append(y - y.mean()); noise.append(g.noise.values)
    x, y, nz = np.concatenate(xs), np.concatenate(ys), np.concatenate(noise)
    lam = float((x * y).sum() / (x * x).sum())
    resid_var = float(np.var(y - lam * x))
    su = float(np.sqrt(max(resid_var - np.median(nz), 0.05)))
    sv, n_sv = item_sd(args.data, set(sb.benchmark))

    # Lookup tables at several granularities (weighted by inverse variance).
    keyrows = sb.drop_duplicates("key")
    table = {}
    agg = {}
    for r in keyrows.itertuples():
        ks = keys(r)
        th, v = theta[r.key], tvar[r.key]
        table[ks["full"]] = [float(th), float(v)]
        for lvl in ("nhe", "ne", "n"):
            agg.setdefault(ks[lvl], []).append((th, v))
    for k, lst in agg.items():
        w = np.array([1 / v for _, v in lst]); t = np.array([t for t, _ in lst])
        table.setdefault(k, [float((w * t).sum() / w.sum()), float(1 / w.sum() + np.var(t))])
    prov = {}
    for p, g in keyrows.groupby(keyrows.provider.map(norm)):
        if not p:
            continue
        t = np.array([theta[k] for k in g.key])
        prov[p] = [float(t.mean()), float(t.var() + 1.0 / len(t))]
    tall = np.array(list(theta.values()))
    prior = {
        "m0": float(cv.mean()), "sc": float(cv.std()), "lam": lam, "su": su, "sv": sv,
        "unknown_var": float(tall.var()) * lam ** 2, "unknown_extra_var": 0.0,
        "theta": table, "provider": prov,
        "meta": {"n_benchmarks": int(sb.benchmark.nunique()), "n_subject_keys": len(theta),
                 "n_rows": int(len(sb)), "resid_s2": s2, "lobo_rows": int(len(x)),
                 "sv_benchmarks": n_sv, "excluded": exclude, "tau2": args.tau2},
    }
    with open(args.out, "w") as fh:
        json.dump(prior, fh)
    print(json.dumps({k: v for k, v in prior.items() if k not in ("theta", "provider")}, indent=1))


if __name__ == "__main__":
    main()
