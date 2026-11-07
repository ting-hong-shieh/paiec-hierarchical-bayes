"""Item-text difficulty prior: ridge regression from hashed item text to the item's pooled log-odds accuracy.

* target: logit((K + 0.5) / (N + 1)) pooled over all subjects' responses to the item
* each benchmark gets equal total weight; at most --per-bench items per benchmark
* out-of-fold predictions by benchmark (GroupKFold) are written for the local replica,
  so a benchmark's text prediction never comes from a model trained on it
* the full-data fit is written as sparse weights for submission/model.py

Usage: python tools/text_prior.py --agg AGG --items-index INDEX.json --out-weights submission/text_prior.json --out-oof results/text_oof.parquet
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import textfeat  # noqa: E402


def build(agg, index, per_bench, seed):
    it = pd.read_parquet(os.path.join(agg, "items.parquet"), columns=["item_id", "k", "n", "benchmark"])
    g = it.groupby(["benchmark", "item_id"], sort=False)[["k", "n"]].sum().reset_index()
    idx = json.load(open(index))
    rows, cols, vals, y, bench, ids = [], [], [], [], [], []
    rng = np.random.default_rng(seed)
    r = 0
    for b, gb in g.groupby("benchmark"):
        p = idx.get(b, "")
        if not str(p).startswith("/"):
            continue
        meta = pd.read_parquet(p, columns=["item_id", "content", "item_features"]).drop_duplicates("item_id")
        m = gb.merge(meta, on="item_id", how="inner")
        if len(m) < 20:
            continue
        if len(m) > per_bench:
            m = m.iloc[rng.choice(len(m), per_bench, replace=False)]
        for rec in m.itertuples(index=False):
            f = textfeat.features(rec.content, rec.item_features, "")
            rows.extend([r] * len(f)); cols.extend(f.keys()); vals.extend(f.values())
            y.append(np.log((rec.k + 0.5) / (rec.n - rec.k + 0.5)))
            bench.append(b); ids.append(rec.item_id)
            r += 1
        print(b, len(m), f"match {len(m)}/{len(gb)}", flush=True)
    X = sp.csr_matrix((vals, (rows, cols)), shape=(r, textfeat.DIM), dtype=np.float32)
    return X, np.array(y), np.array(bench), np.array(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--items-index", required=True)
    ap.add_argument("--out-weights", required=True)
    ap.add_argument("--out-oof", required=True)
    ap.add_argument("--per-bench", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    X, y, bench, ids = build(args.agg, args.items_index, args.per_bench, args.seed)
    ub, inv, cnt = np.unique(bench, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv]
    w = w * len(w) / w.sum()
    print("items", X.shape, "benchmarks", len(ub), f"{time.time() - t0:.0f}s", flush=True)
    gkf = GroupKFold(n_splits=5)
    res = {}
    for alpha in (3.0, 10.0, 30.0, 100.0):
        oof = np.zeros_like(y)
        for tr, te in gkf.split(X, y, bench):
            mdl = Ridge(alpha=alpha, solver="sparse_cg", max_iter=300).fit(X[tr], y[tr], sample_weight=w[tr])
            oof[te] = mdl.predict(X[te])
        # weighted R2 at the item level and at the benchmark-mean level
        r2 = 1 - np.sum(w * (y - oof) ** 2) / np.sum(w * (y - np.average(y, weights=w)) ** 2)
        bm = pd.DataFrame({"b": bench, "y": y, "p": oof}).groupby("b").mean()
        r2b = 1 - np.sum((bm.y - bm.p) ** 2) / np.sum((bm.y - bm.y.mean()) ** 2)
        within = pd.DataFrame({"b": bench, "y": y, "p": oof})
        within["yc"] = within.y - within.groupby("b").y.transform("mean")
        within["pc"] = within.p - within.groupby("b").p.transform("mean")
        cw = np.corrcoef(within.yc, within.pc)[0, 1]
        res[alpha] = (r2, r2b, cw, oof)
        print(f"alpha {alpha}: item R2 {r2:.3f}  bench-mean R2 {r2b:.3f}  within-bench corr {cw:.3f}", flush=True)
    best = max(res, key=lambda a: res[a][0])
    oof = res[best][3]
    os.makedirs(os.path.dirname(args.out_oof), exist_ok=True)
    pd.DataFrame({"benchmark": bench, "item_id": ids, "y": y, "oof": oof}).to_parquet(args.out_oof)
    mdl = Ridge(alpha=best, solver="sparse_cg", max_iter=300).fit(X, y, sample_weight=w)
    coef = mdl.coef_.astype(float)
    nz = np.nonzero(np.abs(coef) > 1e-4)[0]
    center = float(np.average(y, weights=w))
    out = {"alpha": best, "intercept": float(mdl.intercept_), "center": center,
           "w": {int(i): round(float(coef[i]), 5) for i in nz},
           "cv": {str(a): {"item_r2": round(v[0], 4), "bench_r2": round(v[1], 4), "within_corr": round(v[2], 4)}
                  for a, v in res.items()}}
    json.dump(out, open(args.out_weights, "w"), separators=(",", ":"))
    print("best alpha", best, "nonzero", len(nz), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
