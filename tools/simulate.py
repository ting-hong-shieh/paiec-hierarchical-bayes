"""Local replica of the PAIEC streaming evaluation on held-out training benchmarks.

Benchmarks are split into K folds. For each fold, the prior is refitted without the
fold's benchmarks (benchmark-level cold start), then every fold benchmark is scored:
  * pairs = subjects with >= 80 distinct items on the benchmark (up to --max-subjects sampled)
  * each pair's items are split 50/50 into acquisition / evaluation pools
    (--split per_pair: independent split per pair; shared: one split per benchmark)
  * random acquisition order; budgets 0,1,3,7,15,31 are nested prefixes
  * labeled = revealed labels of ALL sampled pairs on that benchmark (as on the platform)
  * a revealed label is one Bernoulli draw from the item's recorded responses
  * Brier is computed over all recorded responses of the evaluation items
Brier ALC = 0.1 B0 + 0.2 (B1 + B3 + B7 + B15) + 0.1 B31, averaged equally over pairs.
Usage: python tools/simulate.py --data data/ --out results/sim.json
"""
import argparse, importlib.util, json, os, random, sys, tempfile, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fit_prior  # noqa: E402

BUDGETS = [0, 1, 3, 7, 15, 31]
WEIGHTS = [0.1, 0.2, 0.2, 0.2, 0.2, 0.1]
SUBJ_COLS = ["normalized_name", "provider", "release_date", "access_date", "harness",
             "harness_version", "reasoning_effort", "subject_features_extra"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


MODEL_PATH = os.path.join(ROOT, "submission", "model.py")


def load_model():
    spec = importlib.util.spec_from_file_location("paiec_model", MODEL_PATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fit_prior_dict(data, exclude, tau2=1.0):
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        path = fh.name
    argv = sys.argv
    sys.argv = ["fit_prior", "--data", data, "--out", path, "--exclude-benchmarks", ",".join(exclude),
                "--tau2", str(tau2)]
    try:
        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            fit_prior.main()
    finally:
        sys.argv = argv
    with open(path) as fh:
        prior = json.load(fh)
    os.unlink(path)
    return prior


def methods(model):
    def const(inp, lab):
        return 0.5

    def emp_mean(inp, lab):
        s, it = inp
        ys = [y for (os_, oi), y in lab if os_ == s and oi["benchmark_id"] == it["benchmark_id"]]
        return sum(ys) / len(ys) if ys else 0.5

    def ours(inp, lab):
        return model.predict(inp, lab)

    def ours_no_pool(inp, lab):
        s = inp[0]
        return model.predict(inp, [e for e in lab if e[0][0] == s])

    return {"const_0.5": const, "empirical_mean": emp_mean, "ours": ours, "ours_no_pool": ours_no_pool}


def run(args):
    sb = pd.read_parquet(f"{args.data}/sb.parquet")
    items = pd.read_parquet(f"{args.data}/items.parquet")
    eligible = fit_prior.load(args.data).benchmark.unique()
    # Benchmarks that can host at least one pair with >= 80 distinct items.
    cand = sorted(set(sb[(sb.n_items >= 80)].benchmark) & set(eligible))
    rng = random.Random(args.seed)
    rng.shuffle(cand)
    folds = [cand[i::args.folds] for i in range(args.folds)]
    model = load_model()
    meths = methods(model)
    text_oof = None
    if args.text_beta and args.text_oof:
        to = pd.read_parquet(args.text_oof)
        text_center = float(to.groupby("benchmark").y.mean().mean())
        text_oof = {b: dict(zip(g.item_id, g.oof)) for b, g in to.groupby("benchmark")}
        text_bmean = to.groupby("benchmark").oof.mean().to_dict()
    probe = None
    if args.llm_probe:
        pr_ = pd.read_parquet(args.llm_probe)
        probe = {b: dict(zip(g.item_id, g.l)) for b, g in pr_.groupby("benchmark")}
    per_pair = []
    t0 = time.time()
    for f, fold in enumerate(folds):
        prior = fit_prior_dict(args.data, fold, args.tau2)
        for hp in ("sc", "su", "sv"):
            prior[hp] *= getattr(args, "scale_" + hp)
        if probe is not None:
            prior["text_beta"] = args.text_beta
            prior["text_center"] = args.llm_center
        if text_oof is not None:
            prior["text_beta"] = args.text_beta
            prior["text_center"] = text_center
        prior_nosubj = dict(prior, theta={}, provider={}, unknown_var=0.0)
        for b in fold:
            if probe is not None and b not in probe:
                continue
            g = sb[(sb.benchmark == b) & (sb.n_items >= 80) & (sb.normalized_name.str.len() > 0)]
            g = g.drop_duplicates(subset=SUBJ_COLS)
            subs = g.to_dict("records")
            rng.shuffle(subs)
            subs = subs[: args.max_subjects]
            ib = items[items.benchmark == b]
            if probe is not None:
                d = probe[b]; bm = float(np.mean(list(d.values())))
                model._TEXT_PRED = {x: d.get(x, bm) for x in ib.item_id.unique()}
            if text_oof is not None:
                d = text_oof.get(b, {})
                bm = text_bmean.get(b, text_center)
                if args.text_center_bench:
                    model._TEXT_PRED = {x: d.get(x, bm) - bm + text_center for x in ib.item_id.unique()}
                else:
                    model._TEXT_PRED = {x: d.get(x, bm) for x in ib.item_id.unique()}
            pairs = []
            shared_perm = None
            for s in subs:
                si = ib[ib.subject_id == s["subject_id"]]
                ids = sorted(si.item_id.unique())
                if args.split == "shared":
                    if shared_perm is None:
                        allids = sorted(ib.item_id.unique()); rng.shuffle(allids)
                        shared_perm = {x: i % 2 for i, x in enumerate(allids)}
                    acq = [x for x in ids if shared_perm[x] == 0]
                    ev = [x for x in ids if shared_perm[x] == 1]
                    rng.shuffle(acq)
                else:
                    rng.shuffle(ids)
                    half = len(ids) // 2
                    acq, ev = ids[:half], ids[half:]
                if probe is not None:
                    ev = [x for x in ev if x in probe[b]]
                ev = ev[: args.max_eval]
                if not ev:
                    continue
                kn = si.set_index("item_id")[["k", "n"]]
                subj = {c: str(s.get(c, "") or "") for c in SUBJ_COLS}
                reveals = []
                for x in acq[:31]:
                    k, n = kn.loc[x, "k"], kn.loc[x, "n"]
                    y = 1 if rng.random() < k / n else 0
                    reveals.append([[subj, {"item_content": x, "item_features": "", "interactors": "",
                                            "benchmark_id": b}], y])
                targets = [({"item_content": x, "item_features": "", "interactors": "", "benchmark_id": b},
                            float(kn.loc[x, "k"]), float(kn.loc[x, "n"])) for x in ev]
                pairs.append((subj, reveals, targets))
            if not pairs:
                continue
            res = {}
            variants = (("full", prior),) if args.only_ours else (("full", prior), ("no_subject_prior", prior_nosubj))
            for vname, pr in variants:
                model._PRIOR = pr
                model._FIT_CACHE.clear()
                for mname, fn in meths.items():
                    if vname != "full" and mname in ("const_0.5", "empirical_mean"):
                        continue
                    if args.only_ours and mname not in ("const_0.5", "ours"):
                        continue
                    label = mname if vname == "full" else f"{mname}+{vname}"
                    for pi, (subj, _, targets) in enumerate(pairs):
                        briers = []
                        for bud in BUDGETS:
                            lab = [e for (_, rv, _) in pairs for e in rv[:bud]]
                            se = tot = 0.0
                            for it, k, n in targets:
                                p = fn([subj, it], lab)
                                se += k * (1 - p) ** 2 + (n - k) * p ** 2
                                tot += n
                            briers.append(se / tot)
                        res.setdefault(pi, {})[label] = briers
            for pi, d in res.items():
                per_pair.append({"benchmark": b, "fold": f, "n_pairs_on_benchmark": len(pairs),
                                 "subject": pairs[pi][0]["normalized_name"], "n_eval": len(pairs[pi][2]),
                                 "briers": d})
            print(f"fold {f} {b}: {len(pairs)} pairs ({time.time() - t0:.0f}s)", flush=True)
    return per_pair


def summarize(per_pair, n_boot=2000, seed=0):
    names = sorted(per_pair[0]["briers"].keys())
    alc = {m: np.array([sum(w * x for w, x in zip(WEIGHTS, r["briers"][m])) for r in per_pair]) for m in names}
    curves = {m: np.array([r["briers"][m] for r in per_pair]).mean(0).tolist() for m in names}
    # Cluster bootstrap over benchmarks (pairs on one benchmark are not independent).
    bench = np.array([r["benchmark"] for r in per_pair])
    ub = np.unique(bench)
    idx = {b: np.where(bench == b)[0] for b in ub}
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in names}
    diffs = {m: [] for m in names}
    for _ in range(n_boot):
        pick = np.concatenate([idx[b] for b in rng.choice(ub, len(ub))])
        for m in names:
            boots[m].append(alc[m][pick].mean())
            diffs[m].append((alc[m][pick] - alc["const_0.5"][pick]).mean())
    out = {"n_pairs": len(per_pair), "n_benchmarks": int(len(ub)), "methods": {}}
    for m in names:
        out["methods"][m] = {"alc": float(alc[m].mean()),
                             "alc_ci95": [float(np.percentile(boots[m], 2.5)), float(np.percentile(boots[m], 97.5))],
                             "diff_vs_const_ci95": [float(np.percentile(diffs[m], 2.5)), float(np.percentile(diffs[m], 97.5))],
                             "curve": curves[m]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-subjects", type=int, default=8)
    ap.add_argument("--max-eval", type=int, default=150)
    ap.add_argument("--split", choices=["per_pair", "shared"], default="per_pair")
    ap.add_argument("--tau2", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    for hp in ("sc", "su", "sv"):
        ap.add_argument("--scale-" + hp, type=float, default=1.0)
    ap.add_argument("--only-ours", action="store_true")
    ap.add_argument("--model", default="", help="model file (default submission/model.py); "
                    "the text/LLM offset runs need experiments/model_with_offset.py")
    ap.add_argument("--text-beta", type=float, default=0.0)
    ap.add_argument("--text-oof", default="")
    ap.add_argument("--text-center-bench", action="store_true")
    ap.add_argument("--llm-probe", default="", help="parquet benchmark,item_id,l: restrict to these benchmarks and eval items")
    ap.add_argument("--llm-center", type=float, default=1.67)
    args = ap.parse_args()
    global MODEL_PATH
    if args.model:
        MODEL_PATH = os.path.abspath(args.model)
    elif args.text_beta:
        raise SystemExit("--text-beta needs --model experiments/model_with_offset.py")
    per_pair = run(args)
    summary = summarize(per_pair)
    summary["args"] = vars(args)
    with open(args.out, "w") as fh:
        json.dump({"summary": summary, "per_pair": per_pair}, fh, indent=1)
    for m, d in summary["methods"].items():
        print(f"{m:32s} ALC {d['alc']:.4f} [{d['alc_ci95'][0]:.4f},{d['alc_ci95'][1]:.4f}]  "
              + " ".join(f"{x:.3f}" for x in d["curve"]))


if __name__ == "__main__":
    main()
