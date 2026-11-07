"""LaTeX macros for the LLM-prior probe (results/llm_*.json, results/llm_probe/)."""
import glob, json, os, re
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
def pp(f): return json.load(open(os.path.join(ROOT, "results", f)))
def alcs(d): return np.array([W @ np.array(r["briers"]["ours"]) for r in d["per_pair"]]), np.array([r["benchmark"] for r in d["per_pair"]])
def diff(a, b, n=2000):
    x, bench = alcs(pp(a)); y, _ = alcs(pp(b)); d = y - x
    ub = np.unique(bench); idx = {u: np.where(bench == u)[0] for u in ub}; rng = np.random.default_rng(0)
    bs = [d[np.concatenate([idx[u] for u in rng.choice(ub, len(ub))])].mean() for _ in range(n)]
    s = lambda v: ("$-$" if v < 0 else "$+$") + f"{abs(v):.4f}"
    return f"{s(d.mean())} [{s(np.percentile(bs, 2.5))}, {s(np.percentile(bs, 97.5))}]", d.mean() / x.mean() * 100
m = {}
base = pp("llm_b0.json"); m["LlmPairs"] = base["summary"]["n_pairs"]; m["LlmBench"] = base["summary"]["n_benchmarks"]
m["LlmAlcBase"] = f"{base['summary']['methods']['ours']['alc']:.4f}"
for tag, f in (("Four", "llm_b04.json"), ("Seven", "llm_b07.json"), ("Ten", "llm_b10.json")):
    m["LlmAlc" + tag] = f"{pp(f)['summary']['methods']['ours']['alc']:.4f}"
    m["LlmDiff" + tag], rel = diff("llm_b0.json", f); m["LlmRel" + tag] = f"{abs(rel):.1f}"
m["LlmBzeroBase"] = f"{base['summary']['methods']['ours']['curve'][0]:.3f}"
for tag, f in (("Four", "llm_b04.json"), ("Seven", "llm_b07.json"), ("Ten", "llm_b10.json")):
    m["LlmBzero" + tag] = f"{pp(f)['summary']['methods']['ours']['curve'][0]:.3f}"
    m["LlmBthree" + tag] = f"{pp(f)['summary']['methods']['ours']['curve'][2]:.3f}"
m["LlmBthreeBase"] = f"{base['summary']['methods']['ours']['curve'][2]:.3f}"
m["TextAlcDiff"], _ = diff("sim_per_pair.json", "exp_text_raw_b0.5.json")
# probe correlations
df = pd.read_parquet(os.path.join(ROOT, "results/llm_probe/llm_logits.parquet"))
w = df.groupby("benchmark").apply(lambda g: np.corrcoef(g.l, g.y)[0, 1])
m["LlmWithinCorr"] = f"{w.mean():.2f}"; m["LlmWithinMin"] = f"{w.min():.2f}"; m["LlmWithinMax"] = f"{w.max():.2f}"
bm = df.groupby("benchmark")[["l", "y"]].mean(); m["LlmBenchCorr"] = f"{np.corrcoef(bm.l, bm.y)[0, 1]:.2f}"
m["LlmItems"] = len(df)
to = pd.read_parquet(os.path.join(ROOT, "results/text_oof.parquet"))
tp = json.load(open(os.path.join(ROOT, "results/text_prior_full.json")))
m["TextItemRtwo"] = f"{max(v['item_r2'] for v in tp['cv'].values()):.2f}"; m["TextWithin"] = f"{max(v['within_corr'] for v in tp['cv'].values()):.2f}"
m["TextBench"] = to.benchmark.nunique()
single = os.path.join(ROOT, "results/llm_probe/single_summary.json")
if os.path.exists(single):
    for k, v in json.load(open(single)).items(): m[k] = v
with open(os.path.join(ROOT, "paper/generated/llm_macros.tex"), "w") as fh:
    for k, v in m.items():
        v = str(v)
        if v.startswith("-"):
            v = "$-$" + v[1:]
        fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(json.dumps(m, indent=1, default=str))
