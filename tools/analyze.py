"""Tables and figure for the report from results/*.json.

Writes paper/generated/*.tex (LaTeX table bodies and macros) and paper/figures/learning_curve.pdf.
Differences use a cluster bootstrap over benchmarks (2,000 resamples).
"""
import glob, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "paper", "figures"))
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
GEN = os.path.join(ROOT, "paper", "generated")
os.makedirs(GEN, exist_ok=True)


def load(name):
    with open(os.path.join(ROOT, "results", name)) as fh:
        return json.load(fh)


def alc_arrays(pp):
    names = pp[0]["briers"].keys()
    return {m: np.array([W @ np.array(r["briers"][m]) for r in pp]) for m in names}


def boot_diff(pp, a, b, n=2000, seed=0):
    al = alc_arrays(pp)
    bench = np.array([r["benchmark"] for r in pp])
    ub = np.unique(bench)
    idx = {x: np.where(bench == x)[0] for x in ub}
    rng = np.random.default_rng(seed)
    d = al[a] - al[b]
    bs = [d[np.concatenate([idx[x] for x in rng.choice(ub, len(ub))])].mean() for _ in range(n)]
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def f4(x):
    return f"{x:.4f}".replace("-", "--") if x < 0 else f"{x:.4f}"


def sgn(x):
    return ("$-$" if x < 0 else "$+$") + f"{abs(x):.4f}"


LABEL = {"const_0.5": "Constant 0.5", "empirical_mean": "Empirical mean (official)",
         "ours": "Ours (full)", "ours+no_subject_prior": "\\quad w/o subject prior",
         "ours_no_pool": "\\quad w/o pooling",
         "ours_no_pool+no_subject_prior": "\\quad w/o both"}
ORDER = ["const_0.5", "empirical_mean", "ours", "ours+no_subject_prior", "ours_no_pool",
         "ours_no_pool+no_subject_prior"]


def main_table():
    rows = []
    d = load("sim_per_pair.json")
    s = d["summary"]["methods"]
    for m in ORDER:
        c = s[m]["curve"]
        lo, hi = s[m]["alc_ci95"]
        rows.append(f"{LABEL[m]} & {s[m]['alc']:.4f} & [{lo:.3f}, {hi:.3f}] & "
                    + " & ".join(f"{x:.3f}" for x in c) + " \\\\")
    with open(os.path.join(GEN, "main_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lcccccccc}\n\\toprule\nPredictor & ALC & 95\\% CI & $B_0$ & $B_1$ & $B_3$ & $B_7$ & $B_{15}$ & $B_{31}$ \\\\\n\\midrule\n"
                 + "\n".join(rows[:2]) + "\n\\midrule\n" + "\n".join(rows[2:]) + "\n\\bottomrule\n\\end{tabular}\n")
    pp = d["per_pair"]
    macros = {
        "NPairs": d["summary"]["n_pairs"], "NBench": d["summary"]["n_benchmarks"],
        "AlcOurs": f"{s['ours']['alc']:.4f}", "AlcConst": f"{s['const_0.5']['alc']:.4f}",
        "AlcMean": f"{s['empirical_mean']['alc']:.4f}",
    }
    for key, (a, b) in {"DiffConst": ("ours", "const_0.5"), "DiffMean": ("ours", "empirical_mean"),
                        "DiffPool": ("ours", "ours_no_pool"), "DiffSubj": ("ours", "ours+no_subject_prior"),
                        "DiffSubjNoPool": ("ours_no_pool", "ours_no_pool+no_subject_prior")}.items():
        m, lo, hi = boot_diff(pp, a, b)
        macros[key] = f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]"
    # Shared split (item effects never identified for evaluation items).
    sh = load("sim_shared.json")
    macros["AlcOursShared"] = f"{sh['summary']['methods']['ours']['alc']:.4f}"
    m, lo, hi = boot_diff(sh["per_pair"], "ours", "ours_no_pool")
    macros["DiffPoolShared"] = f"{sgn(m)} [{sgn(lo)}, {sgn(hi)}]"
    # Seeds.
    seeds = [s["ours"]["alc"]]
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "sim_per_pair_seed*.json"))):
        seeds.append(json.load(open(f))["summary"]["methods"]["ours"]["alc"])
    macros["NSeeds"] = len(seeds)
    macros["SeedRange"] = f"{min(seeds):.4f}--{max(seeds):.4f}"
    # Per-pair win rate of ours vs const at each budget.
    al = alc_arrays(pp)
    macros["WinConst"] = f"{(al['ours'] < al['const_0.5']).mean() * 100:.0f}"
    macros["WinMean"] = f"{(al['ours'] < al['empirical_mean']).mean() * 100:.0f}"
    with open(os.path.join(GEN, "macros.tex"), "w") as fh:
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    return d, macros


def sens_table():
    base = load("sim_per_pair.json")["summary"]["methods"]["ours"]["alc"]
    rows = []
    for hp, tex in (("sc", "$\\sigma_c$"), ("su", "$\\sigma_u$"), ("sv", "$\\sigma_v$")):
        cells = []
        for x in ("0.5", "2.0"):
            f = os.path.join(ROOT, "results", f"sens_{hp}_{x}.json")
            if os.path.exists(f):
                cells.append(f"{json.load(open(f))['summary']['methods']['ours']['alc']:.4f}")
            else:
                cells.append("n/a")
        rows.append(f"{tex} & {cells[0]} & {base:.4f} & {cells[1]} \\\\")
    with open(os.path.join(GEN, "sens_table.tex"), "w") as fh:
        fh.write("\\begin{tabular}{lccc}\n\\toprule\nScale & $\\times 0.5$ & $\\times 1$ (submitted) & $\\times 2$ \\\\\n\\midrule\n"
                 + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")


def figure(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import figstyle as fs
    plt.rcParams.update(fs.rc(7.5))
    s = d["summary"]["methods"]
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    series = [("const_0.5", fs.GREY, "Constant 0.5", "s", ":"),
              ("empirical_mean", fs.VERMILION, "Empirical mean", "^", "-"),
              ("ours_no_pool", fs.BLUE, "Ours, no pooling", "D", "-"),
              ("ours", fs.NAVY, "Ours (full)", "o", "-")]
    pp = d["per_pair"]
    for key, col, lab, mk, ls in series:
        arr = np.array([r["briers"][key] for r in pp])
        mean = arr.mean(0)
        ax.plot(x, mean, color=col, marker=mk, ms=3, ls=ls, lw=1.1)
        se = arr.std(0, ddof=1) / np.sqrt(len(arr))
        ax.fill_between(x, mean - 1.96 * se, mean + 1.96 * se, color=col, alpha=0.15, lw=0)
    # Direct labels on the right.
    ends = {"const_0.5": 0.250, "ours_no_pool": 0.2160, "empirical_mean": 0.1975, "ours": 0.1790}
    for key, col, lab, mk, ls in series:
        ax.text(5.15, ends[key], lab, color=col, va="center", fontsize=7.2,
                bbox=dict(fc="white", ec="none", pad=0.3))
    ax.set_xticks(x, ["0", "1", "3", "7", "15", "31"])
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(0.165, 0.39)
    ax.set_xlabel("Label budget per pair (log$_2$(1+n) spacing)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.grid(axis="y")
    ax.spines["bottom"].set_bounds(0, 5)
    fig.savefig(os.path.join(ROOT, "paper", "figures", "learning_curve.pdf"))
    fig.savefig(os.path.join(ROOT, "paper", "figures", "learning_curve.png"), dpi=300)


if __name__ == "__main__":
    d, macros = main_table()
    sens_table()
    figure(d)
    print(json.dumps(macros, indent=1))
