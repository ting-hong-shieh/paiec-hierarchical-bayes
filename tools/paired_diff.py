"""Paired cluster-bootstrap difference in Brier ALC of 'ours' between two replica result files (same seed/pairs)."""
import json, sys
import numpy as np
W = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.1])
def load(f):
    pp = json.load(open(f))["per_pair"]
    return [(r["benchmark"], W @ np.array(r["briers"]["ours"])) for r in pp]
base = load(sys.argv[1])
for f in sys.argv[2:]:
    new = load(f)
    assert [b for b, _ in new] == [b for b, _ in base], f
    bench = np.array([b for b, _ in base]); d = np.array([n - o for (_, n), (_, o) in zip(new, base)])
    ub = np.unique(bench); idx = {x: np.where(bench == x)[0] for x in ub}
    rng = np.random.default_rng(0)
    bs = [d[np.concatenate([idx[x] for x in rng.choice(ub, len(ub))])].mean() for _ in range(2000)]
    m = np.mean([o for _, o in new])
    print(f"{f}: ALC {m:.4f}  diff {d.mean():+.4f} [{np.percentile(bs,2.5):+.4f}, {np.percentile(bs,97.5):+.4f}]  rel {d.mean()/np.mean([o for _,o in base])*100:+.2f}%")
