"""Stress-test submission/model.py under platform-like load (no network, synthetic data).

Simulates one formative run: 30 subject-benchmark pairs on 6 benchmarks, 31 labels each,
~1000 targets, six budgets, predictions from a thread pool; also malformed inputs.
"""
import concurrent.futures as cf, copy, importlib.util, json, math, os, random, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("m", os.path.join(ROOT, "submission", "model.py"))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
prior = json.load(open(os.path.join(ROOT, "submission", "prior.json")))
names = [k.split("|", 1)[1] for k in prior["theta"] if k.startswith("n|")][:20] + ["never seen model"]
rng = random.Random(0)


def subj(name):
    return {"normalized_name": name, "provider": rng.choice(["OpenAI", "Anthropic", ""]), "release_date": "2025-01-01",
            "access_date": "", "harness": rng.choice(["", "claude-code"]), "harness_version": "",
            "reasoning_effort": rng.choice(["", "high"]), "subject_features_extra": ""}


pairs = [(subj(rng.choice(names)), f"benchmark_{i % 6}") for i in range(30)]
items = {b: [{"item_content": f"{b} question {j} " + "x" * rng.randint(10, 3000), "item_features": "",
              "interactors": "", "benchmark_id": b} for j in range(120)] for b in {p[1] for p in pairs}}
budgets = [0, 1, 3, 7, 15, 31]
acq = {i: rng.sample(items[b][:60], 31) for i, (s, b) in enumerate(pairs)}
targets = [[s, it] for i, (s, b) in enumerate(pairs) for it in items[b][60:60 + 33]]
t0 = time.time()
for bud in budgets:
    labeled = [[[pairs[i][0], it], rng.randint(0, 1)] for i in acq for it in acq[i][:bud]]
    with cf.ThreadPoolExecutor(8) as ex:
        ps = list(ex.map(lambda t: m.predict(copy.deepcopy(t), labeled), targets))
    assert all(isinstance(p, float) and math.isfinite(p) and 0 <= p <= 1 for p in ps)
    print(f"budget {bud:2d}: {len(targets)} preds, {len(labeled)} labels, mean p {sum(ps)/len(ps):.3f}, "
          f"{time.time() - t0:.1f}s")
# Malformed / edge inputs must not raise.
s0 = pairs[0][0]
odd = [
    ([{}, {}], None), ([s0, {"item_content": ""}], []),
    ([s0, items["benchmark_0"][0]], [[[s0, {"benchmark_id": "benchmark_0"}], "1"]]),
    ([s0, items["benchmark_0"][0]], [[[s0, items["benchmark_0"][1]], True], [[{}, {"benchmark_id": "benchmark_0"}], 0.0]]),
    ([s0, items["benchmark_0"][0]], [[[s0, items["benchmark_0"][1]], 2]]),
]
for inp, lab in odd:
    p = m.predict(inp, lab)
    assert 0 <= p <= 1, p
print("edge cases ok; total", f"{time.time() - t0:.1f}s")
