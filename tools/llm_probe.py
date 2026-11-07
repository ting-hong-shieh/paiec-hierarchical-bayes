"""Build the LLM-difficulty probe: 80 random items from each of 8 held-out benchmarks, prompts in
batches of 20 (results/llm_probe/batch_*.txt) and one-item prompts for 10 items per benchmark
(results/llm_probe/single/*.txt). The prompts were answered with `codex exec` (GPT-6-astra);
`--collect` parses the answers into results/llm_probe/llm_logits.parquet.
Prompt files contain item text from measurement-db and are not redistributed.
Usage: python tools/llm_probe.py --items-index INDEX.json [--collect]"""
import argparse, glob, json, os, re
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "llm_probe")
BENCHES = ["medhelm_med_qa", "babilong", "clasheval", "evouna", "sgrades", "tulu_human_eval", "agi_elo", "medhelm_med_mcqa"]
BATCH = ("You estimate item difficulty for AI evaluation. Below are items from one benchmark. For each item, estimate the probability "
         "that a typical AI system evaluated on this benchmark (a mix of 2023-2026 LLMs and agents, weak to frontier) gets it right. "
         "Use the task type, answer format, required reasoning depth, and length. Do not solve the items in detail. "
         "Return ONLY a JSON object mapping each item id to a probability between 0.01 and 0.99.\n\n")
SINGLE = ("Estimate the probability that a typical AI system evaluated on this benchmark (a mix of 2023-2026 LLMs and agents, weak to frontier) "
          "answers the following item correctly. Consider the task type, answer format, required reasoning depth and length. Do not solve it in detail. "
          "Reply with only a number between 0.01 and 0.99.\n\nITEM:\n")

def build(index):
    to = pd.read_parquet(os.path.join(ROOT, "results/text_oof.parquet")); idx = json.load(open(index)); out = []
    for b in BENCHES:
        meta = pd.read_parquet(idx[b], columns=["item_id", "content"]).drop_duplicates("item_id")
        g = to[to.benchmark == b].merge(meta, on="item_id").sample(n=80, random_state=0)
        out += [{"benchmark": b, "item_id": r.item_id, "y": float(r.y), "content": str(r.content)[:1500]} for r in g.itertuples()]
    df = pd.DataFrame(out); os.makedirs(os.path.join(OUT, "single"), exist_ok=True); df.to_parquet(os.path.join(OUT, "items.parquet"))
    k = 0
    for b, g in df.groupby("benchmark"):
        for i in range(0, len(g), 20):
            items = "\n\n".join(f"### ITEM {r.item_id}\n{r.content}" for r in g.iloc[i:i + 20].itertuples())
            open(os.path.join(OUT, f"batch_{k:03d}.txt"), "w").write(BATCH + items); k += 1
    for r in df.groupby("benchmark").head(10).itertuples():
        open(os.path.join(OUT, "single", f"{r.item_id}.txt"), "w").write(SINGLE + r.content)

def collect():
    df = pd.read_parquet(os.path.join(OUT, "items.parquet")); P = {}
    for f in glob.glob(os.path.join(OUT, "batch_*.txt.out")):
        P.update(json.loads(re.search(r"\{.*\}", open(f).read(), re.S).group(0)))
    df["p"] = df.item_id.map(P).clip(0.01, 0.99); df["l"] = np.log(df.p / (1 - df.p))
    df[["benchmark", "item_id", "y", "p", "l"]].to_parquet(os.path.join(OUT, "llm_logits.parquet"))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--items-index", default=""); ap.add_argument("--collect", action="store_true")
    a = ap.parse_args(); collect() if a.collect else build(a.items_index)
