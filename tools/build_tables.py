"""Aggregate measurement-db binary item-level benchmarks into compact tables.

Output (parquet, in --out):
  sb.parquet    one row per (benchmark, subject): k successes, n responses, subject attributes
  items.parquet one row per (benchmark, subject, item): k, n  (used by the local simulator)
Usage: python tools/build_tables.py --cache /mnt/d/Data/paiec-measurement-db --out data/
"""
import argparse, glob, os
import pandas as pd

SUBJ_COLS = ["normalized_name", "provider", "release_date", "access_date", "harness",
             "harness_version", "reasoning_effort", "subject_features_extra"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    bfiles = sorted(glob.glob(os.path.join(a.cache, "datasets--*/snapshots/*/*/formatted_tables/benchmarks.parquet")))
    sb_parts, it_parts, cat = [], [], []
    for bf in bfiles:
        d = os.path.dirname(bf)
        b = pd.read_parquet(bf).iloc[0]
        ok = b["response_type"] == "binary" and b["granularity"] == "item" and b["release"] == "public"
        cat.append({"dir": d.split("/")[-2], "benchmark_id": b["benchmark_id"], "eligible": bool(ok),
                    "n_subjects": int(b["n_subjects"]), "n_items": int(b["n_items"]),
                    "n_responses": int(b["n_responses"]), "license": b["license"]})
        rf, sf = os.path.join(d, "responses.parquet"), os.path.join(d, "subjects.parquet")
        if not ok or not (os.path.exists(rf) and os.path.exists(sf)):
            continue
        r = pd.read_parquet(rf, columns=["subject_id", "item_id", "response"])
        r = r[r["response"].isin([0.0, 1.0])]
        if r.empty:
            continue
        s = pd.read_parquet(sf)
        for c in SUBJ_COLS:
            if c not in s.columns:
                s[c] = ""
        s = s[["subject_id"] + SUBJ_COLS].fillna("").astype(str).drop_duplicates("subject_id")
        g = r.groupby(["subject_id", "item_id"])["response"].agg(k="sum", n="count").reset_index()
        g["benchmark"] = b["benchmark_id"]
        it_parts.append(g)
        sb = g.groupby("subject_id")[["k", "n"]].sum().reset_index()
        sb["n_items"] = g.groupby("subject_id").size().values
        sb["benchmark"] = b["benchmark_id"]
        sb_parts.append(sb.merge(s, on="subject_id", how="left"))
    pd.DataFrame(cat).to_csv(os.path.join(a.out, "catalog.csv"), index=False)
    sb = pd.concat(sb_parts, ignore_index=True)
    for c in SUBJ_COLS:
        sb[c] = sb[c].fillna("").astype(str)
    sb.to_parquet(os.path.join(a.out, "sb.parquet"))
    pd.concat(it_parts, ignore_index=True).to_parquet(os.path.join(a.out, "items.parquet"))
    print(len(cat), "benchmarks;", sb["benchmark"].nunique(), "eligible with data;", len(sb), "subject-benchmark rows")

if __name__ == "__main__":
    main()
