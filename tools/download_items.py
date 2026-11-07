"""Download items.parquet (item_content etc.) for the benchmarks used in sb.parquet. Skips files > 300 MB."""
import os, sys, json, concurrent.futures as cf
import pandas as pd, requests
from huggingface_hub import hf_hub_download
REPO = "aims-foundations/measurement-db"; SHA = "54074453da7b9911be0f48e4c560be547bc84d21"
tok = os.environ["HF_TOKEN"]; out = sys.argv[1]; agg = sys.argv[2]
benches = sorted(pd.read_parquet(os.path.join(agg, "sb.parquet"))["benchmark"].unique())
def one(b):
    t = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{SHA}/{b}/formatted_tables",
                     headers={"Authorization": "Bearer " + tok}, timeout=60).json()
    f = [x for x in t if isinstance(x, dict) and x.get("path", "").endswith("items.parquet")]
    if not f: return b, "MISSING"
    if f[0]["size"] > 300e6: return b, f"SKIP {f[0]['size']}"
    try: return b, hf_hub_download(REPO, f[0]["path"], repo_type="dataset", revision=SHA, cache_dir=out, token=tok)
    except Exception as e: return b, "ERR " + type(e).__name__
res = {}
with cf.ThreadPoolExecutor(12) as ex:
    for b, p in ex.map(one, benches):
        res[b] = p; print(b, p if not p.startswith("/") else "ok", flush=True)
json.dump(res, open(os.path.join(out, "items_index.json"), "w"), indent=1)
print("done", len(res), flush=True)
