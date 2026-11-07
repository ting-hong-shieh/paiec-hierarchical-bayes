import os, sys, json, concurrent.futures as cf, requests
from huggingface_hub import hf_hub_download
REPO="aims-foundations/measurement-db"; SHA="54074453da7b9911be0f48e4c560be547bc84d21"
tok=os.environ["HF_TOKEN"]
top=requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/{SHA}",headers={"Authorization":"Bearer "+tok}).json()
dirs=sorted(x['path'] for x in top if x['type']=='directory')
print(len(dirs),flush=True)
out=sys.argv[1]
def one(d):
    got={}
    for t in ("benchmarks","subjects","responses"):
        fn=f"{d}/formatted_tables/{t}.parquet"
        try: got[t]=hf_hub_download(REPO,fn,repo_type="dataset",revision=SHA,cache_dir=out)
        except Exception as e: got[t]="ERR "+type(e).__name__
    return d,got
res={}
with cf.ThreadPoolExecutor(16) as ex:
    for i,(d,got) in enumerate(ex.map(one,dirs)):
        res[d]=got
        if i%40==0: print(i,flush=True)
json.dump({"sha":SHA,"files":res},open(os.path.join(out,"index2.json"),"w"),indent=1)
print("done",flush=True)
