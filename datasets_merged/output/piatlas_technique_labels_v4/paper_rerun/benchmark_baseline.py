# App. C 'Grouping by benchmark as a baseline' and the AMI comparison of Finding G1: scores the
# benchmark (source) grouping and the five-seed k=33 k-means clusterings of each encoder against the
# technique-set labels, with and without ASB. Prints only.  Usage: python3 benchmark_baseline.py
import os, numpy as np
from sklearn.metrics import adjusted_mutual_info_score as AMI, normalized_mutual_info_score as NMI, adjusted_rand_score as ARI
import rerun_paper_experiments as R
SP = os.environ.get("SP", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cluster_cache"))
old, ds, new, idx = R.load_inputs()
def src(i):
    s=ds[str(i)] if isinstance(ds,dict) else ds[i]
    return "|".join(sorted(set(R.source_name(x) for x in s))) if isinstance(s,list) else R.source_name(s)
S=np.array([src(i) for i in idx]); Sy=R.catify(S.tolist())
L=R.catify([R.tech(new[i]) for i in idx]); na=S!="ASB"
f=lambda a,b: (NMI(a,b),ARI(a,b),AMI(a,b))
print("benchmark grouping vs labels, all:    NMI %.4f ARI %.4f AMI %.4f"%f(L,Sy))
print("benchmark grouping vs labels, no ASB: NMI %.4f ARI %.4f AMI %.4f"%f(L[na],Sy[na]))
for enc in ["f2llm_steer_noll","f2llm_plain_noll","qwen_noll"]:
    a=[];b=[]
    for s in [0,1,2,3,42]:
        C=np.load(f"{SP}/{enc}_kmeans_ninit10_seed{s}_k33.npy")[idx]
        a.append(f(L,C)); b.append(f(L[na],C[na]))
    a=np.mean(a,0); b=np.mean(b,0)
    print(f"{enc:18s} five-seed all: NMI {a[0]:.4f} ARI {a[1]:.4f} AMI {a[2]:.4f} | no ASB: NMI {b[0]:.4f} ARI {b[1]:.4f} AMI {b[2]:.4f}")
