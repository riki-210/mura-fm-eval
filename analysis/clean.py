from load import *
import json
from sklearn.metrics import roc_auc_score,cohen_kappa_score
rng=np.random.default_rng(7);B=10000;seeds=(1,2,3)
def cor(r,n,s): return ((sp(r,f'{n}_s{s}')>=.5)==labels(r)).astype(float)
def acc(n,regs):
    return [np.concatenate([cor(r,n,s) for r in regs]).mean()*100 for s in seeds]
def pair(a,b,regs): return np.concatenate([np.mean([cor(r,a,s)-cor(r,b,s) for s in seeds],0) for r in regs])
def boot(d):
    n=len(d);m=d.mean();bs=np.array([d[rng.integers(0,n,n)].mean() for _ in range(B)])
    return m*100,np.percentile(bs,2.5)*100,np.percentile(bs,97.5)*100,min(1,2*min((bs<=0).mean(),(bs>=0).mean()))
res={}
for lab,regs in [('pooled',R)]+[(r.split('_')[1].lower(),[r]) for r in R]:
    print('==',lab)
    for n in ['dinov2_base','dinov2_base_clean','dinov2_base_randexcl']:
        a=acc(n,regs);print(f'{n:22s} seeds {np.round(a,2)} mean {np.mean(a):.2f} sd {np.std(a,ddof=1):.2f}')
    for k,(a,b) in {'clean vs base':('dinov2_base_clean','dinov2_base'),'randexcl vs base':('dinov2_base_randexcl','dinov2_base'),'clean vs randexcl':('dinov2_base_clean','dinov2_base_randexcl')}.items():
        m,lo,hi,p=boot(pair(a,b,regs));res[f'{lab}|{k}']=[m,lo,hi,p]
        print(f'  {k:20s}{m:6.2f} [{lo:5.2f},{hi:5.2f}] p={p:.4f}')
json.dump(res,open('clean.json','w'))
