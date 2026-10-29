from load import *
import json
from sklearn.metrics import roc_auc_score,roc_curve
rng=np.random.default_rng(1)
B=10000
seeds=(1,2,3)
def cor(r,name,s): return ((sp(r,f'{name}_s{s}')>=.5)==labels(r)).astype(float)
def cor1(r,name): return ((sp(r,name)>=.5)==labels(r)).astype(float)
def pair(A,Bn,regions=R,single=False):
    # diff per study averaged over seeds (matched), pooled over regions
    d=[]
    for r in regions:
        if single: x=cor1(r,A[0])-cor1(r,Bn)  # not used
        x=np.mean([cor(r,A,s)-cor(r,Bn,s) for s in seeds],axis=0)
        d.append(x)
    return np.concatenate(d)
def boot(d):
    n=len(d);m=d.mean()
    bs=np.array([d[rng.integers(0,n,n)].mean() for _ in range(B)])
    p=min(1,2*min((bs<=0).mean(),(bs>=0).mean()))
    return m,np.percentile(bs,2.5),np.percentile(bs,97.5),p
comps={'DINOv2 all body parts vs base':('dinov2_all7','dinov2_base'),
 'Study-level loss vs base':('dinov2_mil2','dinov2_base'),
 'DINOv2-large vs base':('dinov2_large','dinov2_base'),
 'RAD-DINO vs base':('rad_dino','dinov2_base'),
 'Random erasing vs base':('dinov2_base_erase','dinov2_base'),
 'Lower learning rate vs base':('dinov2_base_lr1e5','dinov2_base'),
 'Study-level loss on all body parts vs base':('dinov2_mil3','dinov2_base'),
 'Study-level loss on all body parts vs all body parts':('dinov2_mil3','dinov2_all7'),
 'Study-level loss vs all body parts':('dinov2_mil2','dinov2_all7')}
out=[]
for k,(a,b) in comps.items():
    for lab,regs in [('pooled',R)]+[(r.split('_')[1].lower(),[r]) for r in R]:
        m,lo,hi,p=boot(pair(a,b,regs));out.append((k,lab,m*100,lo*100,hi*100,p))
        print(f'{k:55s}{lab:9s}{m*100:6.2f} [{lo*100:5.2f},{hi*100:5.2f}] p={p:.4f}')
json.dump(out,open('comps.json','w'))
