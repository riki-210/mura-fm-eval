from load import *
import json
rng=np.random.default_rng(2);B=10000
def avg(r,name): return np.mean([sp(r,f'{name}_s{s}') for s in (1,2,3)],axis=0)
def acc(r,n): return ((sp(r,n)>=.5)==labels(r)).astype(float)
d=[np.mean([acc(r,f'dinov2_base_s{s}')-acc(r,f'graz_d169_224_s{s}') for s in (1,2,3)],axis=0) for r in R]
d=np.concatenate(d);n=len(d);bs=np.array([d[rng.integers(0,n,n)].mean() for _ in range(B)])
p0=min(1,2*min((bs<=0).mean(),(bs>=0).mean()));print('dino vs D169',d.mean()*100,np.percentile(bs,[2.5,97.5])*100,p0)
pb=json.load(open('../cb/paired_bootstrap.json'))
prim=[('Transfer: '+k,v['diff']*100,v['p_bootstrap']) for k,v in pb['pooled'].items()]
prim.append(('DINOv2-base vs GRAZ-DenseNet169 (224 px)',d.mean()*100,max(p0,1/B)))
c=json.load(open('comps.json'))
prim+=[(k,m,p) for k,l,m,lo,hi,p in c if l=='pooled']
expl=[(f'{k} [{l}]',m,p) for k,l,m,lo,hi,p in c if l!='pooled']
for r in pb['per_region']: expl.append((f"Transfer {r['backbone']} [{r['region'][3:].lower()}]",r['diff']*100,r['p_bootstrap']))
def adj(fam):
    ps=np.array([x[2] for x in fam]);m=len(ps);o=np.argsort(ps)
    holm=np.empty(m);run=0
    for rank,i in enumerate(o): run=max(run,min(1,(m-rank)*ps[i]));holm[i]=run
    bh=np.empty(m);run=1
    for rank in range(m-1,-1,-1):
        i=o[rank];run=min(run,ps[i]*m/(rank+1));bh[i]=run
    return holm,bh
res={}
for nm,fam in [('primary',prim),('exploratory',expl)]:
    h,b=adj(fam);res[nm]=[(f[0],f[1],f[2],h[i],b[i]) for i,f in enumerate(fam)]
    print(nm,len(fam),'n sig raw',sum(f[2]<.05 for f in fam),'holm',(h<.05).sum(),'bh',(b<.05).sum())
    for x in res[nm]: print(f'  {x[0]:60s}{x[1]:6.2f} p={x[2]:.4f} holm={x[3]:.4f} bh={x[4]:.4f}')
json.dump(res,open('corr.json','w'))
