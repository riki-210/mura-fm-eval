from load import *
import json,matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,brier_score_loss
def avg(r,n): return np.mean([sp(r,f'{n}_s{s}') for s in (1,2,3)],axis=0)
A=['dinov2_base','dinov2_base_lr1e5','dinov2_large','rad_dino','dinov2_mil2']
def ens(r,names): return np.mean([sp(r,f'{n}_s{s}') for n in names for s in (1,2,3)],axis=0)
M={'GRAZ-pretrained DenseNet169':lambda r:sp(r,'graz_pretrained_densenet169'),
 'DINOv2-base':lambda r:avg(r,'dinov2_base'),
 'DINOv2-base, all body parts':lambda r:avg(r,'dinov2_all7'),
 'DINOv2-base, study-level loss':lambda r:avg(r,'dinov2_mil2'),
 'Ensemble C':lambda r:ens(r,A+['dinov2_all7'])}
y={r:labels(r) for r in R}
P={m:{r:f(r) for r in R} for m,f in M.items()}
for m in P:
    ys=np.concatenate([y[r] for r in R]);ps=np.concatenate([P[m][r] for r in R])
    print(m,'pooled acc %.3f auc %.3f'%(((ps>=.5)==ys).mean(),roc_auc_score(ys,ps)))
# ROC figure
cols={'GRAZ-pretrained DenseNet169':'#888888','DINOv2-base':'#0072B2','DINOv2-base, all body parts':'#009E73','DINOv2-base, study-level loss':'#E69F00','Ensemble C':'#D55E00'}
fig,ax=plt.subplots(1,3,figsize=(10.5,3.6),sharey=True)
for a,r in zip(ax,R):
    for m in P:
        fpr,tpr,_=roc_curve(y[r],P[m][r]);a.plot(fpr,tpr,color=cols[m],lw=1.6,label=f'{m} ({roc_auc_score(y[r],P[m][r]):.3f})')
    a.plot([0,1],[0,1],':',color='#bbbbbb');a.set_title(r[3:].capitalize());a.set_xlabel('1 − specificity');a.set_aspect('equal')
    for s in ('top','right'):a.spines[s].set_visible(False)
ax[0].set_ylabel('Sensitivity')
h,l=ax[0].get_legend_handles_labels()
fig.legend(h,[x for x in l],loc='lower center',ncol=3,fontsize=7.5,frameon=False)
fig.tight_layout(rect=(0,0.17,1,1));fig.savefig('fig_roc.pdf');fig.savefig('fig_roc.png',dpi=200)
# calibration, Ensemble C pooled + per model ECE
def ece(yv,p,k=10):
    q=np.quantile(p,np.linspace(0,1,k+1));q[-1]+=1e-9;e=0;pts=[]
    for i in range(k):
        m=(p>=q[i])&(p<q[i+1])
        if m.sum(): pts.append((p[m].mean(),yv[m].mean(),m.sum()));e+=m.sum()/len(p)*abs(p[m].mean()-yv[m].mean())
    return e,pts
cal={}
for m in P:
    ys=np.concatenate([y[r] for r in R]);ps=np.concatenate([P[m][r] for r in R])
    e,pts=ece(ys,ps);cal[m]=(e,brier_score_loss(ys,ps),pts);print('cal',m,'ECE %.3f Brier %.3f'%(e,cal[m][1]))
fig,a=plt.subplots(figsize=(4.2,4.2))
for m in ['DINOv2-base','Ensemble C']:
    pts=np.array(cal[m][2]);a.plot(pts[:,0],pts[:,1],'o-',color=cols[m],lw=1.4,ms=4,label=f'{m} (ECE {cal[m][0]:.3f})')
a.plot([0,1],[0,1],':',color='#aaaaaa');a.set_xlabel('Mean predicted probability');a.set_ylabel('Observed fraction abnormal');a.legend(fontsize=8,frameon=False)
for s in ('top','right'):a.spines[s].set_visible(False)
fig.tight_layout();fig.savefig('fig_cal.pdf');fig.savefig('fig_cal.png',dpi=200)
# operating points of Ensemble C from test ROC
res={}
for r in R+['pooled']:
    ys=np.concatenate([y[x] for x in R]) if r=='pooled' else y[r]
    ps=np.concatenate([P['Ensemble C'][x] for x in R]) if r=='pooled' else P['Ensemble C'][r]
    fpr,tpr,th=roc_curve(ys,ps)
    i=np.argmax(tpr>=.9);j=np.where(fpr<=.10)[0][-1];k=np.where(fpr<=.05)[0][-1]
    res[r]=dict(spec_at_sens90=1-fpr[i],thr90=th[i],sens_at_spec90=tpr[j],sens_at_spec95=tpr[k])
    print(r,{a:round(float(b),3) for a,b in res[r].items()})
# overlap
def corr(r,n): return (avg(r,n)>=.5)==y[r]
out={}
for scope,regs in [('pooled',R),('elbow',['XR_ELBOW'])]:
    c={n:np.concatenate([corr(r,n) for r in regs]) for n in ['dinov2_base','dinov2_all7','dinov2_mil2','dinov2_mil3']}
    b=c['dinov2_base']
    fix={n:(~b)&c[n] for n in c if n!='dinov2_base'};brk={n:b&~c[n] for n in fix}
    print(scope,{n:(int(fix[n].sum()),int(brk[n].sum())) for n in fix})
    f1,f2=fix['dinov2_all7'],fix['dinov2_mil2']
    print(' all7 vs mil2: both fix',int((f1&f2).sum()),'only all7',int((f1&~f2).sum()),'only mil2',int((f2&~f1).sum()),'jaccard %.2f'%((f1&f2).sum()/(f1|f2).sum()))
    s1,s2=brk['dinov2_all7'],brk['dinov2_mil2'];print(' broken both',int((s1&s2).sum()),'only all7',int((s1&~s2).sum()),'only mil2',int((s2&~s1).sum()))
    m3=c['dinov2_mil3'];print(' mil3 correct count',int(m3.sum()),'all7',int(c['dinov2_all7'].sum()),'mil2',int(c['dinov2_mil2'].sum()),'base',int(b.sum()))
    # of studies fixed by all7 or mil2 (union), how many fixed by mil3
    u=f1|f2;print(' union fixed',int(u.sum()),'mil3 fixes',int((u&m3).sum()))
    out[scope]=dict(fix={k:int(v.sum()) for k,v in fix.items()},brk={k:int(v.sum()) for k,v in brk.items()},both=int((f1&f2).sum()),only_all7=int((f1&~f2).sum()),only_mil2=int((f2&~f1).sum()))
json.dump(dict(cal={m:(cal[m][0],cal[m][1]) for m in cal},op=res,overlap=out),open('more.json','w'),default=float)
