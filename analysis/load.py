import numpy as np,pandas as pd,os,glob
R=['XR_ELBOW','XR_HAND','XR_SHOULDER']
T=os.environ.get('PRED_DIR','../predictions')+'/'
ti=pd.read_csv(os.environ.get('TEST_INDEX','../data/test_index.csv'))
ti['study']=ti.image_path.map(os.path.dirname)
idx={r:ti[ti.region==r].reset_index(drop=True) for r in R}
studies={}
for r in R:
    d=idx[r];g=d.groupby('study',sort=False)
    studies[r]=(g.ngroup().values,g.fracture_label.first().values)
def sp(r,name,tta=True):
    f=T+f'tier2_{r}_{name}/'+('test_probs.npy')
    p=np.load(f)[:,1]
    inv,y=studies[r]
    s=np.bincount(inv,weights=p)/np.bincount(inv)
    return s
def labels(r): return studies[r][1]
