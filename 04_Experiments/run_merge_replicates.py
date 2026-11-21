import sys
sys.path.insert(0,'/mnt/data/KLCR_current_experiments')
from run_klcr import klcr, merge_r2, fit_ols, pooled_metrics
import numpy as np, pandas as pd
SEED=20261121; N=400; SIG=.35; B=100
names=['Parallel','Crossing','Near-parallel','Change-point','Three-pattern']
def generate(name,seed):
 r=np.random.default_rng(seed); x=r.uniform(-2,2,N)
 if name=='Parallel': z=r.integers(0,2,N); y=2+1.5*x+5*z+r.normal(0,SIG,N)
 elif name=='Crossing': z=r.integers(0,2,N); y=np.where(z==0,2+1.5*x,2-1.5*x)+r.normal(0,SIG,N)
 elif name=='Near-parallel': z=r.integers(0,2,N); y=np.where(z==0,2+1.0*x,2.8+1.1*x)+r.normal(0,SIG,N)
 elif name=='Change-point': z=(x>0).astype(int); y=np.where(x<=0,2+1.5*x,2-.8*x)+r.normal(0,SIG,N)
 else: z=r.choice(3,N,p=[.7,.2,.1]); ab=np.array([[1,1.5],[5,.2],[9,-1.]]); y=ab[z,0]+ab[z,1]*x+r.normal(0,SIG,N)
 return x,y,z
rows=[]
for ni,name in enumerate(names):
 for b in range(B):
  x,y,z=generate(name,SEED+10000*(1+ni)+b)
  for scale in ['Raw','Standardized']:
   xx=x.copy(); yy=y.copy()
   if scale=='Standardized': xx=(xx-xx.mean())/xx.std(ddof=1); yy=(yy-yy.mean())/yy.std(ddof=1)
   f=klcr(xx,yy,3,3)
   if f['status']!='converged':
    for tau in [.01,.025,.05,.10]: rows.append({'Structure':name,'Replicate':b+1,'Scale':scale,'Threshold':tau,'Status':f['status'],'K_after':np.nan,'R2_loss':np.nan,'Merge_count':np.nan})
    continue
   labels=f['labels']; models=f['models']; r2=pooled_metrics(xx,yy,labels,models)[0]
   for tau in [.01,.025,.05,.10]:
    ml,steps=merge_r2(xx,yy,labels,tau); mm=fit_ols(xx,yy,ml,min_size=2); mr=pooled_metrics(xx,yy,ml,mm)[0]
    rows.append({'Structure':name,'Replicate':b+1,'Scale':scale,'Threshold':tau,'Status':'converged','K_after':len(mm),'R2_loss':r2-mr,'Merge_count':len(steps)})
df=pd.DataFrame(rows); df.to_csv('/mnt/data/KLCR_current_experiments/results/merge_replicates.csv',index=False)
s=df[df.Status=='converged'].groupby(['Structure','Scale','Threshold']).agg(n=('Replicate','count'),mean_K=('K_after','mean'),sd_K=('K_after','std'),p_K1=('K_after',lambda x:(x==1).mean()),p_K2=('K_after',lambda x:(x==2).mean()),p_K3=('K_after',lambda x:(x==3).mean()),mean_R2_loss=('R2_loss','mean'),sd_R2_loss=('R2_loss','std'),mean_merges=('Merge_count','mean')).reset_index()
s.to_csv('/mnt/data/KLCR_current_experiments/results/merge_replicate_summary.csv',index=False)
print(s.to_string(index=False))
