import numpy as np,pandas as pd,sys
sys.path.insert(0,'/mnt/data/KLCR_current_experiments')
from run_klcr import klcr
ROOT='/mnt/data/KLCR_current_experiments'; SEED=20261121; N=400; SIG=.35; B=30
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
   for K in [2,3,4]:
    f=klcr(xx,yy,K,3); lab=f.get('labels') if f['status']=='converged' else None
    from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
    rows.append({'Structure':name,'Replicate':b+1,'Scale':scale,'Design':'K sensitivity','Value':K,'Status':f['status'],'Effective_K':len(np.unique(lab)) if lab is not None else np.nan,'ARI':adjusted_rand_score(z,lab) if lab is not None else np.nan,'NMI':normalized_mutual_info_score(z,lab) if lab is not None else np.nan,'Iterations':f.get('iterations',np.nan)})
   for m in [2,3,5]:
    f=klcr(xx,yy,3,m); lab=f.get('labels') if f['status']=='converged' else None
    rows.append({'Structure':name,'Replicate':b+1,'Scale':scale,'Design':'Minimum-size sensitivity','Value':m,'Status':f['status'],'Effective_K':len(np.unique(lab)) if lab is not None else np.nan,'ARI':adjusted_rand_score(z,lab) if lab is not None else np.nan,'NMI':normalized_mutual_info_score(z,lab) if lab is not None else np.nan,'Iterations':f.get('iterations',np.nan)})
df=pd.DataFrame(rows); df.to_csv(ROOT+'/results/design_sensitivity_replicates.csv',index=False)
s=df.groupby(['Structure','Scale','Design','Value']).agg(n=('Replicate','count'),convergence_rate=('Status',lambda x:(x=='converged').mean()),mean_effective_K=('Effective_K','mean'),sd_effective_K=('Effective_K','std'),mean_ARI=('ARI','mean'),sd_ARI=('ARI','std'),mean_NMI=('NMI','mean'),mean_iterations=('Iterations','mean')).reset_index()
s.to_csv(ROOT+'/results/design_sensitivity_summary.csv',index=False)
print(s.to_string(index=False))
