import sys
sys.path.insert(0,'/mnt/data/KLCR_current_experiments')
from run_klcr import klcr
import numpy as np,pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
SEED=20261121; N=400; SIG=.35; B=100
rows=[]
def generate(name,seed):
 r=np.random.default_rng(seed); x=r.uniform(-2,2,N)
 if name=='Parallel':
  z=r.integers(0,2,N); y=2+1.5*x+5*z+r.normal(0,SIG,N)
 elif name=='Crossing':
  z=r.integers(0,2,N); y=np.where(z==0,2+1.5*x,2-1.5*x)+r.normal(0,SIG,N)
 elif name=='Near-parallel':
  z=r.integers(0,2,N); y=np.where(z==0,2+1.0*x,2.8+1.1*x)+r.normal(0,SIG,N)
 elif name=='Change-point':
  z=(x>0).astype(int); y=np.where(x<=0,2+1.5*x,2-.8*x)+r.normal(0,SIG,N)
 else:
  z=r.choice(3,N,p=[.7,.2,.1]); ab=np.array([[1,1.5],[5,.2],[9,-1.]]) ; y=ab[z,0]+ab[z,1]*x+r.normal(0,SIG,N)
 return x,y,z
for name in ['Parallel','Crossing','Near-parallel','Change-point','Three-pattern']:
 for b in range(B):
  x,y,z=generate(name,SEED+10000*(1+['Parallel','Crossing','Near-parallel','Change-point','Three-pattern'].index(name))+b)
  for scale in ['Raw','Standardized']:
   xx=x.copy(); yy=y.copy()
   if scale=='Standardized': xx=(xx-xx.mean())/xx.std(ddof=1); yy=(yy-yy.mean())/yy.std(ddof=1)
   f=klcr(xx,yy,3,3)
   rows.append({'Dataset':name,'Replicate':b+1,'Scale':scale,'Status':f['status'],'Iterations':f.get('iterations',np.nan),'K':len(np.unique(f['labels'])) if f.get('labels') is not None else np.nan,'ARI':adjusted_rand_score(z,f['labels']) if f['status']=='converged' else np.nan,'NMI':normalized_mutual_info_score(z,f['labels']) if f['status']=='converged' else np.nan,'Cluster_sizes':','.join(map(str,pd.Series(f['labels']).value_counts().sort_index().tolist())) if f['status']=='converged' else ''})
df=pd.DataFrame(rows); df.to_csv('/mnt/data/KLCR_current_experiments/results/simulation_replicates.csv',index=False)
s=df.groupby(['Dataset','Scale']).agg(replicates=('Replicate','count'),converged=('Status',lambda x:(x=='converged').sum()),convergence_rate=('Status',lambda x:(x=='converged').mean()),mean_iterations=('Iterations','mean'),median_iterations=('Iterations','median'),mean_ARI=('ARI','mean'),sd_ARI=('ARI','std'),median_ARI=('ARI','median'),mean_NMI=('NMI','mean'),sd_NMI=('NMI','std')).reset_index()
s.to_csv('/mnt/data/KLCR_current_experiments/results/simulation_replicate_summary.csv',index=False)
print(s.to_string(index=False))
