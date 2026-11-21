import numpy as np,pandas as pd,sys
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
sys.path.insert(0,'/mnt/data/KLCR_current_experiments')
from run_klcr import fit_ols
ROOT='/mnt/data/KLCR_current_experiments'; SEED=20261121; N=400; SIG=.35; B=100
names=['Parallel','Crossing','Near-parallel','Change-point','Three-pattern']
def gen(name,seed):
 r=np.random.default_rng(seed); x=r.uniform(-2,2,N)
 if name=='Parallel': z=r.integers(0,2,N); y=2+1.5*x+5*z+r.normal(0,SIG,N)
 elif name=='Crossing': z=r.integers(0,2,N); y=np.where(z==0,2+1.5*x,2-1.5*x)+r.normal(0,SIG,N)
 elif name=='Near-parallel': z=r.integers(0,2,N); y=np.where(z==0,2+1.0*x,2.8+1.1*x)+r.normal(0,SIG,N)
 elif name=='Change-point': z=(x>0).astype(int); y=np.where(x<=0,2+1.5*x,2-.8*x)+r.normal(0,SIG,N)
 else: z=r.choice(3,N,p=[.7,.2,.1]); ab=np.array([[1,1.5],[5,.2],[9,-1.]]); y=ab[z,0]+ab[z,1]*x+r.normal(0,SIG,N)
 return x,y,z
def canon(z):
 u=sorted(np.unique(z)); mp={v:i for i,v in enumerate(u)}; return np.array([mp[v] for v in z])
def run(x,y,K=3):
 centers=np.array([x.min()+j/(K+1)*(x.max()-x.min()) for j in range(1,K+1)])
 lab=np.argmin(np.abs(x[:,None]-centers),axis=1); seen=set()
 for it in range(1,501):
  try: models=fit_ols(x,y,lab,min_size=3)
  except: return None,'invalid',it
  labs=sorted(models); D=np.column_stack([np.abs(models[j][0]+models[j][1]*x-y) for j in labs])
  new=canon(np.array([labs[k] for k in np.argmin(D,axis=1)]))
  if np.array_equal(new,lab): return new,'converged',it
  if tuple(new) in seen: return None,'cycle',it
  seen.add(tuple(lab)); lab=new
 return None,'limit',500
rows=[]
for ni,name in enumerate(names):
 for b in range(B):
  x,y,z=gen(name,SEED+10000*(1+ni)+b)
  for scale in ['Raw','Standardized']:
   xx=x.copy(); yy=y.copy()
   if scale=='Standardized': xx=(xx-xx.mean())/xx.std(ddof=1); yy=(yy-yy.mean())/yy.std(ddof=1)
   pred,status,it=run(xx,yy)
   rows.append({'Structure':name,'Replicate':b+1,'Scale':scale,'Method':'OLS-residual CLR','Status':status,'ARI':adjusted_rand_score(z,pred) if pred is not None else np.nan,'NMI':normalized_mutual_info_score(z,pred) if pred is not None else np.nan,'Iterations':it})
df=pd.DataFrame(rows); df.to_csv(ROOT+'/results/ols_residual_comparator_replicates.csv',index=False)
s=df.groupby(['Structure','Scale','Method']).agg(replicates=('Replicate','count'),converged=('Status',lambda x:(x=='converged').sum()),convergence_rate=('Status',lambda x:(x=='converged').mean()),mean_ARI=('ARI','mean'),sd_ARI=('ARI','std'),median_ARI=('ARI','median'),mean_NMI=('NMI','mean'),sd_NMI=('NMI','std')).reset_index()
s.to_csv(ROOT+'/results/ols_residual_comparator_summary.csv',index=False)
print(s.to_string(index=False))
