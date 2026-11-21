import numpy as np, pandas as pd
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from pathlib import Path
ROOT=Path('/mnt/data/KLCR_current_experiments'); OUT=ROOT/'results'
from run_klcr import klcr
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

def canon(z):
 u=sorted(np.unique(z)); mp={v:i for i,v in enumerate(u)}; return np.array([mp[v] for v in z])
def fit_tls(x,y,labels,minsize=3):
 out={}
 for g in sorted(np.unique(labels)):
  ix=np.where(labels==g)[0]
  if len(ix)<minsize: raise ValueError('small cluster')
  P=np.column_stack([x[ix],y[ix]]); mu=P.mean(0); Q=P-mu
  if np.linalg.matrix_rank(Q)<1: raise ValueError('degenerate')
  _,_,vt=np.linalg.svd(Q,full_matrices=False); d=vt[0]
  # line represented by normal n=(-dy,dx); n dot (p-mu)=0
  n=np.array([-d[1],d[0]])
  out[g]=(mu,n)
 return out
def tls_assign(x,y,models):
 P=np.column_stack([x,y]); ds=[]
 for mu,n in models.values(): ds.append(np.abs((P-mu)@n))
 return canon(np.argmin(np.column_stack(ds),axis=1))
def tls_klines(x,y,K=3,max_iter=500):
 centers=np.array([x.min()+j/(K+1)*(x.max()-x.min()) for j in range(1,K+1)])
 lab=np.argmin(np.abs(x[:,None]-centers),axis=1); seen=set()
 for it in range(1,max_iter+1):
  try: m=fit_tls(x,y,lab)
  except: return None,'invalid',it
  new=tls_assign(x,y,m)
  if np.array_equal(new,lab): return new,'converged',it
  key=tuple(new)
  if key in seen: return None,'cycle',it
  seen.add(tuple(lab)); lab=new
 return None,'limit',max_iter
def ols_residual_klines(x,y,K=3,max_iter=500):
 from run_klcr import fit_ols
 centers=np.array([x.min()+j/(K+1)*(x.max()-x.min()) for j in range(1,K+1)])
 lab=np.argmin(np.abs(x[:,None]-centers),axis=1); seen=set()
 for it in range(1,max_iter+1):
  try: models=fit_ols(x,y,lab,min_size=3)
  except: return None,'invalid',it
  labs=sorted(models)
  D=np.column_stack([np.abs(models[j][0]+models[j][1]*x-y) for j in labs])
  new=canon(np.array([labs[k] for k in np.argmin(D,axis=1)]))
  if np.array_equal(new,lab): return new,'converged',it
  if tuple(new) in seen: return None,'cycle',it
  seen.add(tuple(lab)); lab=new
 return None,'limit',max_iter

rows=[]
for nm_i,name in enumerate(names):
 for b in range(B):
  x,y,z=generate(name,SEED+10000*(1+nm_i)+b)
  for scale in ['Raw','Standardized']:
   xx=x.copy(); yy=y.copy()
   if scale=='Standardized': xx=(xx-xx.mean())/xx.std(ddof=1); yy=(yy-yy.mean())/yy.std(ddof=1)
   methods={}
   f=klcr(xx,yy,3,3); methods['K-LCR']=(f.get('labels') if f['status']=='converged' else None,f['status'],f.get('iterations',np.nan))
   lab,st,it=tls_klines(xx,yy); methods['TLS K-lines']=(lab,st,it)
   lab,st,it=ols_residual_klines(xx,yy); methods['OLS-residual CLR']=(lab,st,it)
   xy=np.column_stack([xx,yy])
   km=KMeans(n_clusters=3,n_init=5,max_iter=100,random_state=SEED+b).fit_predict(xy); methods['K-means']=(km,'converged',np.nan)
   Z=linkage(xy,method='ward',metric='euclidean'); wd=fcluster(Z,3,criterion='maxclust')-1; methods['Ward']=(wd,'converged',np.nan)
   for m,(pred,status,iterations) in methods.items():
    rows.append({'Structure':name,'Replicate':b+1,'Scale':scale,'Method':m,'Status':status,'ARI':adjusted_rand_score(z,pred) if pred is not None and status=='converged' else np.nan,'NMI':normalized_mutual_info_score(z,pred) if pred is not None and status=='converged' else np.nan,'Iterations':iterations})
df=pd.DataFrame(rows); df.to_csv(OUT/'comparator_replicates.csv',index=False)
s=df.groupby(['Structure','Scale','Method']).agg(replicates=('Replicate','count'),converged=('Status',lambda x:(x=='converged').sum()),convergence_rate=('Status',lambda x:(x=='converged').mean()),mean_ARI=('ARI','mean'),sd_ARI=('ARI','std'),median_ARI=('ARI','median'),mean_NMI=('NMI','mean'),sd_NMI=('NMI','std')).reset_index()
s.to_csv(OUT/'comparator_summary.csv',index=False)
print(s.to_string(index=False))
