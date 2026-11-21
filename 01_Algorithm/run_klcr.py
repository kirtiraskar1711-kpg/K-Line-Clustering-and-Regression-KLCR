"""Reproducible K-LCR experiment using vertical initialization, OLS refits,
perpendicular-distance reassignment, and pooled-R2-loss merging.

Primary operational safeguard: minimum cluster size = 3. Sensitivity runs
also use 2 and 5. Seed and data-generation details are recorded in README.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'; OUT=ROOT/'results'; FIG=ROOT/'figures'
for p in (DATA,OUT,FIG): p.mkdir(exist_ok=True,parents=True)
SEED=20261121


def fit_ols(X, y, labels, min_size=3):
    models={}
    for lab in sorted(np.unique(labels)):
        ix=np.where(labels==lab)[0]
        if len(ix)<min_size: raise ValueError(f'cluster {lab} has n={len(ix)} < min_size={min_size}')
        xx=X[ix]
        if np.ptp(xx)<=1e-12: raise ValueError(f'cluster {lab} has no predictor variation')
        b=np.sum((xx-xx.mean())*(y[ix]-y[ix].mean()))/np.sum((xx-xx.mean())**2)
        a=y[ix].mean()-b*xx.mean()
        models[lab]=(float(a),float(b))
    return models

def assign_perp(X,y,models):
    labs=list(models)
    D=np.column_stack([np.abs(models[j][0]+models[j][1]*X-y)/np.sqrt(1+models[j][1]**2) for j in labs])
    # np.argmin gives deterministic lowest current label in ties
    return np.array([labs[k] for k in np.argmin(D,axis=1)],dtype=int)

def canonicalize(labels):
    uniq=sorted(np.unique(labels)); mp={v:i+1 for i,v in enumerate(uniq)}
    return np.array([mp[v] for v in labels],dtype=int)

def klcr(x,y,K=3,min_size=3,max_iter=500):
    x=np.asarray(x,float); y=np.asarray(y,float); n=len(x)
    if len(np.unique(x))<2: return {'status':'invalid_x','labels':None}
    xmin,xmax=x.min(),x.max()
    centers=np.array([xmin+j/(K+1)*(xmax-xmin) for j in range(1,K+1)])
    # Initial predictor-only assignment to equally spaced vertical lines
    labels=np.argmin(np.abs(x[:,None]-centers[None,:]),axis=1)+1
    labels=canonicalize(labels)
    seen=set(); history=[]
    for it in range(1,max_iter+1):
        try: models=fit_ols(x,y,labels,min_size)
        except Exception as e: return {'status':'invalid_fit','reason':str(e),'labels':labels,'iterations':it-1}
        new=canonicalize(assign_perp(x,y,models))
        key=tuple(new.tolist())
        if np.array_equal(new,labels):
            labels=new
            try: models=fit_ols(x,y,labels,min_size)
            except Exception as e: return {'status':'invalid_fit','reason':str(e),'labels':labels,'iterations':it}
            return {'status':'converged','labels':labels,'models':models,'iterations':it,'history':history,'centers':centers}
        if key in seen:
            return {'status':'cycle','labels':new,'iterations':it,'history':history,'centers':centers}
        seen.add(tuple(labels.tolist())); history.append({'iteration':it,'sizes':pd.Series(labels).value_counts().sort_index().to_dict()})
        labels=new
    return {'status':'limit','labels':labels,'iterations':max_iter,'history':history,'centers':centers}

def pooled_metrics(x,y,labels,models=None):
    if models is None: models=fit_ols(x,y,labels,min_size=2)
    pred=np.empty(len(y))
    for lab,(a,b) in models.items(): pred[labels==lab]=a+b*x[labels==lab]
    sse=np.sum((y-pred)**2); sst=np.sum((y-y.mean())**2)
    r2=1-sse/sst if sst>0 else np.nan
    return float(r2),float(np.sqrt(np.mean((y-pred)**2))),float(sse)

def merge_r2(x,y,labels,tau):
    labels=canonicalize(labels.copy()); steps=[]
    while len(np.unique(labels))>1:
        labs=sorted(np.unique(labels)); before_models=fit_ols(x,y,labels,min_size=2)
        r2_before=pooled_metrics(x,y,labels,before_models)[0]
        candidates=[]
        for ii,a in enumerate(labs):
            for b in labs[ii+1:]:
                trial=labels.copy(); trial[trial==b]=a; trial=canonicalize(trial)
                try:
                    tm=fit_ols(x,y,trial,min_size=2); r2_after=pooled_metrics(x,y,trial,tm)[0]
                except Exception: continue
                loss=r2_before-r2_after
                candidates.append((loss,a,b,trial,r2_after))
        if not candidates: break
        loss,a,b,trial,r2_after=min(candidates,key=lambda z:(z[0],z[1],z[2]))
        if loss<=tau+1e-12:
            steps.append({'merged_pair':f'{a}+{b}','r2_before':r2_before,'r2_after':r2_after,'r2_loss':loss})
            labels=trial
        else: break
    return labels,steps

def make_sims():
    rng=np.random.default_rng(SEED); n=400; sig=.35
    x=rng.uniform(-2,2,n)
    # labels and equations follow the working simulation plan; same n/noise across designs
    z=rng.integers(0,2,n)
    sims={}
    y=2+1.5*x+5*z+rng.normal(0,sig,n)
    sims['Parallel']=(x,y,z)
    z=rng.integers(0,2,n); y=np.where(z==0,2+1.5*x,2-1.5*x)+rng.normal(0,sig,n)
    sims['Crossing']=(x,y,z)
    z=rng.integers(0,2,n); y=np.where(z==0,2+1.0*x,2.8+1.1*x)+rng.normal(0,sig,n)
    sims['Near-parallel']=(x,y,z)
    y=np.where(x<=0,2+1.5*x,2-.8*x)+rng.normal(0,sig,n); z=(x>0).astype(int)
    sims['Change-point']=(x,y,z)
    z=rng.choice(3,n,p=[.7,.2,.1]); lines=np.array([[1,1.5],[5,.2],[9,-1.]])
    y=lines[z,0]+lines[z,1]*x+rng.normal(0,sig,n)
    sims['Three-pattern']=(x,y,z)
    return sims

def empirical():
    iris=load_iris(); X=iris.data[:,2]; Y=iris.data[:,3]
    out={'Iris':(X,Y,iris.target)}
    # R datasets carried in the reproducibility archive
    old=pd.read_csv(DATA/'Old_Faithful.csv'); out['Old Faithful']=(old.iloc[:,0].to_numpy(),old.iloc[:,1].to_numpy(),None)
    fuel=pd.read_csv(DATA/'Fuel.csv'); out['Fuel']=(fuel.iloc[:,0].to_numpy(),fuel.iloc[:,1].to_numpy(),None)
    cr=pd.read_csv(DATA/'Crabs.csv'); out['Crabs']=(cr['FL'].to_numpy(),cr['RW'].to_numpy(),None)
    hyp=pd.read_csv(DATA/'Hypothetical.csv'); out['Hypothetical']=(hyp.iloc[:,0].to_numpy(),hyp.iloc[:,1].to_numpy(),None)
    return out

def run_suite():
    sim=make_sims(); emp=empirical(); allsets={**sim,**emp}
    summary=[]; equations=[]; merge_rows=[]; assignments=[]
    for name,(x,y,true_lab) in allsets.items():
      for scale in ['Raw','Standardized']:
        xx=np.asarray(x,float); yy=np.asarray(y,float)
        if scale=='Standardized':
          xx=(xx-xx.mean())/xx.std(ddof=1); yy=(yy-yy.mean())/yy.std(ddof=1)
        for min_size in [3]:
          res=klcr(xx,yy,K=3,min_size=min_size)
          if res['status']!='converged':
            summary.append({'Dataset':name,'Scale':scale,'Min cluster size':min_size,'Status':res['status'],'Reason':res.get('reason',''),'Iterations':res.get('iterations',np.nan)})
            continue
          labels=res['labels']; models=res['models']; r2,rmse,sse=pooled_metrics(xx,yy,labels,models)
          summary.append({'Dataset':name,'Scale':scale,'Min cluster size':min_size,'Status':'converged','Iterations':res['iterations'],'K before merge':len(models),'Cluster sizes':','.join(map(str,[int(np.sum(labels==j)) for j in sorted(models)])),'Pooled R2 before merge':r2,'RMSE before merge':rmse})
          for j,(a,b) in models.items(): equations.append({'Dataset':name,'Scale':scale,'Cluster':j,'n':int(np.sum(labels==j)),'Intercept':a,'Slope':b})
          for tau in [.01,.025,.05,.10]:
            ml,steps=merge_r2(xx,yy,labels,tau)
            mm=fit_ols(xx,yy,ml,min_size=2); mr,mrmse,_=pooled_metrics(xx,yy,ml,mm)
            merge_rows.append({'Dataset':name,'Scale':scale,'Threshold':tau,'K before':len(models),'K after':len(mm),'Cluster sizes after':','.join(map(str,[int(np.sum(ml==j)) for j in sorted(mm)])),'Accepted merges':len(steps),'Pooled R2 before':r2,'Pooled R2 after':mr,'Total R2 loss':r2-mr,'RMSE after':mrmse,'Merge history':json.dumps(steps)})
          for i in range(len(xx)):
            assignments.append({'Dataset':name,'Scale':scale,'row':i+1,'x':xx[i],'y':yy[i],'true_pattern':None if true_lab is None else int(true_lab[i]),'cluster':int(labels[i])})
    pd.DataFrame(summary).to_csv(OUT/'klcr_summary.csv',index=False)
    pd.DataFrame(equations).to_csv(OUT/'klcr_equations.csv',index=False)
    pd.DataFrame(merge_rows).to_csv(OUT/'merge_sensitivity.csv',index=False)
    pd.DataFrame(assignments).to_csv(OUT/'assignments.csv',index=False)
    # concise simulation agreement diagnostics
    simdiag=[]
    for name,(x,y,z) in sim.items():
      for scale in ['Raw','Standardized']:
        xx=x.copy(); yy=y.copy()
        if scale=='Standardized': xx=(xx-xx.mean())/xx.std(ddof=1); yy=(yy-yy.mean())/yy.std(ddof=1)
        for m in [3,2,5]:
          r=klcr(xx,yy,3,min_size=m)
          simdiag.append({'Dataset':name,'Scale':scale,'Minimum size':m,'Status':r['status'],'Iterations':r.get('iterations',np.nan),'K':len(np.unique(r['labels'])) if r.get('labels') is not None else np.nan,'ARI':adjusted_rand_score(z,r['labels']) if r['status']=='converged' else np.nan,'NMI':normalized_mutual_info_score(z,r['labels']) if r['status']=='converged' else np.nan,'Reason':r.get('reason','')})
    pd.DataFrame(simdiag).to_csv(OUT/'simulation_diagnostics.csv',index=False)
    # Figures: simulations raw and standardized, fitted lines before merging
    for name,(x,y,z) in sim.items():
      for scale in ['Raw','Standardized']:
        xx=x.copy(); yy=y.copy()
        if scale=='Standardized': xx=(xx-xx.mean())/xx.std(ddof=1); yy=(yy-yy.mean())/yy.std(ddof=1)
        r=klcr(xx,yy,3,3)
        if r['status']!='converged': continue
        fig,ax=plt.subplots(figsize=(7,5))
        ax.scatter(xx,yy,c=r['labels'],s=18,alpha=.75)
        for j,(a,b) in r['models'].items():
          xs=np.linspace(xx.min(),xx.max(),200); ax.plot(xs,a+b*xs,lw=2,label=f'C{j}: n={sum(r["labels"]==j)}')
        ax.set(title=f'{name} — {scale} K-LCR (K=3)',xlabel='Predictor' + (' (standardized)' if scale=='Standardized' else ''),ylabel='Response' + (' (standardized)' if scale=='Standardized' else ''))
        ax.legend(); fig.tight_layout(); fig.savefig(FIG/f'{name.lower().replace(" ","_")}_{scale.lower()}.png',dpi=180); plt.close(fig)
    print('Finished. Outputs:',*[str(p) for p in OUT.iterdir()],sep='\n')

if __name__=='__main__': run_suite()
