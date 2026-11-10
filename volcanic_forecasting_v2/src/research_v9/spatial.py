"""Separate historical spatial basis fitting from latent forecasting algorithms."""
from .common import *
from ..stage1.models import fill_past
from ..research_v6.data import score
from ..research_v7.point import temporal as latent_temporal
from sklearn.decomposition import PCA,TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,HuberRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF,WhiteKernel
from sklearn.cross_decomposition import PLSRegression
from scipy.ndimage import gaussian_filter
import joblib,time
sys.path.append(str(REPO/'mixture_experiment/vendor'))

def fit_basis(rep,m,a,ids):
 rows=[];orig=m.iloc[ids].drop_duplicates(['source_id','origin_day']).index.to_numpy()
 for i in orig:
  for h in a['history'][i]:rows.append((h-a['last'][i])/a['scale'][i])
 raw=np.array(rows);flat=raw.reshape(len(raw),-1);valid=np.isfinite(flat);means=np.nan_to_num(np.nanmean(flat,0));xx=np.where(valid,flat,means);rank=8 if rep=='pca8' else 16 if rep=='block16' else 4
 if rep=='robust_pca4':med=np.median(xx,0);mad=np.maximum(np.median(abs(xx-med),0)*1.4826,.1);xx=np.clip(xx,med-3*mad,med+3*mad)
 if rep=='multiscale4':xx=np.array([gaussian_filter(row.reshape(32,32),1).ravel() for row in xx])
 if rep=='mask_pca4':
  for _ in range(3):pca=PCA(n_components=rank,svd_solver='randomized',random_state=SEED).fit(xx);xx=np.where(valid,flat,pca.inverse_transform(pca.transform(xx)))
 if rep=='block16':
  b=[]
  for y in range(0,32,8):
   for x in range(0,32,8):
    patch=xx.reshape(-1,32,32)[:,y:y+8,x:x+8].reshape(len(xx),64);p=PCA(n_components=1,random_state=SEED).fit(patch);full=np.zeros((32,32));full[y:y+8,x:x+8]=p.components_[0].reshape(8,8);b.append(full.ravel())
  basis=np.array(b)
 elif rep=='svd4':basis=TruncatedSVD(n_components=rank,random_state=SEED).fit(xx).components_
 else:basis=PCA(n_components=rank,svd_solver='randomized',random_state=SEED).fit(xx).components_
 artifact=dict(representation=rep,basis=basis,training_ids=ids.tolist(),basis_target='Historical train maps relative to historical last only; no future targets')
 if rep=='geometry4':
  artifact['geometry']={}
  for asc in [False,True]:
   sub=[i for i in orig if (a['context'][i,0,16]<0)==asc];hh=np.concatenate([np.nan_to_num((a['history'][i]-a['last'][i])/a['scale'][i]).reshape(6,1024) for i in sub]);artifact['geometry'][str(asc)]=PCA(n_components=4,svd_solver='randomized',random_state=SEED).fit(hh).components_ if len(hh)>4 else basis
 return artifact

def encode(art,m,a,ids,targets=False):
 xs=[];ys=[];bs=[];coeffs=[];rec=[]
 for i in ids:
  basis=art['geometry'][str(a['context'][i,0,16]<0)] if art['representation']=='geometry4' and a['context'][i,0,19]>0 else art['basis'];f,v=fill_past(a['history'][i]);h=(f-a['last'][i])/a['scale'][i];h=np.nan_to_num(h).reshape(6,-1);c=h@basis.T;t=a['times'][i];row=np.r_[c.ravel(),m.iloc[i].horizon_days/180,np.log1p(m.iloc[i].horizon_days)/6,np.ptp(t)/365,np.median(np.diff(t))/30,np.log(a['spacing'][i]),a['context'][i,0,16:20],np.log1p(a['scale'][i])];xs.append(row);bs.append(basis);coeffs.append(c);rec.append(float(abs(h-c@basis).mean()*a['scale'][i]))
  if targets:
   valid=np.isfinite(a['truth'][i]).ravel()&np.isfinite(a['last'][i]).ravel();b=basis[:,valid];dy=((a['truth'][i]-a['last'][i])/a['scale'][i]).ravel()[valid];ys.append(np.linalg.solve(b@b.T+np.eye(len(b))*.001,b@dy))
 return np.array(xs),np.array(ys),np.array(bs),np.array(coeffs),np.array(rec)

def fit(rep,method,m,a,ids):
 art=fit_basis(rep,m,a,ids);x,y,_,_,_=encode(art,m,a,ids,True);art['method']=method;art['scaler']=StandardScaler().fit(x);art['target_scaler']=StandardScaler().fit(y);xx=art['scaler'].transform(x);yy=art['target_scaler'].transform(y)
 if method in ['constant','linear','ema','kalman']:return art
 if method=='ridge':head=Ridge(alpha=100)
 elif method=='huber':head=MultiOutputRegressor(HuberRegressor(alpha=.1,max_iter=300))
 elif method=='RandomForest':head=RandomForestRegressor(n_estimators=60,max_depth=6,min_samples_leaf=8,random_state=SEED,n_jobs=1)
 elif method=='ExtraTrees':head=ExtraTreesRegressor(n_estimators=60,max_depth=6,min_samples_leaf=8,random_state=SEED,n_jobs=1)
 elif method=='LightGBM':
  from lightgbm import LGBMRegressor
  head=MultiOutputRegressor(LGBMRegressor(n_estimators=60,num_leaves=7,min_child_samples=20,reg_lambda=10,verbosity=-1,n_jobs=1,random_state=SEED))
 elif method=='GaussianProcess':
  head=GaussianProcessRegressor(kernel=RBF(length_scale=np.sqrt(x.shape[1]))+WhiteKernel(.1),alpha=.1,optimizer=None,normalize_y=False,random_state=SEED);ix=np.random.default_rng(SEED).choice(len(x),min(256,len(x)),replace=False);xx,yy=xx[ix],yy[ix]
 elif method=='PLS':head=PLSRegression(n_components=min(4,x.shape[1],len(ids)-1),scale=False,max_iter=300)
 head.fit(xx,yy);art['head']=head;return art

def predict(art,m,a,ids):
 x,_,basis,coeff,recon=encode(art,m,a,ids,False);method=art['method']
 if method in ['constant','linear','ema','kalman']:
  zs=np.array([latent_temporal(coeff[j],a['times'][i],m.iloc[i].horizon_days,method) for j,i in enumerate(ids)])
 else:zs=art['target_scaler'].inverse_transform(art['head'].predict(art['scaler'].transform(x)))
 pred=np.einsum('nr,nrp->np',zs,basis).reshape(-1,32,32)*a['scale'][ids,None,None]+a['last'][ids];return pred.astype('float32'),recon

def run():
 assert done('features');m,a,_=data();fs=folds(m);tr,cal,dev,test=partitions(m);threshold=readjson(ROOT/'manifests/thresholds.json')['pixel_extreme_mm'];reps=['pca4','pca8','svd4','robust_pca4','mask_pca4','block16','multiscale4','geometry4'];rows=[];timing=[]
 def evaluate(rep,method,folds0,stage):
  metrics=[];oof=np.full_like(a['truth'],np.nan)
  for fold,fitids,va in folds0:
   tick=time.perf_counter();art=fit(rep,method,m,a,fitids);ts=time.perf_counter()-tick;tick=time.perf_counter();p,recon=predict(art,m,a,va);inf=time.perf_counter()-tick;oof[va]=p;f=pd.DataFrame([dict(volcano=m.iloc[i].volcano,**score(a['truth'][i],p[j],a['last'][i],a['spacing'][i],threshold)) for j,i in enumerate(va)]);z=macro(f,['mae','extreme_mae','gradient_mae']);metrics.append(dict(model_id='spatial_'+rep+'_'+method,representation=rep,coefficient_algorithm=method,fold=fold,stage=stage,training_pairs=len(fitids),validation_pairs=len(va),historical_reconstruction_MAE=recon.mean(),training_seconds=ts,inference_seconds=inf,**z.to_dict()))
  return metrics,oof
 for rep in reps:
  rr,_=evaluate(rep,'ridge',[fs[1]],'representation_screen');rows.extend(rr);print('V9 REPRESENTATION',rep,rr[0]['mae'],flush=True)
 ranked=pd.DataFrame(rows).sort_values('mae');chosen=ranked.representation.head(2).tolist();final=[];preds={}
 for rep in chosen:
  for method in ['constant','linear','ridge','huber','ema','kalman','RandomForest','ExtraTrees','LightGBM','GaussianProcess','PLS']:
   try:
    rr,of=evaluate(rep,method,fs,'coefficient_CV');rows.extend(rr);final.append(dict(model_id='spatial_'+rep+'_'+method,representation=rep,method=method,mae=np.mean([r['mae'] for r in rr]),gradient_mae=np.mean([r['gradient_mae'] for r in rr]),extreme_mae=np.mean([r['extreme_mae'] for r in rr])));np.savez_compressed(ROOT/'predictions'/('OOF_spatial_'+rep+'_'+method+'.npz'),pred=of)
   except Exception as exc:rows.append(dict(model_id='spatial_'+rep+'_'+method,representation=rep,coefficient_algorithm=method,stage='failed',failure=str(exc)))
   history('spatial',rows)
 rank=pd.DataFrame(final).sort_values('mae');short=rank.head(1);eligible=rank[rank.mae<=rank.mae.min()*1.1];specialist=eligible.sort_values(['gradient_mae','extreme_mae']).head(1);short=pd.concat([short,specialist,rank[rank.coefficient_algorithm=='ridge'].head(1) if 'coefficient_algorithm' in rank else rank[rank.method=='ridge'].head(1)]).drop_duplicates('model_id')
 for r in short.itertuples():
  tick=time.perf_counter();art=fit(r.representation,r.method,m,a,tr);ts=time.perf_counter()-tick;joblib.dump(art,ROOT/'checkpoints'/(r.model_id+'.joblib'));tick=time.perf_counter();p,_=predict(art,m,a,np.arange(len(m)));preds[r.model_id]=p;timing.append(dict(model_id=r.model_id,training_seconds=ts,inference_seconds=time.perf_counter()-tick,training_pairs=len(tr),device='CPU'))
 f=csv(ROOT/'spatial_branch_results.csv',rows);csv(ROOT/'lowrank_representation_results.csv',f[f.stage=='representation_screen']);csv(ROOT/'lowrank_coefficient_forecasting.csv',f[f.stage!='representation_screen']);np.savez_compressed(ROOT/'predictions/spatial.npz',**preds);register(ROOT/'manifests/spatial_selection.json',dict(finalists=short.to_dict('records'),all_CV=final,criterion='Training-internal MAE champion plus gradient specialist within10% MAE and Ridge reference; no2023 labels'));csv(ROOT/'diagnostics/spatial_runtime.csv',timing);finish('spatial');print('V9 SPATIAL SEARCH COMPLETE',flush=True)
if __name__=='__main__':run()
