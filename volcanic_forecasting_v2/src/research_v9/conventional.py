"""Conventional sklearn models with fold-local preprocessing and target formulations."""
from .common import *
from .features import load,subset_indices,NAMES
from .protocol import ml_space
from ..research_v6.data import score
from sklearn import linear_model,ensemble,tree,neighbors,svm,set_config
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,RobustScaler
import joblib,time,inspect,warnings,copy
set_config(working_memory=64);sys.path.append(str(REPO/'mixture_experiment/vendor'));warnings.filterwarnings('ignore',message='X does not have valid feature names')

def estimator(cfg):
 fam=cfg['family'];p=dict(cfg['params'])
 if fam=='LightGBM':
  from lightgbm import LGBMRegressor
  return LGBMRegressor(**p,n_jobs=1,verbosity=-1,random_state=SEED)
 cls=getattr(linear_model,fam,None) or getattr(ensemble,fam,None) or getattr(tree,fam,None) or getattr(neighbors,fam,None) or getattr(svm,fam,None)
 if fam in ['RandomForestRegressor','ExtraTreesRegressor']:p.update(n_jobs=1,random_state=SEED)
 if fam in ['DecisionTreeRegressor','HistGradientBoostingRegressor']:p.update(random_state=SEED)
 if fam=='HistGradientBoostingRegressor':p['early_stopping']=False
 if fam=='KNeighborsRegressor':p.update(n_jobs=1,algorithm='brute')
 return cls(**p)

def parent_maps(a):
 old=np.load(V6/'deterministic/point_predictions.npz');return {'absolute':np.zeros_like(a['last']),'delta':a['last'],'normalized_delta':a['last'],'residual_EMA':old['ema_half30'],'residual_Kalman':old['kalman_localtrend']}

def fit(cfg,m,a,x,ids,parents):
 rng=np.random.default_rng(SEED);cols=subset_indices(cfg['features']);xx=[];yy=[];ww=[];counts=m.iloc[ids].volcano.value_counts()
 for i in ids:
  valid=np.flatnonzero(np.isfinite(a['truth'][i]).ravel()&np.isfinite(a['last'][i]).ravel());ix=rng.choice(valid,min(32,len(valid)),replace=False);xx.append(x[i][ix][:,cols]);yy.append((a['truth'][i]-parents[cfg['target']][i]).ravel()[ix]);ww.extend(np.full(len(ix),1/(counts[m.iloc[i].volcano]*len(ix))))
 xx=np.concatenate(xx);yy=np.concatenate(yy);ww=np.array(ww);maxrows=cfg['max_rows']
 if len(yy)>maxrows:
  ix=rng.choice(len(yy),maxrows,replace=False,p=ww/ww.sum());xx,yy,ww=xx[ix],yy[ix],ww[ix]
 offset=float(yy.mean()) if cfg['target']=='normalized_delta' else 0.;scale=max(float(yy.std()),1.) if cfg['target']=='normalized_delta' else 1.;model=estimator(cfg);scaler=StandardScaler() if cfg['scaler']=='standard' else RobustScaler();pipe=Pipeline([('scaler',scaler),('model',model)]);args={}
 if 'sample_weight' in inspect.signature(model.fit).parameters:args['model__sample_weight']=ww*len(ww)/ww.sum()
 with warnings.catch_warnings(record=True) as caught:
  pipe.fit(xx,(yy-offset)/scale,**args)
 return dict(pipeline=pipe,columns=cols,config=cfg,target_offset=offset,target_scale=scale,training_rows=len(yy),fit_warnings=sorted(set(str(w.message) for w in caught)),training_examples=ids.tolist())

def predict(art,a,x,ids,parents):
 out=[]
 for i in ids:
  z=art['pipeline'].predict(x[i][:,art['columns']]).reshape(32,32)*art['target_scale']+art['target_offset']+parents[art['config']['target']][i];z[~np.isfinite(a['last'][i])]=np.nan;out.append(z)
 return np.array(out,dtype='float32')

def cv(cfg,m,a,x,parents,folds0,screen=False):
 records=[];oof=np.full_like(a['truth'],np.nan);threshold=readjson(ROOT/'manifests/thresholds.json')['pixel_extreme_mm']
 for name,tr,va in folds0:
  if screen:va=m.iloc[va].sort_values('target_day').drop_duplicates(['source_id','origin_day']).index.to_numpy()
  tick=time.perf_counter();art=fit(cfg,m,a,x,tr,parents);trainsec=time.perf_counter()-tick;tick=time.perf_counter();p=predict(art,a,x,va,parents);infer=time.perf_counter()-tick;oof[va]=p;case=pd.DataFrame([dict(volcano=m.iloc[i].volcano,**score(a['truth'][i],p[j],a['last'][i],a['spacing'][i],threshold)) for j,i in enumerate(va)]);z=macro(case,['mae','extreme_mae','gradient_mae']);records.append(dict(model_id=cfg['id'],family=cfg['family'],fold=name,stage='screen' if screen else 'refined_CV',config=json.dumps(cfg),training_pairs=len(tr),training_rows=art['training_rows'],validation_pairs=len(va),training_seconds=trainsec,inference_seconds=infer,warnings=' | '.join(art['fit_warnings']),**z.to_dict()))
 return records,oof

def run():
 assert done('features');m,a,_=data();x=load();parents=parent_maps(a);fs=folds(m);tr,cal,dev,test=partitions(m);space=ml_space();rows=[];attempts=[];screen=[];oofdict={}
 for cfg in space:
  cache=ROOT/'search'/('ML_'+cfg['id']+'_screen.json')
  try:
   if cache.exists():rr=readjson(cache)
   else:rr,_=cv(cfg,m,a,x,parents,[fs[1]],True);writejson(cache,rr)
   rows.extend(rr);screen.append(dict(model_id=cfg['id'],family=cfg['family'],mae=np.mean([r['mae'] for r in rr])));print('V9 ML SCREEN',cfg['id'],rr[0]['mae'],flush=True)
  except Exception as exc:rows.append(dict(model_id=cfg['id'],family=cfg['family'],stage='screen_failed',config=json.dumps(cfg),failure=str(exc)))
  history('conventional',rows)
 rank=pd.DataFrame(screen).sort_values('mae');byid={c['id']:c for c in space};best=rank.groupby('family',sort=False).head(1);refinements=[]
 for mid in rank.groupby('family',sort=False).head(1).head(4).model_id:
  original=byid[mid]
  for j in range(2):
   c=copy.deepcopy(original);c['id']=original['id']+'_refine'+str(j);c['scaler']='robust' if j else 'standard'
   if 'alpha' in c['params']:c['params']['alpha']*=.3 if j==0 else 3
   elif 'min_samples_leaf' in c['params']:c['params']['min_samples_leaf']=max(8,int(c['params']['min_samples_leaf']*(.5 if j==0 else 2)))
   elif 'min_child_samples' in c['params']:c['params']['min_child_samples']=max(20,int(c['params']['min_child_samples']*(.5 if j==0 else 2)))
   c['target']='delta' if j==0 else 'residual_EMA';refinements.append(c)
 for c in refinements:
  try:rr,_=cv(c,m,a,x,parents,[fs[1]],True);rows.extend(rr);screen.append(dict(model_id=c['id'],family=c['family'],mae=np.mean([v['mae'] for v in rr])));byid[c['id']]=c
  except Exception as exc:rows.append(dict(model_id=c['id'],family=c['family'],stage='refine_failed',failure=str(exc),config=json.dumps(c)))
  history('conventional',rows)
 rank=pd.DataFrame(screen).sort_values('mae');best=rank.groupby('family',sort=False).head(1);final=[];allpred={};timing=[]
 for r in best.itertuples():
  cfg=byid[r.model_id];oofpath=ROOT/'predictions'/('OOF_ML_'+cfg['id']+'.npz');cache=ROOT/'search'/('ML_'+cfg['id']+'_cv.json')
  if cache.exists() and oofpath.exists():rr=readjson(cache);oof=np.load(oofpath)['pred']
  else:rr,oof=cv(cfg,m,a,x,parents,fs);writejson(cache,rr);np.savez_compressed(oofpath,pred=oof)
  rows.extend(rr);final.append(dict(model_id=cfg['id'],family=cfg['family'],CV_MAE=np.mean([v['mae'] for v in rr]),CV_extreme_MAE=np.mean([v['extreme_mae'] for v in rr]),config=cfg));path=ROOT/'checkpoints'/('ML_'+cfg['id']+'.joblib');tick=time.perf_counter()
  if path.exists():art=joblib.load(path)
  else:art=fit(cfg,m,a,x,tr,parents);joblib.dump(art,path)
  training=time.perf_counter()-tick;tick=time.perf_counter();p=predict(art,a,x,np.arange(len(m)),parents);allpred['ML_'+cfg['id']]=p;timing.append(dict(model_id='ML_'+cfg['id'],training_seconds=training,inference_seconds=time.perf_counter()-tick,training_rows=art['training_rows'],training_pairs=len(tr),device='CPU'));print('V9 ML FAMILY FINALIST',cfg['family'],flush=True);history('conventional',rows)
 register(ROOT/'manifests/conventional_selection.json',dict(finalists=sorted(final,key=lambda r:r['CV_MAE']),criterion='Training-internal3fold macro-volcano MAE only; no2023 or assessment selection',refinement_configs=refinements));np.savez_compressed(ROOT/'predictions/conventional.npz',**allpred);csv(ROOT/'conventional_ml_results.csv',rows);csv(ROOT/'target_formulation_results.csv',[r for r in rows if 'Ridge_target_' in r.get('model_id','')]);csv(ROOT/'feature_ablation_results.csv',[r for r in rows if 'Ridge_features_' in r.get('model_id','')]);csv(ROOT/'diagnostics/conventional_runtime.csv',timing);finish('conventional');print('V9 CONVENTIONAL SEARCH COMPLETE',flush=True)
if __name__=='__main__':run()
