"""Matched deterministic historical LOS forecasts; old artifacts remain untouched."""
from .common import *
from .population import load
from ..stage1.models import CONTROLS,LocalFit,GlobalRidge,features,fill_past
from scipy.ndimage import gaussian_filter
from scipy.linalg import solve
import time,joblib

EXTRA=[dict(id='v3_robust_trend',family='robust_trend',history=6),dict(id='v3_kalman_q001',family='kalman',history=6,q=.001),dict(id='v3_kalman_q01',family='kalman',history=6,q=.01),dict(id='v3_gp_l30',family='gp',history=6,length=30),dict(id='v3_gp_l90',family='gp',history=6,length=90)]

def custom(m,t,h,cfg):
 f,v=fill_past(m);last=f[-1];t=np.array(t,float);dt=np.diff(t);noise=max(float(np.median(np.std(np.diff(f,axis=0),axis=0))),1.)
 if cfg['family']=='robust_trend':p=last+np.median(np.diff(f,axis=0)/dt[:,None,None],axis=0)*h*np.exp(-h/90)
 elif cfg['family']=='gp':
  dist=abs(t[:,None]-t[None,:]);l=cfg['length'];k=(1+np.sqrt(3)*dist/l)*np.exp(-np.sqrt(3)*dist/l)+np.eye(len(t))*.1;distance=t[-1]+h-t;cross=(1+np.sqrt(3)*distance/l)*np.exp(-np.sqrt(3)*distance/l);p=last+(cross@np.linalg.solve(k,(f-last).reshape(len(t),-1))).reshape(last.shape)
 else:
  level=f[0].copy();velocity=np.zeros_like(last);p00=np.full(last.shape,noise**2*4);p01=np.zeros_like(last);p11=np.full(last.shape,noise**2/900);r=noise**2;q=r*cfg['q']/30
  for j,d in enumerate(dt,1):
   level=level+velocity*d;p00=p00+2*d*p01+d*d*p11+q*d**3/3;p01=p01+d*p11+q*d*d/2;p11=p11+q*d;innovation=np.clip(f[j]-level,-3*np.sqrt(p00+r),3*np.sqrt(p00+r));den=p00+r;k0=p00/den;k1=p01/den;valid=v[j];level=np.where(valid,level+k0*innovation,level);velocity=np.where(valid,velocity+k1*innovation,velocity);old01=p01.copy();p00=np.where(valid,(1-k0)*p00,p00);p01=np.where(valid,(1-k0)*p01,p01);p11=np.where(valid,p11-k1*old01,p11)
  p=level+velocity*h*np.exp(-h/90)
 p=np.asarray(p,dtype='float32');p[~v.any(0)]=np.nan;return p

def gradient_error(y,p,valid,sp):
 errors=[]
 for axis,delta in [(0,sp[0]),(1,sp[1])]:
  a=[slice(None),slice(None)];b=a.copy();a[axis]=slice(1,None);b[axis]=slice(None,-1);a=tuple(a);b=tuple(b);mask=valid[a]&valid[b]
  if mask.any():errors.append(abs((p[a]-p[b])-(y[a]-y[b]))[mask]/delta)
 return float(np.concatenate(errors).mean()) if errors else np.nan

def score(y,p,last,sp):
 v=np.isfinite(y)&np.isfinite(p)&np.isfinite(last);err=(p-y)[v];change=(y-last)[v];predchange=(p-last)[v];ext=abs(change)>=np.quantile(abs(change),.9);gy,gx=np.gradient(np.nan_to_num(y),*sp);py,px=np.gradient(np.nan_to_num(p),*sp)
 return dict(mae=float(abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),extreme_mae=float(abs(err[ext]).mean()),gradient_mae_mm_km=gradient_error(y,p,v,sp),magnitude_bias=float((abs(predchange)-abs(change)).mean()),forecast_coverage=float(np.isfinite(p).mean()))

def macro(frame):return frame.groupby(['source_id','volcano'])[['mae','rmse','extreme_mae','gradient_mae_mm_km']].mean().groupby('volcano').mean().mean()

def prepare():
 p=experiment(19);sources=readjson(ROOT/'manifests/insar_sources.json');split=pd.read_csv(ROOT/'splits/stage1_v3.csv');hist=[];times=[];truth=[];last=[];sp=[];records=[]
 for r in sources:
  g=split[split.source_id==r['source_id']]
  if g.empty:continue
  maps,t=load(r)
  for row in g.itertuples():
   m=maps[row.origin_index-5:row.origin_index+1];filled,v=fill_past(m);seen=v.any(0);h=filled.copy();h[:,~seen]=np.nan;hist.append(m);times.append(t[row.origin_index-5:row.origin_index+1]);truth.append(maps[row.target_index]);last.append(np.where(seen,filled[-1],np.nan));sp.append([r['spacing_y_km'],r['spacing_x_km']]);records.append(row._asdict())
 meta=pd.DataFrame(records).drop(columns=['Index'],errors='ignore');np.savez_compressed(ROOT/'processed/stage1_examples.npz',history=np.array(hist),times=np.array(times),truth=np.array(truth),last=np.array(last),spacing=np.array(sp));csv(ROOT/'processed/stage1_examples.csv',meta);return meta,np.array(hist),np.array(times),np.array(truth),np.array(last),np.array(sp)

def run():
 p=experiment(19)
 for folder in ['configs','models','predictions']:(p/folder).mkdir(exist_ok=True)
 meta,m,t,y,last,spacing=prepare();train=np.flatnonzero(meta.role=='train');validation=np.flatnonzero(meta.role=='validation');rng=np.random.default_rng(SEED);x=[];target=[];weights=[];sample_indices={};counts=meta.iloc[train].volcano.value_counts();runtime=[]
 for i in train:
  ix=np.flatnonzero(np.isfinite(y[i]).ravel()&np.isfinite(last[i]).ravel());ix=rng.choice(ix,min(32,len(ix)),replace=False);sample_indices[i]=ix;x.append(features(m[i],t[i],meta.iloc[i].horizon_days,spacing[i])[ix]);target.append((y[i]-last[i]).ravel()[ix]);weights.extend([1/(counts[meta.iloc[i].volcano]*len(ix))]*len(ix))
 xx=np.concatenate(x);yy=np.concatenate(target);weights=np.array(weights);ridge={}
 for alpha in [1,10,100]:
  tick=time.perf_counter();mid=f'v3_ridge_a{alpha}';ridge[mid]=GlobalRidge().fit(xx,yy,weights,alpha);runtime.append(dict(model_id=mid,fit_seconds=time.perf_counter()-tick,fit_type='global_train_only',training_pairs=len(train),training_pixels=len(yy)))
 # Recent-training ablation shares the same evaluation population and fixed alpha10; no old headline is reinterpreted.
 late=[i for i in train if meta.iloc[i].origin_day>=day('2017-01-01')]
 if len(late)>=30:
  rows=[];tg=[];ww=[];late_counts=meta.iloc[late].volcano.value_counts()
  for i in late:
   ix=sample_indices[i];rows.append(features(m[i],t[i],meta.iloc[i].horizon_days,spacing[i])[ix]);tg.append((y[i]-last[i]).ravel()[ix]);ww.extend([1/(late_counts[meta.iloc[i].volcano]*max(len(ix),1))]*len(ix))
  tick=time.perf_counter();ridge['v3_ridge_recent2017']=GlobalRidge().fit(np.concatenate(rows),np.concatenate(tg),np.array(ww),10);runtime.append(dict(model_id='v3_ridge_recent2017',fit_seconds=time.perf_counter()-tick,fit_type='recent_training_ablation',training_pairs=len(late)))
 configs=CONTROLS+EXTRA;allmodels=[c['id'] for c in configs]+list(ridge);predictions={mid:np.zeros_like(y) for mid in allmodels};rows=[];fitcache={};wall=time.perf_counter()
 # Freeze model selection based on validation only BEFORE scoring either new evaluation role.
 order=[*train,*validation,*np.flatnonzero(meta.role.isin(['temporal_evaluation','unseen_volcano']))];selection_written=False;validation_rows=[]
 for j,i in enumerate(order):
  r=meta.iloc[i];is_eval=r.role in ['temporal_evaluation','unseen_volcano']
  if is_eval and not selection_written:
   scores=pd.DataFrame(validation_rows);summary=[]
   for mid,g in scores.groupby('model_id'):summary.append(dict(model_id=mid,**macro(g).to_dict()))
   ranking=csv(p/'validation_selection.csv',summary).sort_values(['mae','model_id']);winner=ranking.iloc[0].model_id;writejson(p/'frozen_selection.json',dict(selected_model=winner,selection='validation macro MAE, source views averaged within volcano',frozen_utc=utc(),dataset_version=VERSION,evaluation_version=str(meta.iloc[0].evaluation_version)));selection_written=True
  for cfg in configs:
   mid=cfg['id'];tick=time.perf_counter()
   if cfg['family'] in ['persistence','ema','lowrank','quality']:
    key=(r.source_id,r.origin_index,mid)
    if key not in fitcache:fitcache[key]=LocalFit(m[i],t[i],cfg)
    pred=fitcache[key].predict(r.horizon_days)
   else:pred=custom(m[i],t[i],r.horizon_days,cfg)
   predictions[mid][i]=pred;record=dict(model_id=mid,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,frame=r.frame,role=r.role,horizon_days=r.horizon_days,origin_day=r.origin_day,inference_seconds=time.perf_counter()-tick,**score(y[i],pred,last[i],spacing[i]));rows.append(record)
   if r.role=='validation':validation_rows.append(record)
  for mid,model in ridge.items():
   tick=time.perf_counter();pred=model.predict(m[i],t[i],r.horizon_days,spacing[i]);predictions[mid][i]=pred;record=dict(model_id=mid,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,frame=r.frame,role=r.role,horizon_days=r.horizon_days,origin_day=r.origin_day,inference_seconds=time.perf_counter()-tick,**score(y[i],pred,last[i],spacing[i]));rows.append(record)
   if r.role=='validation':validation_rows.append(record)
  if j%300==0:print('Stage1cases',j,'/',len(order),flush=True)
 if not selection_written:raise RuntimeError('No evaluation cases; check protocol')
 f=csv(p/'predictions/point_predictions.csv',rows);summ=[]
 for (role,mid),g in f.groupby(['role','model_id']):summ.append(dict(role=role,model_id=mid,**macro(g).to_dict(),pairs=len(g),volcanoes=g.volcano.nunique()))
 csv(p/'metrics.csv',summ);csv(p/'per_volcano_metrics.csv',f.groupby(['role','model_id','volcano'])[['mae','rmse','extreme_mae','gradient_mae_mm_km']].mean().reset_index());csv(p/'horizon_metrics.csv',f.groupby(['role','model_id','horizon_days'])[['mae','rmse','extreme_mae','gradient_mae_mm_km']].mean().reset_index());np.savez_compressed(ROOT/'processed/stage1_point_predictions.npz',**predictions)
 for mid,model in ridge.items():joblib.dump(model,p/'models'/f'{mid}.joblib')
 for c in configs:writejson(p/'configs'/f"{c['id']}.json",dict(version=VERSION,config=c,fit_type='new local fit per historical origin; no old weights changed'))
 csv(p/'training_times.csv',runtime);writejson(p/'runtime.json',dict(total_prediction_seconds=time.perf_counter()-wall,examples=len(meta),unique_origins=len(meta[['source_id','origin_index']].drop_duplicates()),hardware='CPU',old_artifacts_unchanged=verify_frozen()));print('STAGE1COMPLETE',winner,flush=True)
if __name__=='__main__':run()
