"""Restoredlocal/globaldeterministicfamilies, four-branchhybrids and compactCNNcopies."""
from .common import *
from .data import examples,score,macro
from ..stage1.models import LocalFit,CONTROLS,fill_past
from ..historical_v3.stage1 import custom
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import time,joblib,torch,torch.nn as nn,copy,warnings
warnings.filterwarnings('ignore',message='X does not have valid feature names')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')

class MapCNN(nn.Module):
 def __init__(self,channels):
  super().__init__();self.net=nn.Sequential(nn.Conv2d(channels,12,3,padding=1),nn.SiLU(),nn.Conv2d(12,12,3,padding=1),nn.SiLU(),nn.Conv2d(12,1,1));nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
 def forward(self,x):return self.net(x)[:,0]

def local_level(m,t):
 f,v=fill_past(m);level=f[0].astype(float);noise=max(float(np.median(abs(np.diff(f,axis=0)))),1.);variance=np.full((32,32),noise**2*10)
 for j,dt in enumerate(np.diff(t),1):
  variance+=.03*noise**2*dt/30;innovation=f[j]-level;obsvar=noise**2*(1+np.minimum((innovation/(3*np.sqrt(variance+noise**2)))**2,25));gain=variance/(variance+obsvar);level=np.where(v[j],level+gain*innovation,level);variance=np.where(v[j],(1-gain)*variance,variance)
 level[~v.any(0)]=np.nan;return level.astype('float32')

def local_configs():
 out=[dict(id='persistence',family='persistence',history=6)]
 for k in [6,12,18]:
  for rank in [1,3,5]:
   for robust in [False,True]:
    for method in ['linear','damped90']:out.append(dict(id=f'lr_h{k}_r{rank}_rob{int(robust)}_{method}',family='lowrank',history=k,rank=rank,robust=robust,alpha=4.,method=method))
 for half in [30,120]:out.append(dict(id=f'ema_half{half}',family='ema',history=6,half=half))
 out.extend([dict(c) for c in CONTROLS if c['id'] in ['quality','quality_v2_h6_half30_w0','lr_v2_h6_r1_rob0_a0.5']]);return out

def fit_context(meta,a,indices,kind='ridge'):
 rng=np.random.default_rng(SEED);xs=[];ys=[];ws=[];counts=meta.iloc[indices].volcano.value_counts()
 for i in indices:
  v=np.flatnonzero(np.isfinite(a['last'][i]).ravel()&np.isfinite(a['truth'][i]).ravel());ix=rng.choice(v,min(48,len(v)),replace=False);xs.append(a['context'][i][ix]);ys.append(((a['truth'][i]-a['last'][i])/a['scale'][i]).ravel()[ix]);ws.extend([1/(counts[meta.iloc[i].volcano]*len(ix))]*len(ix))
 x=np.concatenate(xs);y=np.concatenate(ys);w=np.array(ws);scaler=RobustScaler().fit(x);z=np.clip(scaler.transform(x),-6,6)
 if kind=='lightgbm':
  sys.path.append(str(REPO/'mixture_experiment/vendor'));from lightgbm import LGBMRegressor
  model=LGBMRegressor(n_estimators=80,num_leaves=15,min_child_samples=100,reg_lambda=10.,n_jobs=1,verbosity=-1,random_state=SEED).fit(z,y,sample_weight=w*len(w)/w.sum())
 else:model=Ridge(alpha=100).fit(z,y,sample_weight=w*len(w)/w.sum())
 return dict(model=model,scaler=scaler,kind=kind)

def context_predict(artifact,a,indices):
 out=[]
 for i in indices:
  p=a['last'][i]+artifact['model'].predict(np.clip(artifact['scaler'].transform(a['context'][i]),-6,6)).reshape(32,32)*a['scale'][i];out.append(p)
 return np.array(out,dtype='float32')

def train_cnn(meta,a,train,cal,epochs=80):
 device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2);torch.manual_seed(SEED);net=MapCNN(a['cnn'].shape[1]).to(device);optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);target=np.nan_to_num((a['truth']-a['last'])/a['scale'][:,None,None]);valid=np.isfinite(a['truth'])&np.isfinite(a['last']);x=torch.tensor(a['cnn'],device=device);y=torch.tensor(target,dtype=torch.float32,device=device);v=torch.tensor(valid,dtype=torch.float32,device=device);best=None;wait=0;curves=[];rng=np.random.default_rng(SEED)
 for epoch in range(epochs):
  net.train();losses=[]
  for ids in np.array_split(rng.permutation(train),max(1,int(np.ceil(len(train)/64)))):
   optim.zero_grad();p=net(x[ids]);loss=(abs(p-y[ids])*v[ids]).sum()/v[ids].sum().clamp_min(1);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optim.step();losses.append(float(loss.detach().cpu()))
  net.eval()
  with torch.no_grad():pp=net(x[cal]);vl=float((abs(pp-y[cal])*v[cal]).sum()/v[cal].sum())
  curves.append(dict(epoch=epoch,train_loss=np.mean(losses),calibration_loss=vl))
  if best is None or vl<best[0]-1e-5:best=(vl,copy.deepcopy(net.state_dict()),epoch);wait=0
  else:wait+=1
  if wait>=15:break
 net.load_state_dict(best[1]);net.eval();pred=[]
 with torch.no_grad():
  for begin in range(0,len(meta),64):pred.extend(net(x[begin:begin+64]).cpu().numpy())
 out=np.array(pred)*a['scale'][:,None,None]+a['last'];return net,out.astype('float32'),curves,device,best[2]

def run():
 start=time.perf_counter();p=ROOT/'deterministic';meta,a,sources=examples();train=np.flatnonzero(meta.role=='train');val=np.flatnonzero(meta.role=='validation');cal=np.array([i for i in val if meta.iloc[i].target_day<day('2023-01-01')]);select=np.array([i for i in val if meta.iloc[i].origin_day>=day('2023-01-01')]);assert len(train)>50 and len(cal)>10 and len(select)>10;register(p/'protocol.json',dict(version=VERSION,model_fitting='Traininglabelsonly; no previouslyprotectedsourcepromotion',early_stopping_and_hybrid_weights='2022developmenttargets',selection='2023developmentMAEmacro; freezebefore2024/geoerrors',models='Persistence,EMA,robust/selectedLR6/12/18ranks1/3/5,quality,Kalman,contextRidge/LightGBM,CNN,four-branchhybrids',CNN_epochs=80,primary_comparison='Common>=18actualmaphistorysubset for allconfigcomparisons; full6reference alsoexported',extremes='Training90thpercentileabsfuturechange; nofuturedatatoprediction',seed=SEED))
 extreme=float(np.nanquantile(abs(a['truth'][train]-a['last'][train]),.9));writejson(p/'thresholds.json',dict(extreme_pixel_mm=extreme,large_case_mean_change_mm=float(np.nanquantile(np.nanmean(abs(a['truth'][train]-a['last'][train]),axis=(1,2)),.9))));pred={};times=[];configs=local_configs();sourcecache={k:load_source(r) for k,r in sources.items()};cache={};allids=np.arange(len(meta))
 for cfg in configs:pred[cfg['id']]=np.full_like(a['truth'],np.nan)
 for mid in ['kalman_localtrend','kalman_locallevel']:pred[mid]=np.full_like(a['truth'],np.nan)
 for i,r in meta.iterrows():
  maps,tt,sp=sourcecache[r.source_id]
  for cfg in configs:
   if r.origin_index+1<cfg['history']:continue
   key=(r.source_id,r.origin_index,cfg['id']);tick=time.perf_counter()
   if key not in cache:cache[key]=LocalFit(maps[r.origin_index-cfg['history']+1:r.origin_index+1],tt[r.origin_index-cfg['history']+1:r.origin_index+1],cfg)
   pred[cfg['id']][i]=cache[key].predict(r.horizon_days);times.append(dict(model_id=cfg['id'],phase='localfit_cached_plusforecast',seconds=time.perf_counter()-tick,example_id=r.example_id))
  for mid,q in [('kalman_localtrend',.01),('kalman_locallevel',.001)]:
   tick=time.perf_counter();pred[mid][i]=local_level(a['history'][i],a['times'][i]) if mid=='kalman_locallevel' else custom(a['history'][i],a['times'][i],r.horizon_days,dict(family='kalman',q=q));times.append(dict(model_id=mid,phase='localfit_plusforecast',seconds=time.perf_counter()-tick,example_id=r.example_id))
  if i%250==0:print('DETERMINISTICLOCAL',i,'/',len(meta),flush=True)
 models={};fail=[]
 for kind in ['ridge','lightgbm']:
  tick=time.perf_counter()
  try:
   artifact=fit_context(meta,a,train,kind);fit_seconds=time.perf_counter()-tick;joblib.dump(artifact,p/(kind+'_context.joblib'));models[kind]=artifact;tick=time.perf_counter();pred['context_'+kind]=context_predict(artifact,a,allids);times.extend([dict(model_id='context_'+kind,phase='training',seconds=fit_seconds),dict(model_id='context_'+kind,phase='inference_all_examples',seconds=time.perf_counter()-tick)])
  except Exception as e:fail.append(dict(model_id=kind,status='unavailable',reason=str(e)))
 tick=time.perf_counter();net,out,curves,device,best=train_cnn(meta,a,train,cal);times.append(dict(model_id='CNN',phase='training_plus_full_prediction',seconds=time.perf_counter()-tick,device=device));pred['CNN']=out;torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},channels=a['cnn'].shape[1],best_epoch=best,version=VERSION),p/'CNN.pt');csv(p/'CNN_training_curves.csv',curves)
 context='context_lightgbm' if 'context_lightgbm' in pred else 'context_ridge';branches=['kalman_localtrend','lr_h6_r3_rob1_linear',context,'quality'];stack=np.stack([pred[mid] for mid in branches]);pred['four_branch_equal']=stack.mean(0)
 # Constrainedweights fitted on2022only with fixedpixelmask; no heldtestoptimization.
 sample=[];target=[];weight=[];counts=meta.iloc[cal].volcano.value_counts()
 for i in cal:
  mask=np.isfinite(a['truth'][i])&np.isfinite(stack[:,i]).all(0);ix=np.flatnonzero(mask.ravel())[::8];sample.append(stack[:,i].reshape(4,-1)[:,ix].T);target.append(a['truth'][i].ravel()[ix]);weight.extend([1/(counts[meta.iloc[i].volcano]*len(ix))]*len(ix))
 xx=np.concatenate(sample);yy=np.concatenate(target);ww=np.array(weight);objective=lambda w:float(np.average(abs(xx@w-yy),weights=ww));fit=minimize(objective,np.full(4,.25),method='SLSQP',bounds=[(0,1)]*4,constraints={'type':'eq','fun':lambda w:w.sum()-1},options={'maxiter':200,'ftol':1e-8});weights=fit.x/fit.x.sum();pred['four_branch_constrained']=np.einsum('b,bnhw->nhw',weights,stack);variability=np.nanmedian(abs(a['history'][:,-1]-a['history'][:,0]),axis=(1,2));threshold=float(np.median(variability[train]));gate=np.clip(variability/(threshold+1),0,1);pred['four_branch_regime']=(1-gate[:,None,None])*pred['quality']+gate[:,None,None]*pred['four_branch_constrained'];writejson(p/'hybrid_weights.json',dict(branches=branches,nonnegative_sum_one_weights=weights.tolist(),fit_targets='2022developmentonly',converged=bool(fit.success),regime_threshold_train=threshold))
 common=meta.origin_index>=17;validation=[]
 for mid,pp in pred.items():
  rows=[]
  for i in select:
   if not common.iloc[i]:continue
   rows.append(dict(source_id=meta.iloc[i].source_id,volcano=meta.iloc[i].volcano,**score(a['truth'][i],pp[i],a['last'][i],a['spacing'][i],extreme)))
  validation.append(dict(model_id=mid,**macro(pd.DataFrame(rows)).to_dict()))
 rank=csv(p/'validation_selection.csv',validation).sort_values('mae');winner=rank.iloc[0].model_id;writejson(p/'frozen_selection.json',dict(selected_model=winner,frozen_utc=utc(),selection='2023developmentMAE; same>=18mapeligiblecohort',version=VERSION,extreme_pixel_threshold=extreme,protected_outcomes_selected=False))
 rows=[]
 for mid,pp in pred.items():
  for i,r in meta.iterrows():
   if not np.isfinite(pp[i]).any():continue
   rows.append(dict(model_id=mid,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,region=r.region,sensor=r.sensor,role=r.role,origin_day=r.origin_day,target_day=r.target_day,horizon_days=r.horizon_days,common18_cohort=bool(common.iloc[i]),**score(a['truth'][i],pp[i],a['last'][i],a['spacing'][i],extreme)))
 f=csv(p/'predictions.csv',rows);summ=[]
 for (role,mid),g in f[f.common18_cohort].groupby(['role','model_id']):summ.append(dict(role=role,model_id=mid,**macro(g).to_dict(),map_pairs=len(g),volcanoes=g.volcano.nunique()))
 mm=csv(ROOT/'deterministic_benchmark.csv',summ);csv(ROOT/'low_rank_experiments.csv',mm[mm.model_id.str.startswith('lr_')]);csv(ROOT/'four_branch_hybrid_experiments.csv',mm[mm.model_id.str.startswith('four_branch')]);csv(p/'per_volcano.csv',f[f.common18_cohort].groupby(['role','model_id','volcano'])[['mae','rmse','bias','extreme_mae','gradient_mae']].mean().reset_index());csv(ROOT/'variable_horizon_results.csv',f[f.common18_cohort].groupby(['role','model_id','horizon_days'])[['mae','rmse','extreme_mae']].mean().reset_index());csv(p/'runtime.csv',times);csv(p/'model_failures.csv',fail);np.savez_compressed(p/'point_predictions.npz',**pred);writejson(p/'status.json',dict(status='completed',selected_model=winner,models=len(pred),wall_seconds=time.perf_counter()-start,CNNdevice=device,CNNparameters=sum(v.numel() for v in net.parameters()),parents=protect()));print('DETERMINISTIC COMPLETE',winner,flush=True)
if __name__=='__main__':
 from .benchmark import run as canonical_run
 canonical_run()
