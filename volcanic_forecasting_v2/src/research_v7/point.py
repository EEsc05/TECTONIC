"""Past-only low-rank temporal decomposition and matched diagnostic controls."""
from .common import *
from ..stage1.models import LocalFit,fill_past,smooth
from ..research_v6.data import score
from ..research_v6.benchmark import local_level
from ..historical_v3.stage1 import custom
from scipy.ndimage import gaussian_filter
import time,joblib

def configurations():
 c=[dict(id=f'lr_rob_r{r}_{t}',rank=r,temporal=t,representation='robust') for r in [1,3] for t in ['constant','linear','damped','ema','kalman','robust_trend']]
 c += [dict(id='lr_plain_r3_'+t,rank=3,temporal=t,representation='ordinary') for t in ['constant','linear','damped','ema','kalman','robust_trend']]
 c += [dict(id='lr_'+rep+'_r3_damped',rank=3,temporal='damped',representation=rep) for rep in ['mask_EM','spatial_normalized','multiscale','shared_training_basis','geometry_conditioned_shared','sparse_residual']]
 return c

def temporal(c,t,h,kind):
 if kind=='constant':return c[-1]
 if kind=='ema':
  w=np.exp(-np.log(2)*(t[-1]-t)/30);return np.average(c,axis=0,weights=w)
 if kind=='kalman':
  level=c[0].copy();var=np.ones(c.shape[1]);noise=np.maximum(np.median(abs(np.diff(c,axis=0)),axis=0)**2,1.)
  for j,dt in enumerate(np.diff(t),1):
   var+=.03*noise*dt/30;gain=var/(var+noise);level+=gain*(c[j]-level);var*=1-gain
  return level
 if kind=='robust_trend':
  slopes=np.array([(c[j]-c[i])/(t[j]-t[i]) for i in range(len(t)) for j in range(i+1,len(t))]);slope=np.median(slopes,0);return np.median(c-(t-t[-1])[:,None]*slope,0)+slope*h*np.exp(-h/90)
 x=np.c_[np.ones(len(t)),(t-t[-1])/90];beta=np.linalg.solve(x.T@x+np.diag([0,4.]),x.T@c);return beta[0]+beta[1]*h/90*(np.exp(-h/90) if kind=='damped' else 1)

def lowrank(m,t,h,cfg,shared=None,los=None):
 f,v=fill_past(m);shape=f.shape[1:];x=f.reshape(len(t),-1).astype(float);rep=cfg['representation'];original=x.copy();norm=np.ones(x.shape[1]);offset=np.zeros(x.shape[1])
 if rep!='ordinary':
  med=np.median(x,0);mad=np.maximum(np.median(abs(x-med),0)*1.4826,1);x=np.clip(x,med-3*mad,med+3*mad)
 if rep=='spatial_normalized':norm=np.maximum(np.std(x,0),1);offset=x.mean(0);x=(x-offset)/norm
 if rep=='multiscale':x=np.stack([gaussian_filter(y.reshape(shape),1).ravel() for y in x])
 mean=x.mean(0);center=x-mean;rank=min(cfg['rank'],len(t)-1)
 if rep in ['shared_training_basis','geometry_conditioned_shared']:
  key='all' if rep=='shared_training_basis' else 'all' if los is None else 'ascending' if los[0]<0 else 'descending';basis=shared.get(key,shared['all'])[:rank];coeff=center@basis.T
 else:
  if rep=='mask_EM':
   valid=v.reshape(len(t),-1)
   for iteration in range(4):
    mean=x.mean(0);z=x-mean;e,u=np.linalg.eigh(z@z.T);ix=np.argsort(e)[::-1][:rank];recon=u[:,ix]@u[:,ix].T@z+mean;x=np.where(valid,original,recon)
   mean=x.mean(0);center=x-mean
  e,u=np.linalg.eigh(center@center.T);ix=np.argsort(e)[::-1][:rank];u=u[:,ix];ss=np.sqrt(np.maximum(e[ix],0));basis=np.divide(u.T@center,ss[:,None],out=np.zeros((rank,x.shape[1])),where=ss[:,None]>1e-8);coeff=center@basis.T
 forecast=(mean+temporal(coeff,t,h,cfg['temporal'])@basis)*norm+offset;recon=(mean+coeff@basis)*norm+offset
 if rep in ['sparse_residual','multiscale']:forecast+=(original[-1]-recon[-1])*np.exp(-h/30)
 out=forecast.reshape(shape).astype('float32');out[~v.any(0)]=np.nan
 return out,float(abs(recon-original)[v.reshape(len(t),-1)].mean())

def controls(m,t,h):
 f,v=fill_past(m);rates=np.array([(f[j]-f[i])/(t[j]-t[i]) for i in range(len(t)) for j in range(i+1,len(t))]);trend=np.median(rates,0);recent=(f[-1]-f[-2])/(t[-1]-t[-2]);level=smooth(m,t,30);med=np.median(f,0);mad=np.maximum(np.median(abs(f-med),0)*1.4826,1);clean=np.clip(f,med-3*mad,med+3*mad);ct=np.median(np.diff(clean,axis=0)/np.diff(t)[:,None,None],0);out={'recent_rate':f[-1]+recent*h,'robust_trend':f[-1]+trend*h,'EMA_trend_damped':level+trend*h*np.exp(-h/90),'quality_trend_damped':smooth(np.where(v,clean,np.nan),t,30)+ct*h*np.exp(-h/90)}
 for key in out:out[key]=np.where(v.any(0),out[key],np.nan).astype('float32')
 return out

def metric_rows(meta,a,pred,ids):
 threshold=readjson(ROOT/'manifests/thresholds.json');out=[]
 for name,p in pred.items():
  for i in ids:
   r=meta.iloc[i];valid=np.isfinite(a['truth'][i])&np.isfinite(p[i])&np.isfinite(a['last'][i]);y=a['truth'][i]-a['last'][i];d=p[i]-a['last'][i];actual=abs(y)>threshold['pixel_extreme_mm'];forecast=abs(d)>threshold['pixel_extreme_mm'];intersection=(actual&forecast&valid).sum();union=((actual|forecast)&valid).sum();event=float(np.nanmean(abs(y))>threshold['map_extreme_mm']);pe=float(np.nanmean(abs(d))>threshold['map_extreme_mm']);out.append(dict(model_id=name,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,region=r.region,role=r.role,origin_day=r.origin_day,horizon_days=r.horizon_days,common18=bool(r.origin_index>=17),map_event=event,event_probability=pe,event_brier=(pe-event)**2,localization_IoU=intersection/union if union else np.nan,**score(a['truth'][i],p[i],a['last'][i],a['spacing'][i],threshold['pixel_extreme_mm'])))
 return out

def run():
 assert done('protocol');start=time.perf_counter();m,a,sources=data();tr,cal,sel,test=partitions(m);v6=np.load(V6/'deterministic/point_predictions.npz');pred={name:v6[name] for name in v6.files};ticks=[];threshold=dict(pixel_extreme_mm=float(np.nanquantile(abs(a['truth'][tr]-a['last'][tr]),.9)),map_extreme_mm=float(np.quantile(np.nanmean(abs(a['truth'][tr]-a['last'][tr]),axis=(1,2)),.9)));register(ROOT/'manifests/thresholds.json',threshold)
 # Shared spatial basis uses historical maps of training examples only, never test future maps.
 shared={};baseids=m.iloc[tr].drop_duplicates(['source_id','origin_day']).index
 for key in ['all','ascending','descending']:
  ids=[i for i in baseids if key=='all' or (sources[m.iloc[i].source_id]['los_e_mean'] is not None and (sources[m.iloc[i].source_id]['los_e_mean']<0)==(key=='ascending'))];xx=np.nan_to_num((a['history'][ids,-1]-a['history'][ids,0])/a['scale'][ids,None,None]).reshape(len(ids),-1);_,_,vt=np.linalg.svd(xx,full_matrices=False);shared[key]=vt[:8]
 joblib.dump(shared,ROOT/'checkpoints/shared_historical_basis.joblib');reconstruction=[];cfgs=configurations();newnames=[c['id'] for c in cfgs]+['recent_rate','robust_trend','EMA_trend_damped','quality_trend_damped'];new={name:np.full_like(a['truth'],np.nan) for name in newnames};cache={};elapsed={k:0. for k in newnames}
 for i,r in m.iterrows():
  los=[sources[r.source_id][k] for k in ['los_e_mean','los_n_mean','los_u_mean']];los=los if all(z is not None for z in los) else None
  for cfg in cfgs:
   t0=time.perf_counter();pp,rec=lowrank(a['history'][i],a['times'][i],r.horizon_days,cfg,shared,los);new[cfg['id']][i]=pp;elapsed[cfg['id']]+=time.perf_counter()-t0;reconstruction.append(dict(example_id=r.example_id,model_id=cfg['id'],role=r.role,volcano=r.volcano,past_only_reconstruction_mae=rec,forecast_MAE=float(np.nanmean(abs(pp-a['truth'][i]))),future_used_for_basis=False))
  t0=time.perf_counter();cc=controls(a['history'][i],a['times'][i],r.horizon_days)
  for name,pp in cc.items():new[name][i]=pp;elapsed[name]+=(time.perf_counter()-t0)/len(cc)
  if i%300==0:print('V7 LOWRANK',i,len(m),flush=True)
 np.savez_compressed(ROOT/'predictions/new_point.npz',**new);pred.update(new);f=csv(ROOT/'predictions/point_metrics.csv',metric_rows(m,a,pred,np.arange(len(m))));csv(ROOT/'low_rank_diagnostics.csv',reconstruction)
 simple=['persistence','ema_half30','ema_half120','quality','quality_v2_h6_half30_w0','kalman_locallevel','kalman_localtrend'];rank=[]
 for name in simple+newnames:
  g=f[(f.model_id==name)&f.example_id.isin(m.iloc[sel].example_id)];score0=macro(g,['mae','extreme_mae']);rank.append(dict(model_id=name,mae=score0.mae,extreme_mae=score0.extreme_mae))
 rank=pd.DataFrame(rank);strong=rank[rank.model_id.isin(simple)].sort_values('mae').iloc[0];winner=rank[rank.model_id.str.startswith('lr_')].sort_values('mae').iloc[0];gain=1-winner.mae/strong.mae;register(ROOT/'manifests/point_selection.json',dict(strongest_simple=strong.model_id,selected_lowrank=winner.model_id,simple_mae=strong.mae,lowrank_mae=winner.mae,relative_gain=gain,continuation_eligible=bool(gain>=.02 and winner.extreme_mae<=strong.extreme_mae*1.05),selection='2023 development only; before reporting assessment',no_frozen_parent_changed=True));csv(ROOT/'low_rank_improvements.csv',rank);csv(ROOT/'diagnostics/point_runtime.csv',[dict(model_id=k,local_fit_plus_inference_seconds=v,cases=len(m),device='CPU') for k,v in elapsed.items()]);finish('point',wall_seconds=time.perf_counter()-start,configurations=len(cfgs));print('V7 POINT COMPLETE',flush=True)
if __name__=='__main__':run()
