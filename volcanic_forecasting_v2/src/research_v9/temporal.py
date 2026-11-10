"""Bounded actual-time smoothing/state-space optimization, training-CV only."""
from .common import *
from ..stage1.models import fill_past,smooth
from ..research_v6.data import score
import time

def configs():
 rng=np.random.default_rng(SEED);c=[]
 for j,half in enumerate([7,15,30,60,120,240]):
  for trend in [False,True]:c.append(dict(id=f'EMA_h{half}_trend{int(trend)}',family='EMA',half=half,window=[2,4,6][j%3],trend=trend,damping=[30,90,180][j%3]))
 for j in range(18):c.append(dict(id='Kalman_'+str(j),family='Kalman',q=[.001,.01,.1][j%3],measurement=[.5,2][j%2],initial=[1,10][(j//2)%2],clip=[2,4][(j//3)%2],state='trend' if j%2 else 'level',damping=[30,90,180][(j//6)%3],window=6))
 return c

def forecast(m,t,h,c):
 f,v=fill_past(m[-c['window']:]);t=t[-c['window']:];last=f[-1];seen=v.any(0)
 if c['family']=='EMA':
  p=smooth(m[-c['window']:],t,c['half'])
  if c['trend']:p+=np.median(np.diff(f,axis=0)/np.diff(t)[:,None,None],0)*h*np.exp(-h/c['damping'])
 else:
  noise=max(float(np.median(abs(np.diff(f,axis=0)))),1.);r=noise**2*c['measurement'];level=f[0].astype(float);vel=np.zeros_like(level);p00=np.full_like(level,noise**2*c['initial']);p01=np.zeros_like(level);p11=np.full_like(level,noise**2/900);q=noise**2*c['q']/30
  for j,dt in enumerate(np.diff(t),1):
   if c['state']=='trend':level+=vel*dt;p00+=2*dt*p01+dt*dt*p11+q*dt**3/3;p01+=dt*p11+q*dt*dt/2;p11+=q*dt
   else:p00+=q*dt
   innovation=np.clip(f[j]-level,-c['clip']*np.sqrt(p00+r),c['clip']*np.sqrt(p00+r));gain=p00/(p00+r);g1=p01/(p00+r);level=np.where(v[j],level+gain*innovation,level);vel=np.where(v[j],vel+g1*innovation,vel);old01=p01.copy();p00=np.where(v[j],(1-gain)*p00,p00);p01=np.where(v[j],(1-gain)*p01,p01);p11=np.where(v[j],p11-g1*old01,p11)
  p=level+(vel*h*np.exp(-h/c['damping']) if c['state']=='trend' else 0)
 p=np.asarray(p,dtype='float32');p[~seen]=np.nan;return p

def run():
 assert done('features');m,a,_=data();fs=folds(m);rows=[];pred={};timing=[];threshold=readjson(ROOT/'manifests/thresholds.json')['pixel_extreme_mm'];val=np.unique(np.concatenate([v for _,_,v in fs]))
 for c in configs():
  path=ROOT/'predictions'/('temporal_'+c['id']+'.npz');tick=time.perf_counter()
  if path.exists():pp=np.load(path)['pred']
  else:pp=np.array([forecast(a['history'][i],a['times'][i],r.horizon_days,c) for i,r in m.iterrows()]);np.savez_compressed(path,pred=pp)
  timing.append(dict(model_id=c['id'],local_fit_and_inference_seconds=time.perf_counter()-tick,cases=len(m),device='CPU'))
  for fold,_,ids in fs:
   f=pd.DataFrame([dict(volcano=m.iloc[i].volcano,**score(a['truth'][i],pp[i],a['last'][i],a['spacing'][i],threshold)) for i in ids]);z=macro(f,['mae','extreme_mae','gradient_mae']);rows.append(dict(model_id=c['id'],family=c['family'],stage='internal_CV',fold=fold,config=json.dumps(c),cases=len(ids),**z.to_dict()))
 f=csv(ROOT/'temporal_branch_results.csv',rows);history('temporal',rows);rank=f.groupby(['model_id','family']).mae.mean().reset_index();chosen=rank.sort_values('mae').groupby('family',sort=False).head(1);register(ROOT/'manifests/temporal_selection.json',dict(selected=chosen.model_id.tolist(),best=rank.sort_values('mae').iloc[0].model_id,configs={c['id']:c for c in configs()},selection='Equal fold-weighted training-internal volcano-macro MAE; no2023 evaluation'));csv(ROOT/'diagnostics/temporal_runtime.csv',timing);finish('temporal');print('V9 TEMPORAL SEARCH COMPLETE',flush=True)
if __name__=='__main__':run()
