"""Replay the six frozen models; no new training, selection or artifact overwriting."""
import os,sys,json,time,hashlib,argparse
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1');sys.dont_write_bytecode=True
import numpy as np,pandas as pd
PKG=Path(__file__).resolve().parent;REPO=PKG.parent
V9=REPO/'volcanic_forecasting_v2/experiments/v9_model_optimization'

def verify_files():
 manifest=json.loads((PKG/'file_manifest.json').read_text());bad=[]
 for r in manifest:
  p=REPO/r['path']
  if not p.is_file():bad.append(r['path']);continue
  with p.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
  if sha!=r['sha256']:bad.append(r['path'])
 assert not bad,bad
 return len(manifest)

def run(output,quick=False):
 start=time.perf_counter();out=Path(output);out.mkdir(parents=True,exist_ok=True);report={'verified_files':verify_files(),'new_training':False,'Stage2':'Not run'};clocks=[]
 from volcanic_forecasting_v2.src.research_v9.common import data,partitions,readjson
 from volcanic_forecasting_v2.src.stage1.models import LocalFit
 from volcanic_forecasting_v2.src.research_v9.temporal import forecast
 from volcanic_forecasting_v2.src.research_v9.spatial import predict as spatial_predict
 from volcanic_forecasting_v2.src.research_v9.conventional import predict as context_predict
 from volcanic_forecasting_v2.src.research_v9.features import build
 from volcanic_forecasting_v2.src.research_v9.neural import VAE,decode
 from volcanic_forecasting_v2.src.research_v6.generative import conditioning
 from volcanic_forecasting_v2.src.research_v7.probability import crps
 from .original_lowrank import lowrank,make_history,score
 import joblib,torch
 torch.set_num_threads(2)
 tick=time.perf_counter();m,a,_=data();tr,cal,dev,test=partitions(m);points=np.load(PKG/'artifacts/v9_points.npz');clocks.append(dict(component='data_loading_and_past_features',seconds=time.perf_counter()-tick))
 # Forecast-local smoothing is recomputed from past-only inputs, then compared with frozen arrays.
 fresh={};configs={'quality':dict(family='quality',history=6,half=30,gain=0,weighted=False),'EMA30':dict(family='ema',history=6,half=30)}
 ids=np.unique(np.r_[cal[0],dev[:2],test[:2]]) if quick else np.arange(len(m))
 for name,cfg in configs.items():
  tick=time.perf_counter();pred=np.array([LocalFit(a['history'][i],a['times'][i],cfg).predict(float(m.iloc[i].horizon_days)) for i in ids]);assert np.allclose(pred,points[name][ids],atol=1e-4,equal_nan=True),name;fresh[name]=pred;clocks.append(dict(component=name+'_past_only_replay',seconds=time.perf_counter()-tick,cases=len(ids)))
 rows=[]
 for name,key in [('quality','quality'),('EMA30','EMA30'),('four_horizon','four_horizon')]:
  for i in dev:
   valid=np.isfinite(a['truth'][i])&np.isfinite(a['last'][i]);rows.append(dict(model=name,volcano=m.iloc[i].volcano,case_index=int(i),mae_mm=float(np.mean(abs(points[key][i][valid]-a['truth'][i][valid])))))
 f=pd.DataFrame(rows);point=f.groupby(['model','volcano']).mae_mm.mean().groupby('model').mean();f.to_csv(out/'v9_point_replay.csv',index=False)
 reference=pd.read_csv(V9/'diagnostics/point_macro.csv').query("role=='validation'").set_index('model_id');mapping={'quality':'ref_quality_v2_h6_half30_w0','EMA30':'ref_ema_half30','four_horizon':'four_horizon'}
 for name,mid in mapping.items():assert np.isclose(point[name],reference.loc[mid,'mae'],atol=1e-6),name
 report['V9_2023_volcano_macro_MAE_mm']=point.to_dict()
 # Check actual checkpoint inference for all four branches on three existing cases.
 tick=time.perf_counter();sample=np.array([int(cal[0]),int(dev[0]),int(test[0])]);x=build(m,a);context=joblib.load(V9/'checkpoints/context_SVR_residual_temporal.joblib');cp=context_predict(context,a,x,sample,{'residual_temporal':points['expert_temporal']});sp,_=spatial_predict(joblib.load(V9/'checkpoints/spatial_block16_linear.joblib'),m,a,sample);tcfg=readjson(V9/'manifests/temporal_selection.json');tcfg=tcfg['configs'][tcfg['best']];tp=np.array([forecast(a['history'][i],a['times'][i],float(m.iloc[i].horizon_days),tcfg) for i in sample]);qp=np.array([LocalFit(a['history'][i],a['times'][i],configs['quality']).predict(float(m.iloc[i].horizon_days)) for i in sample]);stack=np.array([tp,sp,cp,qp]);weights=readjson(PKG/'artifacts/four_horizon_weights.json');w=np.array([next(r['weights'] for r in weights['bins'] if r['bin']==int(np.digitize(m.iloc[i].horizon_days,[31,91]))) for i in sample]);hp=np.einsum('nb,bnhw->nhw',w,stack);assert np.allclose(hp,points['four_horizon'][sample],atol=1e-4,equal_nan=True),'Hybrid checkpoint replay';report['four_branch_checkpoint_replay_cases']=sample.tolist();clocks.append(dict(component='four_branch_checkpoint_replay_including_feature_build',seconds=time.perf_counter()-tick,cases=3));del x
 # Exact original64px rank3 and persistence evaluation, with original zero/missing convention.
 tick=time.perf_counter();native=[]
 for r in readjson(PKG/'data/original64/inventory.json'):
  z=np.load(PKG/'data/original64'/(r['name']+'.npz'));maps=z['maps'];days=z['days'];space=z['spacing'];origins=r['origins']['test'][:1] if quick else r['origins']['test']
  for i in origins:
   ix=i+np.array([1,2,4]);h=make_history(maps,days,i,6,days[ix],space,64);pred=lowrank(h);truth=maps[ix[0]].copy();truth[truth==0]=np.nan;mask=np.isfinite(truth)&h['seen']
   for name,p in [('lowrank_r3_robust',pred[0]),('persistence',h['last'])]:
    z0=score(p,truth,h['last'],mask,space)
    if z0:native.append(dict(model=name,group=r['group'],volcano=r['name'],origin_index=i,mae_mm=z0['mae_mm']))
 nf=pd.DataFrame(native);native_mean=nf.groupby(['model','group']).mae_mm.mean().groupby('model').mean();nf.to_csv(out/'original64_replay.csv',index=False);report['original64_frame_macro_MAE_mm']=native_mean.to_dict();original=pd.read_csv(PKG/'results/original64_summary.csv').query('horizon==1').set_index('model')
 if not quick:
  for name in native_mean.index:assert np.isclose(native_mean[name],original.loc[name,'mae_mm'],atol=1e-6),name
 clocks.append(dict(component='original64_lowrank_and_persistence_fit_inference_scoring',seconds=time.perf_counter()-tick,cases=len(nf)//2))
 # Recompute CRPS from original draws. Separately exercise generative inference from frozen checkpoints.
 cv=[];art=joblib.load(V9/'checkpoints/neural_preprocessor.joblib');cond=np.clip(art['cscaler'].transform(conditioning(m,a)),-6,6).astype('float32');tick=time.perf_counter()
 for seed in [20261206,20261207,20261208]:
  mid='cvae_coarse3_seed'+str(seed);draws=np.load(V9/'predictions'/(mid+'_draws.npz'));ii=draws['indices'];samples=draws['samples'];wanted=set(dev)
  for j,i in enumerate(ii):
   if int(i) in wanted:
    cv.append(dict(seed=seed,case_index=int(i),volcano=m.iloc[i].volcano,crps_mm=crps(samples[j],a['truth'][i])))
    if quick:break
  ck=torch.load(V9/'checkpoints'/(mid+'.pt'),map_location='cpu',weights_only=False);cfg=ck['config'];net=VAE(ck['latent_dim'],ck['condition_dim'],cfg);net.load_state_dict(ck['state_dict']);net.eval();p=decode(net,cfg,art,cond,points['EMA30'],a,int(dev[0]),n=32);assert p.shape==(32,32,32);valid=np.isfinite(a['last'][dev[0]]);assert np.isfinite(p[:,valid]).all();assert np.allclose(p,decode(net,cfg,art,cond,points['EMA30'],a,int(dev[0]),n=32),equal_nan=True);np.savez_compressed(out/(mid+'_example.npz'),samples=p,truth=a['truth'][dev[0]],case_index=int(dev[0]));del samples
 cf=pd.DataFrame(cv);crps_mean=cf.groupby(['seed','volcano']).crps_mm.mean().groupby('seed').mean().mean();report['CVAE_2023_mean_seed_volcano_macro_CRPS_mm']=float(crps_mean)
 if not quick:assert np.isclose(crps_mean,8.172473710107143,atol=1e-6)
 cf.to_csv(out/'cvae_replay.csv',index=False);clocks.append(dict(component='CVAE_original_draw_scoring_and_frozen_checkpoint_generation',seconds=time.perf_counter()-tick,seeds=3));report['quick']=quick;report['seconds']=time.perf_counter()-start;report['status']='passed';pd.DataFrame(clocks).to_csv(out/'runtime.csv',index=False);(out/'reproduction.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--quick',action='store_true',help='Smoke test only; does not claim full cohort metrics');p.add_argument('--output',type=Path,default=PKG/'generated');a=p.parse_args();run(a.output,a.quick)
