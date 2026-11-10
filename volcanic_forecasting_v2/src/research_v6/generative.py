"""Globalprobabilisticlearningcurves onfixedcohorts; no diffusion expansion."""
from .common import *
from .data import examples
from ..research_v4.representation import SpatialBasis
from ..research_v4.neural import CVAE,Flow
from ..historical_v3.generative import scores
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import Ridge
from scipy.ndimage import gaussian_filter
import torch,joblib,time,copy,warnings
warnings.filterwarnings('ignore',message='Mean of empty slice');warnings.filterwarnings('ignore',message='Degrees of freedom <= 0');os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')

def conditioning(meta,a):
 change=np.nan_to_num(a['history'][:,-1]-a['history'][:,0])/a['scale'][:,None,None];coarse=change.reshape(-1,8,4,8,4).mean((2,4));return np.c_[coarse.reshape(len(meta),-1),a['cnn'][:,-11:,0,0]]

def ordered_sites(meta,indices):
 groups={r:sorted(g.volcano.unique()) for r,g in meta.iloc[indices].groupby('region')};ans=[]
 while any(groups.values()):
  for r in sorted(groups):
   if groups[r]:ans.append(groups[r].pop(0))
 return ans

def decode_samples(artifact,c,base,scale,sp,h,n=32,seed=SEED,model=None,kind='gaussian'):
 rng=np.random.default_rng(seed);cx=np.clip(artifact['condition_scaler'].transform(c[None]),-6,6);basis=artifact['basis'];ls=artifact['latent_scaler'];z=artifact['head'].predict(cx)[0];rank=len(z)
 if kind=='independent_gaussian':
  binid=min(int(h//30),5);std=artifact['independent_std'][binid];samples=base[None]+rng.normal(size=(n,32,32))*std*scale
 else:
  if model is not None:
   device=next(model.parameters()).device;torch.manual_seed(seed)
   with torch.no_grad():latent=model.sample(torch.tensor(cx,dtype=torch.float32,device=device),n).cpu().numpy()
  else:
   eps=rng.normal(size=(n,rank))@artifact['chol'].T
   if kind=='student':eps/=np.sqrt(rng.chisquare(4,(n,1))/2)
   latent=z+eps
  maps=basis.decode(ls.inverse_transform(latent));white=rng.normal(size=(n,32,32))
  for j in range(n):white[j]=gaussian_filter(white[j],sigma=max(.5,1./max(sp)),mode='reflect');white[j]/=max(white[j].std(),1e-6)
  samples=base[None]+(maps+white*artifact['noise'])*scale
 return samples.astype('float32')

def run():
 p=ROOT/'generative';start=time.perf_counter();meta,a,_=examples();alltrain=np.flatnonzero(meta.role=='train');val=np.flatnonzero(meta.role=='validation');cal=np.array([i for i in val if meta.iloc[i].target_day<day('2023-01-01')]);select=np.array([i for i in val if meta.iloc[i].origin_day>=day('2023-01-01')]);test=np.flatnonzero(meta.role.isin(['temporal_evaluation','geographic_evaluation']));base_id='ema_half30';base=np.load(ROOT/'deterministic/point_predictions.npz')[base_id];res=(a['truth']-base)/a['scale'][:,None,None];cond=conditioning(meta,a);sites=ordered_sites(meta,alltrain);levels=sorted(set([min(2,len(sites)),max(2,len(sites)//2),len(sites)]));register(p/'protocol.json',dict(version=VERSION,dataset_version=DATASET,selection='2022scale/earlystop;2023CRPSforfinalcandidateselection;2024andgeographyafterfreeze',training_levels=levels,site_order=sites,basis='Training-onlyblockPCA128normalizedpastamplituderesiduals',models=['independentGaussian','localizedGaussian','localizedStudent','CVAE','flowmatching'],width=128,max_epochs=80,samples=32,full_training_seeds=[SEED,SEED+1,SEED+2],horizon_variance='Gaussianbinvariancesestimatedtrainingonly; no imposedmonotonicity',Peru='Stillunopeneduntilglobalselectionfreeze',seed=SEED))
 device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2);predrows=[];timing=[];curves=[];selection=[];artifacts={};threshold=readjson(ROOT/'deterministic/thresholds.json')['extreme_pixel_mm']
 for level in levels:
  tr=np.array([i for i in alltrain if meta.iloc[i].volcano in sites[:level]]);tick=time.perf_counter();basis=SpatialBasis('block_pca',8).fit(res[tr]);zs=StandardScaler().fit(basis.encode(res[tr]));z=zs.transform(basis.encode(res));cs=RobustScaler().fit(cond[tr]);cx=np.clip(cs.transform(cond),-6,6);head=Ridge(alpha=100).fit(cx[tr],z[tr]);rr=z[tr]-head.predict(cx[tr]);cov=np.cov(rr,rowvar=False);cov=.5*cov+.5*np.diag(np.diag(cov))+np.eye(cov.shape[0])*.001;noise=np.maximum(np.nanstd(res[tr]-basis.decode(basis.encode(res[tr])),axis=0),.05);indstd=[]
  for binid in range(6):
   ids=[i for i in tr if min(int(meta.iloc[i].horizon_days//30),5)==binid];indstd.append(np.maximum(np.nan_to_num(np.nanstd(res[ids if len(ids)>=10 else tr],axis=0),nan=1.),.1))
  artifact=dict(basis=basis,latent_scaler=zs,condition_scaler=cs,head=head,chol=np.linalg.cholesky(cov),noise=noise,independent_std=np.array(indstd),training_sites=sites[:level],training_pairs=len(tr),parent_point_model=base_id,version=VERSION,lookvectors='KnownstaticLOSprojectionmeans, no3Dtarget');joblib.dump(artifact,p/f'level{level}_probabilistic.joblib');timing.append(dict(model_id=f'probabilistic_level{level}',training_seconds=time.perf_counter()-tick,training_pairs=len(tr),volcanoes=level,device='CPU'));models={}
  for family in ['cvae','flow']:
   for seed in ([SEED,SEED+1,SEED+2] if level==len(sites) else [SEED]):
    torch.manual_seed(seed);net=(CVAE if family=='cvae' else Flow)(z.shape[1],cx.shape[1]).to(device);optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);cc=torch.tensor(cx,dtype=torch.float32,device=device);zz=torch.tensor(z,dtype=torch.float32,device=device);best=None;wait=0;rng=np.random.default_rng(seed);tick=time.perf_counter();mid=f'{family}_level{level}_seed{seed}';counts=meta.iloc[tr].volcano.value_counts();w=np.array([1/counts[meta.iloc[i].volcano] for i in tr]);w/=w.sum()
    for epoch in range(80):
     net.train();losses=[];ids=rng.choice(tr,len(tr),replace=True,p=w)
     for begin in range(0,len(ids),64):
      ix=ids[begin:begin+64];optim.zero_grad();loss,*_=net.loss(zz[ix],cc[ix]);loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),5);optim.step();losses.append(float(loss.detach().cpu()))
     net.eval();torch.manual_seed(seed+1000)
     with torch.no_grad():vl=float(net.loss(zz[cal],cc[cal])[0].cpu())
     curves.append(dict(model_id=mid,epoch=epoch,train_loss=np.mean(losses),calibration_loss=vl))
     if best is None or vl<best[0]-1e-4:best=(vl,copy.deepcopy(net.state_dict()),epoch);wait=0
     else:wait+=1
     if wait>=15:break
    net.load_state_dict(best[1]);net.eval();models[mid]=net;torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},family=family,latent_dim=z.shape[1],condition_dim=cx.shape[1],artifact=str(p/f'level{level}_probabilistic.joblib'),training_sites=sites[:level],training_pairs=len(tr),best_epoch=best[2],seed=seed,version=VERSION),p/(mid+'.pt'));timing.append(dict(model_id=mid,training_seconds=time.perf_counter()-tick,epochs=epoch+1,parameters=sum(v.numel() for v in net.parameters()),training_pairs=len(tr),volcanoes=level,device=device));print('GLOBALGENERATOR TRAINED',mid,flush=True)
  kinds=[('independent_gaussian',None),('localized_gaussian',None),('localized_student',None),*models.items()];scales={}
  for name,net in kinds:
   best=None
   for factor in [.75,1.,1.5]:
    vals=[]
    for i in cal:
     samples=decode_samples(artifact,cond[i],base[i],a['scale'][i],a['spacing'][i],meta.iloc[i].horizon_days,seed=SEED+int(meta.iloc[i].origin_day),model=net,kind='independent_gaussian' if name=='independent_gaussian' else 'student' if name=='localized_student' else 'gaussian');mean=np.nanmean(samples,0);samples=mean+(samples-mean)*factor;vals.append(scores(samples,a['truth'][i],a['last'][i],a['spacing'][i],threshold)['crps'])
    value=float(np.mean(vals))
    if best is None or value<best[0]:best=(value,factor)
   scales[name]=best[1]
   vals=[]
   for i in select:
    samples=decode_samples(artifact,cond[i],base[i],a['scale'][i],a['spacing'][i],meta.iloc[i].horizon_days,seed=SEED+int(meta.iloc[i].origin_day),model=net,kind='independent_gaussian' if name=='independent_gaussian' else 'student' if name=='localized_student' else 'gaussian');mean=np.nanmean(samples,0);samples=mean+(samples-mean)*best[1];vals.append(scores(samples,a['truth'][i],a['last'][i],a['spacing'][i],threshold))
   selection.append(dict(model_id=name if net else f'{name}_level{level}',level=level,CRPS2023=np.mean([v['crps'] for v in vals]),coverage90=np.mean([v['coverage90'] for v in vals]),scale=best[1]))
  if level==len(sites):
   full=pd.DataFrame(selection);full=full[full.level==len(sites)].sort_values('CRPS2023');writejson(p/'frozen_selection.json',dict(selected_model=full.iloc[0].model_id,frozen_before_Global_test_and_Peru_utc=utc(),selection='2023developmentCRPSonly',full_training_volcanoes=len(sites),maximum_trained_horizon=180,parent_point_model=base_id))
  writejson(p/f'level{level}_frozen_scales.json',dict(scales=scales,frozen_utc=utc(),calibration_targets='2022only',selection_targets='2023only',never_fit_Peru=True))
  # Evaluationafterper-levelscale/candidateconfigurationfreeze; no adaptationfromtest errors.
  for name,net in kinds:
   mid=name if net else f'{name}_level{level}';tick=time.perf_counter()
   for i in np.r_[select,test]:
    raw=decode_samples(artifact,cond[i],base[i],a['scale'][i],a['spacing'][i],meta.iloc[i].horizon_days,seed=SEED+int(meta.iloc[i].origin_day),model=net,kind='independent_gaussian' if name=='independent_gaussian' else 'student' if name=='localized_student' else 'gaussian');mean=np.nanmean(raw,0);sample=mean+(raw-mean)*scales[name];s=scores(sample,a['truth'][i],a['last'][i],a['spacing'][i],threshold);r=meta.iloc[i];predrows.append(dict(model_id=mid,training_volcanoes=level,training_pairs=len(tr),example_id=r.example_id,volcano=r.volcano,region=r.region,role=r.role,horizon_days=r.horizon_days,origin_day=r.origin_day,**s))
    if i==test[0] and (net is not None or name=='localized_gaussian'):np.savez_compressed(p/f'{mid}_sample_maps.npz',samples=sample,truth=a['truth'][i],last=a['last'][i],horizon_days=r.horizon_days)
   timing.append(dict(model_id=mid,sampling_and_scoring_seconds=time.perf_counter()-tick,cases=len(select)+len(test),samples_per_case=32,device=device if net else 'CPU'))
  csv(p/'training_curves.csv',curves);csv(p/'runtime.csv',timing);csv(p/'predictions_partial.csv',predrows)
 ranking=csv(p/'validation_selection.csv',selection);full=ranking[ranking.level==len(sites)].sort_values('CRPS2023');winner=full.iloc[0].model_id;assert readjson(p/'frozen_selection.json')['selected_model']==winner
 f=csv(p/'predictions.csv',predrows);mm=f.groupby(['role','model_id','training_volcanoes','volcano'])[['mae','rmse','crps','coverage90','interval_width90','energy_score','extreme_mae','gradient_mae','spatial_roughness_mm','sample_diversity_mm']].mean().groupby(['role','model_id','training_volcanoes']).mean().reset_index();csv(ROOT/'generative_benchmark.csv',mm);csv(ROOT/'probabilistic_calibration.csv',f.groupby(['role','model_id','horizon_days'])[['crps','coverage90','interval_width90']].mean().reset_index());csv(ROOT/'scaling/generative_learning_curves.csv',mm);csv(p/'runtime.csv',timing);writejson(p/'status.json',dict(status='completed',selected_model=winner,wall_seconds=time.perf_counter()-start,device=device,peak_GPU_memory_bytes=int(torch.cuda.max_memory_allocated()) if device=='cuda' else 0,parents=protect()));print('GLOBALGENERATIVE COMPLETE',winner,flush=True)
if __name__=='__main__':run()
