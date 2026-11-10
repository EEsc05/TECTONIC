"""Bounded conditional VAE, flow-matching and latent diffusion pilots on all518eligible pairs."""
from .common import *
from .representation import SpatialBasis,past_features
from ..historical_v3.generative import scores as map_scores
from sklearn.preprocessing import RobustScaler,StandardScaler
import torch,torch.nn as nn,torch.nn.functional as F
import joblib,time,copy

def mlp(a,b,width=128):return nn.Sequential(nn.Linear(a,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU(),nn.Linear(width,b))
class CVAE(nn.Module):
 def __init__(self,d,c):
  super().__init__();self.prior=mlp(c,16);self.posterior=mlp(d+c,16);self.decoder=mlp(8+c,d);self.log_noise=nn.Parameter(torch.full((d,),-1.))
 def loss(self,z,c):
  qm,ql=self.posterior(torch.cat([z,c],-1)).chunk(2,-1);pm,pl=self.prior(c).chunk(2,-1);ql=ql.clamp(-6,3);pl=pl.clamp(-6,3);eps=torch.randn_like(qm);latent=qm+eps*torch.exp(.5*ql);out=self.decoder(torch.cat([latent,c],-1));noise=self.log_noise.clamp(-4,2);recon=((z-out)**2*torch.exp(-2*noise)+2*noise).mean()/2;kl=.5*(pl-ql+(ql.exp()+(qm-pm)**2)/pl.exp()-1).mean();return recon+.1*kl,recon,kl
 def sample(self,c,n):
  pm,pl=self.prior(c).chunk(2,-1);pm=pm.repeat_interleave(n,0);pl=pl.clamp(-6,3).repeat_interleave(n,0);cc=c.repeat_interleave(n,0);z=pm+torch.randn_like(pm)*torch.exp(.5*pl);out=self.decoder(torch.cat([z,cc],-1));return out+torch.randn_like(out)*self.log_noise.clamp(-4,2).exp()
 def reconstruction(self,z,c):
  qm,_=self.posterior(torch.cat([z,c],-1)).chunk(2,-1);return self.decoder(torch.cat([qm,c],-1))
class Flow(nn.Module):
 def __init__(self,d,c):super().__init__();self.net=mlp(d+c+3,d)
 def forward(self,z,c,t):return self.net(torch.cat([z,c,t,torch.sin(t*np.pi),torch.cos(t*np.pi)],-1))
 def loss(self,z,c):
  e=torch.randn_like(z);t=torch.rand((len(z),1),device=z.device);xt=(1-t)*e+t*z;loss=F.mse_loss(self(xt,c,t),z-e);return loss,loss,loss*0
 def sample(self,c,n):
  c=c.repeat_interleave(n,0);z=torch.randn((len(c),self.net[-1].out_features),device=c.device);steps=16
  for j in range(steps):
   t=torch.full((len(z),1),(j+.5)/steps,device=z.device);z=z+self(z,c,t)/steps
  return z
class Diffusion(Flow):
 def loss(self,z,c):
  t=torch.rand((len(z),1),device=z.device)*.98+.01;alpha=torch.cos(t*np.pi/2);sigma=torch.sin(t*np.pi/2);eps=torch.randn_like(z);loss=F.mse_loss(self(alpha*z+sigma*eps,c,t),eps);return loss,loss,loss*0
 def sample(self,c,n):
  c=c.repeat_interleave(n,0);z=torch.randn((len(c),self.net[-1].out_features),device=c.device);steps=20
  for j in range(steps,0,-1):
   t=torch.full((len(z),1),j/steps*.98,device=z.device);tn=(j-1)/steps*.98;alpha=torch.cos(t*np.pi/2);sigma=torch.sin(t*np.pi/2);eps=self(z,c,t);x0=(z-sigma*eps)/alpha.clamp_min(.03);z=np.cos(tn*np.pi/2)*x0+np.sin(tn*np.pi/2)*eps
  return z

def sampling_weights(meta,indices,mode,scale):
 g=meta.iloc[indices];w=np.ones(len(indices))
 if mode=='origin_balanced':w=1/g.groupby(['source_id','origin_day']).origin_day.transform('size').to_numpy()
 elif mode=='volcano_balanced':w=1/g.volcano.map(g.volcano.value_counts()).to_numpy()
 elif mode=='temporally_separated':
  bins=g.origin_day//61;keys=list(zip(g.volcano,bins));counts=pd.Series(keys).value_counts();w=np.array([1/counts[k] for k in keys])
 elif mode=='regime_balanced':
  bins=pd.qcut(scale[indices],3,labels=False,duplicates='drop');counts=pd.Series(bins).value_counts();w=np.array([1/counts[k] for k in bins])
 return w/w.sum()

def run():
 p=task(3);meta,a,base=stage1_data();train=np.flatnonzero(meta.role=='train');val=np.flatnonzero(meta.role=='validation');cal=np.array([i for i in val if meta.iloc[i].target_day<day('2019-01-01')]);select=np.array([i for i in val if meta.iloc[i].history_start_day>=day('2019-01-01')]);device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 register(p/'splits/protocol.json',dict(dataset_version='historical_v3_immutable',split_version='v4_neural_exploratory',all_training_pairs=518,epochs=160,max_epochs=160,patience=25,batch_size=64,representations=['localized_blockPCA128','fixed_DCT144'],models=['conditional_VAE','conditional_flow_matching','latent_DDPM'],training_seeds=[SEED,SEED+1,SEED+2],sampling_modes=['uniform','origin_balanced','volcano_balanced','temporally_separated','regime_balanced'],selection='2018validation early stopping/scale;2019validation CRPS selection; protected old temporal/geographic scores diagnostic only after freeze',geographic_validation='Separate leave-one-development-volcano-out pilot with basis/scaler/head refit excluding azumayama, prior protected sites not used',draws=32,flow_steps=16,diffusion_steps=20,authorization='Explicit new task permits exploratory training below historical_v3confirmatorygates',seed=SEED))
 x,geom,scale=past_features(a,meta);condition=np.c_[x,geom];scaler=RobustScaler().fit(condition[train]);condition=np.clip(scaler.transform(condition),-6,6).astype('float32');res=(a['truth']-base)/scale[:,None,None];basis=SpatialBasis('block_pca',8).fit(res[train]);joblib.dump(dict(basis=basis,condition_scaler=scaler),p/'models/preprocessor.joblib');zraw=basis.encode(res);latent_scaler=StandardScaler().fit(zraw[train]);z=latent_scaler.transform(zraw).astype('float32');reconstruction=basis.decode(zraw);tail=np.nan_to_num(res-reconstruction);noise=np.maximum(np.nanstd(tail[train],axis=0),.05);event_threshold=float(np.nanquantile(abs(a['truth'][train]-a['last'][train]),.9));c=torch.tensor(condition,device=device);zz=torch.tensor(z,device=device);d=z.shape[1];curves=[];timing=[];validation=[];artifacts={};start=time.perf_counter()
 configs=[(family,seed,'origin_balanced') for family in ['cvae','flow'] for seed in [SEED,SEED+1,SEED+2]]+ [('cvae',SEED,mode) for mode in ['uniform','volcano_balanced','temporally_separated','regime_balanced']]+[('diffusion',SEED,'origin_balanced')]
 for family,seed,mode in configs:
  torch.manual_seed(seed);np.random.seed(seed);model={'cvae':CVAE,'flow':Flow,'diffusion':Diffusion}[family](d,c.shape[1]).to(device);optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4);rng=np.random.default_rng(seed);weights=sampling_weights(meta,train,mode,scale);best=None;wait=0;tick=time.perf_counter();mid=f'{family}_seed{seed}_{mode}'
  for epoch in range(160):
   model.train();losses=[];ids=rng.choice(train,len(train),replace=True,p=weights)
   for offset in range(0,len(ids),64):
    idx=ids[offset:offset+64];optim.zero_grad();loss,recon,kl=model.loss(zz[idx],c[idx]);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optim.step();losses.append(float(loss.detach().cpu()))
   model.eval();torch.manual_seed(seed+1000)
   with torch.no_grad():vl=float(model.loss(zz[cal],c[cal])[0].cpu())
   curves.append(dict(model_id=mid,epoch=epoch,training_loss=float(np.mean(losses)),validation_loss=vl,validation_scope='2018calibrationonly'))
   if best is None or vl<best[0]-1e-4:best=(vl,copy.deepcopy(model.state_dict()),epoch);wait=0
   else:wait+=1
   if wait>=25:break
  model.load_state_dict(best[1]);model.eval();artifact=dict(family=family,seed=seed,sampling_mode=mode,state_dict={k:v.cpu() for k,v in model.state_dict().items()},latent_dimension=d,condition_dimension=c.shape[1],best_epoch=best[2],latent_scaler=latent_scaler,basis=basis,condition_scaler=scaler,noise=noise,scales='past-only amplitude',dataset_version='historical_v3_immutable',split_version='v4_neural_exploratory');torch.save(artifact,p/'models'/f'{mid}.pt');artifacts[mid]=(model,artifact);timing.append(dict(model_id=mid,training_seconds=time.perf_counter()-tick,epochs=epoch+1,best_epoch=best[2],parameters=sum(v.numel() for v in model.parameters()),training_pairs=len(train),unique_origins=len(meta.iloc[train][['source_id','origin_day']].drop_duplicates()),device=device,training_seed=seed));print('NEURAL TRAINED',mid,'epochs',epoch+1,flush=True)
  # Early diagnostics use only untouched-within-this-run2019development, never prior geography.
  values=[]
  with torch.no_grad():
   torch.manual_seed(seed+2000)
   for begin in range(0,len(select),32):
    idx=select[begin:begin+32];latent=model.sample(c[idx],32).cpu().numpy();decoded=basis.decode(latent_scaler.inverse_transform(latent)).reshape(len(idx),32,32,32)
    for j,i in enumerate(idx):
     rng2=np.random.default_rng(seed+int(meta.iloc[i].origin_day));samples=base[i][None]+(decoded[j]+rng2.normal(size=decoded[j].shape)*noise)*scale[i];values.append(map_scores(samples,a['truth'][i],a['last'][i],a['spacing'][i],event_threshold))
  validation.append(dict(model_id=mid,family=family,training_seed=seed,sampling_mode=mode,crps=float(np.mean([v['crps'] for v in values])),coverage90=float(np.mean([v['coverage90'] for v in values])),mae=float(np.mean([v['mae'] for v in values]))));csv(p/'training_curves.csv',curves);csv(p/'training_times.csv',timing);csv(p/'validation_selection.csv',validation)
 # Freeze comparisons before any reuse of previous held-out outcomes.
 ranking=pd.DataFrame(validation);winner=ranking.sort_values('crps').iloc[0].model_id;writejson(p/'models/frozen_selection.json',dict(selected_model=winner,frozen_utc=utc(),selection='2019developmentCRPS, exploratory only',previous_holdouts_used_for_selection=False,models=list(artifacts),seed_stability='Three independently trained seeds for primaryCVAE/flow; additional sampling schemes seed0only',old_gates='Remain unmet; explicitly waived solely for bounded exploratory fits'))
 # Score primaryseedreplicates plus diffusion; sampling ablations retain full validation results.
 eval_ids=np.flatnonzero(meta.role.isin(['validation','temporal_evaluation','unseen_volcano']));rows=[];reconrows=[]
 for mid,(model,artifact) in artifacts.items():
  if artifact['sampling_mode']!='origin_balanced':continue
  torch.manual_seed(artifact['seed']+3000)
  with torch.no_grad():
   for begin in range(0,len(eval_ids),32):
    idx=eval_ids[begin:begin+32];latent=model.sample(c[idx],32).cpu().numpy();decoded=basis.decode(latent_scaler.inverse_transform(latent)).reshape(len(idx),32,32,32)
    for j,i in enumerate(idx):
     rng=np.random.default_rng(artifact['seed']+int(meta.iloc[i].origin_day));samples=base[i][None]+(decoded[j]+rng.normal(size=decoded[j].shape)*noise)*scale[i];sc=map_scores(samples,a['truth'][i],a['last'][i],a['spacing'][i],event_threshold);rows.append(dict(model_id=mid,family=artifact['family'],training_seed=artifact['seed'],role=meta.iloc[i].role,volcano=meta.iloc[i].volcano,example_id=meta.iloc[i].example_id,horizon_days=int(meta.iloc[i].horizon_days),evaluation_status='reused_exploratory' if meta.iloc[i].role!='validation' else 'development',**sc))
     if begin==0 and j==0:np.savez_compressed(p/'figures'/f'{mid}_samples.npz',samples=samples,mean=samples.mean(0),truth=a['truth'][i],last=a['last'][i],origin_day=meta.iloc[i].origin_day,horizon_days=meta.iloc[i].horizon_days)
   if isinstance(model,CVAE):
    zr=model.reconstruction(zz[val],c[val]).cpu().numpy();rec=basis.decode(latent_scaler.inverse_transform(zr))*scale[val,None,None]+base[val]
    for j,i in enumerate(val):reconrows.append(dict(model_id=mid,example_id=meta.iloc[i].example_id,posterior_reconstruction_mae=float(np.nanmean(abs(rec[j]-a['truth'][i]))),warning='Posterior uses target for reconstruction diagnostic only; forecast uses conditional prior'))
  print('NEURAL SCORED',mid,flush=True)
 f=csv(p/'probabilistic_predictions.csv',rows);metrics=f.groupby(['role','model_id','volcano']).mean(numeric_only=True).groupby(['role','model_id']).mean().reset_index();csv(p/'probabilistic_metrics.csv',metrics);csv(p/'horizon_metrics.csv',f.groupby(['role','model_id','horizon_days']).mean(numeric_only=True).reset_index());csv(p/'reconstruction_diagnostics.csv',reconrows);csv(p/'per_volcano_metrics.csv',f.groupby(['role','model_id','volcano']).mean(numeric_only=True).reset_index())
 baselines=pd.read_csv(BASE/'experiments/20_stage1_generative/runs/historical_v3/probabilistic_metrics.csv');csv(p/'frozen_gaussian_baselines.csv',baselines);det=pd.read_csv(BASE/'experiments/19_stage1_deterministic_expansion/runs/historical_v3/metrics.csv');csv(p/'frozen_deterministic_controls.csv',det)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(9,4))
 for mid,g in pd.DataFrame(curves).groupby('model_id'):ax.plot(g.epoch,g.validation_loss,label=mid)
 ax.set_xlabel('Epoch');ax.set_ylabel('2018validation objective (family-specific)');ax.legend(fontsize=5);fig.tight_layout();fig.savefig(p/'figures/training_curves.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(9,4));subset=metrics[metrics.role=='validation'];ax.bar(subset.model_id,subset.coverage90);ax.axhline(.9,color='black',ls='--');ax.tick_params(axis='x',rotation=70,labelsize=6);ax.set_ylabel('Marginal90%coverage, validation');fig.tight_layout();fig.savefig(p/'figures/calibration.png',dpi=150);plt.close(fig)
 (p/'summary.md').write_text('# Exploratory neural generative pilots\n\nSmall conditionalVAE, conditionalflowmatching and latentdiffusion were actually trained. Three independent trainingseeds for CVAE/flow differ from repeated sampling of one fit. All518eligible pairs contribute, with overlap-aware sampling comparisons. The localized128coefficient train-only basis avoids exclusive reliance on rank8PCA; geometry features represent spacing/cadence, not unavailable full LOS vectors. Full3Dmotion is not inferred.\n\n'+mdtable(metrics[['role','model_id','mae','crps','coverage90','interval_width90','energy_score','extreme_mae']])+'\n\nAll previous temporal/geographic cohorts are reused exploratory diagnostics, never fresh confirmatory tests or hyperparameter selection. Generator uncertainty is not automatically calibrated. Diffusion is a bounded latentpilot, not a pixelspace model. The old confirmatory gates still govern advancement; exploratory authorization does not erase them. Posterior reconstruction may use target supervision; prediction-time conditional prior and flow inputs never use future maps or extreme masks.\n')
 writejson(p/'status.json',dict(status='completed_exploratory',wall_seconds=time.perf_counter()-start,device=device,peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()) if device=='cuda' else 0,models_trained=len(artifacts),selected_model=winner,training_pairs=len(train),independent_training_intervals=32,parent_unchanged=parent_check()));print('TASK3 COMPLETE',flush=True)
if __name__=='__main__':
 from .neural import run as canonical_run
 canonical_run()
