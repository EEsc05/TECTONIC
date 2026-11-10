"""Matched spatial probability diagnostics with train-defined events and development calibration."""
from .common import *
from ..research_v6.generative import conditioning,decode_samples
from ..research_v4.neural import CVAE,Flow
from ..research_v4.representation import SpatialBasis
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.linear_model import Ridge
from scipy.ndimage import gaussian_filter
import torch,torch.nn as nn,torch.nn.functional as F,joblib,time,copy,warnings
warnings.filterwarnings('ignore',message='Mean of empty slice');warnings.filterwarnings('ignore',message='All-NaN slice encountered');os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')

class DiagnosticVAE(nn.Module):
 def __init__(self,d,c,latent=8,beta=.01,huber=False):
  super().__init__();from ..research_v4.neural import mlp
  self.prior=mlp(c,latent*2);self.posterior=mlp(d+c,latent*2);self.decoder=mlp(latent+c,d);self.log_noise=nn.Parameter(torch.full((d,),-1.));self.beta=beta;self.huber=huber
 def loss(self,z,c):
  qm,ql=self.posterior(torch.cat([z,c],-1)).chunk(2,-1);pm,pl=self.prior(c).chunk(2,-1);ql=ql.clamp(-6,3);pl=pl.clamp(-6,3);out=self.decoder(torch.cat([qm+torch.randn_like(qm)*(.5*ql).exp(),c],-1));noise=self.log_noise.clamp(-4,2);err=(z-out)*(-noise).exp();rec=(F.huber_loss(err,torch.zeros_like(err),reduction='none')+noise).mean() if self.huber else (.5*err**2+noise).mean();kl=.5*(pl-ql+(ql.exp()+(qm-pm)**2)/pl.exp()-1).mean();return rec+self.beta*kl,rec,kl
 def sample(self,c,n):
  pm,pl=self.prior(c).chunk(2,-1);pm=pm.repeat_interleave(n,0);pl=pl.clamp(-6,3).repeat_interleave(n,0);cc=c.repeat_interleave(n,0);return self.decoder(torch.cat([pm+torch.randn_like(pm)*(.5*pl).exp(),cc],-1))+torch.randn((len(cc),self.log_noise.numel()),device=c.device)*self.log_noise.clamp(-4,2).exp()
 def reconstruction(self,z,c):
  qm,_=self.posterior(torch.cat([z,c],-1)).chunk(2,-1);return self.decoder(torch.cat([qm,c],-1))

class StepFlow(Flow):
 def sample(self,c,n,steps=16):
  c=c.repeat_interleave(n,0);z=torch.randn((len(c),self.net[-1].out_features),device=c.device)
  for j in range(steps):z=z+self(z,c,torch.full((len(c),1),(j+.5)/steps,device=c.device))/steps
  return z

def crps(samples,y):
 v=np.isfinite(y)&np.isfinite(samples).all(0);s=samples[:,v];truth=y[v];n=len(s);return float(np.mean(abs(s-truth))-np.mean(np.sum((2*np.arange(1,n+1)-n-1)[:,None]*np.sort(s,axis=0),axis=0)/n**2))

def scores(samples,y,last,sp,threshold):
 v=np.isfinite(y)&np.isfinite(last)&np.isfinite(samples).all(0);s=samples[:,v];truth=y[v];n=len(s);mean=np.nanmean(samples,axis=0);err=mean[v]-truth;ext=abs(truth-last[v])>=threshold['pixel_extreme_mm'];z=s-last[v];pixelprob=(abs(z)>threshold['pixel_extreme_mm']).mean(0);outcome=ext;low,hi=np.quantile(s,[.05,.95],axis=0);event=float(np.nanmean(abs(y-last))>threshold['map_extreme_mm']);past=np.isfinite(last)&np.isfinite(samples).all(0);events=np.nanmean(abs(samples[:,past]-last[past]),axis=1)>threshold['map_extreme_mm'];p=(events.sum()+.5)/(n+1);energy=np.linalg.norm(s-truth,axis=1).mean()/np.sqrt(len(truth))-.5*np.linalg.norm(s[:n//2]-s[n//2:2*(n//2)],axis=1).mean()/np.sqrt(len(truth));d=s-mean[v];neighbor=[]
 for axis in [1,2]:
  aa=[slice(None)]*3;bb=aa.copy();aa[axis]=slice(1,None);bb[axis]=slice(None,-1);dd=samples-np.nanmean(samples,0);u=dd[tuple(aa)].ravel();vv=dd[tuple(bb)].ravel();ok=np.isfinite(u)&np.isfinite(vv);neighbor.append(float(np.corrcoef(u[ok],vv[ok])[0,1]) if ok.sum()>2 else np.nan)
 # Variogram score uses a fixed physical pixel lattice and sqrt increments.
 yy=y[::4,::4].ravel();ss=samples[:,::4,::4].reshape(n,-1);ii,jj=np.triu_indices(len(yy),1);ok=np.isfinite(yy[ii])&np.isfinite(yy[jj])&np.isfinite(ss[:,ii]).all(0)&np.isfinite(ss[:,jj]).all(0);variogram=float(np.mean((np.sqrt(abs(yy[ii[ok]]-yy[jj[ok]]))-np.mean(np.sqrt(abs(ss[:,ii[ok]]-ss[:,jj[ok]])),0))**2))
 result=dict(mae=float(abs(err).mean()),crps=crps(samples,y),extreme_mae=float(abs(err[ext]).mean()) if ext.any() else np.nan,coverage90=float(((truth>=low)&(truth<=hi)).mean()),interval_width90=float((hi-low).mean()),energy_score=float(energy),variogram_score=variogram,neighbor_correlation=float(np.nanmean(neighbor)),pixel_event_brier=float(((pixelprob-outcome)**2).mean()),map_event=event,event_probability=float(p),event_brier=float((p-event)**2),event_log_loss=float(-event*np.log(p)-(1-event)*np.log1p(-p)),simultaneous90=float(((truth>=low)&(truth<=hi)).all()))
 for nominal in [.5,.8,.95]:
  lo,hi=np.quantile(s,[(1-nominal)/2,(1+nominal)/2],axis=0);result['coverage'+str(int(nominal*100))]=float(((truth>=lo)&(truth<=hi)).mean());result['width'+str(int(nominal*100))]=float((hi-lo).mean())
 return result

def fit_artifact(m,a,tr,base,cond):
 tick=time.perf_counter();res=(a['truth']-base)/a['scale'][:,None,None];basis=SpatialBasis('block_pca',4).fit(res[tr]);zs=StandardScaler().fit(basis.encode(res[tr]));z=zs.transform(basis.encode(res));cs=RobustScaler().fit(cond[tr]);cx=np.clip(cs.transform(cond),-6,6);head=Ridge(alpha=100).fit(cx[tr],z[tr]);rr=z[tr]-head.predict(cx[tr]);cov=np.cov(rr,rowvar=False);cov=.5*cov+.5*np.diag(np.diag(cov))+np.eye(cov.shape[0])*.001;noise=np.maximum(np.nan_to_num(np.nanstd(res[tr]-basis.decode(basis.encode(res[tr])),axis=0),nan=.1),.05);std=[]
 for k in range(6):
  ids=[i for i in tr if min(m.iloc[i].horizon_days//30,5)==k];std.append(np.maximum(np.nan_to_num(np.nanstd(res[ids if len(ids)>=10 else tr],axis=0),nan=1),.1))
 hs=np.c_[np.log1p(m.horizon_days),np.log1p(a['scale']),np.isfinite(a['history']).mean((1,2,3))];hsc=RobustScaler().fit(hs[tr]);het=Ridge(alpha=100).fit(hsc.transform(hs[tr]),np.log(np.maximum(np.nanmean(res[tr]**2,axis=(1,2)),.01)));pred=np.exp(.5*het.predict(hsc.transform(hs)));pred/=np.median(pred[tr]);return dict(basis=basis,latent_scaler=zs,condition_scaler=cs,head=head,chol=np.linalg.cholesky(cov),noise=noise,independent_std=np.array(std),heteroscedastic_factor=np.clip(pred,.5,3),training_pairs=len(tr)),time.perf_counter()-tick

def sample(artifact,cond,base,a,m,i,kind,net=None,steps=16,disagreement=None):
 if net is not None and isinstance(net,StepFlow):
  cx=np.clip(artifact['condition_scaler'].transform(cond[i,None]),-6,6);torch.manual_seed(SEED+int(m.iloc[i].origin_day));device=next(net.parameters()).device
  with torch.no_grad():zz=net.sample(torch.tensor(cx,dtype=torch.float32,device=device),32,steps).cpu().numpy()
  maps=artifact['basis'].decode(artifact['latent_scaler'].inverse_transform(zz));rng=np.random.default_rng(SEED+int(m.iloc[i].origin_day));noise=rng.normal(size=maps.shape)
  for j in range(32):noise[j]=gaussian_filter(noise[j],max(.5,1/max(a['spacing'][i])));noise[j]/=max(noise[j].std(),1e-6)
  result=base[i]+(maps+noise*artifact['noise'])*a['scale'][i]
 else:result=decode_samples(artifact,cond[i],base[i],a['scale'][i],a['spacing'][i],m.iloc[i].horizon_days,kind='independent_gaussian' if kind in ['independent','heteroscedastic','disagreement'] else 'student' if kind=='student' else 'gaussian',model=net,seed=SEED+int(m.iloc[i].origin_day))
 if kind=='heteroscedastic':result=base[i]+(result-base[i])*artifact['heteroscedastic_factor'][i]
 if kind=='disagreement':
  factor=np.sqrt(1+(disagreement[i]/np.maximum(np.mean(artifact['independent_std'][min(m.iloc[i].horizon_days//30,5)])*a['scale'][i],1))**2);result=base[i]+(result-base[i])*factor
 return result.astype('float32')

def run():
 assert done('hybrid');start=time.perf_counter();torch.set_num_threads(2);m,a,_=data();tr,cal,sel,test=partitions(m);ids=np.unique(np.r_[cal,sel,test]);threshold=readjson(ROOT/'manifests/thresholds.json');cond=conditioning(m,a);v6=np.load(V6/'deterministic/point_predictions.npz');new=np.load(ROOT/'predictions/new_point.npz');hy=np.load(ROOT/'predictions/hybrid.npz');ps=readjson(ROOT/'manifests/point_selection.json');hs=readjson(ROOT/'manifests/hybrid_selection.json');parents={'EMA':v6['ema_half30'],'lowrank':new[ps['selected_lowrank']],'hybrid':hy[hs['selected_model']]};registry=[];timing=[];diagnostics=[];device='cuda' if torch.cuda.is_available() else 'cpu';torch.manual_seed(SEED);arts={}
 for name,base in parents.items():
  artifact,seconds=fit_artifact(m,a,tr,base,cond);arts[name]=artifact;joblib.dump(artifact,ROOT/'checkpoints'/('prob_'+name+'.joblib'));timing.append(dict(model_id='prob_'+name,training_seconds=seconds,device='CPU',training_pairs=len(tr)))
  for kind in (['independent','structured','student','heteroscedastic'] if name!='hybrid' else ['independent','structured','student','disagreement']):registry.append((name+'_'+kind,artifact,base,kind,None,16))
 # Frozen V6 generators are read-only reference copies, same conditions and parent.
 oldart=joblib.load(V6/'generative/level15_probabilistic.joblib')
 for kind in ['independent','structured','student']:registry.append(('V6_'+kind,oldart,parents['EMA'],kind,None,16))
 for family,seed in [('cvae',20261109),('flow',20261108)]:
  ck=torch.load(V6/f'generative/{family}_level15_seed{seed}.pt',map_location='cpu',weights_only=False);net=(CVAE if family=='cvae' else StepFlow)(ck['latent_dim'],ck['condition_dim']).to(device);net.load_state_dict(ck['state_dict']);net.eval()
  for steps in ([16] if family=='cvae' else [8,16,32]):registry.append((f'V6_{family}_steps{steps}',oldart,parents['EMA'],'structured',net,steps))
 # Three justified pilot variants; extra seeds continue only after registered 2023 gain.
 artifact=arts['EMA'];cx=np.clip(artifact['condition_scaler'].transform(cond),-6,6);res=(a['truth']-parents['EMA'])/a['scale'][:,None,None];z=artifact['latent_scaler'].transform(artifact['basis'].encode(res));cc=torch.tensor(cx,dtype=torch.float32,device=device);zz=torch.tensor(z,dtype=torch.float32,device=device);curves=[];pilot=[]
 def train_variant(label,seed):
  d=z.shape[1];c=cx.shape[1];torch.manual_seed(seed);net=(DiagnosticVAE(d,c,8,.01,False) if label=='cvae_beta01' else DiagnosticVAE(d,c,16,.1,True) if label=='cvae_huber16' else StepFlow(d,c)).to(device);ckpath=ROOT/'checkpoints'/(label+'_seed'+str(seed)+'.pt')
  if ckpath.exists():
   ck=torch.load(ckpath,map_location='cpu',weights_only=False);net.load_state_dict(ck['state_dict']);net.eval();print('V7 NEURAL CHECKPOINT REUSED',label,seed,flush=True);return net
  optim=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.0001);rng=np.random.default_rng(seed);counts=m.iloc[tr].volcano.value_counts();w=np.array([1/counts[m.iloc[i].volcano] for i in tr]);w/=w.sum();best=None;wait=0;tick=time.perf_counter()
  for epoch in range(60):
   net.train();losses=[];ii=rng.choice(tr,len(tr),replace=True,p=w)
   for begin in range(0,len(ii),64):
    ix=ii[begin:begin+64];optim.zero_grad();target=zz[ix].clamp(-4,4) if label=='flow_robust' else zz[ix];loss,rec,kl=net.loss(target,cc[ix]);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optim.step();losses.append([float(v.detach().cpu()) for v in [loss,rec,kl]])
   net.eval();torch.manual_seed(seed+1000)
   with torch.no_grad():val=float(net.loss(zz[cal],cc[cal])[0].cpu())
   curves.append(dict(model_id=label+'_seed'+str(seed),epoch=epoch,training_loss=np.mean(losses,0)[0],reconstruction_loss=np.mean(losses,0)[1],KL=np.mean(losses,0)[2],calibration_loss=val))
   if best is None or val<best[0]-1e-4:best=(val,copy.deepcopy(net.state_dict()),epoch);wait=0
   else:wait+=1
   if wait>=12:break
  net.load_state_dict(best[1]);net.eval();timing.append(dict(model_id=label+'_seed'+str(seed),training_seconds=time.perf_counter()-tick,epochs=epoch+1,parameters=sum(p.numel() for p in net.parameters()),device=device));torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},label=label,seed=seed,latent_dim=d,condition_dim=c,best_epoch=best[2],parent='EMA',artifact=str(ROOT/'checkpoints/prob_EMA.joblib')),ROOT/'checkpoints'/(label+'_seed'+str(seed)+'.pt'));return net
 for label in ['cvae_beta01','cvae_huber16','flow_robust']:
  net=train_variant(label,SEED);registry.append((label+'_seed'+str(SEED),artifact,parents['EMA'],'structured',net,16));print('V7 NEURAL PILOT TRAINED',label,flush=True)
 # Cache one distribution per candidate. Frozen predictions and selection are unchanged.
 allrows=[];calrecords=[];selection=[];continuations=[]
 def continuation(mid,f):
  if mid not in [x+'_seed'+str(SEED) for x in ['cvae_beta01','cvae_huber16','flow_robust']]:return
  label=mid.rsplit('_seed',1)[0];reference='V6_flow_steps16' if label=='flow_robust' else 'V6_cvae_steps16';ref=pd.DataFrame(allrows);ref=ref[(ref.model_id==reference)&(ref.role=='validation')&(ref.calibration=='raw')];cand=f[(f.role=='validation')&(f.calibration=='raw')];gain=1-macro(cand,['crps']).crps/macro(ref,['crps']).crps;continuations.append(dict(model_id=mid,reference=reference,relative_CRPS_gain=gain,continue_two_seeds=bool(gain>=.01)))
  if gain>=.01:
   for ss in [SEED+1,SEED+2]:
    name=label+'_seed'+str(ss)
    if name not in [r[0] for r in registry]:registry.append((name,arts['EMA'],parents['EMA'],'structured',train_variant(label,ss),16))
  csv(ROOT/'generative/pilot_continuation.csv',continuations)

 for mid,artifact,base,kind,net,steps in registry:
  checkpoint=ROOT/'generative'/(mid+'_metrics.csv')
  if checkpoint.exists():
   previous=pd.read_csv(checkpoint);allrows.extend(previous.to_dict('records'));continuation(mid,previous);continue
  tick=time.perf_counter();raw={i:sample(artifact,cond,base,a,m,i,kind,net,steps,hy['branch_disagreement']) for i in ids};timing.append(dict(model_id=mid,sampling_seconds=time.perf_counter()-tick,cases=len(ids),samples_per_case=32,device=device if net is not None else 'CPU'));best=None
  for factor in [.75,1.,1.5,2.]:
   ff=[]
   for i in cal:
    mean=raw[i].mean(0);ff.append(dict(volcano=m.iloc[i].volcano,crps=crps(mean+(raw[i]-mean)*factor,a['truth'][i])))
   value=macro(pd.DataFrame(ff),['crps']).crps
   if best is None or value<best[0]:best=(value,factor)
  bias=np.nanmedian(np.stack([a['truth'][i]-raw[i].mean(0) for i in cal]),axis=0);bias=np.nan_to_num(bias);q=[];maxscores=[];last_by_source={}
  for i in sorted(cal,key=lambda j:m.iloc[j].target_day):
   r=m.iloc[i]
   if r.history_start_day<=last_by_source.get(r.source_id,-np.inf):continue
   last_by_source[r.source_id]=r.target_day;lo,hi=np.quantile(raw[i],[.05,.95],axis=0);score=np.maximum(lo-a['truth'][i],a['truth'][i]-hi);q.append(float(np.nanquantile(score,.9)));sd=np.maximum(raw[i].std(0),.1);maxscores.append(float(np.nanmax(abs(a['truth'][i]-raw[i].mean(0))/sd)))
  radius=max(float(np.quantile(q,min(1,np.ceil((len(q)+1)*.9)/len(q)))) if q else 0,0);simq=float(np.quantile(maxscores,min(1,np.ceil((len(maxscores)+1)*.9)/len(maxscores)))) if maxscores else np.nan
  register(ROOT/'checkpoints'/(mid+'_calibration.json'),dict(scale=best[1],bias_map_source='2022only',quantile_radius=radius,nonoverlap_calibration_blocks=len(q),simultaneous_standardized_radius=simq,exchangeability_guaranteed=False));np.savez_compressed(ROOT/'checkpoints'/(mid+'_bias.npz'),bias=bias);rows=[]
  for i in np.r_[sel,test]:
   r=m.iloc[i];common=dict(model_id=mid,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,region=r.region,role=r.role,horizon_days=r.horizon_days,origin_day=r.origin_day)
   for method in ['raw','scale_CRPS','bias_corrected','quantile_additive','block_max_field']:
    mean=raw[i].mean(0);ss=mean+(raw[i]-mean)*(best[1] if method=='scale_CRPS' else 1);ss=ss+bias if method=='bias_corrected' else ss;sc0=scores(ss,a['truth'][i],a['last'][i],a['spacing'][i],threshold)
    if method in ['quantile_additive','block_max_field']:
     lo,hi=np.quantile(raw[i],[.05,.95],axis=0)
     if method=='quantile_additive':lo-=radius;hi+=radius
     else:sd=np.maximum(raw[i].std(0),.1);lo=mean-simq*sd;hi=mean+simq*sd
     valid=np.isfinite(a['truth'][i])&np.isfinite(lo);sc0['coverage90']=float(((a['truth'][i]>=lo)&(a['truth'][i]<=hi))[valid].mean());sc0['interval_width90']=float((hi-lo)[valid].mean());sc0['simultaneous90']=float(((a['truth'][i]>=lo)&(a['truth'][i]<=hi))[valid].all());sc0['interval_only_adjustment']=True
    rows.append(dict(**common,calibration=method,**sc0))
   if i==test[0]:np.savez_compressed(ROOT/'generative'/(mid+'_samples.npz'),samples=raw[i],truth=a['truth'][i],last=a['last'][i],example_id=r.example_id)
  f=csv(checkpoint,rows);allrows.extend(rows);continuation(mid,f);tick=time.perf_counter();csv(ROOT/'generative/runtime_partial.csv',timing);csv(ROOT/'generative/training_curves.csv',curves);print('V7 PROBABILITY SCORED',mid,flush=True)
 f=csv(ROOT/'predictions/probability_metrics.csv',allrows);csv(ROOT/'calibration_results.csv',f.groupby(['role','model_id','calibration','volcano'])[['crps','coverage50','coverage80','coverage90','coverage95','interval_width90','energy_score','variogram_score','simultaneous90','event_brier']].mean().groupby(['role','model_id','calibration']).mean().reset_index());csv(ROOT/'generative_improvements.csv',f[(f.role=='validation')&f.calibration.isin(['raw','scale_CRPS','bias_corrected'])].groupby(['model_id','calibration','volcano'])[['crps','coverage90','interval_width90','energy_score','event_brier']].mean().groupby(['model_id','calibration']).mean().reset_index());csv(ROOT/'generative/runtime.csv',timing);csv(ROOT/'generative/training_curves.csv',curves);rank=pd.read_csv(ROOT/'generative_improvements.csv').sort_values('crps');register(ROOT/'manifests/probability_selection.json',dict(selected_model=rank.iloc[0].model_id,calibration=rank.iloc[0].calibration,selection='2023developmentCRPS only; interval-only methods not comparable distribution candidates',Peru_used=False));finish('probability',wall_seconds=time.perf_counter()-start);print('V7 PROBABILITY COMPLETE',flush=True)
if __name__=='__main__':run()
