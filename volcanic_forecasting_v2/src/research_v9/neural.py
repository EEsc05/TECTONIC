"""Controlled CNN/CVAE/flow staged HPO on training-internal future forecasts."""
from .common import *
from .protocol import neural_space
from ..research_v4.representation import SpatialBasis
from ..research_v6.generative import conditioning
from ..research_v7.probability import crps
import torch,torch.nn as nn,torch.nn.functional as F,joblib,time,copy

def mlp(d,out,width,depth):
 layers=[]
 for _ in range(depth):layers.extend([nn.Linear(d,width),nn.SiLU()]);d=width
 return nn.Sequential(*layers,nn.Linear(d,out))
class CNN(nn.Module):
 def __init__(self,cfg):
  super().__init__();w=cfg['width'];k=cfg['kernel'];self.cfg=cfg;self.input=nn.Conv2d(23,w,k,padding=k//2);self.layers=nn.ModuleList([nn.Conv2d(w,w,k,padding=k//2) for _ in range(cfg['depth']-1)]);self.out=nn.Conv2d(w,1,1);nn.init.zeros_(self.out.weight);nn.init.zeros_(self.out.bias)
 def forward(self,x):
  y=F.silu(self.input(x))
  for layer in self.layers:y=F.silu(layer(y))+(y if self.cfg['residual'] else 0)
  return self.out(y)[:,0]
class VAE(nn.Module):
 def __init__(self,d,c,cfg):
  super().__init__();self.cfg=cfg;z=cfg['latent'];self.prior=mlp(c,2*z,cfg['width'],cfg['depth']);self.posterior=mlp(d+c,2*z,cfg['width'],cfg['depth']);self.decoder=mlp(z+c,d,cfg['width'],cfg['depth']);self.lognoise=nn.Parameter(torch.full((d,),-1.))
 def loss(self,y,c,epoch):
  qm,ql=self.posterior(torch.cat([y,c],-1)).chunk(2,-1);pm,pl=self.prior(c).chunk(2,-1)
  if not self.cfg.get('conditional_prior',True):pm=pm*0;pl=pl*0
  ql=ql.clamp(-6,3);pl=pl.clamp(-6,3);pred=self.decoder(torch.cat([qm+torch.randn_like(qm)*(.5*ql).exp(),c],-1));e=(y-pred)*(-self.lognoise.clamp(-4,2)).exp();rec=(F.huber_loss(e,torch.zeros_like(e),delta=self.cfg['huber_delta'],reduction='none')+self.lognoise.clamp(-4,2)).mean() if self.cfg['loss']=='huber' else (.5*e**2+self.lognoise.clamp(-4,2)).mean();kl=.5*(pl-ql+(ql.exp()+(qm-pm)**2)/pl.exp()-1).mean();beta=self.cfg['beta']*min(1,(epoch+1)/10) if self.cfg['anneal'] else self.cfg['beta'];return rec+beta*kl,rec,kl
 def sample(self,c,n,steps=16):
  pm,pl=self.prior(c).chunk(2,-1)
  if not self.cfg.get('conditional_prior',True):pm=pm*0;pl=pl*0
  pm=pm.repeat_interleave(n,0);pl=pl.clamp(-6,3).repeat_interleave(n,0);cc=c.repeat_interleave(n,0);out=self.decoder(torch.cat([pm+torch.randn_like(pm)*(.5*pl).exp(),cc],-1));return out+torch.randn_like(out)*self.lognoise.clamp(-4,2).exp()
class Flow(nn.Module):
 def __init__(self,d,c,cfg):super().__init__();self.cfg=cfg;self.dim=d;self.net=mlp(d+c+3,d,cfg['width'],cfg['depth'])
 def forward(self,z,c,t):return self.net(torch.cat([z,c,t,torch.sin(t*np.pi),torch.cos(t*np.pi)],-1))
 def loss(self,y,c,epoch):
  e=torch.randn_like(y);t=torch.rand((len(y),1),device=y.device);p=self((1-t)*e+t*y,c,t);v=y-e;loss=F.huber_loss(p,v,delta=self.cfg['huber_delta']) if self.cfg['loss']=='huber' else abs(p-v).mean() if self.cfg['loss']=='mae' else ((p-v)**2).mean();return loss,loss,loss*0
 def sample(self,c,n,steps=16):
  c=c.repeat_interleave(n,0);z=torch.randn((len(c),self.dim),device=c.device);dt=1/steps
  for j in range(steps):
   t=torch.full((len(c),1),j/steps,device=c.device)
   if self.cfg.get('integrator')=='heun':v=self(z,c,t);z=z+.5*dt*(v+self(z+dt*v,c,t+dt))
   else:z=z+dt*self(z,c,t+dt/2)
  return z

def prep(m,a,ids):
 unique=m.iloc[ids].drop_duplicates(['source_id','origin_day']).index;past=np.concatenate([np.nan_to_num((a['history'][i]-a['last'][i])/a['scale'][i]) for i in unique]);basis=SpatialBasis('block_pca',4).fit(past);from sklearn.preprocessing import StandardScaler,RobustScaler
 base=np.load(V6/'deterministic/point_predictions.npz')['ema_half30'];res=(a['truth']-base)/a['scale'][:,None,None];raw=basis.encode(res);zs=StandardScaler().fit(raw[ids]);cond=conditioning(m,a);cs=RobustScaler().fit(cond[ids]);cx=np.clip(cs.transform(cond),-6,6);z=zs.transform(raw);tail=np.maximum(np.nan_to_num(np.nanstd(res[ids]-basis.decode(raw[ids]),axis=0),nan=.1),.05);return dict(basis=basis,zscaler=zs,cscaler=cs,noise=tail,basis_fit='Historical training maps only',train_ids=ids.tolist()),z.astype('float32'),cx.astype('float32'),base

def decode(net,cfg,artifact,c,base,a,i,n=16,steps=16,sample_seed=None):
 torch.manual_seed((SEED if sample_seed is None else sample_seed)+int(i));device=next(net.parameters()).device;cc=c[i].copy()
 if cfg['condition']=='no_geometry':cc[-5:-1]=0
 with torch.no_grad():z=net.sample(torch.tensor(cc[None],device=device),n,steps).cpu().numpy()
 if cfg['normalization']=='asinh':z=2*np.sinh(np.clip(z/2,-10,10))
 fields=artifact['basis'].decode(artifact['zscaler'].inverse_transform(z));rng=np.random.default_rng((SEED if sample_seed is None else sample_seed)+i);noise=rng.normal(size=fields.shape);from scipy.ndimage import gaussian_filter
 for j in range(n):noise[j]=gaussian_filter(noise[j],max(.5,1/max(a['spacing'][i])));noise[j]/=max(noise[j].std(),1e-6)
 return (base[i]+(fields+noise*artifact['noise'])*a['scale'][i]).astype('float32')

def train(cfg,m,a,tr,val,artifact=None,z=None,c=None,base=None,seed=SEED,epochs=None):
 device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2);torch.manual_seed(seed);family=cfg['family'];limit=epochs or cfg['max_epochs'];net=CNN(cfg) if family=='cnn' else (VAE if family=='cvae' else Flow)(z.shape[1],c.shape[1],cfg);net=net.to(device);opt=torch.optim.AdamW(net.parameters(),lr=cfg['lr'],weight_decay=cfg['decay']);rng=np.random.default_rng(seed);best=None;wait=0;curves=[]
 if family=='cnn':
  xx=torch.tensor(a['cnn'],device=device);target=np.nan_to_num((a['truth']-a['last'])/a['scale'][:,None,None]);mask=torch.tensor(np.isfinite(a['truth'])&np.isfinite(a['last']),device=device);yy=torch.tensor(target,dtype=torch.float32,device=device)
  if cfg['normalization']=='asinh':xx=2*torch.asinh(xx/2);yy=2*torch.asinh(yy/2)
 else:
  xx=torch.tensor(c,device=device);target=z.copy()
  if cfg['condition']=='no_geometry':xx=xx.clone();xx[:,-5:-1]=0
  if cfg['normalization']=='asinh':target=2*np.arcsinh(target/2)
  yy=torch.tensor(target,dtype=torch.float32,device=device)
 tick=time.perf_counter()
 for epoch in range(limit):
  net.train();losses=[]
  for offset in range(0,len(tr),cfg['batch']):
   ids=rng.permutation(tr)[offset:offset+cfg['batch']];opt.zero_grad()
   if family=='cnn':
    p=net(xx[ids]);err=p-yy[ids];loss=(abs(err)*mask[ids]).sum()/mask[ids].sum() if cfg['loss']=='mae' else ((err**2)*mask[ids]).sum()/mask[ids].sum() if cfg['loss']=='mse' else (F.huber_loss(p,yy[ids],delta=cfg['huber_delta'],reduction='none')*mask[ids]).sum()/mask[ids].sum();rec=loss;kl=loss*0
   else:loss,rec,kl=net.loss(yy[ids],xx[ids],epoch)
   loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append([float(v.detach().cpu()) for v in [loss,rec,kl]])
  net.eval();metric=np.nan
  if val is not None and (family=='cnn' or epoch%5==4 or epoch==limit-1):
   if family=='cnn':
    with torch.no_grad():p=net(xx[val]);vl=float((abs(p-yy[val])*mask[val]).sum()/mask[val].sum());metric=vl
   else:
    values=[dict(volcano=m.iloc[i].volcano,crps=crps(decode(net,cfg,artifact,c,base,a,i),a['truth'][i])) for i in val];metric=float(macro(pd.DataFrame(values),['crps']).crps)
   if best is None or metric<best[0]-1e-4:best=(metric,copy.deepcopy(net.state_dict()),epoch);wait=0
   else:wait+=1 if family=='cnn' else 5
   if wait>=cfg['patience']:break
  curves.append(dict(epoch=epoch,training_loss=np.mean(losses,0)[0],reconstruction_loss=np.mean(losses,0)[1],KL=np.mean(losses,0)[2],internal_validation_score=metric))
 if best is not None:net.load_state_dict(best[1])
 net.eval();return net,curves,dict(training_seconds=time.perf_counter()-tick,epochs=epoch+1,best_epoch=best[2] if best else epoch,device=device,parameters=sum(p.numel() for p in net.parameters()),full_fit_early_stop='Fixed internal-best epoch count; no2022/2023 labels' if val is None else 'Internal training fold only')

def cnn_predict(net,cfg,a,ids):
 device=next(net.parameters()).device;out=[]
 with torch.no_grad():
  for start in range(0,len(ids),64):
   ix=ids[start:start+64];x=torch.tensor(a['cnn'][ix],device=device);x=2*torch.asinh(x/2) if cfg['normalization']=='asinh' else x;p=net(x).cpu().numpy();p=2*np.sinh(np.clip(p/2,-10,10)) if cfg['normalization']=='asinh' else p;out.extend(a['last'][ix]+p*a['scale'][ix,None,None])
 return np.asarray(out,dtype='float32')

def run():
 assert done('features');m,a,_=data();tr,cal,dev,test=partitions(m);fold,fitids,valall=folds(m)[1];val=m.iloc[valall].sort_values('target_day').drop_duplicates(['source_id','origin_day']).index.to_numpy();art,z,c,base=prep(m,a,fitids);fullart,fullz,fullc,fullbase=prep(m,a,tr);joblib.dump(fullart,ROOT/'checkpoints/neural_preprocessor.joblib');records=[];allcurves=[];timing=[];selections={};cnn_preds={}
 registered={}
 for fam in ['cnn','cvae','flow']:
  cs=neural_space(fam)
  for j,cfg in enumerate(cs):cfg['conditional_prior']=j!=4;cfg['integrator']='heun' if j in [3,5] else 'euler'
  registered[fam]=cs
 register(ROOT/'manifests/neural_exact_implementation.json',dict(configs=registered,validation='Same94 metadata-thinned2018–19 origins, training before2018; target CRPS or actual future MAE; no2022/2023 tuning',refinement='Top2 per family, learning_rate x.3/x3,40maxepochs; finalists chosen internally before final assessment'))
 for family in ['cnn','cvae','flow']:
  scored=[]
  def screen(cfg,stage):
   cache=ROOT/'search'/(cfg['id']+'.json')
   if cache.exists():r=readjson(cache);records.append(r);scored.append(r);return
   net,curve,clock=train(cfg,m,a,fitids,val,art,z,c,base);tick=time.perf_counter()
   if family=='cnn':p=cnn_predict(net,cfg,a,val);value=float(macro(pd.DataFrame([dict(volcano=m.iloc[i].volcano,mae=np.nanmean(abs(p[j]-a['truth'][i]))) for j,i in enumerate(val)]),['mae']).mae)
   else:value=float(macro(pd.DataFrame([dict(volcano=m.iloc[i].volcano,crps=crps(decode(net,cfg,art,c,base,a,i),a['truth'][i])) for i in val]),['crps']).crps)
   r=dict(model_id=cfg['id'],family=family,stage=stage,config=cfg,internal_score=value,objective='MAE' if family=='cnn' else 'CRPS',fold=fold,validation_pairs=len(val),**clock,inference_scoring_seconds=time.perf_counter()-tick);writejson(cache,r);records.append(r);scored.append(r);allcurves.extend([dict(model_id=cfg['id'],family=family,**r) for r in curve]);history('neural',records);csv(ROOT/'diagnostics/neural_learning_curves.csv',allcurves);print('V9 NEURAL SCREEN',cfg['id'],value,flush=True)
  for cfg in registered[family]:screen(cfg,'coarse')
  top=sorted(scored,key=lambda r:r['internal_score'])[:2]
  for j,r in enumerate(top):cfg=copy.deepcopy(r['config']);cfg['id']+='_refine';cfg['lr']*=.3 if j==0 else 3;cfg['max_epochs']=40;cfg['patience']=8;screen(cfg,'refinement')
  shortlist=sorted(scored,key=lambda r:r['internal_score'])[:2];selections[family]=[dict(config=r['config'],score=r['internal_score'],full_epochs=max(5,min(60,r['best_epoch']+1))) for r in shortlist]
  for rank,item in enumerate(selections[family]):
   cfg=item['config']
   for seed in [SEED,SEED+1,SEED+2]:
    mid=cfg['id']+'_seed'+str(seed);path=ROOT/'checkpoints'/(mid+'.pt');torch.manual_seed(seed)
    if path.exists():ck=torch.load(path,map_location='cpu',weights_only=False);net=CNN(cfg) if family=='cnn' else (VAE if family=='cvae' else Flow)(fullz.shape[1],fullc.shape[1],cfg);net.load_state_dict(ck['state_dict']);net.eval()
    else:
     net,curve,clock=train(cfg,m,a,tr,None,fullart,fullz,fullc,fullbase,seed,epochs=item['full_epochs']);torch.save(dict(state_dict={k:v.cpu() for k,v in net.state_dict().items()},config=cfg,seed=seed,latent_dim=fullz.shape[1],condition_dim=fullc.shape[1],version=VERSION),path);timing.append(dict(model_id=mid,**clock));allcurves.extend([dict(model_id=mid,family=family,**r) for r in curve]);csv(ROOT/'diagnostics/neural_runtime.csv',timing)
    if family=='cnn':tick=time.perf_counter();cnn_preds[mid]=cnn_predict(net,cfg,a,np.arange(len(m)));timing.append(dict(model_id=mid,inference_seconds=time.perf_counter()-tick,cases=len(m),device=next(net.parameters()).device.type))
    print('V9 NEURAL FULL FIT',mid,flush=True)
  csv(ROOT/(family+'_hyperparameter_results.csv'),[r for r in records if r['family']==family])
 register(ROOT/'manifests/neural_selection.json',dict(finalists=selections,selection='Training-internal future forecast scores only, two configurations x3full training seeds per family; no2023 tuning'));np.savez_compressed(ROOT/'predictions/cnn.npz',**cnn_preds);csv(ROOT/'diagnostics/neural_runtime.csv',timing);csv(ROOT/'diagnostics/neural_learning_curves.csv',allcurves);history('neural',records);finish('neural');print('V9 NEURAL HPO COMPLETE',flush=True)
if __name__=='__main__':run()
