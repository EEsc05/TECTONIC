"""Conventional predictive distributions and locked neural finalists; no final-label tuning."""
from .common import *
from .features import load
from .conventional import fit as fit_pixel,predict as predict_pixel,parent_maps
from .neural import CNN,VAE,Flow,decode
from .spatial import fit as fit_spatial,encode as spatial_encode
from ..research_v7.probability import crps,scores as raw_scores
from ..research_v6.data import score as point_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from scipy.ndimage import gaussian_filter
import torch,joblib,time

def score(samples,a,i,threshold):
 z=raw_scores(samples,a['truth'][i],a['last'][i],a['spacing'][i],threshold);mean=np.nanmean(samples,0);v=np.isfinite(mean)&np.isfinite(a['truth'][i])&np.isfinite(a['last'][i]);z['rmse']=float(np.sqrt(np.mean((mean[v]-a['truth'][i][v])**2)));z['gradient_mae']=point_score(a['truth'][i],mean,a['last'][i],a['spacing'][i],threshold['pixel_extreme_mm'])['gradient_mae'];z['future_coverage']=v.sum()/max(np.isfinite(a['last'][i]).sum(),1)
 if z['future_coverage']<.9:z['map_event']=np.nan;z['event_brier']=np.nan;z['event_log_loss']=np.nan
 return z

def residual_artifact(m,a,mean,oof,ids,basis,shrink=.5,df=4,alpha=100,horizon_bins=False):
 train=np.array([i for i in ids if np.isfinite(oof[i]).sum()>128]);err=(a['truth'][train]-oof[train])/a['scale'][train,None,None];sigma=np.maximum(np.nan_to_num(np.nanstd(err,axis=0),nan=.5),.1);flat=np.nan_to_num(err).reshape(len(train),-1);latent=np.concatenate([p.transform(np.where(np.isfinite(err[:,y:y+8,x:x+8]),err[:,y:y+8,x:x+8],np.nan_to_num(np.nanmean(err[:,y:y+8,x:x+8],axis=0))).reshape(len(train),64)) for p,(y,x) in zip(basis.models,[(y,x) for y in range(0,32,8) for x in range(0,32,8)])],axis=1);latent-=latent.mean(0);cov=np.cov(latent,rowvar=False);cov=(1-shrink)*cov+shrink*np.diag(np.diag(cov))+np.eye(cov.shape[0])*.001;features=np.c_[np.log1p(m.horizon_days),np.log1p(a['scale']),np.ptp(a['times'],axis=1)/365,np.isfinite(a['history']).mean((1,2,3))];sc=RobustScaler().fit(features[train]);head=Ridge(alpha=alpha).fit(sc.transform(features[train]),np.log(np.maximum(np.nanmean(err**2,axis=(1,2)),.01)));factor=np.exp(.5*head.predict(sc.transform(features)));factor/=np.median(factor[train]);hstd=[]
 for lo,hi in [(1,30),(31,90),(91,180)]:
  ii=[j for j,i in enumerate(train) if lo<=m.iloc[i].horizon_days<=hi];hstd.append(np.maximum(np.nan_to_num(np.nanstd(err[ii],axis=0),nan=.5),.1) if len(ii)>=5 else sigma)
 return dict(sigma=sigma,horizon_std=np.array(hstd),horizon_bins=horizon_bins,student_df=df,shrink=shrink,basis=basis,horizons=m.horizon_days.to_numpy(),chol=np.linalg.cholesky(cov),latent_dim=len(cov),hetero=np.clip(factor,.5,3),OOF_training_ids=train.tolist(),variance_head=head,variance_scaler=sc)

def gaussian_samples(art,mean,a,i,kind,seed=SEED):
 rng=np.random.default_rng(seed+i);sigma=art['sigma'];n=32
 if art.get('horizon_bins'):sigma=art['horizon_std'][np.digitize(float(art['horizons'][i]),[31,91])]
 if kind in ['independent','heteroscedastic']:
  field=rng.normal(size=(n,32,32))*sigma
  if kind=='heteroscedastic':field*=art['hetero'][i]
 else:
  z=rng.normal(size=(n,art['latent_dim']))@art['chol'].T
  if kind=='student':z/=np.sqrt(rng.chisquare(art.get('student_df',4),(n,1))/(art.get('student_df',4)-2))
  b=art['basis'];zero=b.decode(np.zeros((1,len(z[0]))))[0];field=b.decode(z)-zero;white=rng.normal(size=field.shape)
  for j in range(n):white[j]=gaussian_filter(white[j],max(.5,1/max(a['spacing'][i])));white[j]/=max(white[j].std(),1e-6)
  field+=white*sigma*.25
 out=mean[i]+field*a['scale'][i];out[:,~np.isfinite(a['last'][i])]=np.nan;return out.astype('float32')

def run():
 assert done('neural_review') and done('modular') and done('fusion_growth');from .variance_search import optimize
 variance=optimize();m,a,_=data();tr,cal,dev,test=partitions(m);x=load();old=np.load(V6/'deterministic/point_predictions.npz');temporal=readjson(ROOT/'manifests/temporal_selection.json')['best'];sp=readjson(ROOT/'manifests/spatial_selection.json')['finalists'][0];conv=readjson(ROOT/'manifests/conventional_selection.json')['finalists'];mod=np.load(ROOT/'predictions/modular.npz');grow=readjson(ROOT/'manifests/hybrid_growth_gate.json')['passed'];hyname='three_regularized' if grow else 'pair_TS_weighted';spatial=np.load(ROOT/'predictions/spatial.npz');ml=np.load(ROOT/'predictions/conventional.npz');linear=min([r for r in conv if r['family'] in ['LinearRegression','Ridge','Lasso','ElasticNet','HuberRegressor','BayesianRidge']],key=lambda r:r['CV_MAE']);tree=min([r for r in conv if r['family'] in ['RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor','LightGBM']],key=lambda r:r['CV_MAE']);means={'EMA':old['ema_half30'],'Kalman':old['kalman_localtrend'],'lowrank':spatial[sp['model_id']],'hybrid':mod[hyname],'linear':ml['ML_'+linear['model_id']],'tree':ml['ML_'+tree['model_id']]};soof=np.load(ROOT/'predictions'/('OOF_'+sp['model_id']+'.npz'))['pred'];loof=np.load(ROOT/'predictions'/('OOF_ML_'+linear['model_id']+'.npz'))['pred'];toof=np.load(ROOT/'predictions'/('OOF_ML_'+tree['model_id']+'.npz'))['pred'];coof=np.load(ROOT/'predictions/context_oof.npz')['pred'];temp=np.load(ROOT/'predictions'/('temporal_'+temporal+'.npz'))['pred'];recipe=readjson(ROOT/'checkpoints/fusion_recipes.json')[hyname];stack=np.stack([temp,soof,coof,old['quality_v2_h6_half30_w0']]);ho=np.einsum('b,bnhw->nhw',recipe['weights'],stack[recipe['branches']]);oofs={'EMA':old['ema_half30'],'Kalman':old['kalman_localtrend'],'lowrank':soof,'hybrid':ho,'linear':loof,'tree':toof};art0=joblib.load(ROOT/'checkpoints/neural_preprocessor.joblib');basis=art0['basis'];registry=[];timing=[]
 for name,mean in means.items():
  tick=time.perf_counter();art=residual_artifact(m,a,mean,oofs[name],tr,basis,**variance['independent']['params']);joblib.dump(art,ROOT/'checkpoints'/('uncertainty_'+name+'.joblib'));timing.append(dict(model_id='uncertainty_'+name,training_seconds=time.perf_counter()-tick,OOF_cases=len(art['OOF_training_ids']),device='CPU'));registry.append((name+'_Gaussian',lambda i,art=art,mean=mean:gaussian_samples(art,mean,a,i,'independent')))
  if name=='EMA':
   for kind in ['heteroscedastic','structured','student']:
    aa=residual_artifact(m,a,mean,oofs[name],tr,basis,**variance[kind]['params']);joblib.dump(aa,ROOT/'checkpoints'/('uncertainty_EMA_'+kind+'.joblib'));registry.append(('EMA_'+kind,lambda i,art=aa,mean=mean,kind=kind:gaussian_samples(art,mean,a,i,kind)))
 # Quantile boosting is a conventional residual distribution, not an assumed Gaussian.
 cfg=dict(id='QuantileEMA',family='HistGradientBoostingRegressor',params=dict(max_iter=80,max_leaf_nodes=15,learning_rate=.05,min_samples_leaf=40,l2_regularization=10,early_stopping=False),target='residual_EMA',scaler='robust',features='full',max_rows=24000);parents=parent_maps(a);qs=[]
 for q in [.05,.5,.95]:
  from .conventional import estimator
  c=copy_config=dict(cfg);c['params']=dict(cfg['params'],loss='quantile',quantile=q);tick=time.perf_counter();head=fit_pixel(c,m,a,x,tr,parents);joblib.dump(head,ROOT/'checkpoints'/('quantile_'+str(q)+'.joblib'));qs.append(predict_pixel(head,a,x,np.arange(len(m)),parents));timing.append(dict(model_id='quantile_'+str(q),training_and_inference_seconds=time.perf_counter()-tick,device='CPU'))
 quant=np.sort(np.stack(qs),axis=0);np.savez_compressed(ROOT/'predictions/quantile_maps.npz',quantiles=quant)
 def quantile_sample(i):
  rng=np.random.default_rng(SEED+i);zz=rng.normal(size=(32,32,32));med=quant[1,i];left=np.maximum((med-quant[0,i])/1.64485362695,.1);right=np.maximum((quant[2,i]-med)/1.64485362695,.1);return (med+zz*np.where(zz<0,left,right)).astype('float32')
 registry.append(('QuantileBoosting_EMA',quantile_sample))
 # BayesianRidge posterior predictive standard deviation, on its bounded pixel feature task.
 br=next(r for r in conv if r['family']=='BayesianRidge');bart=joblib.load(ROOT/'checkpoints'/('ML_'+br['model_id']+'.joblib'));bmeans=[];bstd=[]
 for i in range(len(m)):
  xx=bart['pipeline'].named_steps['scaler'].transform(x[i][:,bart['columns']]);mu,sigma=bart['pipeline'].named_steps['model'].predict(xx,return_std=True);bmeans.append(mu.reshape(32,32)*bart['target_scale']+bart['target_offset']+parents[bart['config']['target']][i]);bstd.append(sigma.reshape(32,32)*bart['target_scale'])
 bmeans=np.array(bmeans);bstd=np.array(bstd);registry.append(('BayesianRidge_predictive',lambda i:bmeans[i]+np.random.default_rng(SEED+i).normal(size=(32,32,32))*bstd[i]))
 # GP posterior sampling is bounded in the latent representation, never full-pixel kernel fitting.
 gpconfigs=[r for r in readjson(ROOT/'manifests/spatial_selection.json')['all_CV'] if r['method']=='GaussianProcess'];gpc=min(gpconfigs,key=lambda r:r['mae']);tick=time.perf_counter();gpart=fit_spatial(gpc['representation'],'GaussianProcess',m,a,tr);joblib.dump(gpart,ROOT/'checkpoints/GP_latent_predictive.joblib');timing.append(dict(model_id='GP_latent_predictive',training_seconds=time.perf_counter()-tick,training_cases_cap=256,device='CPU'));gx,_,gbasis,_,_=spatial_encode(gpart,m,a,np.arange(len(m)));gmu,gstd=gpart['head'].predict(gpart['scaler'].transform(gx),return_std=True)
 def gp_sample(i):
  rng=np.random.default_rng(SEED+i);mu=np.asarray(gmu[i]);sd=np.asarray(gstd[i]);zs=mu+rng.normal(size=(32,len(mu)))*sd;zz=gpart['target_scaler'].inverse_transform(zs);field=(zz@gbasis[i]).reshape(32,32,32)*a['scale'][i]+a['last'][i];return field.astype('float32')
 registry.append(('GP_latent_predictive',gp_sample))
 # All neural finalists/configurations are already fixed by internal CV, before2023 labels.
 from ..research_v6.generative import conditioning
 cond=np.clip(art0['cscaler'].transform(conditioning(m,a)),-6,6).astype('float32');base=old['ema_half30'];nsel=readjson(ROOT/'manifests/neural_selection.json')['finalists'];device='cuda' if torch.cuda.is_available() else 'cpu';torch.set_num_threads(2)
 for family in ['cvae','flow']:
  for item in nsel[family]:
   cfg=item['config']
   for seed in [SEED,SEED+1,SEED+2]:
    mid=cfg['id']+'_seed'+str(seed);ck=torch.load(ROOT/'checkpoints'/(mid+'.pt'),map_location='cpu',weights_only=False);net=(VAE if family=='cvae' else Flow)(ck['latent_dim'],ck['condition_dim'],cfg).to(device);net.load_state_dict(ck['state_dict']);net.eval();registry.append((mid,lambda i,net=net,cfg=cfg:decode(net,cfg,art0,cond,base,a,i,n=32)))
 register(ROOT/'manifests/probability_candidate_registry.json',dict(candidates=[r[0] for r in registry],calibration='2022 CRPS scale factors .75/1/1.5/2, no2023 selection',quantiles='Three train-only HGB quantiles; Gaussian-Z piecewise interpolation with linear tail extension; no Gaussian truth claim',Stage2='Suspended'))
 # Cache sampling for2022 and final examples without accessing final labels for selection.
 predictids=np.unique(np.r_[cal,dev,test]);threshold=readjson(ROOT/'manifests/thresholds.json');allrows=[];calibrations={}
 for mid,sampler in registry:
  samplepath=ROOT/'predictions'/(mid+'_draws.npz');calpath=ROOT/'checkpoints'/(mid+'_calibration.json');tick=time.perf_counter()
  if samplepath.exists():raw=np.load(samplepath)['samples']
  else:raw=np.array([sampler(int(i)) for i in predictids],dtype='float32');np.savez_compressed(samplepath,samples=raw,indices=predictids)
  sampling=time.perf_counter()-tick;tick=time.perf_counter();lookup={i:j for j,i in enumerate(predictids)};best=None
  for factor in [.75,1,1.5,2]:
   vals=[]
   for i in cal:
    ss=raw[lookup[i]];mean=ss.mean(0);vals.append(dict(volcano=m.iloc[i].volcano,crps=crps(mean+(ss-mean)*factor,a['truth'][i])))
   value=macro(pd.DataFrame(vals),['crps']).crps
   if best is None or value<best[0]:best=(value,factor)
  register(calpath,dict(scale=best[1],criterion='2022 CRPS only',calibration_cases=len(cal),hyperparameters_selected_before_2023=True));calibrations[mid]=best[1];timing.append(dict(model_id=mid,sampling_seconds=sampling,calibration_seconds=time.perf_counter()-tick,cases=len(predictids),draws=32,device=device if '_seed' in mid else 'CPU'));csv(ROOT/'diagnostics/probability_runtime.csv',timing);print('V9 DISTRIBUTION SAMPLED/CALIBRATED',mid,flush=True)
 register(ROOT/'manifests/probability_lock.json',dict(candidates=list(calibrations),scales=calibrations,all_hyperparameters_from_training_CV=True,no_final_2023_outcomes_used=True));csv(ROOT/'diagnostics/probability_runtime.csv',timing);finish('probability');print('V9 PROBABILITY CANDIDATES LOCKED',flush=True)
if __name__=='__main__':run()
