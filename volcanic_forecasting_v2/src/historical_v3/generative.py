"""Training-only residual PCA and conditional latent Gaussian/Student ensembles."""
from .common import *
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
import time,joblib

class LatentForecast:
 def features(self,m,t,h,sp):
  filled,valid=fill_past(m);last=filled[-1];change=last-filled[0];previous=filled[-2]-filled[0];state=self.pca.components_@change.ravel();rate=(state-self.pca.components_@previous.ravel())/max(float(t[-1]-t[-2]),1);noise=float(np.nanmedian(np.nanstd(m,axis=0)));base=np.array([h,np.log1p(h),t[-1]-t[0],np.median(np.diff(t)),np.log1p(max(noise,0)),np.log(sp[0]),np.log(sp[1]),valid.mean(),np.mean(abs(change))]);return np.r_[base,state,rate*h]
 def fit(self,m,t,h,sp,y,base,rank=8):
  from ..stage1.models import fill_past
  error=(y-base).reshape(len(y),-1);self.pixel_mean=np.nanmean(error,axis=0);self.pixel_mean=np.nan_to_num(self.pixel_mean);a=np.where(np.isfinite(error),error,self.pixel_mean);self.imputation_fraction=float((~np.isfinite(error)).mean());self.pca=PCA(n_components=min(rank,len(a)-2),svd_solver='randomized',random_state=SEED).fit(a);z=self.pca.transform(a);x=np.array([self.features(mm,tt,hh,ss) for mm,tt,hh,ss in zip(m,t,h,sp)]);self.scaler=StandardScaler().fit(x);xx=self.scaler.transform(x);self.model=Ridge(alpha=10).fit(xx,z);res=z-self.model.predict(xx);cov=np.cov(res,rowvar=False);self.cov=.5*cov+.5*np.diag(np.diag(cov))+np.eye(cov.shape[0])*1e-4;self.chol=np.linalg.cholesky(self.cov);reconstructed=self.pca.inverse_transform(z);residual=a-reconstructed;self.residual_std=np.maximum(np.std(residual,axis=0),.5);self.full_std=np.maximum(np.nanstd(error,axis=0),1);self.leverage_inv=np.linalg.inv(xx.T@xx+np.eye(xx.shape[1])*10);self.reference_h=float(np.mean(h));self.reference_noise=float(np.median([np.nanmedian(np.nanstd(mm,axis=0)) for mm in m]));self.train_examples=len(m);return self
 def sample(self,m,t,h,sp,base,n=32,kind='gaussian_latent',scale=1.,seed=SEED):
  if not 1<=h<=60:raise ValueError('Horizon1–60days')
  rng=np.random.default_rng(seed);x=self.scaler.transform(self.features(m,t,h,sp)[None]);latent=self.model.predict(x)[0];leverage=float((x@self.leverage_inv@x.T).item());noise=float(np.nanmedian(np.nanstd(m,axis=0)));factor=np.sqrt(h/self.reference_h)*np.clip(np.sqrt((noise+1)/(self.reference_noise+1)),.5,2.)*scale
  if kind=='independent_gaussian':samples=base[None]+rng.normal(size=(n,*base.shape))*self.full_std.reshape(base.shape)*factor
  else:
   innovation=rng.normal(size=(n,len(latent)))@self.chol.T
   if kind=='student_latent':innovation/=np.sqrt(rng.chisquare(4,size=(n,1))/2) # df4, unit variance standardized
   coeff=latent[None]+innovation*np.sqrt(1+leverage)*factor;residual=self.pca.inverse_transform(coeff).reshape(n,*base.shape);white=rng.normal(size=(n,*base.shape))
   if kind!='gaussian_unsmoothed':
    # Soft physical-scale covariance prior, not a universal elastic-source constraint.
    for j in range(n):white[j]=gaussian_filter(white[j],sigma=max(.5,1./max(sp)),mode='reflect');white[j]/=max(white[j].std(),1e-6)
   samples=base[None]+residual+white*self.residual_std.reshape(base.shape)*factor
  seen=np.isfinite(m).any(0);samples[:,~seen]=np.nan;return samples.astype('float32')

from ..stage1.models import fill_past
from .stage1 import gradient_error

def scores(samples,y,last,sp,event_threshold):
 valid=np.isfinite(y)&np.isfinite(samples).all(0)&np.isfinite(last);s=samples[:,valid];truth=y[valid];mean=s.mean(0);err=mean-truth;change=abs(truth-last[valid]);ext=change>=np.quantile(change,.9);sorted_s=np.sort(s,axis=0);n=len(s);weights=(2*np.arange(1,n+1)-n-1)[:,None];crps=np.mean(abs(s-truth))-np.mean(np.sum(weights*sorted_s,axis=0)/n**2);low,high=np.quantile(s,[.05,.95],axis=0);energy=np.linalg.norm(s-truth,axis=1).mean()/np.sqrt(len(truth))-.5*np.linalg.norm(s[:n//2]-s[n//2:2*(n//2)],axis=1).mean()/np.sqrt(len(truth));threshold=event_threshold;prob=(abs(s-last[valid])>=threshold).mean(0);outcome=change>=threshold
 # Threshold is preregistered from training changes, not chosen from the observed future map.
 pg=np.nanmean(samples,0);gy,gx=np.gradient(np.nan_to_num(y),*sp);py,px=np.gradient(np.nan_to_num(pg),*sp);rough=np.nanmean(abs(np.diff(samples,axis=1)))+np.nanmean(abs(np.diff(samples,axis=2)))
 return dict(mae=float(abs(err).mean()),rmse=float(np.sqrt(np.mean(err**2))),extreme_mae=float(abs(err[ext]).mean()),crps=float(crps),coverage90=float(((truth>=low)&(truth<=high)).mean()),interval_width90=float((high-low).mean()),energy_score=float(energy),gradient_mae=gradient_error(y,pg,valid,sp),sample_diversity_mm=float(np.std(s,axis=0).mean()),spatial_roughness_mm=float(rough),change_exceedance_brier=float(np.mean((prob-outcome)**2)))

def run():
 p=experiment(20)
 for name in ['configs','probabilistic_lowrank','conditional_vae','conditional_flow','latent_diffusion','physics_constraints','predictions','sample_maps']:(p/name).mkdir(exist_ok=True)
 meta=pd.read_csv(ROOT/'processed/stage1_examples.csv');data=np.load(ROOT/'processed/stage1_examples.npz');m,t,y,last,sp=[data[k] for k in ['history','times','truth','last','spacing']];point=np.load(ROOT/'processed/stage1_point_predictions.npz');selected=readjson(experiment(19)/'frozen_selection.json')['selected_model'];base=point[selected];train=np.flatnonzero(meta.role=='train');val=np.flatnonzero(meta.role=='validation');start=time.perf_counter();model=LatentForecast().fit(m[train],t[train],meta.iloc[train].horizon_days.to_numpy(),sp[train],y[train],base[train]);model.event_threshold=float(np.nanquantile(abs(y[train]-last[train]),.9));joblib.dump(model,p/'probabilistic_lowrank/artifact.joblib');writejson(p/'probabilistic_lowrank/training_manifest.json',dict(dataset_version=VERSION,parent_deterministic=selected,basis_fit_role='train_only',rank=len(model.pca.components_),train_pairs=len(train),imputation_fraction=model.imputation_fraction,missingness='Training pixel-mean imputation for PCA only, never treated as observed truth; all scoring masked',fit_seconds=time.perf_counter()-start))
 # Reconstruction diagnostics do not conflate encoding quality and forecasting quality.
 reconstruction=[]
 for role in ['train','validation']:
  idx=np.flatnonzero(meta.role==role);err=(y[idx]-base[idx]).reshape(len(idx),-1);dense=np.where(np.isfinite(err),err,model.pixel_mean);decoded=model.pca.inverse_transform(model.pca.transform(dense));v=np.isfinite(err);threshold=np.nanquantile(abs(err),.9,axis=1);ext=v&(abs(err)>=threshold[:,None]);reconstruction.append(dict(role=role,residual_reconstruction_mae=float(abs(decoded-err)[v].mean()),extreme_residual_reconstruction_mae=float(abs(decoded-err)[ext].mean()),extreme_residual_magnitude=float(abs(err)[ext].mean()),extreme_fraction=float(abs(decoded-err)[ext].mean()/max(abs(err)[ext].mean(),1e-6))))
 csv(p/'reconstruction_metrics.csv',reconstruction)
 # Choose scale on2018validation acquisitions; select candidate by2019validation CRPS.
 calibration=[i for i in val if meta.iloc[i].target_day<day('2019-01-01')];selection=[i for i in val if meta.iloc[i].history_start_day>=day('2019-01-01')]
 if len(calibration)<20 or len(selection)<20:raise RuntimeError('Insufficient separate calibration/selection subsets')
 kinds=['independent_gaussian','gaussian_latent','student_latent','gaussian_unsmoothed'];scale_results=[];chosen_scales={}
 for kind in kinds:
  best=None
  for scale in [.5,1.,2.]:
   values=[]
   for i in calibration:
    samples=model.sample(m[i],t[i],meta.iloc[i].horizon_days,sp[i],base[i],kind=kind,scale=scale,seed=SEED+int(meta.iloc[i].origin_day));s=scores(samples,y[i],last[i],sp[i],model.event_threshold);values.append(s)
   score=float(np.mean([v['crps'] for v in values]));cov=float(np.mean([v['coverage90'] for v in values]));scale_results.append(dict(kind=kind,scale=scale,calibration_crps=score,coverage90=cov))
   if best is None or score<best[0]:best=(score,scale)
  chosen_scales[kind]=best[1]
 csv(p/'calibration_scale_search.csv',scale_results);rows=[];selection_results=[]
 for kind in kinds:
  rr=[]
  for i in selection:
   samples=model.sample(m[i],t[i],meta.iloc[i].horizon_days,sp[i],base[i],kind=kind,scale=chosen_scales[kind],seed=SEED+int(meta.iloc[i].origin_day));rr.append(scores(samples,y[i],last[i],sp[i],model.event_threshold))
  selection_results.append(dict(model_id=kind,scale=chosen_scales[kind],crps=float(np.mean([v['crps'] for v in rr])),coverage90=float(np.mean([v['coverage90'] for v in rr]))))
 selection_df=csv(p/'validation_model_selection.csv',selection_results);best=selection_df.sort_values('crps').iloc[0];baseline=selection_df[selection_df.model_id=='independent_gaussian'].iloc[0];improvement=1-best.crps/baseline.crps
 # Count disjoint full historical intervals after collapsing all source views within a volcano.
 blocks=0
 for site,g in meta[meta.role=='train'].groupby('volcano'):
  end=-10**9
  for r in g.sort_values('origin_day').itertuples():
   if r.history_start_day>end:blocks+=1;end=r.target_day
 rec=next(v for v in reconstruction if v['role']=='validation');gate=blocks>=50 and meta.loc[meta.role=='train','volcano'].nunique()>=4 and rec['extreme_fraction']<=.25 and improvement>=.05 and .8<=best.coverage90<=.98
 writejson(p/'frozen_selection.json',dict(selected_model=best.model_id,scales=chosen_scales,evaluation_version=str(meta.iloc[0].evaluation_version),selection_role='2019validation',calibration_role='2018validation',frozen_utc=utc(),neural_advance=bool(gate),independent_training_blocks=blocks,validation_extreme_reconstruction_fraction=rec['extreme_fraction'],validation_crps_improvement_fraction=float(improvement),reason='Neural training requires all preregistered block-count, representation-extreme, CRPS and calibration conditions; no criteria relaxed.'))
 for i,r in meta.iterrows():
  if r.role=='train':continue
  for kind in kinds:
   tick=time.perf_counter();samples=model.sample(m[i],t[i],r.horizon_days,sp[i],base[i],kind=kind,scale=chosen_scales[kind],seed=SEED+int(r.origin_day));metrics=scores(samples,y[i],last[i],sp[i],model.event_threshold);rows.append(dict(model_id=kind,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,frame=r.frame,role=r.role,horizon_days=r.horizon_days,origin_day=r.origin_day,sampling_seconds=time.perf_counter()-tick,**metrics))
   if i%113==0 and kind in ['gaussian_latent','student_latent']:np.savez_compressed(p/'sample_maps'/f'{kind}_{i}.npz',samples=samples,truth=y[i],past_last=last[i],horizon_days=r.horizon_days,spatial_spacing_km=sp[i])
  if i%250==0:print('Generativecases',i,'/',len(meta),flush=True)
 for role in ['temporal_evaluation','unseen_volcano']:
  idx=np.flatnonzero(meta.role==role);err=(y[idx]-base[idx]).reshape(len(idx),-1);dense=np.where(np.isfinite(err),err,model.pixel_mean);decoded=model.pca.inverse_transform(model.pca.transform(dense));v=np.isfinite(err);threshold=np.nanquantile(abs(err),.9,axis=1);ext=v&(abs(err)>=threshold[:,None]);reconstruction.append(dict(role=role,residual_reconstruction_mae=float(abs(decoded-err)[v].mean()),extreme_residual_reconstruction_mae=float(abs(decoded-err)[ext].mean()),extreme_residual_magnitude=float(abs(err)[ext].mean()),extreme_fraction=float(abs(decoded-err)[ext].mean()/max(abs(err)[ext].mean(),1e-6))))
 csv(p/'reconstruction_metrics.csv',reconstruction)
 f=csv(p/'predictions/probabilistic_predictions.csv',rows);metrics=f.groupby(['role','model_id','volcano'])[['mae','rmse','extreme_mae','crps','coverage90','interval_width90','energy_score','gradient_mae','sample_diversity_mm','spatial_roughness_mm']].mean().groupby(['role','model_id']).mean().reset_index();csv(p/'probabilistic_metrics.csv',metrics);csv(p/'point_metrics.csv',metrics[['role','model_id','mae','rmse','extreme_mae','gradient_mae']]);csv(p/'horizon_calibration.csv',f.groupby(['role','model_id','horizon_days'])[['crps','coverage90','interval_width90']].mean().reset_index());csv(p/'volcano_calibration.csv',f.groupby(['role','model_id','volcano'])[['crps','coverage90','interval_width90']].mean().reset_index());writejson(p/'runtime.json',dict(wall_seconds=time.perf_counter()-start,samples_per_forecast=32,parameters='PCA8 plus linear latent dynamics and residual covariance; no large image network',neural_advance=bool(gate)))
 for name in ['conditional_vae','conditional_flow','latent_diffusion']:
  writejson(p/name/'status.json',dict(status='pending_bounded_pilot' if gate else 'not_trained_preregistered_gate_failed',reason='See frozen_selection.json. Software/literature feasibility does not establish data support.',no_synthetic_results=True))
 print('GENERATIVE COMPLETE; advance=',gate,flush=True)
if __name__=='__main__':
 from .generative import run as canonical_run
 canonical_run()
