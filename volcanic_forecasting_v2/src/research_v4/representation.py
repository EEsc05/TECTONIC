"""Train-only localized/robust spatial representations and conditioning diagnostics."""
from .common import *
from scipy.fft import dctn,idctn
from sklearn.decomposition import PCA,MiniBatchDictionaryLearning
from sklearn.linear_model import Ridge
from sklearn.preprocessing import RobustScaler
from ..historical_v3.stage1 import score,gradient_error
import joblib,time

class SpatialBasis:
 def __init__(self,kind='block_pca',rank=8):self.kind=kind;self.rank=rank
 def fit(self,images):
  a=np.nan_to_num(np.asarray(images,float));self.mean=a.mean(0);self.models=[]
  if self.kind in ['block_pca','robust_block_pca']:
   if self.kind=='robust_block_pca':
    center=np.median(a,axis=0);scale=np.median(abs(a-center),axis=0)*1.4826+.1;a=np.clip(a,center-4*scale,center+4*scale)
   for yy in range(0,32,8):
    for xx in range(0,32,8):self.models.append(PCA(n_components=self.rank,random_state=SEED).fit(a[:,yy:yy+8,xx:xx+8].reshape(len(a),64)))
  elif self.kind in ['pca','robust_pca']:
   if self.kind=='robust_pca':
    center=np.median(a,axis=0);scale=np.median(abs(a-center),axis=0)*1.4826+.1;a=np.clip(a,center-4*scale,center+4*scale)
   self.model=PCA(n_components=min(self.rank,len(a)-2),svd_solver='randomized',random_state=SEED).fit(a.reshape(len(a),-1))
  elif self.kind=='sparse_dictionary':self.model=MiniBatchDictionaryLearning(n_components=self.rank,batch_size=64,max_iter=40,random_state=SEED,transform_algorithm='lasso_lars',transform_alpha=.1).fit(a.reshape(len(a),-1))
  return self
 def encode(self,a):
  a=np.nan_to_num(np.asarray(a,float));n=len(a)
  if self.kind in ['block_pca','robust_block_pca']:return np.concatenate([m.transform(a[:,yy:yy+8,xx:xx+8].reshape(n,64)) for m,(yy,xx) in zip(self.models,[(y,x) for y in range(0,32,8) for x in range(0,32,8)])],axis=1)
  if self.kind in ['pca','robust_pca','sparse_dictionary']:return self.model.transform(a.reshape(n,-1))
  if self.kind=='dct':return dctn(a-self.mean,axes=(-2,-1),norm='ortho')[:,:self.rank,:self.rank].reshape(n,-1)
  if self.kind=='haar':
   # Full two-level Haar analysis; retain coarse and spatially localized details at a fixed rank.
   c=a-self.mean;out=[]
   for _ in range(2):
    aa,bb,cc,dd=c[:,::2,::2],c[:,::2,1::2],c[:,1::2,::2],c[:,1::2,1::2];out.append(np.stack([(aa-bb+cc-dd)/2,(aa+bb-cc-dd)/2,(aa-bb-cc+dd)/2],1));c=(aa+bb+cc+dd)/2
   return np.concatenate([c.reshape(n,-1),out[-1].reshape(n,-1)],axis=1) #256coefficients; omit finest detail
  if self.kind=='multiscale':
   # Coarse DCT plus local residual PCA, preserving locality rather than a global rank truncation.
   return self.sub.encode(a)
 def decode(self,z):
  n=len(z)
  if self.kind in ['block_pca','robust_block_pca']:
   out=np.zeros((n,32,32))
   for i,(m,(yy,xx)) in enumerate(zip(self.models,[(y,x) for y in range(0,32,8) for x in range(0,32,8)])):out[:,yy:yy+8,xx:xx+8]=m.inverse_transform(z[:,i*self.rank:(i+1)*self.rank]).reshape(n,8,8)
   return out
  if self.kind in ['pca','robust_pca']:return self.model.inverse_transform(z).reshape(n,32,32)
  if self.kind=='sparse_dictionary':return (z@self.model.components_).reshape(n,32,32)
  if self.kind=='dct':
   a=np.zeros((n,32,32));a[:,:self.rank,:self.rank]=z.reshape(n,self.rank,self.rank);return idctn(a,axes=(-2,-1),norm='ortho')+self.mean
  if self.kind=='haar':
   c=z[:,:64].reshape(n,8,8);details=z[:,64:].reshape(n,3,8,8)
   for level in range(2):
    hh,vv,dd=details[:,0],details[:,1],details[:,2];out=np.zeros((n,c.shape[1]*2,c.shape[2]*2));out[:,::2,::2]=(c+hh+vv+dd)/2;out[:,::2,1::2]=(c-hh+vv-dd)/2;out[:,1::2,::2]=(c+hh-vv-dd)/2;out[:,1::2,1::2]=(c-hh-vv+dd)/2;c=out;details=np.zeros((n,3,c.shape[1],c.shape[2]))
   return c+self.mean

def past_features(a,meta):
 history=a['history'];filled=np.nan_to_num(history);changes=filled[:,-1]-filled[:,0];scale=np.maximum(np.nanmedian(abs(history[:,-1]-history[:,0]),axis=(1,2)),1.);changes=np.nan_to_num((history[:,-1]-history[:,0])/scale[:,None,None]);coarse=changes.reshape(-1,8,4,8,4).mean((2,4));times=a['times'];base=np.c_[coarse.reshape(len(meta),-1),meta.horizon_days.to_numpy()/60,np.log1p(meta.horizon_days),np.log1p(np.ptp(times,axis=1)),np.log1p(np.median(np.diff(times,axis=1),axis=1)),np.log1p(scale),np.isfinite(history).mean((1,2,3))];geometry=np.c_[np.log(a['spacing']),np.ptp(times,axis=1)/100];return np.nan_to_num(base),np.nan_to_num(geometry),scale

def run():
 p=task(2);meta,a,base=stage1_data();train=np.flatnonzero(meta.role=='train');val=np.flatnonzero(meta.role=='validation');register(p/'splits/protocol.json',dict(dataset_version='historical_v3_immutable',split_version='v4_representation_dev_only',train='Inherited518trainpairs only; all basis/scalers fit train only',selection='2018–2019development validation reconstruction/extreme/gradient tradeoff',protected='No previous geographic or temporal outcome is used for new hyperparameters; old errors inspected only as labelled diagnostics',candidates=['PCA8/32/64/128','robustPCA64','blockPCA128/256','robustblockPCA128','DCT144/256','Haar256','sparse64'],neural_recommendation='Select train-fitted blockPCA128 as preregistered initial spatially-localized neural pilot; other alternatives diagnostic only',seed=SEED))
 x,geometry,scale=past_features(a,meta);res=(a['truth']-base)/scale[:,None,None];rows=[];runtime=[];models={}
 for kind,rank in [('pca',8),('pca',32),('pca',64),('pca',128),('robust_pca',64),('block_pca',8),('block_pca',16),('robust_block_pca',8),('dct',12),('dct',16),('haar',0),('sparse_dictionary',64)]:
  tick=time.perf_counter();model=SpatialBasis(kind,rank).fit(res[train]);mid=f'{kind}_{rank}';models[mid]=model;decoded=model.decode(model.encode(res[val]))*scale[val,None,None]+base[val];runtime.append(dict(model_id=mid,fit_seconds=time.perf_counter()-tick,latent_dimension=model.encode(res[train[:1]]).shape[1]));joblib.dump(model,p/'models'/f'{mid}.joblib')
  for j,i in enumerate(val):
   truth=a['truth'][i];v=np.isfinite(truth)&np.isfinite(base[i]);error=abs(decoded[j]-truth);change=abs(truth-a['last'][i]);threshold=np.nanquantile(abs(truth-base[i]),.9);ext=v&(abs(truth-base[i])>=threshold);pastmask=np.isfinite(a['history'][i]).any(0);rows.append(dict(model_id=mid,example_id=meta.iloc[i].example_id,volcano=meta.iloc[i].volcano,frame=meta.iloc[i].frame,role='development_validation',reconstruction_mae=float(error[v].mean()),extreme_reconstruction_mae=float(error[ext].mean()),extreme_fraction=float(error[ext].mean()/max(abs(truth-base[i])[ext].mean(),1e-6)),gradient_error=gradient_error(truth,decoded[j],v,a['spacing'][i]),uplift_mae=float(error[v&(truth-a['last'][i]>0)].mean()) if (v&(truth-a['last'][i]>0)).any() else np.nan,subsidence_mae=float(error[v&(truth-a['last'][i]<0)].mean()) if (v&(truth-a['last'][i]<0)).any() else np.nan,valid_fraction=float(v.mean())))
 f=csv(p/'reconstruction_diagnostics.csv',rows);csv(p/'representation_comparison.csv',f.groupby('model_id').mean(numeric_only=True).reset_index());csv(p/'per_volcano_reconstruction.csv',f.groupby(['model_id','volcano']).mean(numeric_only=True).reset_index());csv(p/'per_track_reconstruction.csv',f.groupby(['model_id','frame']).mean(numeric_only=True).reset_index());csv(p/'training_times.csv',runtime)
 # Lightweight transferable heads; clipped standardized covariates prevent unconstrained metadata extrapolation.
 forecast=[];timing=[]
 for mid in ['block_pca_8','robust_block_pca_8','dct_12']:
  basis=models[mid];z=basis.encode(res);targets=z[train]
  for geo in [False,True]:
   xx=np.c_[x,geometry] if geo else x;scaler=RobustScaler().fit(xx[train]);xt=np.clip(scaler.transform(xx),-6,6);tick=time.perf_counter();reg=Ridge(alpha=100.).fit(xt[train],targets);pred=basis.decode(reg.predict(xt[val]))*scale[val,None,None]+base[val];model_id=mid+('_geometry' if geo else '_no_geometry');joblib.dump(dict(basis=basis,scaler=scaler,head=reg,geometry=geo),p/'models'/f'{model_id}_forecast.joblib');timing.append(dict(model_id=model_id,fit_seconds=time.perf_counter()-tick))
   for j,i in enumerate(val):forecast.append(dict(model_id=model_id,example_id=meta.iloc[i].example_id,volcano=meta.iloc[i].volcano,role='validation',**score(a['truth'][i],pred[j],a['last'][i],a['spacing'][i])))
 ff=csv(p/'forecast_metrics_by_example.csv',forecast);csv(p/'forecast_comparison.csv',ff.groupby(['model_id','volcano']).mean(numeric_only=True).groupby('model_id').mean().reset_index());csv(p/'forecast_training_times.csv',timing)
 # Diagnose already-published failures without reselecting models on them.
 prior=np.load(PARENT/'processed/stage1_point_predictions.npz');summ=[];center=np.median(x[train],axis=0);mad=np.median(abs(x[train]-center),axis=0)+.01;dist=abs(x-center)/mad
 for role,g in meta.groupby('role'):
  ids=g.index.to_numpy();summ.append(dict(role=role,cases=len(ids),past_feature_99pct_robust_z=float(np.quantile(dist[ids],.99)),amplitude_median=float(np.median(scale[ids])),pixel_spacing_min=float(a['spacing'][ids].min()),pixel_spacing_max=float(a['spacing'][ids].max()),missing_fraction=float((~np.isfinite(a['history'][ids])).mean()),label='Previously inspected diagnostic only' if role in ['unseen_volcano','temporal_evaluation'] else 'Development'))
 csv(p/'feature_distribution_shift.csv',summ)
 # Leave-one-development-volcano-out forecasts: basis and head both refit without that volcano.
 loo=[]
 for held in sorted(meta.iloc[train].volcano.unique()):
  fitidx=np.array([i for i in train if meta.iloc[i].volcano!=held]);testidx=np.array([i for i in train if meta.iloc[i].volcano==held]);basis=SpatialBasis('block_pca',8).fit(res[fitidx]);scaler=RobustScaler().fit(np.c_[x,geometry][fitidx]);xt=np.clip(scaler.transform(np.c_[x,geometry]),-6,6);reg=Ridge(alpha=100).fit(xt[fitidx],basis.encode(res[fitidx]));pred=basis.decode(reg.predict(xt[testidx]))*scale[testidx,None,None]+base[testidx]
  for j,i in enumerate(testidx):loo.append(dict(volcano=held,example_id=meta.iloc[i].example_id,mae=float(np.nanmean(abs(pred[j]-a['truth'][i]))),local_control_mae=float(np.nanmean(abs(base[i]-a['truth'][i]))),basis_fit_without_site=True))
 csv(p/'leave_one_development_volcano_out.csv',loo);writejson(p/'recommendation.json',dict(representation='block_pca_8',latent_dimensions=128,uses_past_amplitude_normalization=True,selection='Preregistered compact neural candidate, not optimized on prior geographic errors',fitted_basis=str(p/'models/block_pca_8.joblib'),neural_autoencoder='Implemented separately by Task3CVAE rather than conflating representation and forecasting losses',physics='No co-registered3D/GNSS targets; no justified universal elastic basis',prior_holdouts='Diagnostic only; never tune these heads on them'))
 # Display development reconstruction only.
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 i=val[len(val)//2];model=models['block_pca_8'];dec=model.decode(model.encode(res[i:i+1]))[0]*scale[i]+base[i];fig,axs=plt.subplots(1,3,figsize=(10,3))
 for ax,img,title in zip(axs,[a['truth'][i]-base[i],dec-base[i],dec-a['truth'][i]],['Observed validation residual','Localized reconstruction','Reconstruction error']):im=ax.imshow(img,cmap='RdBu_r',vmin=-30,vmax=30);ax.set_title(title);ax.axis('off')
 fig.colorbar(im,ax=axs.ravel().tolist(),label='mm LOS');fig.savefig(p/'figures/reconstruction.png',dpi=150);plt.close(fig)
 (p/'summary.md').write_text('# Spatial representation pilot\n\nTrain-only representations and past-amplitude normalization; no protected outcome selects a model.\n\n'+mdtable(pd.read_csv(p/'representation_comparison.csv'))+'\n\n'+mdtable(pd.read_csv(p/'forecast_comparison.csv'))+'\n\nGlobal unbounded residual regression is vulnerable to reference/grid/amplitude covariate shift. These diagnostics separate compression from forecasting and use development-only leave-one-volcano-out tests; geometric metadata here includes spacing and cadence, not unavailable full LOS look vectors. Atmospheric and true-source heterogeneity are not identifiable separately. No frozen baseline is replaced.\n')
 writejson(p/'status.json',dict(status='completed',parent_unchanged=parent_check(),representations=len(models),training_pairs=len(train),validation_pairs=len(val)));print('REPRESENTATION COMPLETE',flush=True)
if __name__=='__main__':
 from .representation import run as canonical_run
 canonical_run()
