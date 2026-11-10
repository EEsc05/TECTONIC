"""Locked final development/old-assessment evaluation; no hyperparameter selection here."""
from .common import *
from .probability import score as probability_score
from ..research_v6.data import score
from scipy.stats import norm
import time

def point_score(p,y,last,sp,threshold):
 v=np.isfinite(y)&np.isfinite(last);assert np.isfinite(p[v]).all(),'Missing forecast inside common evaluation mask';z=score(y,p,last,sp,threshold['pixel_extreme_mm']);actual=abs(y-last)>=threshold['pixel_extreme_mm'];forecast=abs(p-last)>=threshold['pixel_extreme_mm'];union=((actual|forecast)&v).sum();z['localization_IoU']=float(((actual&forecast)&v).sum()/union) if union else np.nan;z['future_coverage']=v.sum()/max(np.isfinite(last).sum(),1);z['map_event']=float(np.nanmean(abs(y-last))>=threshold['map_extreme_mm']) if z['future_coverage']>=.9 else np.nan;return z

def run():
 assert done('probability') and done('importance');m,a,sources=data();tr,cal,dev,test=partitions(m);ids=np.r_[dev,test];threshold=readjson(ROOT/'manifests/thresholds.json');pred={};old=np.load(V6/'deterministic/point_predictions.npz')
 refs=['persistence','ema_half30','ema_half120','quality_v2_h6_half30_w0','kalman_locallevel','kalman_localtrend','lr_h6_r3_rob1_linear','context_ridge','context_lightgbm','CNN','four_branch_equal','four_branch_constrained']
 for mid in refs:pred['ref_'+mid]=old[mid]
 v7=np.load(V7/'predictions/new_point.npz');pred['ref_V7_lowrank']=v7['lr_rob_r3_ema'];pred['ref_robust_temporal']=v7['robust_trend'];pred['ref_V7_hybrid']=np.load(V7/'predictions/hybrid.npz')['v7_four_ridge_gate'];p8=np.load(V8/'predictions/specialists.npz');pred['ref_V8_spatial']=p8['spatial_mask_r3_km1'];pred['ref_V8_hybrid']=p8['specialist_four_equal']
 ts=readjson(ROOT/'manifests/temporal_selection.json')
 for mid in ts['selected']:pred['temporal_'+mid]=np.load(ROOT/'predictions'/('temporal_'+mid+'.npz'))['pred']
 for file in ['conventional','spatial','modular','cnn']:
  p=np.load(ROOT/'predictions'/(file+'.npz'));pred.update({k:p[k] for k in p.files})
 checkpoints={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'checkpoints').glob('*') if p.is_file()};selectors={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'manifests').glob('*selection.json')};register(ROOT/'manifests/finalist_lock.json',dict(point_candidates=list(pred),probability_candidates=readjson(ROOT/'manifests/probability_lock.json'),checkpoints=checkpoints,selectors=selectors,locked_before_final_2023_scoring=True,no_assessment_based_tuning=True,Stage2='Suspended'))
 rows=[];timing=[]
 for mid,pp in pred.items():
  tick=time.perf_counter()
  for i in ids:
   r=m.iloc[i];rows.append(dict(model_id=mid,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,region=r.region,role=r.role,horizon_days=r.horizon_days,origin_day=r.origin_day,assessment_status='Locked2023 development assessment' if r.role=='validation' else 'Previously inspected exploratory assessment',**point_score(pp[i],a['truth'][i],a['last'][i],a['spacing'][i],threshold)))
  timing.append(dict(model_id=mid,scoring_seconds=time.perf_counter()-tick,cases=len(ids),device='CPU'))
 f=csv(ROOT/'predictions/point_metrics.csv',rows);pointmacro=f.groupby(['role','model_id','volcano'])[['mae','rmse','extreme_mae','gradient_mae','localization_IoU']].mean().groupby(['role','model_id']).mean().reset_index();csv(ROOT/'diagnostics/point_macro.csv',pointmacro);csv(ROOT/'hybrid_combination_results.csv',f[f.model_id.str.startswith(('pair_','three_','four_'))]);csv(ROOT/'hybrid_ablation_results.csv',f[f.model_id.str.startswith('ablate_')]);csv(ROOT/'baseline_registry.csv',[dict(model_id=k,reference=k.startswith('ref_'),observations='Identical six historical acquisitions and actual requested lag',units='signed LOSmm',evaluation_masks='Same finite last-history support and observed future pixels',role='Stage1 deformation only') for k in pred]);prows=[];plock=readjson(ROOT/'manifests/probability_lock.json');want=set(ids.tolist())
 for mid in plock['candidates']:
  cache=ROOT/'predictions'/(mid+'_scored.csv')
  if cache.exists():prows.extend(pd.read_csv(cache).to_dict('records'));continue
  p=np.load(ROOT/'predictions'/(mid+'_draws.npz'));draws=p['samples'];ii=p['indices'];tick=time.perf_counter();rr=[];factor=plock['scales'][mid]
  for j,i in enumerate(ii):
   if int(i) not in want:continue
   r=m.iloc[i];raw=draws[j];v=np.isfinite(a['truth'][i])&np.isfinite(a['last'][i]);assert np.isfinite(raw[:,v]).all();mean=raw.mean(0);s0=probability_score(raw,a,int(i),threshold)
   for method in ['raw','scale_CRPS']:
    ss=raw if method=='raw' else mean+(raw-mean)*factor;sc=s0 if method=='raw' or factor==1 else probability_score(ss,a,int(i),threshold);rr.append(dict(model_id=mid,calibration=method,example_id=r.example_id,source_id=r.source_id,volcano=r.volcano,region=r.region,role=r.role,horizon_days=r.horizon_days,origin_day=r.origin_day,**sc))
   if int(i)==int(test[0]):np.savez_compressed(ROOT/'predictions'/(mid+'_sample_maps.npz'),samples=raw,truth=a['truth'][i],last=a['last'][i],example_id=r.example_id)
  csv(cache,rr);prows.extend(rr);timing.append(dict(model_id=mid,scoring_seconds=time.perf_counter()-tick,cases=len(want),draws=32,device='CPU'));csv(ROOT/'diagnostics/assessment_runtime.csv',timing);print('V9 LOCKED PROBABILITY ASSESSED',mid,flush=True)
 p=csv(ROOT/'predictions/probability_metrics.csv',prows);pm=p.groupby(['role','model_id','calibration','volcano'])[['mae','rmse','crps','coverage50','coverage80','coverage90','coverage95','interval_width90','energy_score','variogram_score','extreme_mae','event_brier']].mean().groupby(['role','model_id','calibration']).mean().reset_index();csv(ROOT/'gaussian_probabilistic_results.csv',pm[~pm.model_id.str.contains('_seed')]);csv(ROOT/'calibration_results.csv',pm);csv(ROOT/'geographic_transfer.csv',pd.concat([pointmacro[pointmacro.role=='geographic_evaluation'],pm[pm.role=='geographic_evaluation']],ignore_index=True));h=f.groupby(['role','model_id','horizon_days']).agg(mae=('mae','mean'),rmse=('rmse','mean'),cases=('example_id','nunique'),volcanoes=('volcano','nunique')).reset_index();h['kind']='point';hp=p.groupby(['role','model_id','calibration','horizon_days']).agg(crps=('crps','mean'),coverage90=('coverage90','mean'),width90=('interval_width90','mean'),cases=('example_id','nunique'),volcanoes=('volcano','nunique')).reset_index();hp['kind']='probability';csv(ROOT/'horizon_results.csv',pd.concat([h,hp],ignore_index=True));seed=p[p.model_id.str.contains('_seed')].copy();seed['configuration']=seed.model_id.str.replace(r'_seed\d+$','',regex=True);s=seed.groupby(['role','configuration','model_id','calibration','volcano'])[['crps','coverage90','event_brier']].mean().groupby(['role','configuration','model_id','calibration']).mean().reset_index();csv(ROOT/'diagnostics/seed_metrics.csv',s);csv(ROOT/'diagnostics/seed_stability.csv',s.groupby(['role','configuration','calibration']).agg(training_seeds=('model_id','nunique'),CRPS_mean=('crps','mean'),CRPS_std=('crps','std'),coverage90=('coverage90','mean')).reset_index());csv(ROOT/'diagnostics/assessment_runtime.csv',timing)
 # Strict six-map common-history temporal blocks, shared frame grouping, vector-valued bootstrap.
 meta=m.iloc[ids].copy();meta['frame']=meta.source_id.map({k:v['frame'] for k,v in sources.items()});blocks=[]
 for (role,frame),g in meta.groupby(['role','frame']):
  z=g.groupby(['source_id','origin_day']).agg(start=('history_start_day','min'),end=('target_day','max')).reset_index().sort_values('end');last=-np.inf
  for r in z.itertuples():
   if r.start>last:blocks.append(dict(role=role,frame=frame,source_id=r.source_id,origin_day=r.origin_day,start=r.start,end=r.end));last=r.end
 blocks=csv(ROOT/'diagnostics/independent_blocks.csv',blocks);cis=[];rng=np.random.default_rng(SEED)
 for family,table,metric,baseline in [('point',f,'mae','ref_quality_v2_h6_half30_w0'),('probability',p.assign(model_id=p.model_id+'::'+p.calibration),'crps','EMA_Gaussian::scale_CRPS')]:
  for role,g in table.groupby('role'):
   pivot=g.pivot(index=['source_id','origin_day','example_id'],columns='model_id',values=metric);delta=pivot.sub(pivot[baseline],axis=0);case=delta.reset_index();case['frame']=case.source_id.map({k:v['frame'] for k,v in sources.items()});orig=case.groupby(['frame','source_id','origin_day'])[delta.columns].mean().reset_index();chosen=orig.merge(blocks[blocks.role==role][['frame','source_id','origin_day']],on=['frame','source_id','origin_day']);clusters=[q[delta.columns].to_numpy() for _,q in chosen.groupby('frame')];M=len(delta.columns);B=2000;K=len(clusters);outer=rng.integers(K,size=(B,K));draw=np.zeros((B,K,M))
   for k,values in enumerate(clusters):mask=outer==k;idx=rng.integers(len(values),size=(mask.sum(),len(values)));draw[mask]=values[idx].mean(1)
   boot=draw.mean(1);center=np.mean([q.mean(0) for q in clusters],0);se=boot.std(0);standard=np.max(abs(boot-center)/np.maximum(se,1e-9),axis=1);critical=np.quantile(standard,.95);zbon=norm.ppf(1-.05/(2*max(M,1)))
   for j,mid in enumerate(delta.columns):lo,hi=np.quantile(boot[:,j],[.025,.975]);cis.append(dict(family=family,role=role,model_id=mid,reference=baseline,matched_cases=len(g[g.model_id==mid]),frame_clusters=K,nonoverlap_sixmap_blocks=len(chosen),frame_block_mean_difference=center[j],marginal_lower95=lo,marginal_upper95=hi,simultaneous_lower95=center[j]-critical*se[j],simultaneous_upper95=center[j]+critical*se[j],bonferroni_normal_lower95=center[j]-zbon*se[j],bonferroni_normal_upper95=center[j]+zbon*se[j],family_size=M,meaning='Exploratory shared-frame/block maxT and Bonferroni-normal sensitivity, not fresh confirmation; few-cluster limits apply'))
 csv(ROOT/'diagnostics/paired_cluster_uncertainty.csv',cis);finish('assessment');print('V9 LOCKED ASSESSMENT COMPLETE',flush=True)
if __name__=='__main__':run()
