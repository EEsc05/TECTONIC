"""New populations and preregistered split versions; no prior assignment is rewritten."""
from .common import *
from ..data.dataset import load as old_load
import json

def source_inventory():
 j=REPO/'volcanic_experiment/japan_expansion';inventory=readjson(j/'data/manifests/japan_dataset_inventory.json');rows=[]
 for r in inventory:
  if r['system']=='fuji':continue
  rows.append(dict(source_id=r['source_id'],system=r['system'],frame=r['frame'],kind='web',path=str(j/r['processed_file']),raw_path=str(j/r['raw_file']),spacing_y_km=r['spacing_y_m']/1000,spacing_x_km=r['spacing_x_m']/1000,reference=r['reference_system']+' '+r['reference_metadata'],units='mm LOS',satellite='Sentinel-1',orbit=r['orbit'],processing=r['processing'],provenance=r['url'],source_sha256=r['source_sha256'],previous_status='Earlier-than2022unused temporal slices of an inherited retrospective source;2022+quarantined',geographic_holdout=False))
 # Reuse Azumayama's already cached pre2022observations; previous future holdout dates remain quarantined.
 for r in readjson(BASE/'data/manifests/unified_dataset_v2.json'):
  if r['system']!='azumayama':continue
  raw=BASE/'data/cache/raw'/(r['source_id']+'_web_x100.json')
  rows.append(dict(source_id=r['source_id'],system=r['system'],frame=r['frame'],kind='web',path=r['path'],raw_path=str(raw),spacing_y_km=r['spacing_y_km'],spacing_x_km=r['spacing_x_km'],reference=r['reference'],units='mm LOS',satellite='Sentinel-1',orbit='descending',processing=r['processing'],provenance=r['source_url'],source_sha256=r.get('sha256',digest(raw)),previous_status='InheritedAzumapre2022contextonly; previously evaluated2025+notused',geographic_holdout=False))
 # New sites are a NEW evaluation version. No old held-out site is promoted to training.
 for site,frame in [('asamayama','112A_05454_131213'),('unzendake','054A_05688_051313')]:
  raw=ROOT/'raw/insar'/f'{site}_{frame}_web_x100.json'
  if not raw.exists():continue
  d=readjson(raw);a=numeric(d['data_raw']);c=a.shape[1]//2;a=a[:,c-16:c+16,c-16:c+16];t=np.array([day(x) for x in d['dates']]);lat=np.array(d['y']);lon=np.array(d['x']);sp=spacing(float(lat.mean()),float(np.median(np.diff(lat))),float(np.median(np.diff(lon))));cache=ROOT/'processed'/f'{site}_{frame}.npz';np.savez_compressed(cache,maps=a,days=t,spacing=sp)
  rows.append(dict(source_id=f'{site}_{frame}',system=site,frame=frame,kind='web',path=str(cache),raw_path=str(raw),spacing_y_km=sp[0],spacing_x_km=sp[1],reference=str(d['refarea'])+' date'+d['dates'][0],units='mm LOS',satellite='Sentinel-1',orbit='ascending',processing='Retrospective LiCSBAS web inversion; unfiltered/GACOS-uncorrected; not historical acquisition-vintage',provenance=f'https://comet-volcanodb.org/data/disp_data/eastern_asia/{raw.name}',source_sha256=digest(raw),previous_status='Newly acquired geographic source; held out from all fitting',geographic_holdout=True))
 # Existing ALOS files are audited/reused, never downloaded twice and never pooled with500mweb representation.
 for r in readjson(REPO/'volcanic_stage2/data/manifests/dataset_inventory.json'):
  if r['volcano']!='shinmoedake' or r['crop']!=32:continue
  rows.append(dict(source_id=r['source_id'],system='kirishima',frame=r['track'],kind='alos',path=str(REPO/'volcanic_stage2'/r['processed_file']),raw_path=str(REPO/r['source_file']),spacing_y_km=r['spacing_y_m']/1000,spacing_x_km=r['spacing_x_m']/1000,reference=r['reference'],units='mm LOS',satellite=r['mission'],orbit=r['orbit'],processing=r['processing'],provenance='https://doi.org/'+r['doi'],source_sha256=r['source_sha256'],previous_status='Previously acquired ALOS; secondary sensor diagnostic only; excluded from learned web PCA',geographic_holdout=False))
 for r in rows:
  m,t=load(r);r.update(start=date(t[0]),end=date(t[-1]),acquisitions=len(t),historical_pre2022_acquisitions=int((t<day('2022-01-01')).sum()),newly_acquired=r['system'] in GEO,primary_spatial_basis_eligible=r['kind']=='web')
 csv(experiment(17)/'insar_coverage.csv',rows);writejson(ROOT/'manifests/insar_sources.json',rows);return rows

def load(r):
 p=np.load(r['path']);return (p['displacement_mm'],p['days']) if r['kind']=='alos' else (p['maps'],p['days'])

def define_protocol():
 extra=dict(stage1=dict(version='historical_eval_v3_stage1',train_end='2018-01-01',validation_start='2018-01-01',validation_end='2020-01-01',temporal_evaluation_start='2020-01-01',temporal_evaluation_end='2022-01-01',context_maps=6,all_windows_entire_history_to_target_in_role=True,primary_grid='web32native500mclass',training_stride=3,evaluation_stride=3),seismic=dict(version='historical_eval_v3_seismic',train_end='2013-01-01',validation_start='2013-01-01',validation_end='2015-01-01',temporal_evaluation_start='2015-01-01',temporal_evaluation_end='2016-09-01',geographic_calibration_start='2008-01-01',geographic_calibration_end='2010-01-01',geographic_evaluation_start='2010-01-01',geographic_evaluation_end='2016-09-01',common_history_days=365,full_interval_purge=True,training_stride=7,evaluation_stride=61),matched=dict(version='historical_eval_v3_matched',train_end='2016-01-01',validation_start='2016-01-01',validation_end='2016-05-01',temporal_evaluation_start='2016-05-01',temporal_evaluation_end='2016-09-01',historical_seismic_days=30,insar_maps=6,population='frozenlocalStage1atactualSARorigins;targetdailycountavailabilitydoesnotrequirefutureSAR',caveat='Short independently versioned matched pilot, not the main365day historical count population'),gas=dict(version='historical_eval_v3_gas',train_end='2012-01-01',validation_start='2012-01-01',validation_end='2015-01-01',temporal_evaluation_start='2015-01-01',temporal_evaluation_end='2022-01-01',history_observations=6),ash=dict(version='historical_eval_v3_ash',train_end='2014-01-01',validation_start='2014-01-01',validation_end='2017-01-01',temporal_evaluation_start='2017-01-01',temporal_evaluation_end='2022-01-01',lookback_days=30),selection='Validation system-macro MAE for point/count models; CRPS primary for probabilistic maps subject to coverage/extreme reconstruction gates. No new temporal/geographic errors are used for selection.',neural_gating='Require>=50nonoverlapping train history-target intervals,>=4training volcanoes,<=25%validation extreme reconstruction loss,>=5%validation CRPS improvement versus Gaussian field baseline,90%coverage in[.80,.98]. If failed, conditionalVAE/flow/diffusion remain untrained.',cross_stage_gating='New globally trained probabilistic Stage1 cannot enter earlier Stage2origins in-sample. Need earlier-only crossfits and independent temporal support; otherwise skip D/E/generative propagation, retain frozen local pathway.',ancestry='Old_v2andstage2_expansion_v1remainimmutable. Historical primary daily totals2008–2016havea distinct definition fromrecentplotcounts; no numerical headline comparisons across versions.')
 return register_protocol(extra)

def role(start,end,site,part):
 protocol=readjson(ROOT/'splits/preregistered_protocol.json')[part]
 if end>=day('2022-01-01'):return None
 if site in GEO:
  if part=='seismic':
   if start>=day('2008-01-01') and end<day('2010-01-01'):return 'geographic_calibration_only'
   return 'unseen_volcano' if start>=day('2010-01-01') and end<day('2016-09-01') else None
  return 'unseen_volcano' if part in ['stage1','matched'] else None
 if end<day(protocol['train_end']):return 'train'
 if start>=day(protocol['validation_start']) and end<day(protocol['validation_end']):return 'validation'
 if start>=day(protocol['temporal_evaluation_start']) and end<day(protocol['temporal_evaluation_end']):return 'temporal_evaluation'
 return None

def windows_stage1(sources):
 define_protocol();rows=[];exclusions=[];amendment=readjson(ROOT/'splits/source_quarantine_amendment_q1.json') if (ROOT/'splits/source_quarantine_amendment_q1.json').exists() else {};blocked=set(amendment.get('quarantined_sources',[]))
 for r in sources:
  maps,t=load(r)
  if r['source_id'] in blocked:
   exclusions.append(dict(source_id=r['source_id'],reason='q1protected_frame_or_geographic_orbit_overlap'));continue
  for i in range(5,len(t)-1,3):
   for j in range(i+1,len(t)):
    h=int(t[j]-t[i])
    if h>60:break
    rr=role(int(t[i-5]),int(t[j]),r['system'],'stage1')
    if not rr or r['kind']=='alos':
     exclusions.append(dict(source_id=r['source_id'],origin_day=int(t[i]),target_day=int(t[j]),reason='ALOSsecondarysensoronly' if r['kind']=='alos' else 'outside_newrole_or2022plusquarantine'));continue
    seen=np.isfinite(maps[i-5:i+1]).any(0);valid=seen&np.isfinite(maps[j])
    if valid.sum()<64:continue
    rows.append(dict(example_id=f"{r['source_id']}:{i}:{j}",dataset_version=VERSION,evaluation_version=amendment.get('evaluation_version','historical_eval_v3_stage1'),source_id=r['source_id'],volcano=r['system'],frame=r['frame'],origin_index=i,target_index=j,history_start_day=int(t[i-5]),origin_day=int(t[i]),target_day=int(t[j]),horizon_days=h,role=rr,valid_pixels=int(valid.sum())))
 f=csv(ROOT/'splits/stage1_v3.csv',rows);csv(ROOT/'splits/stage1_quarantine.csv',exclusions);return f
if __name__=='__main__':
 sources=source_inventory();define_protocol();f=windows_stage1(sources);print(f.groupby('role').size().to_dict())
