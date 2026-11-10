"""Read immutable allowed sources; split complete history-to-target intervals."""
from ..common import *
JROOT=REPO/'volcanic_experiment/japan_expansion'
def construct():
 rows=[]
 for r in readjson(REPO/'volcanic_experiment/data/inventory.json'):
  if r['role']!='train':continue
  rows.append(dict(source_id='original_'+r['name'],system=r['name'],frame=r['source_file'],kind='original',path=str(REPO/'volcanic_experiment/data'/f"{r['name']}_native256.npy"),coordinates=str(REPO/'volcanic_experiment/data'/f"{r['name']}_coordinates.npz"),spacing_y_km=r['spacing_y_m']/1000,spacing_x_km=r['spacing_x_m']/1000,latitude=r['latitude'],longitude=r['longitude'],source_url=r['source'],license=r['license'],units='mm',reference=r['displacement'],processing=r['processing'],historical_role='train',asof_available=False))
 for r in readjson(JROOT/'data/manifests/japan_dataset_inventory.json'):
  if not r['eligible_for_recent_evaluation']:continue
  rows.append(dict(source_id=r['source_id'],system=r['system'],frame=r['frame'],kind='existing_japan',path=str(JROOT/r['processed_file']),spacing_y_km=r['spacing_y_m']/1000,spacing_x_km=r['spacing_x_m']/1000,latitude=r['latitude'],longitude=r['longitude'],source_url=r['url'],license=r['license'],units='mm',reference=r['reference_system']+'; '+r['reference_metadata'],processing=r['processing'],historical_role='development2022–23; validation2024; evaluated2025+ protected',asof_available=False))
 for rr in readjson(ROOT/'data/manifests/new_japanese_downloads.json'):
  if rr['status']=='unavailable':continue
  p=ROOT/'data/cache/raw'/rr['name'];d=readjson(p);site=rr['name'].split('_')[0];fid='_'.join(rr['name'].split('_')[1:4]);lat=np.array(d['y']);lon=np.array(d['x']);sp=spacing(float(np.mean(lat)),float(np.median(np.diff(lat))),float(np.median(np.diff(lon))))
  out=ROOT/'data/processed'/f'{site}_{fid}_32.npz';a=numeric(d['data_raw']);cy,cx=a.shape[1]//2,a.shape[2]//2;a=a[:,cy-16:cy+16,cx-16:cx+16];days=np.array([day(t) for t in d['dates']]);np.savez_compressed(out,maps=a,days=days,spacing=sp,lat=lat[cy-16:cy+16],lon=lon[cx-16:cx+16])
  rows.append(dict(source_id=f'{site}_{fid}',system=site,frame=fid,kind='new_japan',path=str(out),spacing_y_km=sp[0],spacing_x_km=sp[1],latitude=float(np.mean(lat)),longitude=float(np.mean(lon)),source_url=rr['url'],license='Portal images CC BY-NC4.0; numerical export terms not explicit',units='mm',reference=str(d['refarea'])+'; epoch '+d['dates'][0],processing='COMET LiCSBAS raw web export; retrospective full-record inversion; no GACOS correction',historical_role='new site, reserved by system and chronology',asof_available=False,sha256=rr['sha256'],producer_timestamp=d['timestamp']))
 manifests=[];pairs=[]
 for r in rows:
  a,t=load(r);r.update(acquisitions=len(t),start=date(t[0]),end=date(t[-1]),height=32,width=32,width_km=32*r['spacing_x_km'],height_km=32*r['spacing_y_km'],area_km2=1024*r['spacing_y_km']*r['spacing_x_km'],missing_fraction=float((~np.isfinite(a)).mean()),median_gap_days=float(np.median(np.diff(t))),max_gap_days=int(np.max(np.diff(t))))
  for i in range(17,len(t)-1):
   for j in range(i+1,len(t)):
    h=int(t[j]-t[i])
    if h>60:break
    role=partition(r,t[i-17],t[j])
    if role is None:continue
    valid=np.isfinite(a[i-17:i+1]).any(0)&np.isfinite(a[j])
    if valid.sum()<32:continue
    pairs.append(dict(pair_id=f"{r['source_id']}:{i}:{j}",source_id=r['source_id'],system=r['system'],frame=r['frame'],origin_index=i,target_index=j,history_start_day=int(t[i-17]),origin_day=int(t[i]),target_day=int(t[j]),horizon_days=h,horizon_bin=hbin(h),role=role,valid_pixels=int(valid.sum()),cadence_days=float(np.median(np.diff(t[i-17:i+1])))))
  manifests.append(r)
 csv(ROOT/'data/manifests/unified_dataset_v2.csv',manifests);writejson(ROOT/'data/manifests/unified_dataset_v2.json',manifests);csv(ROOT/'data/manifests/source_provenance.csv',manifests)
 f=csv(ROOT/'data/splits/stage1_split_manifest.csv',pairs);csv(ROOT/'data/manifests/dataset_counts.csv',f.groupby(['role','system','horizon_bin']).size().rename('actual_target_pairs').reset_index())
 # Full interval check across temporal roles, source and shared-frame systems.
 checks=[]
 for frame,g in f.groupby('frame'):
  spans=g.groupby('role').agg(start=('history_start_day','min'),end=('target_day','max'))
  for x in spans.index:
   for y in spans.index:
    if x>=y:continue
    overlap=max(spans.loc[x,'start'],spans.loc[y,'start'])<=min(spans.loc[x,'end'],spans.loc[y,'end']);checks.append(dict(frame=frame,role_a=x,role_b=y,overlap=bool(overlap)))
 csv(ROOT/'data/splits/interval_checks.csv',checks);assert not any(r['overlap'] for r in checks),'Cross-partition frame/time overlap'
 return manifests,f

def load(r,size=32):
 if r['kind']=='original':
  a=np.load(r['path'],mmap_mode='r');c=a.shape[1]//2;a=np.array(a[:,c-size//2:c+size//2,c-size//2:c+size//2]);a[a==0]=np.nan;t=np.load(r['coordinates'])['days']
 else:
  d=np.load(r['path']);a=d['maps'];t=d['days']
  if size!=32:raise ValueError('Alternate grid requires native source export')
 return a,np.array(t,dtype=int)

def partition(r,start,end):
 if r['kind']=='original':
  if end<day('2018-01-01'):return 'train'
  if start>=day('2018-01-01'):return 'validation'
 elif r['kind']=='existing_japan':
  if start>=day('2022-01-01') and end<day('2024-01-01'):return 'train'
  if start>=day('2024-02-01') and end<day('2025-01-01'):return 'validation'
 elif r['system']=='azumayama':
  if start>=day('2022-01-01') and end<day('2025-01-01'):return 'validation'
  if start>=day('2025-01-01') and end<day('2026-11-06'):return 'internal_holdout'
 elif start>=day('2022-01-01') and end<day('2026-11-06'):return 'unseen_volcano'
 return None
