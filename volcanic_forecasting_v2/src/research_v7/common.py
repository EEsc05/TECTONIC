"""Isolated V7 diagnostics; V3–V6 artifacts are read-only parents."""
from ..common import *
BASE=ROOT;ROOT=BASE/'experiments/v7_model_diagnostics';V6=BASE/'experiments/v6_global_volcanic_forecasting/runs/v6_global_v2';VERSION='v7_diagnostics_v1';DATASET='global_comet_LOS_v6_2_immutable';SPLIT='v7_development_2022cal_2023select_v1';SEED=20261116
for sub in ['manifests','logs','checkpoints','predictions','figures','diagnostics','tests','stage2','scaling','generative']:(ROOT/sub).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
def register(p,x):
 if p.exists():assert readjson(p)==x,f'Immutable registration changed: {p}'
 else:writejson(p,x)
def protect():
 counts={}
 for folder,name in [('historical_v3','frozen_reference_hashes.json'),('historical_v3','primary_numerical_results_hashes.json'),('research_v4','numerical_results_hashes.json'),('research_v5','numerical_results_hashes.json')]:
  h=readjson(BASE/folder/'manifests'/name);assert all((REPO/k).exists() and digest(REPO/k)==v for k,v in h.items()),folder;counts[folder+'/'+name]=len(h)
 h=readjson(V6/'manifests/numerical_results_hashes.json');assert all(digest(REPO/k)==v for k,v in h.items()),'V6 numerical files';counts['V6_frozen']=len(h)
 snapshot=ROOT/'manifests/parent_snapshot.json'
 if snapshot.exists():
  hh=readjson(snapshot);assert all(digest(REPO/k)==v for k,v in hh.items()),'V6 full parent snapshot';counts['V6_snapshot']=len(hh)
 return counts

def data():
 from ..research_v6.data import examples
 m,a,s=examples();return m,a,s

def partitions(m):
 train=np.flatnonzero(m.role=='train');cal=np.flatnonzero((m.role=='validation')&(m.target_day<day('2023-01-01'))&(m.origin_index>=17));select=np.flatnonzero((m.role=='validation')&(m.origin_day>=day('2023-01-01'))&(m.origin_index>=17));test=np.flatnonzero(m.role.isin(['temporal_evaluation','geographic_evaluation'])&(m.origin_index>=17));return train,cal,select,test

def macro(f,cols):return f.groupby('volcano')[cols].mean().mean()
def markdown(f):
 if f.empty:return 'No eligible observations.\n'
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/') for x in r)+' |' for r in f.itertuples(index=False,name=None))+'\n'
def done(name):return (ROOT/'manifests'/(name+'_status.json')).exists()
def finish(name,**kwargs):writejson(ROOT/'manifests'/(name+'_status.json'),dict(status='completed',**kwargs))
