"""V6versioned globalexperiment; inherited artifacts remain read-only."""
from ..common import *
BASE=ROOT;PROJECT=BASE/'experiments/v6_global_volcanic_forecasting';VERSION=os.environ.get('V6_RUN_VERSION','v6_global_v2');ROOT=PROJECT if VERSION=='v6_global_v1' else PROJECT/'runs'/VERSION;SEED=20261108
DATASET='global_comet_LOS_v6_1' if VERSION=='v6_global_v1' else 'global_comet_LOS_v6_2'
SPLIT='global_target_isolated_v6_1' if VERSION=='v6_global_v1' else 'global_target_isolated_v6_2'
for part in ['acquisition','preprocessing','deterministic','generative','horizons','scaling','geographic_transfer','stage2_integration','manifests','logs','reports','checkpoints','tests']:(ROOT/part).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
def register(p,obj):
 if p.exists():assert readjson(p)==obj,f'Immutableversion differs:{p}'
 else:writejson(p,obj)
def protect():
 answer={}
 for folder,name in [('historical_v3','frozen_reference_hashes.json'),('historical_v3','primary_numerical_results_hashes.json'),('research_v4','numerical_results_hashes.json'),('research_v5','numerical_results_hashes.json')]:
  h=readjson(BASE/folder/'manifests'/name);bad=[k for k,v in h.items() if digest(REPO/k)!=v];assert not bad,bad[:3];answer[folder+'/'+name]=len(h)
 return answer

def table(f):
 def fmt(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in r)+' |' for r in f.itertuples(index=False,name=None))+'\n'

def load_source(r):
 a=np.load(r['processed_path']);return a['maps'],a['days'],a['spacing']
