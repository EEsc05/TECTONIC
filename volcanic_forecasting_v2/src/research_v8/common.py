"""V8 isolated specialization, with immutable V3–V7 parents."""
from ..common import *
BASE=ROOT;ROOT=BASE/'experiments/v8_specialized_forecasting';V7=BASE/'experiments/v7_model_diagnostics';V6=BASE/'experiments/v6_global_volcanic_forecasting/runs/v6_global_v2';SEED=20261126;VERSION='v8_specialization_v1';DATASET='global_comet_LOS_v6_2_immutable';SPLIT='v8_2022cal_2023select_v1'
for d in ['manifests','logs','checkpoints','predictions','diagnostics','figures','evidence','external','tests','scaling']:(ROOT/d).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
def register(p,x):
 if p.exists():assert readjson(p)==x,f'Immutable registration differs: {p}'
 else:writejson(p,x)
def done(phase):return (ROOT/'manifests'/(phase+'_status.json')).exists()
def finish(phase,**kwargs):writejson(ROOT/'manifests'/(phase+'_status.json'),dict(status='completed',**kwargs))
def data():
 from ..research_v6.data import examples
 return examples()
def partitions(m):
 from ..research_v7.common import partitions as split
 return split(m)
def protect():
 from ..research_v7.common import protect as oldcheck
 answer=oldcheck();h=readjson(V7/'manifests/numerical_results_hashes.json');assert all(digest(REPO/k)==v for k,v in h.items()),'V7 parent changed';answer['V7_frozen']=len(h)
 p=ROOT/'manifests/parent_snapshot.json'
 if p.exists():assert all(digest(REPO/k)==v for k,v in readjson(p).items()),'V7 full snapshot changed'
 return answer
def macro(f,columns):return f.groupby('volcano')[columns].mean().mean()
def md(f):
 if f.empty:return 'No eligible observations.\n'
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/') for x in r)+' |' for r in f.itertuples(index=False,name=None))+'\n'
