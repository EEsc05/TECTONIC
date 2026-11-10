"""Stage1-only V9; all prior artifacts read-only."""
from ..common import *
BASE=ROOT;ROOT=BASE/'experiments/v9_model_optimization';V8=BASE/'experiments/v8_specialized_forecasting';V7=BASE/'experiments/v7_model_diagnostics';V6=BASE/'experiments/v6_global_volcanic_forecasting/runs/v6_global_v2';VERSION='v9_optimization_v1';DATASET='global_comet_LOS_v6_2_immutable';SPLIT='v9_nested_training_cv_locked2023_v1';SEED=20261206
for d in ['manifests','logs','checkpoints','features','predictions','search','figures','diagnostics','tests','code']:(ROOT/d).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
def register(p,x):
 if p.exists():assert readjson(p)==x,f'Immutable registration differs: {p}'
 else:writejson(p,x)
def finish(p,**kw):writejson(ROOT/'manifests'/(p+'_status.json'),dict(status='completed',**kw))
def done(p):return (ROOT/'manifests'/(p+'_status.json')).exists()
def protect():
 from ..research_v8.common import protect as prior
 a=prior();h=readjson(V8/'manifests/artifact_hashes.json');bad=[k for k,v in h.items() if not (REPO/k).exists() or digest(REPO/k)!=v];assert not bad,bad[:4];a['V8_frozen']=len(h);return a

def data():
 from ..research_v6.data import examples
 return examples()
def partitions(m):
 tr=np.flatnonzero(m.role=='train');cal=np.flatnonzero((m.role=='validation')&(m.target_day<day('2023-01-01'))&(m.origin_index>=17));dev=np.flatnonzero((m.role=='validation')&(m.origin_day>=day('2023-01-01'))&(m.origin_index>=17));test=np.flatnonzero(m.role.isin(['temporal_evaluation','geographic_evaluation'])&(m.origin_index>=17));return tr,cal,dev,test

def folds(m):
 tr,_,_,_=partitions(m);r=m.iloc[tr];out=[]
 for name,cut,end,held in [('early2016','2016-01-01','2017-01-01',None),('late2018','2018-01-01','2020-01-01',None),('region2018','2018-01-01','2020-01-01','southeast_asia')]:
  fit=r[(r.target_day<day(cut))&((r.region!=held) if held else True)].index.to_numpy();val=r[(r.origin_day>=day(cut))&(r.target_day<day(end))&((r.region==held) if held else True)&(r.origin_index>=17)].index.to_numpy();assert len(fit)>=30 and len(val)>=5,(name,len(fit),len(val));out.append((name,fit,val))
 return out

def macro(f,columns):return f.groupby('volcano')[columns].mean().mean()
def md(f):
 if f.empty:return 'No eligible observations.\n'
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/') for x in r)+' |' for r in f.itertuples(index=False,name=None))+'\n'
def history(phase,rows):csv(ROOT/'search'/(phase+'_history.csv'),rows)
