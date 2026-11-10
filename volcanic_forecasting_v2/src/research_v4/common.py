"""Isolated v4 research inputs and task directories. No inherited writer entry points imported."""
from ..common import *
BASE=ROOT;PARENT=BASE/'historical_v3';ROOT=BASE/'research_v4';VERSION='research_v4';SEED=20261106
TASKS={1:'28_broad_historical_expansion',2:'29_stage1_representation',3:'30_stage1_neural_generative',4:'31_stage2_seismic_normalization',5:'32_cross_stage_integration'}
ROOT.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
for name in ['manifests','logs','reports','tests']:(ROOT/name).mkdir(parents=True,exist_ok=True)
def task(n):
 p=BASE/'experiments'/TASKS[n]/'runs'/VERSION
 for sub in ['raw','processed','models','splits','figures','logs','reports']:(p/sub).mkdir(parents=True,exist_ok=True)
 return p

def register(path,obj):
 if path.exists():assert readjson(path)==obj,f'Immutable registration differs: {path}'
 else:writejson(path,obj)

def parent_check():
 for name in ['frozen_reference_hashes.json','primary_numerical_results_hashes.json']:
  h=readjson(PARENT/'manifests'/name);bad=[p for p,v in h.items() if digest(REPO/p)!=v];assert not bad,bad[:3]
 return 866

def stage1_data():
 f=pd.read_csv(PARENT/'processed/stage1_examples.csv');a=np.load(PARENT/'processed/stage1_examples.npz');b=np.load(PARENT/'processed/stage1_point_predictions.npz');selected='quality_v2_h6_half30_w0';return f,{k:a[k] for k in a.files},b[selected]

def mdtable(f):
 def fmt(x):return f'{x:.4f}' if isinstance(x,(float,np.floating)) else str(x).replace('|','/')
 return '| '+' | '.join(f.columns)+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(fmt(x) for x in row)+' |' for row in f.itertuples(index=False,name=None))+'\n'
