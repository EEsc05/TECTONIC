import os,sys,json,hashlib,datetime
from pathlib import Path
sys.dont_write_bytecode=True;os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/cache/matplotlib'))
import numpy as np
import pandas as pd
P=json.loads((ROOT/'configs/protocol.json').read_text())
EXPS=['00_repository_audit','01_dataset_construction','02_stage1_variable_horizon','03_stage1_comparison','04_stage2_event_audit','05_monitoring_baseline','06_deformation_baseline','07_parallel_weighted','08_parallel_learned','09_cascade','10_hazard_model','11_horizon_ablation','12_generalization','13_final_comparison']
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def readjson(p):return json.loads(Path(p).read_text())
def writejson(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,ensure_ascii=False,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
def csv(p,x,columns=None):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);f=x if isinstance(x,pd.DataFrame) else pd.DataFrame(x,columns=columns);f.to_csv(p,index=False);return f

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def day(s):return int(np.datetime64(str(s)[:10],'D').astype(int))
def date(d):return str(np.datetime64(int(d),'D'))
def exp(i):return ROOT/'experiments'/EXPS[i]
def new_run(i):
 p=exp(i)/'runs'/utc().replace(':','').replace('+','_')
 for part in ['logs','metrics','predictions','figures','outputs']:(p/part).mkdir(parents=True,exist_ok=True)
 (p/'config.yaml').write_text(json.dumps(P,indent=2));writejson(exp(i)/'latest_run.json',dict(path=str(p.relative_to(ROOT)),created_utc=utc()));return p

def numeric(a):
 a=np.array(a,dtype=object);a[a=='null']=np.nan;a[a=='NaN']=np.nan;return a.astype('float32')
def fill_past(maps):
 valid=np.isfinite(maps);filled=np.zeros_like(maps,dtype='float32')
 for i in range(len(maps)):filled[i]=np.where(valid[i],maps[i],filled[i-1] if i else 0)
 return filled,valid

def spacing(lat,dy,dx):
 a=6378137.;e=6.69437999014e-3;p=np.deg2rad(lat)
 return np.array([a*(1-e)/(1-e*np.sin(p)**2)**1.5*np.deg2rad(abs(dy)),a/np.sqrt(1-e*np.sin(p)**2)*np.cos(p)*np.deg2rad(abs(dx))])/1000

def hbin(h):
 if h<=7:return '1–7'
 if h<=14:return '8–14'
 if h<=30:return '15–30'
 if h<=45:return '31–45'
 return '46–60'
