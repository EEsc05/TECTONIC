"""Isolated experiment helpers and compatibility interface for frozen forecasters."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','1')
import sys,json,dataclasses
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
REPO=ROOT.parent
sys.path.append(str(REPO/'mixture_experiment/vendor'))
sys.path.append(str(REPO/'mixture_experiment/scripts'))
sys.path.append(str(REPO/'overnight/scripts'))
@dataclasses.dataclass
class BranchOutput:
    forecast_mm: object
    valid: object
    uncertainty_mm: object=None
    auxiliary: dict=dataclasses.field(default_factory=dict)
    metadata: dict=dataclasses.field(default_factory=dict)
def fill_past(maps):
    valid=np.isfinite(maps); filled=np.zeros_like(maps)
    for j in range(len(maps)): filled[j]=np.where(valid[j],maps[j],filled[j-1] if j else 0)
    return filled,valid
def readjson(p): return json.loads(Path(p).read_text())
def writejson(p,x): Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.item() if isinstance(v,np.generic) else str(v)))
def crop(a,n):
    h,w=a.shape[-2:]
    assert n<=min(h,w) and (h-n)%2==0 and (w-n)%2==0
    return a[..., (h-n)//2:(h+n)//2,(w-n)//2:(w+n)//2]
def spacing(latitude,delta_lat,delta_lon):
    # WGS84 local meridional and prime-vertical radii; cell-edge extents.
    rad=np.deg2rad(latitude);e2=6.69437999014e-3;a=6378137.
    m=a*(1-e2)/(1-e2*np.sin(rad)**2)**1.5;v=a/np.sqrt(1-e2*np.sin(rad)**2)
    return np.array([np.deg2rad(abs(delta_lat))*m,np.deg2rad(abs(delta_lon))*v*np.cos(rad)])/1000

def make_history(values,days,i,n,future,space,size):
    assert i-n+1>=0 and np.all(future>days[i])
    maps=crop(np.array(values[i-n+1:i+1]),size)
    maps[maps==0]=np.nan
    filled,valid=fill_past(maps)
    return dict(maps=maps,times=np.array(days[i-n+1:i+1]),future=np.array(future),last=filled[-1],seen=valid.any(0),spacing=np.array(space))
def origins(days,role,split,protocol):
    ds=days.astype('datetime64[D]'); ans=[]
    for i in range(protocol['max_history']-1,len(ds)-4,protocol['origin_stride_acquisitions']):
        if split=='validation': ok=role=='validation' and ds[i-17]>=np.datetime64(protocol['validation_input_start']) and ds[i+4]<np.datetime64(protocol['validation_target_end_exclusive'])
        elif split=='calibration':ok=role=='train' and ds[i+4]<np.datetime64(protocol['calibration_target_end_exclusive'])
        elif split=='test':ok=role=='test' and ds[i]>=np.datetime64(protocol['test_origin_start'])
        elif split=='historical_challenge':ok=role=='test' and ds[i]<np.datetime64('2019-01-01')
        else: raise ValueError(split)
        if ok:ans.append(i)
    return ans

def activity(name,date):
    # All labels are retrospective evaluation metadata, NOT model inputs.
    for e in readjson(ROOT/'data/activity_annotations.json'):
        if e['volcano']==name and e['start']<=str(np.datetime64(int(date),'D'))<=e['end']:return e['label'],e['episode']
    return 'unknown','unknown'

def calendar_targets(days,origin,horizons=(12,24,48),tolerance=6):
    days=np.asarray(days);goals=days[origin]+np.asarray(horizons)
    if origin>=len(days)-1:raise ValueError('No strictly future acquisition')
    ix=np.array([origin+1+np.argmin(abs(days[origin+1:]-goal)) for goal in goals])
    return ix,abs(days[ix]-goals)<=tolerance
