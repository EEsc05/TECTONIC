"""Common past-only pixel features from exactly six irregular acquisitions."""
from .common import *
from ..stage1.models import fill_past,smooth
import time
NAMES=['recent_rate_x_h','median_rate_x_h','overall_rate_x_h','acceleration_x_h2','EMA120_minus_last','past_std','gradient_y','gradient_x','spatial_contrast','availability','horizon','log_horizon','history_span','cadence','log_spacing_y','log_spacing_x','LOS_e','LOS_n','LOS_u','geometry_known','last','median','MAD','EMA30_minus_last','EMA120_minus_last_duplicate_control','robust_pairwise_trend_x_h','valid_count','rank3_historical_residual','rank3_component1','rank3_component2','rank3_component3']
def build(m,a):
 x=a['context'].copy();x[:,:,:9]*=a['scale'][:,None,None];extras=[]
 for i,r in m.iterrows():
  raw=a['history'][i];t=a['times'][i];f,v=fill_past(raw);last=f[-1];med=np.median(f,0);mad=np.median(abs(f-med),0);trend=np.median(np.array([(f[k]-f[j])/(t[k]-t[j]) for j in range(6) for k in range(j+1,6)]),0);z=f.reshape(6,-1).astype(float);z-=z.mean(0);eig,u=np.linalg.eigh(z@z.T);ii=np.argsort(eig)[::-1][:3];ss=np.sqrt(np.maximum(eig[ii],0));vt=np.divide(u[:,ii].T@z,ss[:,None],out=np.zeros((3,1024)),where=ss[:,None]>1e-8);coeff=z@vt.T;components=coeff[-1,:,None]*vt;recon=z.mean(0)+coeff@vt;residual=(z[-1]-recon[-1]).reshape(32,32);extra=[last,med,mad,smooth(raw,t,30)-last,smooth(raw,t,120)-last,trend*r.horizon_days,v.sum(0),residual,*components.reshape(3,32,32)];extras.append(np.stack(extra,-1).reshape(1024,-1))
 return np.nan_to_num(np.concatenate([x,np.array(extras)],axis=-1)).astype('float32')
def subset_indices(name):
 if name=='base':return [0,1,2,4,5,9,10,11,12,13,20,23,24,25,26]
 omit={'no_geometry':list(range(14,20)),'no_spatial':[6,7,8,27,28,29,30],'no_timing':[12,13]}.get(name,[]);return [i for i in range(len(NAMES)) if i not in omit]
def load():return np.load(ROOT/'features/past_features.npy',mmap_mode='r')
def run():
 assert done('protocol');m,a,_=data();tick=time.perf_counter();x=build(m,a);np.save(ROOT/'features/past_features.npy',x);csv(ROOT/'features/feature_library.csv',[dict(index=i,name=name,source='six past maps, actual days, requested horizon or known geometry',future_target_used=False) for i,name in enumerate(NAMES)]);csv(ROOT/'diagnostics/features_runtime.csv',[dict(model_id='shared_past_feature_library',feature_construction_seconds=time.perf_counter()-tick,cases=len(m),columns=x.shape[-1],device='CPU')]);finish('features');print('V9 FEATURE LIBRARY COMPLETE',x.shape,flush=True)
if __name__=='__main__':run()
