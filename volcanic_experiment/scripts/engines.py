"""Frozen prior branches, exact accelerated low rank, and finite declared ablations."""
from common import *
import models
import time,joblib,torch
from dataset import features
from networks import Network
class Engines:
    def __init__(self):
        torch.set_num_threads(4)
        self.tree=joblib.load(REPO/'mixture_experiment/checkpoints/fold1_tree_correction_15.joblib')
        cp=torch.load(REPO/'overnight/checkpoints/matched_seed7.pt',map_location='cpu',weights_only=False)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.cnn=Network(cp['history'],cp['kind'],cp['width'],cp['drop']).to(self.device).eval();self.cnn.load_state_dict(cp['state'])
        prior=readjson(REPO/'mixture_experiment/selection_manifest.json')['folds'][0]
        self.scale=prior['scale_mm'];self.noise=prior['noise_mm']
        self.weights=readjson(REPO/'mixture_experiment/fold1_heterogeneous_constrained_contributions.json')['constant_weights']
        self.official=readjson(REPO/'mixture_experiment/fold1_standalone_ensemble_constrained_contributions.json')['constant_weights']
    def predict(self,h,selected=None,with_cnn=True):
        out={};times={}
        def call(name,fn):
            start=time.perf_counter();out[name]=fn().astype('float32');times[name]=time.perf_counter()-start
        call('persistence',lambda:np.repeat(h['last'][None],3,0))
        call('exponential_level',lambda:np.repeat(models.smooth(h['maps'],h['times'],120)[None],3,0))
        call('lowrank_r3_robust',lambda:lowrank(h,3,True))
        call('temporal',lambda:models.kalman(h,noise=self.noise).forecast_mm)
        call('context',lambda:self.tree.predict(h).forecast_mm)
        call('quality',lambda:models.quality_forecast(h,noise=self.noise).forecast_mm)
        bs=[out[n] for n in ['temporal','lowrank_r3_robust','context','quality']]
        call('heterogeneous_equal',lambda:np.mean(bs,0))
        call('heterogeneous_constrained',lambda:sum(w*b for w,b in zip(self.weights,bs)))
        call('temporal_spatial',lambda:(bs[0]+bs[1])/2)
        call('without_context',lambda:(bs[0]+bs[1]+bs[3])/3)
        call('without_quality',lambda:(bs[0]+bs[1]+bs[2])/3)
        if selected:
            hs,rank,robust=selected
            # caller supplies selected's longer history separately
            call('selected_lowrank',lambda:lowrank(h,rank,robust))
        if with_cnn:
            def cnn():
                x,last,_=features(h['maps'],h['times'],h['future'],self.scale,h['spacing'])
                with torch.no_grad(): pred=self.cnn(torch.as_tensor(x[None],device=self.device))[0][0].cpu().numpy()
                return last+pred*self.scale
            call('matched_cnn',cnn)
            call('standalone_ensemble_constrained',lambda:sum(w*b for w,b in zip(self.official,[bs[0],bs[2],out['matched_cnn']])))
        return out,times

def basis(h,robust):
    a=fill_past(h['maps'])[0].reshape(len(h['maps']),-1).astype('float64')
    if robust:
        med=np.median(a,0);mad=np.maximum(np.median(abs(a-med),0)*1.4826,1.)
        a=np.clip(a,med-3*mad,med+3*mad)
    mean=a.mean(0);a-=mean
    eigen,u=np.linalg.eigh(a@a.T);u=u[:,np.argsort(eigen)[::-1]]
    return a,mean,u

def from_basis(h,b,rank):
    a,mean,u=b;u=u[:,:min(rank,len(a)-1)]
    x=np.stack([np.ones(len(a)),(h['times']-h['times'][-1])/90],1)
    xf=np.stack([np.ones(3),(h['future']-h['times'][-1])/90],1)
    weights=xf@np.linalg.solve(x.T@x+np.diag([0.,4.]),x.T)@u@u.T
    return (mean+weights@a).reshape(3,*h['last'].shape).astype('float32')

def lowrank(h,rank=3,robust=True):return from_basis(h,basis(h,robust),rank)
