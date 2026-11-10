"""Complete both registered growth-gate arms; preserve original pair-stage outputs."""
from .common import *
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import shutil,joblib

def run():
 assert done('modular');m,a,_=data();auditpath=ROOT/'manifests/hybrid_growth_gain_audit.json'
 if not auditpath.exists():
  info=readjson(ROOT/'manifests/modular_shortlist.json');pp=np.load(ROOT/'predictions/modular.npz');spid=info['branches']['spatial']['model_id'];oo=np.load(ROOT/'predictions'/('OOF_'+spid+'.npz'))['pred'];ctx=np.load(ROOT/'predictions/context_oof.npz')['pred'];tt=pp['expert_temporal'];ids=readjson(ROOT/'manifests/quality_selection.json')['OOF_training_ids'];values=[tt,oo,ctx];rr=[]
  for j,k in [(0,1),(0,2),(1,2)]:
   f=pd.DataFrame([dict(volcano=m.iloc[i].volcano,a=np.nanmean(abs(values[j][i]-a['truth'][i])),b=np.nanmean(abs(values[k][i]-a['truth'][i])),pair=np.nanmean(abs((values[j][i]+values[k][i])/2-a['truth'][i]))) for i in ids]);z=macro(f,['a','b','pair']);rr.append(dict(pair=f'{j}{k}',OOF_gain=1-z['pair']/min(z.a,z.b),pair_MAE=z['pair']))
  register(auditpath,dict(pairs=rr,passed=any(r['OOF_gain']>=.005 for r in rr),criterion='Other registered arm: equal two-expert training OOF MAE gain>=.5%; no2023 labels used'))
 audit=readjson(auditpath);initial=readjson(ROOT/'diagnostics/initial_correlation_gate.json') if (ROOT/'diagnostics/initial_correlation_gate.json').exists() else readjson(ROOT/'manifests/hybrid_growth_gate.json');passed=bool(initial['passed'] or audit['passed']);tr,cal,_,_=partitions(m);p=ROOT/'predictions/modular.npz';r=np.load(p);pred={k:r[k] for k in r.files};recipes=readjson(ROOT/'checkpoints/fusion_recipes.json');stack=np.stack([pred['expert_temporal'],pred['expert_spatial'],pred['expert_context'],pred['expert_quality']]);trust=np.load(ROOT/'predictions/reliability.npz')['trust']
 for source,dest in [(p,ROOT/'diagnostics/initial_pair_stage_predictions.npz'),(ROOT/'checkpoints/fusion_recipes.json',ROOT/'diagnostics/initial_pair_fusion_recipes.json'),(ROOT/'manifests/hybrid_growth_gate.json',ROOT/'diagnostics/initial_correlation_gate.json')]:
  if not dest.exists():shutil.copy2(source,dest)
 def weights(which,ids=cal):
  xx=[];yy=[]
  for i in ids:
   good=np.isfinite(a['truth'][i])&np.isfinite(stack[which,i]).all(0);ix=np.flatnonzero(good.ravel())[::16];xx.append(stack[which,i].reshape(len(which),-1)[:,ix].T);yy.append(a['truth'][i].ravel()[ix])
  x=np.concatenate(xx);y=np.concatenate(yy);w0=np.full(len(which),1/len(which));fit=minimize(lambda w:np.mean(abs(x@w-y))+.25*np.sum((w-w0)**2),w0,bounds=[(0,1)]*len(which),constraints={'type':'eq','fun':lambda w:w.sum()-1},method='SLSQP');return fit.x/fit.x.sum() if fit.success else w0
 if passed:
  for name,which in [('three',[0,1,2]),('four',[0,1,2,3])]:
   w=weights(which);pred[name+'_equal']=stack[which].mean(0);pred[name+'_regularized']=np.einsum('b,bnhw->nhw',w,stack[which]);recipes[name+'_regularized']=dict(branches=which,weights=w.tolist());q=trust[which]/trust[which].sum(0,keepdims=True);pred[name+'_reliability']=np.sum(q*stack[which],0)
   for kind,labels in [('horizon',np.digitize(m.horizon_days,[31,91])),('regime',(a['scale']>np.median(a['scale'][tr])).astype(int))]:
    ws=np.tile(w,(len(m),1))
    for value in np.unique(labels):
     ids=np.array([i for i in cal if labels[i]==value]);n=m.iloc[ids][['source_id','origin_day']].drop_duplicates().shape[0];local=weights(which,ids) if n>=8 else w;ws[labels==value]=(n*local+20*w)/(n+20)
    pred[name+'_'+kind]=np.einsum('nb,bnhw->nhw',ws,stack[which])
   x=np.c_[np.log1p(m.horizon_days),np.log1p(a['scale']),np.isfinite(a['history']).mean((1,2,3))];sc=StandardScaler().fit(x[cal]);target=np.array([[-np.nanmean(abs(stack[j,i]-a['truth'][i]))/a['scale'][i] for j in which] for i in cal]);head=Ridge(alpha=100).fit(sc.transform(x[cal]),target);q=np.exp(np.clip(head.predict(sc.transform(x)),-1,1));q/=q.sum(1,keepdims=True);q=.5*q+.5*w;pred[name+'_smallgate']=np.einsum('nb,bnhw->nhw',q,stack[which]);joblib.dump(dict(head=head,scaler=sc,weights=w),ROOT/'checkpoints'/('gate_'+name+'.joblib'))
  for omit in range(4):which=[j for j in range(4) if j!=omit];w=weights(which);pred['ablate_'+str(omit)]=np.einsum('b,bnhw->nhw',w,stack[which])
 np.savez_compressed(p,**pred);writejson(ROOT/'checkpoints/fusion_recipes.json',recipes);writejson(ROOT/'manifests/hybrid_growth_gate.json',dict(passed=passed,correlation_arm=initial['passed'],OOF_pair_gain_arm=audit['passed'],criterion='Registered OR: residual correlation<.95 OR equal-pair OOF gain>=.5%; initial correlation-only component archived'));register(ROOT/'manifests/effective_modular_shortlist.json',dict(point_models=list(pred),growth=passed,selection='Completed registered growth gate, OOF training outcomes and2022 blending targets only, before final2023 assessment',original_pair_stage_preserved=True));finish('fusion_growth');print('V9 REGISTERED FUSION GROWTH COMPLETE',passed,flush=True)
if __name__=='__main__':run()
