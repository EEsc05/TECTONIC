"""Scoring sees future maps only after all forecasts have been generated."""
from common import *
from scipy import ndimage

def coherent(a,mask,threshold=5.):
    lab,n=ndimage.label((abs(a)>=threshold)&mask)
    sizes=np.bincount(lab.ravel());good=sizes>=4;good[0]=False
    return good[lab]

def score(pred,target,last,mask,space,halfwidth=None):
    if mask.sum()<16:return None
    error=(pred-target)[mask].astype(float);abs_error=abs(error);delta=(target-last);forecast=pred-last
    actual=delta[mask].astype(float);f=forecast[mask].astype(float)
    unusual=abs(actual)>=np.quantile(abs(actual),.9)
    meaningful=abs(actual)>=5
    def correlation(a,b):return float(np.corrcoef(a,b)[0,1]) if np.std(a)>1e-8 and np.std(b)>1e-8 else np.nan
    tmask=coherent(delta,mask);pmask=coherent(forecast,mask)
    tp=(tmask&pmask).sum();union=(tmask|pmask).sum()
    gy,gx=np.gradient(np.where(mask,delta,0),*space);py,px=np.gradient(np.where(mask,forecast,0),*space)
    interior=ndimage.binary_erosion(mask)
    validgrad=interior.sum()>0
    observedcent=np.array(ndimage.center_of_mass(tmask)) if tmask.any() else None
    predcent=np.array(ndimage.center_of_mass(pmask)) if pmask.any() else None
    edge=np.zeros_like(mask);edge[:4]=True;edge[-4:]=True;edge[:,:4]=True;edge[:,-4:]=True
    r=dict(n_pixels=int(mask.sum()),mae_mm=float(abs_error.mean()),rmse_mm=float(np.sqrt(np.mean(error**2))),median_ae_mm=float(np.median(abs_error)),p90_ae_mm=float(np.quantile(abs_error,.9)),q90_change_mae_mm=float(abs_error[unusual].mean()),observed_change_mae_mm=float(abs(actual).mean()),observed_change_p90_mm=float(np.quantile(abs(actual),.9)),predicted_change_mean_mm=float(f.mean()),predicted_change_max_mm=float(abs(f).max()),predicted_change_absmean_mm=float(abs(f).mean()),change_magnitude_bias_mm=float(np.mean(abs(f)-abs(actual))),change_pattern_correlation=correlation(f,actual),change_cosine=float(np.dot(f,actual)/max(np.linalg.norm(f)*np.linalg.norm(actual),1e-12)),sign_agreement=float(np.mean(np.sign(f[meaningful])==np.sign(actual[meaningful]))) if meaningful.any() else np.nan,region_precision=float(tp/pmask.sum()) if pmask.any() else np.nan,region_recall=float(tp/tmask.sum()) if tmask.any() else np.nan,region_iou=float(tp/union) if union else np.nan,predicted_coherent_area_km2=float(pmask.sum()*np.prod(space)),observed_coherent_area_km2=float(tmask.sum()*np.prod(space)),predicted_positive_area_km2=float(((forecast>=5)&mask).sum()*np.prod(space)),predicted_negative_area_km2=float(((forecast<=-5)&mask).sum()*np.prod(space)),gradient_mae_mm_km=float(np.mean((abs(py-gy)+abs(px-gx))[interior])) if validgrad else np.nan,predicted_gradient_mean_mm_km=float(np.mean(np.hypot(py,px)[interior])) if validgrad else np.nan,centre_error_km=float(np.linalg.norm((predcent-observedcent)*space)) if predcent is not None and observedcent is not None else np.nan,observed_region_touches_edge=bool((tmask&edge).any()),bias_mm=float(error.mean()),error_std_mm=float(error.std()))
    if halfwidth is not None:
        r.update(interval_coverage=float(np.mean(abs_error<=halfwidth)),unusual_interval_coverage=float(np.mean(abs_error[unusual]<=halfwidth)),interval_width_mm=float(2*halfwidth))
    return r
