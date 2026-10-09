"""Crystallographic objective helpers for Bayesian optimisation.

Evaluates one PtyREX reconstruction (including trial_NNN/split_1 or split_2),
uses symmetry_weighted_score as the default reward, writes per-trial JSON and
cumulative CSV diagnostics, and contains no FRC calculation.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence
from datetime import datetime, timezone
import csv, json, math, os, warnings
import numpy as np

from crystal_fourier_metrics import (
    load_complex_object, compute_fourier_power, compute_radial_power_spectrum,
    find_radial_peaks, compute_azimuthal_profile, find_azimuthal_peaks,
    match_angular_peak_family, fit_peak_family, validate_peak_family_fits,
    score_peak_family,
)

class ObjectiveMetric(str, Enum):
    SYMMETRY = "symmetry_weighted_score"
    SHARPNESS = "sharpness_score"
    INTENSITY = "sum_integrated_intensity"
    GEOMETRY = "geometry_weighted_score"

class FailureCode(str, Enum):
    LOAD="load_failed"; FOURIER="fourier_failed"; NO_RADIAL_PEAK="no_radial_peak"
    AZIMUTHAL="azimuthal_peak_detection_failed"; FAMILY_MATCH="family_match_failed"
    PEAK_FIT="peak_fit_failed"; VALIDATION="fit_validation_failed"
    SCORING="metric_computation_failed"; NONFINITE_OBJECTIVE="nonfinite_objective"

@dataclass(frozen=True)
class CrystalEvaluationConfig:
    q_search: tuple[float,float]=(4.4,6.2)
    angle_spacing_deg: float=60.0
    radial_half_width: float=0.12
    n_angles: int=1440
    n_radial: int=11
    azimuthal_prominence: Optional[float]=None
    min_distance_deg: float=8.0
    azimuthal_smooth_deg: float=0.5
    family_tolerance_deg: float=12.0
    require_matches: Optional[int]=None
    score_column: str="prominence"
    use_measured_angles: bool=True
    position_tolerance: float=0.35
    fit_half_width: float=0.7
    background_model: str="plane"
    peak_model: str="gaussian"
    require_all_valid: bool=False
    validation_kwargs: Mapping[str,Any]=field(default_factory=lambda:{
        "min_success_fraction":0.5,"min_sigma":0.01,"max_sigma":1.0,
        "max_relative_rmse":0.3})
    objective_metric: ObjectiveMetric=ObjectiveMetric.SYMMETRY
    minimise: bool=True
    failure_penalty: float=1e30
    hdf_pattern: str="*.hdf"
    dataset: str="entry_1/process_1/output_1/object"
    window: str="hann"
    subtract_mean: bool=True
    normalise_power: bool=False
    def __post_init__(self):
        lo,hi=map(float,self.q_search)
        if not (math.isfinite(lo) and math.isfinite(hi) and 0<=lo<hi): raise ValueError("invalid q_search")
        if self.failure_penalty<=0 or not math.isfinite(self.failure_penalty): raise ValueError("invalid failure_penalty")
    @property
    def minimum_family_matches(self):
        if self.require_matches is not None: return int(self.require_matches)
        n=max(1,int(round(360/self.angle_spacing_deg))); return max(2,n-2) if n>=2 else 1

@dataclass
class CrystalMetricResult:
    trial:int; parameter_value:float; reconstruction_path:str; success:bool
    objective:float; objective_metric:str; failure_code:Optional[str]=None
    failure_reason:Optional[str]=None; q0:Optional[float]=None
    dy:Optional[float]=None; dx:Optional[float]=None
    n_azimuthal_peaks:int=0; n_family_expected:int=0; n_family_matched:int=0
    mean_angular_error_deg:Optional[float]=None
    metrics:dict[str,Any]=field(default_factory=dict)
    parameters:dict[str,Any]=field(default_factory=dict)
    def to_dict(self): return _json_safe(asdict(self))
    @property
    def loss(self): return float(self.objective)

def _json_safe(v):
    if isinstance(v,Mapping): return {str(k):_json_safe(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_json_safe(x) for x in v]
    if isinstance(v,np.ndarray): return _json_safe(v.tolist())
    if isinstance(v,np.generic): v=v.item()
    if isinstance(v,Path): return str(v)
    if isinstance(v,float) and not math.isfinite(v): return None
    return v

def latest_hdf(recon_dir,trial,pattern="*.hdf",split_name=None):
    """Newest HDF under trial_NNN, optionally under split_1 or split_2."""
    folder=Path(recon_dir)/f"trial_{int(trial):03d}"
    if split_name is not None:
        if split_name not in {"split_1","split_2"}: raise ValueError("split_name must be split_1 or split_2")
        folder/=split_name
    files=sorted(folder.glob(pattern),key=lambda p:p.stat().st_mtime)
    if not files: raise FileNotFoundError(f"Trial {trial:03d}: no {pattern!r} in {folder}")
    return files[-1]

def objective_from_metrics(metrics,metric=ObjectiveMetric.SYMMETRY,*,minimise=True):
    key=metric.value if isinstance(metric,ObjectiveMetric) else str(metric)
    reward=float(metrics[key])
    if not math.isfinite(reward): raise ValueError(f"non-finite objective {key}: {reward}")
    return -reward if minimise else reward

def penalty_result(trial,parameter_value,path,config,code,reason,parameters=None,partial=None):
    p=dict(partial or {})
    return CrystalMetricResult(int(trial),float(parameter_value),str(path),False,
        float(config.failure_penalty),config.objective_metric.value,code.value,str(reason),
        p.get("q0"),p.get("dy"),p.get("dx"),int(p.get("n_azimuthal_peaks",0)),
        int(p.get("n_family_expected",0)),int(p.get("n_family_matched",0)),
        p.get("mean_angular_error_deg"),dict(p.get("metrics",{})),dict(parameters or {}))

def evaluate_crystal_trial(reconstruction_path,*,trial,parameter_value,config,
                           parameters=None,loader=load_complex_object,raise_on_failure=False):
    path=Path(reconstruction_path); partial={}
    def fail(code,exc):
        if raise_on_failure: raise exc
        return penalty_result(trial,parameter_value,path,config,code,
            f"{type(exc).__name__}: {exc}",parameters,partial)
    try:
        obj,dy,dx=loader(path,dataset=config.dataset); obj=np.asarray(obj).squeeze()
        if obj.ndim!=2: raise ValueError(f"expected 2-D object, got {obj.shape}")
        partial.update(dy=float(dy),dx=float(dx))
    except Exception as e: return fail(FailureCode.LOAD,e)
    try:
        power=compute_fourier_power(np.angle(obj),window=config.window,
            subtract_mean=config.subtract_mean,normalise=config.normalise_power)
        q,radial,_=compute_radial_power_spectrum(power,dy,dx)
    except Exception as e: return fail(FailureCode.FOURIER,e)
    try:
        rp=find_radial_peaks(q,radial,config.q_search)
        if rp.empty: raise RuntimeError("no radial peak")
        q0=float(rp.iloc[0]["q"]); partial["q0"]=q0
    except Exception as e: return fail(FailureCode.NO_RADIAL_PEAK,e)
    try:
        theta,az=compute_azimuthal_profile(power,dy,dx,q0,
            radial_half_width=config.radial_half_width,n_angles=config.n_angles,n_radial=config.n_radial)
        peaks=find_azimuthal_peaks(theta,az,prominence=config.azimuthal_prominence,
            min_distance_deg=config.min_distance_deg,smooth_deg=config.azimuthal_smooth_deg)
        partial["n_azimuthal_peaks"]=len(peaks)
    except Exception as e: return fail(FailureCode.AZIMUTHAL,e)
    try:
        fam=match_angular_peak_family(peaks,angle_spacing_deg=config.angle_spacing_deg,
            tolerance_deg=config.family_tolerance_deg,score_column=config.score_column,
            require_matches=config.minimum_family_matches)
        partial.update(n_family_expected=int(fam["n_peaks"]),n_family_matched=int(fam["n_matched"]))
        errs=np.asarray(fam["angular_errors_deg"],float); errs=errs[np.isfinite(errs)]
        partial["mean_angular_error_deg"]=float(errs.mean()) if len(errs) else None
    except Exception as e: return fail(FailureCode.FAMILY_MATCH,e)
    try:
        fits=fit_peak_family(power,dy,dx,q0,fam,use_measured_angles=config.use_measured_angles,
            position_tolerance=config.position_tolerance,fit_half_width=config.fit_half_width,
            background_model=config.background_model,peak_model=config.peak_model)
    except Exception as e: return fail(FailureCode.PEAK_FIT,e)
    try:
        validation=validate_peak_family_fits(fits,**dict(config.validation_kwargs))
        if config.require_all_valid and not bool(validation["valid"]): raise RuntimeError("fit validation failed")
    except Exception as e: return fail(FailureCode.VALIDATION,e)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",message="Mean of empty slice")
            warnings.filterwarnings("ignore",message="All-NaN slice encountered")
            metrics=score_peak_family(fits,require_all_valid=config.require_all_valid,
                                      validation_kwargs=dict(config.validation_kwargs))
        for k in ("mean_position_error","max_position_error"):
            if k in metrics and not np.isfinite(metrics[k]): metrics[k]=None
        partial["metrics"]=metrics
        objective=objective_from_metrics(metrics,config.objective_metric,minimise=config.minimise)
    except Exception as e: return fail(FailureCode.SCORING,e)
    return CrystalMetricResult(int(trial),float(parameter_value),str(path),True,objective,
        config.objective_metric.value,q0=q0,dy=float(dy),dx=float(dx),
        n_azimuthal_peaks=len(peaks),n_family_expected=int(fam["n_peaks"]),
        n_family_matched=int(fam["n_matched"]),
        mean_angular_error_deg=partial["mean_angular_error_deg"],metrics=dict(metrics),
        parameters=dict(parameters or {}))

def save_trial_diagnostics(result,output_dir):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    dst=out/f"trial_{result.trial:03d}_metrics.json"; tmp=dst.with_suffix(".json.tmp")
    with tmp.open("w",encoding="utf-8") as f: json.dump(result.to_dict(),f,indent=2,sort_keys=True,allow_nan=False)
    os.replace(tmp,dst); return dst

def result_to_csv_row(result):
    row={"recorded_utc":datetime.now(timezone.utc).isoformat(),"trial":result.trial,
         "parameter_value":result.parameter_value,"reconstruction_path":result.reconstruction_path,
         "success":result.success,"objective":result.objective,"objective_metric":result.objective_metric,
         "failure_code":result.failure_code,"failure_reason":result.failure_reason,"q0":result.q0,
         "dy":result.dy,"dx":result.dx,"n_azimuthal_peaks":result.n_azimuthal_peaks,
         "n_family_expected":result.n_family_expected,"n_family_matched":result.n_family_matched,
         "mean_angular_error_deg":result.mean_angular_error_deg}
    for prefix,values in (("metric_",result.metrics),("parameter_",result.parameters)):
        for k,v in values.items():
            v=_json_safe(v)
            if isinstance(v,(dict,list)): v=json.dumps(v,sort_keys=True,allow_nan=False)
            row[prefix+str(k)]="" if v is None else v
    return {k:("" if v is None else v) for k,v in row.items()}

def append_cumulative_csv(result,csv_path):
    """Atomically append/update by (trial,reconstruction_path). Sequential writers only."""
    dst=Path(csv_path); dst.parent.mkdir(parents=True,exist_ok=True); new=result_to_csv_row(result)
    rows=[]; fields=[]
    if dst.exists() and dst.stat().st_size:
        with dst.open(newline="",encoding="utf-8") as f:
            rd=csv.DictReader(f); fields=list(rd.fieldnames or []); rows=list(rd)
    ident=(str(result.trial),str(result.reconstruction_path))
    rows=[r for r in rows if (str(r.get("trial","")),str(r.get("reconstruction_path","")))!=ident]
    rows.append(new); fields += [k for k in new if k not in fields]
    tmp=dst.with_suffix(dst.suffix+".tmp")
    with tmp.open("w",newline="",encoding="utf-8") as f:
        wr=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore"); wr.writeheader(); wr.writerows(rows)
    os.replace(tmp,dst); return dst

def evaluate_trial(trial,parameter_value,*,recon_dir,config,split_name=None,parameters=None,
                   diagnostics_dir=None,cumulative_csv=None,loader=load_complex_object,raise_on_failure=False):
    try: path=latest_hdf(recon_dir,trial,config.hdf_pattern,split_name)
    except Exception as e:
        path=Path(recon_dir)/f"trial_{trial:03d}"/(split_name or "")
        result=penalty_result(trial,parameter_value,path,config,FailureCode.LOAD,
                              f"{type(e).__name__}: {e}",parameters)
        if raise_on_failure: raise
    else:
        pars=dict(parameters or {}); 
        if split_name is not None: pars.setdefault("split",split_name)
        result=evaluate_crystal_trial(path,trial=trial,parameter_value=parameter_value,
            config=config,parameters=pars,loader=loader,raise_on_failure=raise_on_failure)
    if diagnostics_dir is not None: save_trial_diagnostics(result,diagnostics_dir)
    if cumulative_csv is not None: append_cumulative_csv(result,cumulative_csv)
    return result

def evaluate_trials(trial_numbers,parameter_values,*,recon_dir,config,split_name=None,
                    parameters=None,diagnostics_dir=None,cumulative_csv=None,verbose=True,**kwargs):
    if len(trial_numbers)!=len(parameter_values): raise ValueError("length mismatch")
    pars=list(parameters) if parameters is not None else [None]*len(trial_numbers)
    results=[]
    for t,v,p in zip(trial_numbers,parameter_values,pars):
        r=evaluate_trial(t,v,recon_dir=recon_dir,config=config,split_name=split_name,parameters=p,
                         diagnostics_dir=diagnostics_dir,cumulative_csv=cumulative_csv,**kwargs)
        results.append(r)
        if verbose: print(f"Trial {t:03d}: {'OK' if r.success else r.failure_code}; objective={r.objective:.6g}")
    return results

def optimiser_evaluation_wrapper(trial,parameter_value,*,recon_dir,config,split_name=None,
                                 parameters=None,diagnostics_dir=None,cumulative_csv=None):
    r=evaluate_trial(trial,parameter_value,recon_dir=recon_dir,config=config,split_name=split_name,
                     parameters=parameters,diagnostics_dir=diagnostics_dir,cumulative_csv=cumulative_csv)
    return r.objective,r.to_dict()

__all__=["CrystalEvaluationConfig","CrystalMetricResult","FailureCode","ObjectiveMetric",
"latest_hdf","objective_from_metrics","penalty_result","evaluate_crystal_trial","evaluate_trial",
"evaluate_trials","save_trial_diagnostics","result_to_csv_row","append_cumulative_csv",
"optimiser_evaluation_wrapper"]
