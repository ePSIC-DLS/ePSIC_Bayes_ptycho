from pathlib import Path
import h5py, numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display


def _collapse_object_phase(a):
    a=np.asarray(a); axes=tuple(range(max(0,a.ndim-2)))
    return np.angle(a.sum(axis=axes) if axes else a)


def _collapse_probe_modulus(a):
    a=np.asarray(a); axes=tuple(range(max(0,a.ndim-2)))
    intensity=(np.abs(a)**2).sum(axis=axes) if axes else np.abs(a)**2
    return np.sqrt(np.maximum(intensity,0))


def _crop_phase(p, activity_fraction=.15, margin=.08):
    p=np.asarray(p,float); centre=np.angle(np.nanmean(np.exp(1j*p)))
    activity=np.abs(np.angle(np.exp(1j*(p-centre))))
    finite=activity[np.isfinite(activity)]
    if not finite.size: return p
    threshold=max(activity_fraction*np.nanpercentile(finite,99),np.nanpercentile(finite,60))
    yy,xx=np.nonzero(np.isfinite(activity)&(activity>threshold))
    if not len(yy): return p
    y0,y1,x0,x1=yy.min(),yy.max()+1,xx.min(),xx.max()+1
    my=max(2,int(margin*(y1-y0))); mx=max(2,int(margin*(x1-x0)))
    y0,y1=max(0,y0-my),min(p.shape[0],y1+my); x0,x1=max(0,x0-mx),min(p.shape[1],x1+mx)
    return p[y0:y1,x0:x1]


def _fourier(phase,dy,dx,bins=None):
    im=np.nan_to_num(phase,nan=np.nanmedian(phase)); im-=im.mean()
    im*=np.outer(np.hanning(im.shape[0]),np.hanning(im.shape[1]))
    power=np.abs(np.fft.fftshift(np.fft.fft2(im)))**2
    ny,nx=power.shape
    qy=np.fft.fftshift(np.fft.fftfreq(ny,d=dy))*1e-9
    qx=np.fft.fftshift(np.fft.fftfreq(nx,d=dx))*1e-9
    r=np.hypot(*np.meshgrid(qx,qy)[::-1])
    bins=bins or max(96,min(ny,nx)//2); edges=np.linspace(0,r.max(),bins+1)
    lab=np.digitize(r.ravel(),edges)-1; valid=(lab>=0)&(lab<bins)
    sums=np.bincount(lab[valid],weights=power.ravel()[valid],minlength=bins)
    cnt=np.bincount(lab[valid],minlength=bins)
    radial=np.divide(sums,cnt,out=np.full(bins,np.nan),where=cnt>0)
    return power,qx,qy,(edges[:-1]+edges[1:])/2,radial


def _history(csv):
    d=pd.read_csv(csv); d['trial']=pd.to_numeric(d['trial']).astype(int); return d


def _load(trial,recon_dir,csv,crop=True,bins=None):
    row=_history(csv).query('trial == @trial').iloc[-1]
    p=Path(str(row.get('reconstruction_path','')))
    if not p.exists():
        fs=list((Path(recon_dir)/f'trial_{trial:03d}').glob('*.hdf'))+list((Path(recon_dir)/f'trial_{trial:03d}').glob('*.h5'))
        if not fs: raise FileNotFoundError(f'No HDF for trial {trial}')
        p=max(fs,key=lambda x:x.stat().st_mtime)
    with h5py.File(p,'r') as h:
        root='entry_1/process_1/output_1'
        phase=_collapse_object_phase(h[root+'/object'][...])
        probe=_collapse_probe_modulus(h[root+'/probe'][...])
        d=np.asarray(h['entry_1/process_1/common_1/dx'][...]).squeeze().ravel()
    dy=dx=float(d[0]) if len(d)==1 else None
    if len(d)>1: dy,dx=float(d[0]),float(d[-1])
    phase=_crop_phase(phase) if crop else phase
    power,qx,qy,q,radial=_fourier(phase,dy,dx,bins)
    return dict(row=row,path=p,phase=phase,probe=probe,power=power,qx=qx,qy=qy,q=q,radial=radial)


def _summary(row,name,unit):
    fields=[('trial','Trial'),('parameter_value',f'{name} ({unit})'),('success','Success'),('objective','Loss'),
    ('metric_symmetry_weighted_score','Symmetry-weighted score'),('metric_sharpness_score','Sharpness score'),
    ('metric_sum_integrated_intensity','Integrated intensity'),('metric_geometry_weighted_score','Geometry-weighted score'),
    ('q0','q0 (nm^-1)'),('n_azimuthal_peaks','Azimuthal peaks'),('n_family_matched','Family matched'),
    ('metric_fit_success_fraction','Fit success fraction'),('failure_code','Failure code'),('failure_reason','Failure reason')]
    return pd.DataFrame([(label,row[k]) for k,label in fields if k in row.index],columns=['Metric','Value'])


def show_reconstruction(trial,*,recon_dir,cumulative_csv,parameter_name,parameter_unit,crop=True,
                        n_radial_bins=None,radial_q_min=1.0,radial_q_max=None,fourier_q_max=10.0):
    d=_load(int(trial),recon_dir,cumulative_csv,crop,n_radial_bins); row=d['row']; summary=_summary(row,parameter_name,parameter_unit)
    fig,ax=plt.subplots(2,3,figsize=(15.5,8.7))
    # Positive electron phase contrast only; raw phase still drives Fourier diagnostics.
    pdsp=np.clip(d['phase']-np.nanmedian(d['phase']),0,None)
    vmax=np.nanpercentile(pdsp[np.isfinite(pdsp)],99.5) if np.isfinite(pdsp).any() else 1
    vmax=max(float(vmax),1e-12)
    im=ax[0,0].imshow(pdsp,cmap='inferno',vmin=0,vmax=vmax); ax[0,0].set_title('Object phase (cropped, positive contrast)')
    fig.colorbar(im,ax=ax[0,0],shrink=.8,label='Phase (rad)')
    lp=np.log10(d['power']+np.finfo(float).tiny); lo,hi=np.nanpercentile(lp,[2,99.8])
    extent=[d['qx'][0],d['qx'][-1],d['qy'][0],d['qy'][-1]]
    im=ax[0,1].imshow(lp,origin='lower',extent=extent,cmap='magma',vmin=lo,vmax=hi)
    ax[0,1].set(title='Object-phase Fourier power (log10)',xlabel=r'$q_x$ (nm$^{-1}$)',ylabel=r'$q_y$ (nm$^{-1}$)')
    if fourier_q_max is not None: ax[0,1].set(xlim=(-fourier_q_max,fourier_q_max),ylim=(-fourier_q_max,fourier_q_max))
    fig.colorbar(im,ax=ax[0,1],shrink=.8)
    phi=max(float(np.nanpercentile(d['probe'],99.5)),1e-12)
    im=ax[0,2].imshow(d['probe'],cmap='gray',vmin=0,vmax=phi); ax[0,2].set_title('Probe modulus (all leading dimensions combined)'); fig.colorbar(im,ax=ax[0,2],shrink=.8)
    m=np.isfinite(d['radial'])&(d['radial']>0)&(d['q']>=radial_q_min)
    if radial_q_max is not None: m&=d['q']<=radial_q_max
    ax[1,0].semilogy(d['q'][m],d['radial'][m]); q0=row.get('q0',np.nan)
    if pd.notna(q0): ax[1,0].axvline(float(q0),c='r',ls='--',label=f'q0={float(q0):.3f}'); ax[1,0].legend()
    ax[1,0].set(title='Radial object-phase Fourier power',xlabel=r'q (nm$^{-1}$)',ylabel='Mean power (log scale)'); ax[1,0].grid(alpha=.25)
    ax[1,1].axis('off'); tab=ax[1,1].table(cellText=summary.values,colLabels=summary.columns,loc='center',cellLoc='left'); tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1,1.22); ax[1,1].set_title('Metric summary')
    ax[1,2].axis('off'); val=row.get('parameter_value',np.nan); score=row.get('metric_symmetry_weighted_score',np.nan)
    ax[1,2].text(.02,.92,f'Trial {int(trial):03d}\n\n{parameter_name}: {val:.6g} {parameter_unit}\nSymmetry score: {score:.6g}\n\nHDF:\n{d["path"].name}',va='top')
    fig.suptitle(f'Trial {int(trial):03d} — {parameter_name}={val:.6g} {parameter_unit}'); fig.tight_layout(); plt.show(); display(summary)
    d.update(figure=fig,summary=summary); return d


def show_current_optimum(*,recon_dir,cumulative_csv,parameter_name,parameter_unit,score_column='metric_symmetry_weighted_score',**kwargs):
    h=_history(cumulative_csv); ok=h['success'].astype(str).str.lower().eq('true'); s=pd.to_numeric(h[score_column],errors='coerce'); v=h.loc[ok&s.notna()]
    if v.empty: raise RuntimeError('No successful finite-score trial is available.')
    trial=int(v.loc[pd.to_numeric(v[score_column]).idxmax(),'trial']); print(f'Current optimum by {score_column}: trial {trial:03d}')
    return show_reconstruction(trial,recon_dir=recon_dir,cumulative_csv=cumulative_csv,parameter_name=parameter_name,parameter_unit=parameter_unit,**kwargs)


def compare_reconstructions(trials,*,recon_dir,cumulative_csv,parameter_name,parameter_unit,**kwargs):
    trials=[int(t) for t in trials]
    if len(trials)<2: raise ValueError('Provide at least two trial numbers.')
    results=[show_reconstruction(t,recon_dir=recon_dir,cumulative_csv=cumulative_csv,parameter_name=parameter_name,parameter_unit=parameter_unit,**kwargs) for t in trials]
    metrics=pd.DataFrame({f'trial_{t:03d}':r['summary'].set_index('Metric')['Value'] for t,r in zip(trials,results)})
    print('Comparison metrics:'); display(metrics)
    return dict(trials=trials,datasets=results,metrics=metrics)
