"""Crystal-specific 2-D Fourier analysis with measured azimuthal peak localisation.

Pipeline:
    reconstruction -> Fourier power -> radial spectrum -> radial peak
    -> azimuthal profile -> measured azimuthal peaks -> symmetry-family matching
    -> local 2-D fits -> candidate scores

Reciprocal-space coordinates returned by this module are in nm^-1.
"""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths


def newest_file(directory, pattern="*.hdf"):
    files = sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} in {directory}")
    return files[-1]


def load_complex_object(hdf_file, dataset="entry_1/process_1/output_1/object"):
    """Return complex object, dy and dx (metres per pixel)."""
    with h5py.File(hdf_file, "r") as handle:
        if dataset not in handle:
            raise KeyError(f"Dataset {dataset!r} not found in {hdf_file}")
        obj = np.asarray(handle[dataset][()]).squeeze()
        spacing = np.asarray(handle["entry_1/process_1/common_1/dx"][()]).ravel()
    if obj.ndim != 2:
        raise ValueError(f"Expected a 2-D object, got {obj.shape}")
    if spacing.size == 1:
        dy = dx = float(spacing[0])
    else:
        dy, dx = float(spacing[0]), float(spacing[1])
    return obj, dy, dx


def compute_fourier_power(image, window="hann", subtract_mean=True, normalise=False):
    """Return centred 2-D Fourier power."""
    work = np.asarray(image, dtype=float).copy()
    if subtract_mean:
        work -= np.nanmean(work)
    if window == "hann":
        work *= np.outer(np.hanning(work.shape[0]), np.hanning(work.shape[1]))
    elif window is not None:
        raise ValueError("window must be 'hann' or None")
    power = np.abs(np.fft.fftshift(np.fft.fft2(work))) ** 2
    if normalise and power.sum() > 0:
        power /= power.sum()
    return power


def frequency_axes(shape, dy, dx):
    """Return fy, fx axes in nm^-1."""
    fy = np.fft.fftshift(np.fft.fftfreq(shape[0], d=dy)) * 1e-9
    fx = np.fft.fftshift(np.fft.fftfreq(shape[1], d=dx)) * 1e-9
    return fy, fx


def compute_radial_power_spectrum(power2d, dy, dx, bin_width=None, statistic="mean"):
    """Return radial q centres, binned power and pixel counts."""
    fy, fx = frequency_axes(power2d.shape, dy, dx)
    FX, FY = np.meshgrid(fx, fy)
    radius = np.hypot(FX, FY)
    if bin_width is None:
        bin_width = min(abs(fx[1] - fx[0]), abs(fy[1] - fy[0]))
    bins = np.arange(0, radius.max() + bin_width, bin_width)
    labels = np.digitize(radius.ravel(), bins) - 1
    valid = (labels >= 0) & (labels < len(bins) - 1) & np.isfinite(power2d.ravel())
    counts = np.bincount(labels[valid], minlength=len(bins) - 1)
    sums = np.bincount(labels[valid], weights=power2d.ravel()[valid], minlength=len(bins) - 1)
    if statistic == "mean":
        values = np.divide(sums, counts, out=np.full_like(sums, np.nan, dtype=float), where=counts > 0)
    elif statistic == "sum":
        values = sums
    else:
        raise ValueError("statistic must be 'mean' or 'sum'")
    return 0.5 * (bins[:-1] + bins[1:]), values, counts


def find_radial_peaks(q, radial, q_range, prominence=None, smooth_points=3):
    """Return radial peak candidates sorted by prominence."""
    q = np.asarray(q); radial = np.asarray(radial)
    use = (q >= q_range[0]) & (q <= q_range[1]) & np.isfinite(radial) & (radial > 0)
    q_local, y_local = q[use], radial[use]
    if q_local.size < 5:
        raise ValueError("Too few radial samples in q_range")
    smooth_points = max(1, int(smooth_points))
    smooth = np.convolve(y_local, np.ones(smooth_points) / smooth_points, mode="same")
    if prominence is None:
        prominence = 0.05 * np.ptp(smooth)
    indices, properties = find_peaks(smooth, prominence=prominence)
    rows = [{"q": float(q_local[i]), "height": float(y_local[i]),
             "smoothed_height": float(smooth[i]),
             "prominence": float(properties["prominences"][j])}
            for j, i in enumerate(indices)]
    columns = ["q", "height", "smoothed_height", "prominence"]
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("prominence", ascending=False).reset_index(drop=True))


def compute_azimuthal_profile(power2d, dy, dx, q0, radial_half_width=0.15,
                              n_angles=1440, n_radial=11, reducer="mean"):
    """Sample an annulus and return angle (degrees) and azimuthal power."""
    if q0 <= radial_half_width:
        raise ValueError("q0 must exceed radial_half_width")
    fy, fx = frequency_axes(power2d.shape, dy, dx)
    theta_rad = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    samples = []
    for radius in np.linspace(q0 - radial_half_width, q0 + radial_half_width, n_radial):
        qx = radius * np.cos(theta_rad)
        qy = radius * np.sin(theta_rad)
        cols = np.interp(qx, fx, np.arange(fx.size))
        rows = np.interp(qy, fy, np.arange(fy.size))
        samples.append(map_coordinates(power2d, [rows, cols], order=1, mode="nearest"))
    samples = np.asarray(samples)
    if reducer == "mean":
        profile = samples.mean(axis=0)
    elif reducer == "max":
        profile = samples.max(axis=0)
    else:
        raise ValueError("reducer must be 'mean' or 'max'")
    return np.degrees(theta_rad), profile


def _circular_smooth(values, points):
    points = max(1, int(points))
    if points == 1:
        return np.asarray(values, float).copy()
    kernel = np.ones(points) / points
    pad = points // 2
    extended = np.pad(np.asarray(values, float), pad, mode="wrap")
    return np.convolve(extended, kernel, mode="same")[pad:pad + len(values)]


def find_azimuthal_peaks(theta_deg, azimuthal, prominence=None,
                         min_distance_deg=3.0, smooth_deg=0.5,
                         min_width_deg=0.0):
    """Detect actual peak centres in a periodic 0-360 degree profile.

    Returns a DataFrame containing measured angle, raw/smoothed height,
    prominence and FWHM. Circular padding prevents loss of peaks near 0 degrees.
    """
    theta = np.asarray(theta_deg, float)
    values = np.asarray(azimuthal, float)
    if theta.ndim != 1 or values.shape != theta.shape or theta.size < 8:
        raise ValueError("theta_deg and azimuthal must be equal-length 1-D arrays")
    step = float(np.median(np.diff(theta)))
    smooth_points = max(1, int(round(smooth_deg / step)))
    smoothed = _circular_smooth(values, smooth_points)
    if prominence is None:
        # Robust default based on profile spread, not the highest peak.
        prominence = max(0.05 * np.ptp(smoothed), 2.0 * np.median(np.abs(smoothed - np.median(smoothed))))
    distance_points = max(1, int(round(min_distance_deg / step)))
    pad = max(distance_points, int(round(10.0 / step)))
    extended = np.r_[smoothed[-pad:], smoothed, smoothed[:pad]]
    peaks_ext, props = find_peaks(extended, prominence=prominence, distance=distance_points)
    keep = (peaks_ext >= pad) & (peaks_ext < pad + theta.size)
    peaks_ext = peaks_ext[keep]
    prominences = props["prominences"][keep]
    widths_samples = peak_widths(extended, peaks_ext, rel_height=0.5)[0]
    rows = []
    for idx_ext, prom, width_samples in zip(peaks_ext, prominences, widths_samples):
        idx = int(idx_ext - pad)
        width_deg = float(width_samples * step)
        if width_deg >= min_width_deg:
            rows.append({"angle_deg": float(theta[idx] % 360),
                         "height": float(values[idx]),
                         "smoothed_height": float(smoothed[idx]),
                         "prominence": float(prom),
                         "fwhm_deg": width_deg,
                         "index": idx})
    columns = ["angle_deg", "height", "smoothed_height", "prominence", "fwhm_deg", "index"]
    return (pd.DataFrame(rows, columns=columns)
            .sort_values("prominence", ascending=False).reset_index(drop=True))


def circular_distance_deg(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0)


def match_angular_peak_family(azimuthal_peaks, angle_spacing_deg,
                              n_peaks=None, tolerance_deg=8.0,
                              score_column="prominence", require_matches=4):
    """Match one regular angular family to measured azimuthal peak centres.

    Every measured peak is tried as an orientation anchor. Expected family
    positions are matched one-to-one to actual detected peaks within tolerance.
    Missing members are retained as NaN and penalised. This selects one family
    in multilayer material without forcing reported centres onto ideal angles.
    """
    if azimuthal_peaks.empty:
        raise ValueError("No azimuthal peaks supplied")
    spacing = float(angle_spacing_deg)
    if not (0 < spacing <= 360):
        raise ValueError("angle_spacing_deg must be in (0, 360]")
    if n_peaks is None:
        inferred = 360.0 / spacing
        if not np.isclose(inferred, round(inferred)):
            raise ValueError("Provide n_peaks when 360/angle_spacing_deg is non-integral")
        n_peaks = int(round(inferred))
    if score_column not in azimuthal_peaks:
        raise KeyError(f"Missing score column {score_column!r}")

    measured = azimuthal_peaks["angle_deg"].to_numpy(float)
    strengths = azimuthal_peaks[score_column].to_numpy(float)
    best = None
    for anchor in measured:
        expected = (anchor + spacing * np.arange(n_peaks)) % 360
        used = set(); matched_indices = []; errors = []
        for target in expected:
            distances = circular_distance_deg(measured, target)
            for idx in np.argsort(distances):
                if int(idx) not in used and distances[idx] <= tolerance_deg:
                    used.add(int(idx)); matched_indices.append(int(idx)); errors.append(float(distances[idx])); break
            else:
                matched_indices.append(None); errors.append(np.nan)
        matched = np.array([i is not None for i in matched_indices])
        n_matched = int(matched.sum())
        family_strength = float(sum(strengths[i] for i in matched_indices if i is not None))
        mean_error = float(np.nanmean(errors)) if n_matched else np.inf
        # Completeness-first score avoids a few strong unrelated peaks winning.
        objective = family_strength * (n_matched / n_peaks) ** 2 / (1.0 + mean_error / tolerance_deg)
        candidate = (objective, n_matched, -mean_error, anchor, expected, matched_indices, errors)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    objective, n_matched, _, anchor, expected, matched_indices, errors = best
    if n_matched < require_matches:
        raise RuntimeError(f"Best family matched only {n_matched}/{n_peaks} peaks; required {require_matches}")
    measured_angles = np.array([measured[i] if i is not None else np.nan for i in matched_indices])
    measured_strengths = np.array([strengths[i] if i is not None else np.nan for i in matched_indices])
    return {"orientation_deg": float(anchor % spacing),
            "angle_spacing_deg": spacing, "n_peaks": int(n_peaks),
            "expected_angles_deg": expected,
            "measured_angles_deg": measured_angles,
            "matched_peak_indices": matched_indices,
            "angular_errors_deg": np.asarray(errors, float),
            "measured_strengths": measured_strengths,
            "n_matched": n_matched, "family_score": float(objective)}


def _peak_model(params, x, y, background_model, peak_model):
    amp, x0, y0, sx, sy, theta = params[:6]
    ct, st = np.cos(theta), np.sin(theta)
    xp = ct * (x - x0) + st * (y - y0)
    yp = -st * (x - x0) + ct * (y - y0)
    radius2 = (xp / sx) ** 2 + (yp / sy) ** 2
    peak = amp * np.exp(-0.5 * radius2) if peak_model == "gaussian" else amp / (1.0 + radius2)
    if background_model == "constant":
        background = params[6]
    elif background_model == "plane":
        background = params[6] + params[7] * (x - x0) + params[8] * (y - y0)
    else:
        raise ValueError("background_model must be 'constant' or 'plane'")
    return peak + background


def fit_fourier_peak(power2d, dy, dx, expected_q, expected_angle_deg,
                     position_tolerance=0.35, fit_half_width=0.7,
                     background_model="plane", peak_model="gaussian"):
    """Fit an elliptical Gaussian/Lorentzian plus local background."""
    fy, fx = frequency_axes(power2d.shape, dy, dx)
    angle = np.radians(expected_angle_deg)
    expected_x, expected_y = expected_q * np.cos(angle), expected_q * np.sin(angle)
    use_x = np.abs(fx - expected_x) <= fit_half_width
    use_y = np.abs(fy - expected_y) <= fit_half_width
    data = power2d[np.ix_(use_y, use_x)]
    x, y = np.meshgrid(fx[use_x], fy[use_y])
    if min(data.shape) < 5:
        raise ValueError("Fit window contains fewer than five pixels on one axis")
    edge = np.r_[data[0], data[-1], data[:, 0], data[:, -1]]
    b0 = float(np.median(edge)); amp0 = max(float(data.max() - b0), np.finfo(float).eps)
    tail0 = [b0] if background_model == "constant" else [b0, 0.0, 0.0]
    p0 = [amp0, expected_x, expected_y, 0.12, 0.12, 0.0] + tail0
    lower_tail = [-np.inf] if background_model == "constant" else [-np.inf] * 3
    upper_tail = [np.inf] if background_model == "constant" else [np.inf] * 3
    lower = [0, expected_x-position_tolerance, expected_y-position_tolerance, 0.01, 0.01, -np.pi/2] + lower_tail
    upper = [np.inf, expected_x+position_tolerance, expected_y+position_tolerance,
             fit_half_width, fit_half_width, np.pi/2] + upper_tail
    fit = least_squares(lambda p: (_peak_model(p, x, y, background_model, peak_model)-data).ravel(),
                        p0, bounds=(lower, upper), loss="soft_l1")
    model = _peak_model(fit.x, x, y, background_model, peak_model)
    amp, x0, y0, sx, sy, theta = fit.x[:6]
    bg_at_peak = fit.x[6]
    integrated = float(2*np.pi*amp*sx*sy) if peak_model == "gaussian" else np.nan
    return {

    "success": bool(
        fit.success
    ),

    # ------------------------------------
    # expected position
    # ------------------------------------

    "expected_qx": float(
        expected_x
    ),

    "expected_qy": float(
        expected_y
    ),

    "expected_q": float(
        expected_q
    ),

    "expected_angle_deg": float(
        expected_angle_deg % 360
    ),

    # ------------------------------------
    # fitted position
    # ------------------------------------

    "qx": float(
        x0
    ),

    "qy": float(
        y0
    ),

    "q": float(
        np.hypot(
            x0,
            y0,
        )
    ),

    "angle_deg": float(
        np.degrees(
            np.arctan2(
                y0,
                x0,
            )
        ) % 360
    ),

    # ------------------------------------
    # fit results
    # ------------------------------------

    "amplitude_raw": float(
        amp + bg_at_peak
    ),

    "amplitude_bg_subtracted": float(
        amp
    ),

    "integrated_intensity": integrated,

    "sigma_x": float(
        sx
    ),

    "sigma_y": float(
        sy
    ),

    "theta_deg": float(
        np.degrees(
            theta
        )
    ),

    "background_at_peak": float(
        bg_at_peak
    ),

    "rmse": float(
        np.sqrt(
            np.mean(
                (model - data) ** 2
            )
        )
    ),

    "model": model,
    "data": data,
    "qx_grid": x,
    "qy_grid": y,
    }



def fit_peak_family(power2d, dy, dx, q0, family, use_measured_angles=True, **fit_kwargs):
    """Fit all matched family members, using measured centres when available."""
    key = "measured_angles_deg" if use_measured_angles else "expected_angles_deg"
    angles = np.asarray(family[key], float)
    fallback = np.asarray(family["expected_angles_deg"], float)
    angles = np.where(np.isfinite(angles), angles, fallback)
    return [fit_fourier_peak(power2d, dy, dx, q0, angle, **fit_kwargs) for angle in angles]

def validate_peak_family_fits(
    fits,
    *,
    min_success_fraction=1.0,
    max_position_error=None,
    min_sigma=0.01,
    max_sigma=None,
    max_relative_rmse=None,
):
    """Validate local peak fits and return per-family diagnostics.

    max_position_error is in nm^-1 and requires each fit to contain
    expected_qx/expected_qy. If absent, this check is skipped.
    max_relative_rmse is RMSE divided by background-subtracted amplitude.
    """
    if not fits:
        raise ValueError("fits must contain at least one fitted peak")

    success = np.array([bool(f.get("success", False)) for f in fits])
    sx = np.array([float(f["sigma_x"]) for f in fits])
    sy = np.array([float(f["sigma_y"]) for f in fits])
    amp = np.array([float(f["amplitude_bg_subtracted"]) for f in fits])
    rmse = np.array([float(f["rmse"]) for f in fits])
    relative_rmse = rmse / np.maximum(amp, np.finfo(float).eps)

    valid = success & np.isfinite(sx) & np.isfinite(sy) & np.isfinite(amp)
    valid &= (sx > min_sigma) & (sy > min_sigma) & (amp > 0)
    if max_sigma is not None:
        valid &= (sx < max_sigma) & (sy < max_sigma)
    if max_relative_rmse is not None:
        valid &= relative_rmse <= max_relative_rmse

    position_errors = np.full(len(fits), np.nan)
    if max_position_error is not None:
        for i, fit in enumerate(fits):
            if "expected_qx" in fit and "expected_qy" in fit:
                position_errors[i] = np.hypot(
                    fit["qx"] - fit["expected_qx"],
                    fit["qy"] - fit["expected_qy"],
                )
        checked = np.isfinite(position_errors)
        valid[checked] &= position_errors[checked] <= max_position_error

    success_fraction = float(valid.mean())
    return {
        "valid": bool(success_fraction >= min_success_fraction),
        "valid_fit_mask": valid,
        "success_fraction": success_fraction,
        "relative_rmse": relative_rmse,
        "position_errors": position_errors,
    }

def score_peak_family(fits, *, require_all_valid=True, validation_kwargs=None):
    """Return transparent intensity, sharpness, symmetry and geometry scores.

    The recommended initial Bayesian objective is symmetry_weighted_score.
    geometry_weighted_score additionally penalises variation in fitted radius;
    it should be used only after confirming that q variation is not genuine
    strain or anisotropic reciprocal-space calibration.
    """
    validation = validate_peak_family_fits(
        fits, **(validation_kwargs or {})
    )
    if require_all_valid and not validation["valid"]:
        raise RuntimeError(
            "Peak-family validation failed: "
            f"{validation['success_fraction']:.1%} fits valid"
        )

    mask = validation["valid_fit_mask"]
    selected = [fit for fit, keep in zip(fits, mask) if keep]
    if not selected:
        raise RuntimeError("No valid peak fits remain for scoring")

    intensities = np.array([f["integrated_intensity"] for f in selected], float)
    amplitudes = np.array([f["amplitude_bg_subtracted"] for f in selected], float)
    widths = np.sqrt([f["sigma_x"] * f["sigma_y"] for f in selected])
    radii = np.array([f["q"] for f in selected], float)
    backgrounds = np.array([f["background_at_peak"] for f in selected], float)

    mean_intensity = float(np.nanmean(intensities))
    intensity_cv = float(np.nanstd(intensities) / mean_intensity) if mean_intensity > 0 else np.inf
    mean_q = float(np.nanmean(radii))
    q_std = float(np.nanstd(radii))
    q_cv = float(q_std / mean_q) if mean_q > 0 else np.inf
    total_intensity = float(np.nansum(intensities))
    total_width = float(np.nansum(widths))
    symmetry_score = total_intensity / (1.0 + intensity_cv)
    position_errors = ( validation["position_errors"] ) 
    good = np.isfinite( position_errors )

    return {
        "n_fits": len(fits),
        "n_valid_fits": len(selected),
        "fit_success_fraction": validation["success_fraction"],
        "mean_amplitude": float(np.nanmean(amplitudes)),
        "sum_integrated_intensity": total_intensity,
        "mean_integrated_intensity": mean_intensity,
        "mean_width": float(np.nanmean(widths)),
        "intensity_cv": intensity_cv,
        "mean_q": mean_q,
        "q_std": q_std,
        "q_cv": q_cv,
        "mean_background": float(np.nanmean(backgrounds)),
        "mean_relative_rmse": float(np.nanmean(validation["relative_rmse"][mask])),
        "sharpness_score": total_intensity / total_width,
        "symmetry_weighted_score": symmetry_score,
        "geometry_weighted_score": symmetry_score / (1.0 + q_cv),
        "mean_position_error": float( np.nanmean( position_errors ) ), 
        "max_position_error": float( np.nanmax( position_errors ) )
    }