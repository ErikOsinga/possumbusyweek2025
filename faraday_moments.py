"""
faraday_moments.py

Tools to compute zeroth, first, and second moments of Faraday depth spectra.

Definitions (following Dickey et al. 2019, Eq. 5-7):

    m0 = ∫ T(phi) dphi
    m1 = (1 / m0) ∫ T(phi) * phi dphi
    m2 = (1 / m0) ∫ T(phi) * (phi - m1)**2 dphi

For discrete spectra we approximate the integrals with trapezoidal rules.

The typical use case is a Faraday depth spectrum T(phi) that is
polarized intensity (e.g. |F(phi)|, or brightness temperature in K)
as a function of Faraday depth phi (rad m^-2).

This module supports:
- complex spectra (using |spectrum| as T(phi) by default),
- uneven phi grids,
- NaN handling,
- thresholding to downweight spurious emission/noise in the wings,
- single or batch spectra.

Author: ChatGPT & Osinga
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ThresholdConfig:
    """
    Configuration for automatic thresholding of the spectrum.

    Parameters
    ----------
    relative :
        Fraction of the peak intensity used as a lower bound.
        If None, no relative threshold is applied.
    absolute :
        Absolute lower bound on intensity. If None, no absolute threshold.
        If both `relative` and `absolute` are provided, the effective
        threshold is max(relative * peak, absolute).
    multi_component :
        If True, attempt to include multiple separated components
        above threshold by iteratively masking around peaks.
        If False, a single global mask (T >= threshold) is used.
    min_peak_factor :
        For multi_component=True, additional peaks are only included
        if their height is >= min_peak_factor * effective_threshold.
    """
    relative: Optional[float] = 0.15
    absolute: Optional[float] = None
    multi_component: bool = False
    min_peak_factor: float = 2.0


def _prepare_spectrum(
    phi: np.ndarray,
    spectrum: np.ndarray,
    use_magnitude: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare phi and spectrum arrays:
    - cast to 1D float arrays,
    - use magnitude if complex,
    - remove NaNs/Infs,
    - sort by phi ascending.

    Returns
    -------
    phi_clean, T_clean
    """
    phi = np.asarray(phi, dtype=float).ravel()
    spectrum = np.asarray(spectrum)

    if spectrum.shape != phi.shape:
        # last resort: attempt to ravel and broadcast check
        if spectrum.size != phi.size:
            raise ValueError(
                "phi and spectrum must have the same length; "
                f"got {phi.size} and {spectrum.size}"
            )
        spectrum = spectrum.ravel()

    if np.iscomplexobj(spectrum) and use_magnitude:
        T = np.abs(spectrum).astype(float)
    else:
        T = np.asarray(spectrum, dtype=float)

    # Mask non-finite values
    finite = np.isfinite(phi) & np.isfinite(T)
    if not np.any(finite):
        return np.array([], dtype=float), np.array([], dtype=float)

    phi = phi[finite]
    T = T[finite]

    # Sort by phi so integration is well-defined
    order = np.argsort(phi)
    phi = phi[order]
    T = T[order]

    return phi, T


def _build_mask_single_component(
    T: np.ndarray,
    config: ThresholdConfig,
) -> np.ndarray:
    """
    Build a simple global mask: T >= effective_threshold.

    Returns
    -------
    mask : bool array
    """
    if T.size == 0:
        return np.zeros_like(T, dtype=bool)

    peak = np.nanmax(T)

    if peak <= 0:
        return np.zeros_like(T, dtype=bool)

    thresholds = []
    if config.relative is not None:
        thresholds.append(config.relative * peak)
    if config.absolute is not None:
        thresholds.append(config.absolute)

    if not thresholds:
        # No thresholding requested: use everything that is strictly > 0
        return T > 0

    thr = max(thresholds)
    return T >= thr


def _build_mask_multi_component(
    T: np.ndarray,
    config: ThresholdConfig,
) -> np.ndarray:
    """
    Multi-component mask:
    1. Compute global effective threshold.
    2. Iteratively find peaks in the *unmasked* spectrum.
    3. For each peak above `min_peak_factor * threshold`, include
       the contiguous region where T >= threshold around that peak.

    This is a simple heuristic intended to emulate the approach in
    Dickey et al. (2019, Section 2.4) without external dependencies.

    Returns
    -------
    mask : bool array
    """
    if T.size == 0:
        return np.zeros_like(T, dtype=bool)

    peak_global = np.nanmax(T)
    if peak_global <= 0:
        return np.zeros_like(T, dtype=bool)

    thresholds = []
    if config.relative is not None:
        thresholds.append(config.relative * peak_global)
    if config.absolute is not None:
        thresholds.append(config.absolute)

    if not thresholds:
        return T > 0

    thr = max(thresholds)
    min_peak_height = config.min_peak_factor * thr

    # Work on a copy we can manipulate
    working = T.copy()
    mask = np.zeros_like(T, dtype=bool)

    while True:
        # Find next peak above current mask
        idx_peak = np.argmax(working)
        height = working[idx_peak]

        if height < min_peak_height:
            break

        # Grow to the left
        left = idx_peak
        while left > 0 and T[left - 1] >= thr:
            left -= 1

        # Grow to the right
        right = idx_peak
        while right < T.size - 1 and T[right + 1] >= thr:
            right += 1

        mask[left : right + 1] = True

        # Exclude this component from further searches
        working[left : right + 1] = -np.inf

        # If there is no remaining positive value, stop
        if np.all(working <= 0):
            break

    return mask


def _build_mask(
    T: np.ndarray,
    config: Optional[ThresholdConfig],
) -> np.ndarray:
    """
    Decide which channels to include, given a ThresholdConfig.

    If config is None, include all channels with T > 0.
    """
    if config is None:
        return T > 0

    if not config.multi_component:
        return _build_mask_single_component(T, config)

    return _build_mask_multi_component(T, config)


def _trapz_moments(
    phi: np.ndarray,
    T: np.ndarray,
    mask: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute m0, m1, sqrt(m2) using trapezoidal integration on masked arrays.

    Returns
    -------
    m0 : float
        Zeroth moment (∫ T dphi).
    m1 : float
        First moment (int T*phi dphi / m0).
    width : float
        sqrt(m2), a measure of the Faraday width in rad m^-2.
        m2 is the second central moment (int T*(phi-m1)^2 dphi / m0).
    """
    if phi.size == 0 or T.size == 0 or not np.any(mask):
        return np.nan, np.nan, np.nan

    phi_m = phi[mask]
    T_m = T[mask]

    if phi_m.size < 2:
        # Cannot integrate with less than 2 points
        return np.nan, np.nan, np.nan

    # m0 = ∫ T dphi
    m0 = np.trapz(T_m, phi_m)

    if m0 == 0:
        return 0.0, np.nan, np.nan

    # m1 = ∫ phi * T dphi / m0
    m1_num = np.trapz(T_m * phi_m, phi_m)
    m1 = m1_num / m0

    # m2 = ∫ T * (phi - m1)^2 dphi / m0
    m2_num = np.trapz(T_m * (phi_m - m1) ** 2, phi_m)
    m2 = m2_num / m0

    width = float(np.sqrt(m2)) if m2 >= 0 else np.nan
    return float(m0), float(m1), width


def compute_faraday_moments(
    phi: np.ndarray,
    spectrum: np.ndarray,
    threshold: Optional[ThresholdConfig] = ThresholdConfig(),
    use_magnitude: bool = True,
) -> Tuple[float, float, float]:
    """
    Compute zeroth, first, and second spectral moments for a single
    Faraday depth spectrum.

    Parameters
    ----------
    phi :
        1D array of Faraday depths (rad m^-2).
    spectrum :
        1D array of spectrum values. May be real or complex.
        If complex and `use_magnitude=True`, |spectrum| is used
        as T(phi) for the moment computation.
    threshold :
        ThresholdConfig instance controlling which channels to include,
        or None to skip thresholding (use T > 0).
    use_magnitude :
        If True and spectrum is complex, use |spectrum|.
        If False, the real part is used.

    Returns
    -------
    m0, m1, width :
        Zeroth, first, and sqrt(second) moment.

        - m0 has units of T_units * rad m^-2.
        - m1 has units of rad m^-2.
        - width has units of rad m^-2 (sqrt of the central second moment).

        If no valid channels remain after thresholding, all values are NaN.
    """
    phi_c, T_c = _prepare_spectrum(phi, spectrum, use_magnitude=use_magnitude)
    mask = _build_mask(T_c, threshold)
    m0, m1, width = _trapz_moments(phi_c, T_c, mask)
    return m0, m1, width


def compute_faraday_moments_batch(
    phi: np.ndarray,
    spectra: np.ndarray,
    threshold: Optional[ThresholdConfig] = ThresholdConfig(),
    use_magnitude: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute moments for a batch of spectra sharing the same phi grid.

    Parameters
    ----------
    phi :
        1D array of Faraday depths (rad m^-2).
    spectra :
        Array with shape (n_spectra, n_phi) or (n_phi,).
        If 1D, this is equivalent to `compute_faraday_moments`.
    threshold :
        ThresholdConfig for all spectra (applied per spectrum).
    use_magnitude :
        If True and spectra are complex, use |spectrum| per spectrum.

    Returns
    -------
    m0, m1, width :
        Arrays of shape (n_spectra,) with zeroth, first, and sqrt(second)
        moments for each spectrum.
    """
    phi = np.asarray(phi, dtype=float).ravel()
    spectra = np.asarray(spectra)

    if spectra.ndim == 1:
        m0, m1, width = compute_faraday_moments(
            phi, spectra, threshold=threshold, use_magnitude=use_magnitude
        )
        return (
            np.array([m0], dtype=float),
            np.array([m1], dtype=float),
            np.array([width], dtype=float),
        )

    if spectra.ndim != 2:
        raise ValueError(
            "spectra must be 1D or 2D; got array with ndim="
            f"{spectra.ndim}"
        )

    n_spectra, n_phi = spectra.shape
    if phi.size != n_phi:
        raise ValueError(
            "phi length and spectra last dimension must match; "
            f"got len(phi)={phi.size}, spectra.shape={spectra.shape}"
        )

    m0_arr = np.empty(n_spectra, dtype=float)
    m1_arr = np.empty(n_spectra, dtype=float)
    width_arr = np.empty(n_spectra, dtype=float)

    for i in range(n_spectra):
        m0, m1, width = compute_faraday_moments(
            phi,
            spectra[i],
            threshold=threshold,
            use_magnitude=use_magnitude,
        )
        m0_arr[i] = m0
        m1_arr[i] = m1
        width_arr[i] = width

    return m0_arr, m1_arr, width_arr


if __name__ == "__main__":
    # Minimal self-test / usage example

    # Construct a toy Gaussian Faraday spectrum:
    phi_grid = np.linspace(-100, 100, 801)  # rad m^-2
    phi0 = 10.0   # true center
    sigma = 20.0  # true width

    T_true = np.exp(-0.5 * ((phi_grid - phi0) / sigma) ** 2)

    # Add some noise and a spurious wing component
    rng = np.random.default_rng(42)
    noise = 0.05 * rng.standard_normal(phi_grid.size)
    T_noisy = T_true + noise
    T_noisy[T_noisy < 0] = 0.0

    # Compute moments with a threshold similar to Dickey+2019
    config = ThresholdConfig(relative=0.15, absolute=None,
                             multi_component=False)

    m0, m1, width = compute_faraday_moments(
        phi_grid, T_noisy, threshold=config
    )

    print("Example spectrum:")
    print(f"  True center: {phi0:.2f} rad m^-2")
    print(f"  True sigma : {sigma:.2f} rad m^-2")
    print(f"           : {m0:.3f}")
    print(f"  m1         : {m1:.3f} rad m^-2")
    print(f"  width      : {width:.3f} rad m^-2")
