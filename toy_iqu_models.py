"""
toy_iq_u_models.py

Toy models for simulating polarized LOFAR-like observations (Appendix C of Erceg+22, 
"Faraday tomography of LoTSS-DR2 data - I. Faraday moments in the high-latitude
outer Galaxy and revealing Loop III in polarisation", A&A 663, A7 (2022)).

Implements:
- Case A: Burn slab (uniform emitting + rotating medium)
- Case B.0: Single emitting layer behind full Faraday-rotating screen
- Case B.1: Multiple emitting layers distributed through full screen
- Case B.2: Multiple emitting layers only in near half of screen
- Case C: Like A, but synchrotron emission attenuated with distance

For a given line of sight, the models produce frequency-dependent Stokes
I, Q, U "cubes" (here: 1D in LOS, 1D in frequency).

RM synthesis is not included yet; this module just gives P(nu) (Q+iU).

You can tweak:
- Frequency coverage
- Number of channels
- Faraday depth span (MGRM = total Faraday depth)
- Number and placement of emitting layers
- Emissivity attenuation in case C
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import numpy as np
from astropy import units as u
from astropy.constants import c as c_light

import matplotlib.pyplot as plt

from faraday_moments import ThresholdConfig, compute_faraday_moments

from RMtools_1D.do_RMsynth_1D import run_rmsynth
from RMtools_1D.do_RMclean_1D import run_rmclean


@dataclass
class FrequencyGrid:
    """
    Simple container for a frequency axis and derived lambda^2.

    Attributes
    ----------
    frequency : astropy.units.Quantity
        1D array of frequencies in Hz.
    lambda_sq : astropy.units.Quantity
        1D array of lambda^2 in m^2 corresponding to frequency.
    channel_width : astropy.units.Quantity
        Approximate channel width (constant) in Hz.
    """

    frequency: u.Quantity
    lambda_sq: u.Quantity
    channel_width: u.Quantity

    @classmethod
    def from_limits(
        cls,
        nu_min_MHz: float = 120.0,
        nu_max_MHz: float = 167.0,
        n_channels: int = 480,
    ) -> "FrequencyGrid":
        """
        Create a linearly spaced frequency grid between nu_min and nu_max.

        Notes
        -----
        - Defaults reflect the LOFAR frequency setup used in the paper,
          but channel width will be slightly different from 97.6 kHz due to
          linear spacing between endpoints. If you prefer exact width,
          use `from_center_and_width`.
        """
        freq = np.linspace(nu_min_MHz, nu_max_MHz, n_channels) * u.MHz
        lambda_sq = (c_light / freq) ** 2
        # Approximate constant channel width
        if n_channels > 1:
            channel_width = (freq[1] - freq[0]).to(u.Hz)
        else:
            channel_width = 0.0 * u.Hz
        return cls(frequency=freq.to(u.Hz), lambda_sq=lambda_sq.to(u.m**2), channel_width=channel_width)

    @classmethod
    def from_center_and_width(
        cls,
        nu_center_MHz: float,
        n_channels: int,
        channel_width_kHz: float,
    ) -> "FrequencyGrid":
        """
        Build a frequency grid from center frequency and constant channel width.

        Parameters
        ----------
        nu_center_MHz : float
            Central frequency.
        n_channels : int
            Number of channels.
        channel_width_kHz : float
            Channel width in kHz.

        Returns
        -------
        FrequencyGrid
        """
        nu_center = nu_center_MHz * u.MHz
        delta_nu = channel_width_kHz * u.kHz
        # Centered grid: indices from -(n-1)/2 to +(n-1)/2
        indices = np.arange(n_channels) - (n_channels - 1) / 2.0
        freq = nu_center + indices * delta_nu
        lambda_sq = (c_light / freq) ** 2
        return cls(frequency=freq.to(u.Hz), lambda_sq=lambda_sq.to(u.m**2), channel_width=delta_nu.to(u.Hz))

    @classmethod
    def possum_band1plus2(
        cls,
    ) -> "FrequencyGrid":
        """
        Create a POSSUM combined band1 + band 2 frequency grid

        Notes
        -----
        """
        freq_band1 = np.linspace(800, 1088, 288) * u.MHz
        freq_band2 = np.linspace(1296, 1440, 144) * u.MHz
        freq = np.concatenate([freq_band1,freq_band2])
        lambda_sq = (c_light / freq) ** 2
        # Approximate constant channel width
        channel_width = (freq[1] - freq[0]).to(u.Hz)
        return cls(frequency=freq.to(u.Hz), lambda_sq=lambda_sq.to(u.m**2), channel_width=channel_width)

@dataclass
class LOSGrid:
    """
    Simple container for the Faraday-depth grid along a line of sight.

    Attributes
    ----------
    phi_cells : np.ndarray
        Faraday depth at cell centers [rad m^-2].
        1D array of length n_cells.
    dphi : float
        Cell spacing in Faraday depth [rad m^-2].
    """

    phi_cells: np.ndarray
    dphi: float


def make_los_grid(
    mgrm: float = 20.0,
    n_cells: int = 1024,
    phi_start: float = 0.0,
) -> LOSGrid:
    """
    Build a uniform Faraday-depth grid along the LOS.

    Parameters
    ----------
    mgrm : float
        Total Galactic RM (Faraday depth at the far side) in rad m^-2.
        In the toy models this is fixed to 20 rad m^-2 in the paper.
    n_cells : int
        Number of cells sampling the LOS.
    phi_start : float
        Faraday depth at the near side (observer). Typically 0.

    Returns
    -------
    LOSGrid
    """
    phi_end = phi_start + mgrm
    phi_edges = np.linspace(phi_start, phi_end, n_cells + 1)
    phi_cells = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    dphi = phi_edges[1] - phi_edges[0]
    return LOSGrid(phi_cells=phi_cells.astype(float), dphi=float(dphi))


def _normalize_weights(weights: np.ndarray, dphi: float) -> np.ndarray:
    """
    Normalize real non-negative weights to unit integral over phi.

    Returns
    -------
    np.ndarray
        Normalized weights so that sum(weights) * dphi = 1.
    """
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights) * dphi
    if total <= 0.0:
        raise ValueError("Emissivity weights must have positive total integral.")
    return weights / total


def build_emissivity_case_A(
    los: LOSGrid,
    intrinsic_pol_angle_deg: float = 0.0,
    intrinsic_pol_fraction: float = 1.0,
) -> np.ndarray:
    """
    Case A: Burn slab (uniform emitting + rotating medium).

    Emissivity is uniform along the whole LOS.

    p(FDEP) = p0 * np.exp(i*chi0) = (const as a function of FDEP.)

    Returns
    -------
    np.ndarray
        Complex emissivity per cell (Q+iU) in arbitrary units.
        Normalized such that integral over phi is 1 in total polarized
        intensity before Faraday rotation is applied.
    """
    weights = np.ones_like(los.phi_cells)
    weights = _normalize_weights(weights, los.dphi)
    chi0_rad = np.deg2rad(intrinsic_pol_angle_deg)
    p0 = intrinsic_pol_fraction * np.exp(2j * chi0_rad)
    emissivity = weights * p0
    return emissivity.astype(complex)


def build_emissivity_case_C(
    los: LOSGrid,
    intrinsic_pol_angle_deg: float = 0.0,
    intrinsic_pol_fraction: float = 1.0,
    attenuation_scale_fraction: float = 0.5,
) -> np.ndarray:
    """
    Case C: Like A, but emissivity attenuated with distance from observer.

    We assume an exponential attenuation with phi (distance proxy):

        j(phi) ∝ exp(-phi / phi_scale)

    where phi_scale = attenuation_scale_fraction * mgrm.

    Parameters
    ----------
    attenuation_scale_fraction : float
        Fraction of total MGRM used as attenuation scale. Smaller values
        produce stronger attenuation with distance.

    Returns
    -------
    np.ndarray
        Complex emissivity per cell.
    """
    phi_min = np.min(los.phi_cells)
    phi_max = np.max(los.phi_cells)
    mgrm = phi_max - phi_min
    phi_scale = attenuation_scale_fraction * mgrm
    if phi_scale <= 0.0:
        raise ValueError("attenuation_scale_fraction must be > 0.")
    # Distance from observer approximated as (phi - phi_min)
    dist = los.phi_cells - phi_min
    weights = np.exp(-dist / phi_scale)
    weights = _normalize_weights(weights, los.dphi)
    chi0_rad = np.deg2rad(intrinsic_pol_angle_deg)
    p0 = intrinsic_pol_fraction * np.exp(2j * chi0_rad)
    emissivity = weights * p0
    return emissivity.astype(complex)


def _layers_to_weights(
    los: LOSGrid,
    phi_layers: np.ndarray,
    intrinsic_pol_angle_deg: float = 0.0,
    intrinsic_pol_fraction: float = 1.0,
) -> np.ndarray:
    """
    Convert a set of layer positions (in phi) into emissivity weights.

    Each layer is modeled as a delta function sitting in the nearest cell.
    Weights are normalized to unit integral in phi.
    """
    weights = np.zeros_like(los.phi_cells, dtype=float)
    for phi_layer in np.atleast_1d(phi_layers):
        idx = np.argmin(np.abs(los.phi_cells - phi_layer))
        weights[idx] += 1.0
    weights = _normalize_weights(weights, los.dphi)
    chi0_rad = np.deg2rad(intrinsic_pol_angle_deg)
    p0 = intrinsic_pol_fraction * np.exp(2j * chi0_rad)
    emissivity = weights * p0
    return emissivity.astype(complex)


def build_emissivity_case_B0(
    los: LOSGrid,
    intrinsic_pol_angle_deg: float = 0.0,
    intrinsic_pol_fraction: float = 1.0,
) -> np.ndarray:
    """
    Case B.0: Single emitting layer behind a Faraday screen.

    Interpretation:
    - Faraday-rotating medium spans full phi range.
    - A single synchrotron-emitting layer located at the far side
      (full path length through the screen).

    Returns
    -------
    np.ndarray
        Complex emissivity per cell.
    """
    phi_layer = np.max(los.phi_cells)
    return _layers_to_weights(
        los=los,
        phi_layers=np.array([phi_layer]),
        intrinsic_pol_angle_deg=intrinsic_pol_angle_deg,
        intrinsic_pol_fraction=intrinsic_pol_fraction,
    )


def build_emissivity_case_B1(
    los: LOSGrid,
    n_layers: int = 5,
    randomize: bool = False,
    seed: Optional[int] = None,
    intrinsic_pol_angle_deg: float = 0.0,
    intrinsic_pol_fraction: float = 1.0,
) -> np.ndarray:
    """
    Case B.1: Multiple emitting layers distributed across full screen.

    Parameters
    ----------
    n_layers : int
        Number of discrete emitting layers.
    randomize : bool
        If True, layer positions are random within [phi_min, phi_max].
        If False, they are evenly spaced.
    seed : int, optional
        Random seed for reproducibility (if randomize=True).

    Returns
    -------
    np.ndarray
        Complex emissivity per cell.
    """
    phi_min = np.min(los.phi_cells)
    phi_max = np.max(los.phi_cells)

    if randomize:
        rng = np.random.default_rng(seed)
        phi_layers = rng.uniform(phi_min, phi_max, size=n_layers)
    else:
        phi_layers = np.linspace(phi_min, phi_max, n_layers)

    return _layers_to_weights(
        los=los,
        phi_layers=phi_layers,
        intrinsic_pol_angle_deg=intrinsic_pol_angle_deg,
        intrinsic_pol_fraction=intrinsic_pol_fraction,
    )


def build_emissivity_case_B2(
    los: LOSGrid,
    n_layers: int = 5,
    randomize: bool = False,
    seed: Optional[int] = None,
    intrinsic_pol_angle_deg: float = 0.0,
    intrinsic_pol_fraction: float = 1.0,
) -> np.ndarray:
    """
    Case B.2: Multiple emitting layers only in the first half of the screen.

    Interpretation
    --------------
    - Faraday-rotating medium spans full phi range.
    - Emitting layers are only in the near half of phi (towards observer).

    Parameters
    ----------
    n_layers : int
        Number of discrete emitting layers.
    randomize : bool
        If True, layer positions are random in the near half.
        If False, they are evenly spaced in the near half.
    seed : int, optional
        Random seed for reproducibility (if randomize=True).

    Returns
    -------
    np.ndarray
        Complex emissivity per cell.
    """
    phi_min = np.min(los.phi_cells)
    phi_max = np.max(los.phi_cells)
    phi_mid = 0.5 * (phi_min + phi_max)

    if randomize:
        rng = np.random.default_rng(seed)
        phi_layers = rng.uniform(phi_min, phi_mid, size=n_layers)
    else:
        phi_layers = np.linspace(phi_min, phi_mid, n_layers)

    return _layers_to_weights(
        los=los,
        phi_layers=phi_layers,
        intrinsic_pol_angle_deg=intrinsic_pol_angle_deg,
        intrinsic_pol_fraction=intrinsic_pol_fraction,
    )


def compute_polarization_spectrum(
    freq_grid: FrequencyGrid,
    los: LOSGrid,
    emissivity: np.ndarray,
) -> np.ndarray:
    """
    Compute complex polarization P(nu) for a given emissivity profile.

    Uses
        P(lambda^2) = integral F(phi) * exp(2i * phi * lambda^2) dphi
    discretized as
        P(lambda^2) ≈ sum_j emissivity_j * exp(2i * phi_j * lambda^2) * dphi

    Parameters
    ----------
    freq_grid : FrequencyGrid
        Frequency and lambda^2 grid.
    los : LOSGrid
        Faraday-depth sampling.
    emissivity : np.ndarray
        Complex emissivity per cell (same length as los.phi_cells).

    Returns
    -------
    np.ndarray
        Complex polarization spectrum P(nu) with shape (n_channels,).
    """
    lambda_sq_vals = freq_grid.lambda_sq.to_value(u.m**2)  # (n_freq,)
    phi_vals = los.phi_cells  # (n_cells,)
    emissivity = np.asarray(emissivity, dtype=complex)

    if emissivity.shape != phi_vals.shape:
        raise ValueError("emissivity and los.phi_cells must have the same shape.")

    # Broadcast to (n_freq, n_cells)
    phase = np.exp(2j * np.outer(lambda_sq_vals, phi_vals))
    # Integrate along phi dimension
    p_lambda_sq = phase @ (emissivity * los.dphi)
    return p_lambda_sq


def compute_stokes_from_polarization(
    freq_grid: FrequencyGrid,
    polarization: np.ndarray,
    spectral_index_I: float = -0.7,
    I_ref: float = 1.0,
    nu_ref_MHz: float = 150.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert complex polarization into Stokes I, Q, U spectra.

    Assumes:
    - Stokes I follows a power-law spectrum: I(nu) ∝ (nu / nu_ref)^alpha.
    - Polarization is expressed as Q + i U in *absolute* units, not
      fractional. To keep things simple, we assume the integral of the
      emissivity was normalized to 1, so the typical polarized intensity
      is of order intrinsic_pol_fraction.

    Parameters
    ----------
    spectral_index_I : float
        Spectral index alpha such that I ∝ nu^alpha.
    I_ref : float
        Reference Stokes I at nu_ref.
    nu_ref_MHz : float
        Reference frequency in MHz.

    Returns
    -------
    I, Q, U : np.ndarray
        Arrays of shape (n_channels,) giving Stokes parameters as functions
        of frequency.
    """
    freq = freq_grid.frequency.to_value(u.Hz)
    nu_ref = nu_ref_MHz * 1e6  # Hz

    # Stokes I power-law
    if nu_ref <= 0.0:
        raise ValueError("nu_ref_MHz must be > 0.")
    I = I_ref * (freq / nu_ref) ** spectral_index_I

    Q = np.real(polarization)
    U = np.imag(polarization)

    return I.astype(float), Q.astype(float), U.astype(float)


def generate_toy_model_spectrum(
    case: str,
    freq_grid: Optional[FrequencyGrid] = None,
    mgrm: float = 20.0,
    n_cells: int = 1024,
    **kwargs,
) -> Dict[str, np.ndarray]:
    """
    High-level helper to generate I, Q, U spectra for a given toy model case.

    Parameters
    ----------
    case : {"A", "B0", "B1", "B2", "C"}
        Label of the toy model configuration.
    freq_grid : FrequencyGrid, optional
        If None, defaults to LOFAR-like 120-167 MHz, 480 channels.
    mgrm : float
        Total Faraday depth along LOS (MGRM) in rad m^-2.
    n_cells : int
        Number of cells in the LOS grid.
    **kwargs :
        Passed down to the emissivity-building function and/or
        `compute_stokes_from_polarization`. For example:
        - intrinsic_pol_angle_deg
        - intrinsic_pol_fraction
        - n_layers, randomize, seed
        - attenuation_scale_fraction
        - spectral_index_I, I_ref, nu_ref_MHz

    Returns
    -------
    dict
        Dictionary with keys:
            "frequency_Hz", "lambda_sq_m2", "I", "Q", "U", "P"
    """
    if freq_grid is None:
        freq_grid = FrequencyGrid.from_limits()

    los = make_los_grid(mgrm=mgrm, n_cells=n_cells)

    case = case.upper()
    if case == "A":
        emissivity = build_emissivity_case_A(los, **_filter_kwargs(kwargs, build_emissivity_case_A))
    elif case in ("B0", "B.0"):
        emissivity = build_emissivity_case_B0(los, **_filter_kwargs(kwargs, build_emissivity_case_B0))
    elif case in ("B1", "B.1"):
        emissivity = build_emissivity_case_B1(los, **_filter_kwargs(kwargs, build_emissivity_case_B1))
    elif case in ("B2", "B.2"):
        emissivity = build_emissivity_case_B2(los, **_filter_kwargs(kwargs, build_emissivity_case_B2))
    elif case == "C":
        emissivity = build_emissivity_case_C(los, **_filter_kwargs(kwargs, build_emissivity_case_C))
    else:
        raise ValueError(f"Unknown case '{case}'. Use one of 'A', 'B0', 'B1', 'B2', 'C'.")

    p_lambda_sq = compute_polarization_spectrum(freq_grid, los, emissivity)

    # Separate kwargs for Stokes spectrum
    I_kwargs = _filter_kwargs(kwargs, compute_stokes_from_polarization)
    I, Q, U = compute_stokes_from_polarization(freq_grid, p_lambda_sq, **I_kwargs)

    return {
        "frequency_Hz": freq_grid.frequency.to_value(u.Hz),
        "lambda_sq_m2": freq_grid.lambda_sq.to_value(u.m**2),
        "I": I,
        "Q": Q,
        "U": U,
        "P": p_lambda_sq,
    }


def _filter_kwargs(kwargs: dict, func) -> dict:
    """
    Filter a kwargs dict to include only arguments accepted by func.
    """
    import inspect

    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def plot_fdfs(
    case: str,
    summary_dict: dict,
    result_dict: dict,
    clean_summary: dict,
    clean_result: dict,
    mgrm: float,
    m1_dirty: float,
    m1_clean: float,
):
    """
    Plot dirty and clean FDF using the dictionaries output by RM-tools
    """

    fdf = np.abs(result_dict['dirtyFDF'])
    phiarr = result_dict['phiArr_radm2']
    phi_rmsf = result_dict['phi2Arr_radm2']
    rmsf = np.abs(result_dict['RMSFArr'])

    fig, axes = plt.subplots(1, 2, figsize=(12,6))
    plt.sca(axes[0])
    plt.plot(phi_rmsf[len(phi_rmsf)//4:len(phi_rmsf)//4*3],
            rmsf[len(phi_rmsf)//4:len(phi_rmsf)//4*3],
            color='k',alpha=0.5,ls='dashed', label='RMSF'
    )
    plt.plot(phiarr, fdf/np.max(fdf), label='FDF')
    plt.axvline(summary_dict['phiPeakPIfit_rm2'], color='r', ls='dashed', label='peak RM')
    plt.xlabel("Fdep [rad/m2]")
    plt.ylabel("FDF (arbitrary)")
    plt.title(f"Case {case}. Peak RM = {summary_dict['phiPeakPIfit_rm2']:.0f}, sigmaAddC = {summary_dict['sigmaAddC']:.0f}")
    plt.legend()
    plt.tight_layout()
    
    # rm ratio = full LOS RM divided by the peak RM probed by synchrotron emission
    rmratio_peak = mgrm / summary_dict['phiPeakPIfit_rm2'] 

    # rm ratio = full LOS RM divided by the first moment RM from the synchrotron emission
    rmratio_moment = mgrm / m1_clean
    
    plt.sca(axes[1])
    plt.title(f"Clean FDF. RM ratio = {rmratio_moment:.1f}")
    plt.plot(clean_result['phiArr_radm2'], np.abs(clean_result['cleanFDF']),label='Cleaned FDF')
    plt.plot(clean_result['phiArr_radm2'], np.abs(clean_result['ccArr']), color='k', label='Clean Components')
    plt.axvline(clean_summary['phiPeakPIfit_rm2'], color='r', ls='dashed', label='peak RM')
    plt.axvline(m1_clean, color='green', ls='dashed', label='Clean first moment')
    plt.xlabel("Fdep [rad/m2]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'./plots_toymodels/case{case}.png')
    plt.show()


def get_defaults(setup='lofar'):
    # Simple sanity check: generate all cases with default settings
    if setup == 'lofar':
        phiMax_radm2 = 50
        dPhi_radm2=0.25
        freq_grid = FrequencyGrid.from_limits()  # LOFAR-like default
    elif setup == 'possum_band1':
        phiMax_radm2 = 400
        dPhi_radm2 = 4
        freq_grid = FrequencyGrid.from_limits(nu_min_MHz=800, nu_max_MHz=1088, n_channels=288)
    elif setup == 'possum_band1plus2':
        phiMax_radm2 = 400
        dPhi_radm2 = 4
        freq_grid = FrequencyGrid.possum_band1plus2()

    else:
        raise NotImplementedError(f"Not implemented {setup=}")

    return phiMax_radm2, dPhi_radm2, freq_grid


if __name__ == "__main__":
    MGRM = 20 # the full LOS RM value assumed. 

    # LOFAR-like default
    # phiMax_radm2, dPhi_radm2, freq_grid = get_defaults('lofar') 

    # POSSUM band 1 freq grid
    # phiMax_radm2, dPhi_radm2, freq_grid = get_defaults('possum_band1') 
    phiMax_radm2, dPhi_radm2, freq_grid = get_defaults('possum_band1plus2') 



    # compute Faraday moments with a 15% relative peak threshold for inclusion in calculation, Dickey+2019
    moment_config = ThresholdConfig(relative=0.15, absolute=None,
                             multi_component=False)

    for case in ["A", "B0", "B1", "B2", "C"]:
    # for case in ["A"]:

        ####################################### Generate IQU spectra ###############################
        result = generate_toy_model_spectrum(case, freq_grid=freq_grid, mgrm=MGRM)
        freq = result["frequency_Hz"]
        I = result["I"]
        Q = result["Q"]
        U = result["U"]
        P = result["P"]

        print(f"Case {case}:")
        print(f"  freq range: {freq[0] / 1e6:.2f} - {freq[-1] / 1e6:.2f} MHz")
        print(f"  I shape, Q shape, U shape: {I.shape}, {Q.shape}, {U.shape}")
        print(f"  |P| min-max: {np.min(np.abs(P)):.3e} - {np.max(np.abs(P)):.3e}")
        print("")

        ####################################### Do RM synthesis ####################################
        # data = [freq_Hz, q, u,  dq, du]
        data = [freq, Q, U, np.ones(len(Q))*np.median(Q)*0.1, np.ones(len(U))*np.median(U)*0.1]
        summary_dict, result_dict = run_rmsynth(
            data=data,
            phiMax_radm2=phiMax_radm2,
            dPhi_radm2=dPhi_radm2,
            weightType='uniform',
        )

        ####################################### Do RM Clean ####################################
        clean_summary, clean_result = run_rmclean(summary_dict, result_dict, cutoff=-8)


        # Compute moments with a threshold similar to Dickey+2019
        m0_dirty, m1_dirty, width_dirty = compute_faraday_moments(
            result_dict['phiArr_radm2'], result_dict['dirtyFDF'], threshold=moment_config
        ) 
        # NOTE: CVANECK: "not a fan of computing moments from dirty FDFs"
        # "because the results are generally very sensitive to the sensitivity threshold you apply."

        m0_clean, m1_clean, width_clean = compute_faraday_moments(
            clean_result['phiArr_radm2'], clean_result['cleanFDF']
        )

        print(f"First dirty moment: {m1_dirty:.1f} rad/m2")
        print(f"First clean moment: {m1_clean:.1f} rad/m2")

        ####################################### Plot FDF ####################################
        plot_fdfs(case,
                  summary_dict, result_dict, clean_summary, clean_result,
                  mgrm=MGRM,
                  m1_dirty=m1_dirty,
                  m1_clean=m1_clean,
        )