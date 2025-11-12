#!/usr/bin/env python3
"""
staps_possum_compare.py

Compare POSSUM RMs to STAPS diffuse RM map at nearest pixel positions.

Examples:
  # Basic comparison (scatter only)
  python staps_possum_compare.py \
      --staps-fits /home/osingae/Documents/postdoc-no-onedrive/doradogroup/other_wavelengths/RMp.fits \
      --possum-catalog /path/to/possum_catalog.fits

  # With residuals vs distance from a reference point
  python staps_possum_compare.py \
      --staps-fits RMp.fits --possum-catalog possum.ecsv \
      --ra 201.365 --dec -43.019 --dist-unit arcmin
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits # type: ignore
from astropy.wcs import WCS # type: ignore
from astropy.coordinates import SkyCoord  # type: ignore
import astropy.units as u  # type: ignore
from astropy.units import UnitsWarning  # type: ignore
from astropy.table import Table  # type: ignore

from astropy.wcs import FITSFixedWarning
import warnings

from scipy import stats # type: ignore

warnings.simplefilter("ignore", FITSFixedWarning)
warnings.simplefilter("ignore", UnitsWarning)

# ----------------------------
# I/O and data prep utilities
# ----------------------------

def load_staps_map(fpath: Path) -> tuple[np.ndarray, WCS]:
    """Load STAPS RM FITS image and return (2D data, celestial WCS)."""
    with fits.open(fpath) as hdul:
        hdu = None
        for h in hdul:
            if hasattr(h, "data") and h.data is not None and h.data.ndim >= 2:
                hdu = h
                break
        if hdu is None:
            raise ValueError("No image HDU with data found in STAPS FITS file.")
        data = np.squeeze(hdu.data)
        wcs = WCS(hdu.header).celestial
    if data.ndim != 2:
        raise ValueError("STAPS image is not 2D after squeeze; check FITS file.")
    return data, wcs


def load_possum_catalog(fpath: Path,
                        ra_col: str = "ra",
                        dec_col: str = "dec",
                        rm_col: str = "rm") -> Table:
    """Load POSSUM catalog (Astropy-readable table)."""
    tbl = Table.read(fpath)
    for col in (ra_col, dec_col, rm_col):
        if col not in tbl.colnames:
            raise KeyError(f"Missing required column '{col}' in POSSUM table.")
    return tbl


def make_plots_dir() -> Path:
    outdir = Path("./plots")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


# ----------------------------
# Core calculations
# ----------------------------

def nearest_pixel_values(
    wcs: WCS,
    image: np.ndarray,
    sky: SkyCoord
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each SkyCoord, find nearest pixel in image and return:
      staps_rm (float array),
      x_pix, y_pix (float arrays for possible debugging/QA).
    Out-of-bounds positions yield NaN in staps_rm.
    """
    # World -> pixel (0-based, x is col, y is row)
    x, y = wcs.world_to_pixel(sky)

    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)

    h, w = image.shape
    inside = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)

    staps_vals = np.full(xi.shape, np.nan, dtype=float)
    staps_vals[inside] = image[yi[inside], xi[inside]]

    return staps_vals, x, y


def compute_distance_to_ref(sky: SkyCoord, ref: SkyCoord, unit: str) -> np.ndarray:
    """
    Angular separation between `sky` and `ref` in requested unit: 'deg', 'arcmin', or 'arcsec'.
    """
    sep = sky.separation(ref)
    if unit == "deg":
        return sep.deg
    if unit == "arcmin":
        return sep.to(u.arcmin).value
    if unit == "arcsec":
        return sep.to(u.arcsec).value
    raise ValueError("dist-unit must be one of: deg, arcmin, arcsec")


def fit_and_corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """
    Return slope a, intercept b from least-squares line y = a*x + b,
    and Pearson correlation (r) with two-sided p-value.
    """
    a, b = np.polyfit(x, y, 1)
    # Pearson r (vectorized, same as scipy.stats.pearsonr for r value)
    # r = np.corrcoef(x, y)[0, 1]
    r, p = stats.pearsonr(x, y)
    return a, b, r, p


# ----------------------------
# Plotting
# ----------------------------

def plot_scatter_rm(possum_rm: np.ndarray,
                    staps_rm: np.ndarray,
                    outdir: Path,
                    outname: Optional[str] = None,
                    logscale: bool = False,
                    rm_lim: float | None = 100
) -> Path:
    """
    Scatter RM_possum vs RM_staps with 1:1 line and best-fit line (y = a x + b).
    Legend shows a, b. Pearson r and p-value shown as figure text.
    """
    m = np.isfinite(possum_rm) & np.isfinite(staps_rm)
    x = staps_rm[m]
    y = possum_rm[m]

    fig, ax = plt.subplots(figsize=(6.8, 6.0))

    print(f"Plotting scatter with {x.size} points.")
    ax.scatter(x, y, s=12, alpha=0.4, edgecolor="none", label="POSSUM sources")

    if x.size and y.size:
        # 1:1 line
        xy = np.concatenate([x, y])
        lo = np.nanpercentile(xy, 2)
        hi = np.nanpercentile(xy, 98)
        ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.2, alpha=0.8, label="1:1",color='k')

        # Best-fit line y = a*x + b
        a, b, r, p = fit_and_corr(x, y)
        xx = np.linspace(lo, hi, 200)
        yy = a * xx + b
        fit_label = f"Best fit: y = {a:.3f} x + {b:.3f}"
        ax.plot(xx, yy, linewidth=1.5, alpha=0.9, label=fit_label, color='C1')

        # Legend (includes a, b via fit_label)
        leg = ax.legend(loc="upper left", frameon=True)

        # Pearson stats text below legend area (bottom-left corner of the axes works well)
        p_str = f"{p:.2e}" if np.isfinite(p) else "n/a"
        ax.text(0.02, 0.02,
                f"Pearson r = {r:.3f}\nTwo-sided p = {p_str}",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6))
    else:
        ax.text(0.5, 0.5, "No finite data to plot", transform=ax.transAxes,
                ha="center", va="center")

    ax.set_xlabel("RM_staps (rad/m^2)")
    ax.set_ylabel("RM_possum (rad/m^2)")
    ax.set_title("POSSUM vs STAPS RM (nearest STAPS pixel)")
    if logscale:
        plt.yscale("symlog", linthresh=10.0)
    
    if rm_lim is not None:
        plt.xlim(-rm_lim, rm_lim)
        plt.ylim(-rm_lim, rm_lim)

    fig.tight_layout()
    outfile = outdir / (outname or "rm_possum_vs_rm_staps.png")
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    return outfile


def plot_residual_vs_distance(possum_rm: np.ndarray,
                              staps_rm: np.ndarray,
                              distances: np.ndarray,
                              unit_label: str,
                              outdir: Path,
                              outname: Optional[str] = None,
                              rm_lim: float = 100
) -> Path:
    """
    Scatter of (RM_possum - RM_staps) vs distance to reference.
    """
    m = np.isfinite(possum_rm) & np.isfinite(staps_rm) & np.isfinite(distances)
    res = possum_rm[m] - staps_rm[m]
    d = distances[m]

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.scatter(d, res, s=12, alpha=0.8, edgecolor="none")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)

    ax.set_xlabel(f"Angular distance to reference ({unit_label})")
    ax.set_ylabel("RM_possum - RM_staps (rad/m^2)")
    ax.set_title("Residual RM vs distance to reference")

    fig.tight_layout()
    plt.ylim(-rm_lim, rm_lim)
    outfile = outdir / (outname or f"rm_residual_vs_distance_{unit_label}.png")
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    return outfile

def compute_galactic_b(sky: SkyCoord) -> np.ndarray:
    """
    Return Galactic latitude b (degrees) for each source.
    """
    return sky.galactic.b.deg


def plot_residual_vs_b(possum_rm: np.ndarray,
                       staps_rm: np.ndarray,
                       b_deg: np.ndarray,
                       outdir: Path,
                       outname: str | None = None,
                       rm_lim: float = 100,
) -> Path:
    """
    Scatter of (RM_possum - RM_staps) vs Galactic latitude b (deg).
    """
    m = np.isfinite(possum_rm) & np.isfinite(staps_rm) & np.isfinite(b_deg)
    res = possum_rm[m] - staps_rm[m]
    b = b_deg[m]

    fig, ax = plt.subplots(figsize=(6.8, 6.0))
    ax.scatter(b, res, s=12, alpha=0.8, edgecolor="none")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.7)
    # ax.axvline(0.0, color="k", linestyle=":", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Galactic latitude b (deg)")
    ax.set_ylabel("RM_possum - RM_staps (rad/m^2)")
    ax.set_title("Residual RM vs Galactic latitude")

    plt.ylim(-rm_lim, rm_lim)
    fig.tight_layout()
    outfile = outdir / (outname or "rm_residual_vs_b.png")
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    return outfile


def select_sources_within_radius(sky: SkyCoord, ref: SkyCoord, radius_deg: float) -> np.ndarray:
    """
    Return boolean mask selecting sources within 'radius_deg' of 'ref'.
    """
    if radius_deg <= 0:
        raise ValueError("--width-deg must be > 0")
    return sky.separation(ref).deg <= radius_deg



# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare POSSUM RMs with STAPS RM map.")
    p.add_argument("--staps-fits", type=Path, required=True,
                   help="Path to STAPS RM FITS image.")
    p.add_argument("--possum-catalog", type=Path, required=True,
                   help="Path to POSSUM catalog (FITS/ECSV/CSV/etc.).")

    # Column names
    p.add_argument("--ra-col", type=str, default="ra",
                   help="RA column name in POSSUM table (deg, ICRS).")
    p.add_argument("--dec-col", type=str, default="dec",
                   help="Dec column name in POSSUM table (deg, ICRS).")
    p.add_argument("--rm-col", type=str, default="rm",
                   help="RM column name in POSSUM table (rad/m^2).")

    # Optional distance-to-reference plot
    p.add_argument("--ra", type=float, help="Reference RA in degrees (ICRS).")
    p.add_argument("--dec", type=float, help="Reference Dec in degrees (ICRS).")
    p.add_argument("--dist-unit", type=str, default="deg",
                   choices=["deg", "arcmin", "arcsec"],
                   help="Unit for distance on residual plot (default: deg).")
    
    # Customisation
    p.add_argument("--logscale", action="store_true")
    
    p.add_argument("--radius-deg", type=float,
               help="Angular radius in degrees. If given with --ra and --dec, only sources within this radius are used for ALL plots.")
    p.add_argument("--rmlim", type=float, default=100.0, help="Axis limit for RM scatter plot (default: 100 rad/m^2).")


    # Optional output name suffixes
    p.add_argument("--scatter-out", type=str, default=None,
                   help="Optional filename for the scatter plot.")
    p.add_argument("--resid-out", type=str, default=None,
                   help="Optional filename for the residual-vs-distance plot.")
    p.add_argument("--b-resid-out", type=str, default=None,
                help="Optional filename for residual-vs-Galactic-latitude plot.")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = make_plots_dir()

    # Load data
    image, wcs = load_staps_map(args.staps_fits)
    possum = load_possum_catalog(args.possum_catalog, args.ra_col, args.dec_col, args.rm_col)

    # POSSUM sky coords and RMs
    sky = SkyCoord(ra=np.asarray(possum[args.ra_col]) * u.deg,
                   dec=np.asarray(possum[args.dec_col]) * u.deg,
                   frame="icrs")
    possum_rm = np.asarray(possum[args.rm_col], dtype=float)

    # POSSUM sky coords and RMs
    sky = SkyCoord(ra=np.asarray(possum[args.ra_col]) * u.deg,
                dec=np.asarray(possum[args.dec_col]) * u.deg,
                frame="icrs")
    possum_rm = np.asarray(possum[args.rm_col], dtype=float)

    # Optional spatial selection: only keep sources within width-deg of (ra, dec)
    ref = None
    if (args.ra is not None) and (args.dec is not None):
        ref = SkyCoord(args.ra * u.deg, args.dec * u.deg, frame="icrs")
        if args.radius_deg is not None:
            sel = select_sources_within_radius(sky, ref, args.radius_deg)
            sky = sky[sel]
            possum_rm = possum_rm[sel]
            if sky.size == 0:
                raise RuntimeError("After applying --width-deg selection, no sources remain. "
                                "Relax the radius or check the center coordinates.")

    # Match to nearest STAPS pixel
    staps_rm, x_pix, y_pix = nearest_pixel_values(wcs, image, sky)

    # Write a table with results for possible further analysis
    if ref is not None and args.radius_deg is not None:
        possum_subtable = possum[sel]
        possum_subtable['rm_staps'] = staps_rm
        outtable_path = outdir / f"possum_staps_comparison_within_{args.radius_deg:.2f}deg.fits"
        possum_subtable.write(outtable_path, overwrite=True)
        print(f"Wrote comparison table for selected sources to: {outtable_path}")


    # Plot 1: RM_possum vs RM_staps
    scatter_file = plot_scatter_rm(possum_rm, staps_rm, outdir, outname=args.scatter_out, logscale=args.logscale, rm_lim=args.rmlim)
    print(f"Saved scatter: {scatter_file}")

    # Plot 2: residual vs distance (only if reference is provided)
    if (args.ra is not None) and (args.dec is not None):
        ref = SkyCoord(args.ra * u.deg, args.dec * u.deg, frame="icrs")
        distances = compute_distance_to_ref(sky, ref, args.dist_unit)
        resid_file = plot_residual_vs_distance(possum_rm, staps_rm, distances,
                                               unit_label=args.dist_unit, outdir=outdir,
                                               outname=args.resid_out, rm_lim=args.rmlim)
        print(f"Saved residual plot: {resid_file}")
    else:
        print("No --ra/--dec provided: skipped residual-vs-distance plot.")


    # Compute Galactic latitude and make residual-vs-b plot (always produced)
    b_deg = compute_galactic_b(sky)
    b_file = plot_residual_vs_b(possum_rm, staps_rm, b_deg,
                                outdir=outdir, outname=args.b_resid_out, rm_lim=args.rmlim)
    print(f"Saved residual vs b: {b_file}")


if __name__ == "__main__":
    main()
