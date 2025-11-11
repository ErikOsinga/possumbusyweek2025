#!/usr/bin/env python3
"""
staps_possum_overlay.py

Overlay POSSUM rotation measures on the STAPS diffuse RM map.

Usage examples:
  # All-sky with defaults
  python staps_possum_overlay.py --possum-catalog /path/to/possum_catalog.fits

  # All-sky with custom color stretch
  python staps_possum_overlay.py --possum-catalog possum.ecsv --vmin -100 --vmax 100

  # Cutout around a target position (square size in degrees)
  python staps_possum_overlay.py --possum-catalog possum.fits \
      --ra 201.365 --dec -43.019 --imwidth-deg 10
"""
from __future__ import annotations

import argparse
from pathlib import Path

import warnings 

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits # type: ignore
from astropy.wcs import WCS # type: ignore
from astropy.nddata import Cutout2D # type: ignore
from astropy.coordinates import SkyCoord # type: ignore
import astropy.units as u # type: ignore
from astropy.units import UnitsWarning # type: ignore
from astropy.table import Table # type: ignore
from astropy.wcs import FITSFixedWarning

import fgutils as futil

warnings.simplefilter("ignore", FITSFixedWarning)
warnings.simplefilter("ignore", UnitsWarning)



# ----------------------------
# I/O and data prep utilities
# ----------------------------

def load_staps_map(fpath: Path) -> tuple[np.ndarray, WCS, fits.Header]:
    """Load STAPS RM FITS image and return (data, celestial WCS, header)."""
    with fits.open(fpath) as hdul:
        # Try to find the first 2D image HDU with celestial WCS
        hdu = None
        for h in hdul:
            if hasattr(h, "data") and h.data is not None and h.data.ndim >= 2:
                hdu = h
                break
        if hdu is None:
            raise ValueError("No image HDU with data found in STAPS FITS file.")
        data = np.squeeze(hdu.data)
        header = hdu.header
        wcs = WCS(header).celestial  # use celestial subset to handle redundant axes
    return data, wcs, header


def load_possum_catalog(fpath: Path,
                        ra_col: str = "ra",
                        dec_col: str = "dec",
                        rm_col: str = "rm") -> Table:
    """Load POSSUM catalog (Table.read infers format) and return Astropy Table."""
    tbl = Table.read(fpath)
    # Basic sanity checks
    for col in (ra_col, dec_col, rm_col):
        if col not in tbl.colnames:
            raise KeyError(f"Missing required column '{col}' in POSSUM catalog.")
    return tbl


# ----------------------------
# Plotting helpers
# ----------------------------

def make_plots_dir() -> Path:
    outdir = Path("./plots")
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def imshow_rm(ax: plt.Axes,
              img: np.ndarray,
              wcs: WCS,
              vmin: float,
              vmax: float,
              cmap: str = "RdBu_r") -> None:
    """Display the STAPS RM image with WCSAxes."""
    im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap,
                #    transform=ax.get_transform(wcs), # type: ignore
                   interpolation="nearest"
    )
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RM (rad/m^2)")


def scatter_possum(ax: plt.Axes,
                   tbl: Table,
                   vmin: float,
                   vmax: float,
                   ra_col: str = "ra",
                   dec_col: str = "dec",
                   rm_col: str = "rm",
                   size: float = 10.0,
                   alpha: float = 0.9,
                   cmap: str = "RdBu_r") -> None:
    """Overlay POSSUM points colored by RM."""
    # Build sky coordinates; catalog RA/Dec are ICRS in degrees.
    sky = SkyCoord(ra=np.asarray(tbl[ra_col]) * u.deg,
                   dec=np.asarray(tbl[dec_col]) * u.deg,
                   frame="icrs")
    rms = np.asarray(tbl[rm_col])

    # Mask out NaNs
    m = np.isfinite(rms) & np.isfinite(sky.ra.deg) & np.isfinite(sky.dec.deg)
    sky = sky[m]
    rms = rms[m]

    sc = ax.scatter(sky.ra.deg, sky.dec.deg, s=size, c=rms, cmap=cmap,
                    vmin=vmin, vmax=vmax, alpha=alpha,
                    transform=ax.get_transform("world"), # type:ignore
                    linewidths=0.1, edgecolor='k')
    # Add a small colorbar for points if desired (kept single shared bar via imshow)
    # Could be enabled by user, but we keep the image colorbar only to avoid duplicates.


def plot_allsky(staps_path: Path,
                possum_path: Path,
                vmin: float,
                vmax: float,
                ra_col: str,
                dec_col: str,
                rm_col: str,
                outname: str | None = None) -> Path:
    """Create all-sky overlay figure."""
    img, wcs, _ = load_staps_map(staps_path)
    possum = load_possum_catalog(possum_path, ra_col, dec_col, rm_col)

    fig = plt.figure(figsize=(10, 6))
    ax = plt.subplot(projection=wcs)

    imshow_rm(ax, img, wcs, vmin=vmin, vmax=vmax)
    scatter_possum(ax, possum, vmin=vmin, vmax=vmax,
                   ra_col=ra_col, dec_col=dec_col, rm_col=rm_col)

    ax.set_title("STAPS RM + POSSUM RM overlay (all-sky)")
    outdir = make_plots_dir()
    if outname is None:
        outname = "staps_possum_allsky.png"
    outfile = outdir / outname
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    return outfile


def plot_cutout(staps_path: Path,
                possum_path: Path,
                center_ra: float,
                center_dec: float,
                width_deg: float,
                vmin: float,
                vmax: float,
                ra_col: str,
                dec_col: str,
                rm_col: str,
                objects_csv: Path | None = None,
                outname: str | None = None) -> Path:
    """Create a cutout overlay figure centered at (RA, Dec) with square width_deg."""
    img, wcs, header = load_staps_map(staps_path)
    possum = load_possum_catalog(possum_path, ra_col, dec_col, rm_col)

    center = SkyCoord(center_ra * u.deg, center_dec * u.deg, frame="icrs")

    # Build cutout (square)
    cutout = Cutout2D(data=img, position=center, size=(width_deg * u.deg, width_deg * u.deg),
                      wcs=wcs)
    fig = plt.figure(figsize=(6.5, 6.5))
    ax = plt.subplot(projection=cutout.wcs)

    imshow_rm(ax, cutout.data, cutout.wcs, vmin=vmin, vmax=vmax)

    # Overlay only the POSSUM points that fall inside the cutout footprint
    sky_all = SkyCoord(ra=np.asarray(possum[ra_col]) * u.deg,
                       dec=np.asarray(possum[dec_col]) * u.deg,
                       frame="icrs")
    # Footprint filter via pixel coords in cutout WCS
    # Use celestial subset to ensure consistent axes
    w = cutout.wcs
    x, y = w.world_to_pixel(sky_all)
    inside = (
        (x >= 0) & (x < cutout.data.shape[1]) &
        (y >= 0) & (y < cutout.data.shape[0])
    )
    possum_in = possum[inside]
    scatter_possum(ax, possum_in, vmin=vmin, vmax=vmax,
                   ra_col=ra_col, dec_col=dec_col, rm_col=rm_col)

    # Decorations
    ax.set_title(f"STAPS RM + POSSUM RM overlay (cutout {width_deg:.2f} deg)")
    ax.grid(color="k", alpha=0.2, linestyle=":", linewidth=0.5)

    # optional overplot of objects from CSV
    if objects_csv is not None:
        futil.overlay_groupmembers(ax, objects_csv, cutout.wcs, header, box_size_arcmin=7.57)

    outdir = make_plots_dir()
    if outname is None:
        outname = f"staps_possum_cutout_ra{center_ra:.3f}_dec{center_dec:.3f}_w{width_deg:.2f}.png"
    outfile = outdir / outname
    fig.tight_layout()
    fig.savefig(outfile, dpi=220)
    plt.close(fig)
    return outfile


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay POSSUM RM points on STAPS diffuse RM map.")
    p.add_argument("--staps-fits", type=Path,
                   default=Path("/home/osingae/Documents/postdoc-no-onedrive/doradogroup/other_wavelengths/RMp.fits"),
                   help="Path to STAPS RM FITS image (default: your provided RMp.fits path).")
    p.add_argument("--possum-catalog", type=Path, required=True,
                   help="Path to POSSUM catalog (Astropy-readable format: FITS/ECSV/CSV).")

    # Column names (in case your table uses different headers)
    p.add_argument("--ra-col", type=str, default="ra", help="RA column name in POSSUM table (deg).")
    p.add_argument("--dec-col", type=str, default="dec", help="Dec column name in POSSUM table (deg).")
    p.add_argument("--rm-col", type=str, default="rm", help="RM column name in POSSUM table (rad/m^2).")

    # Color stretch
    p.add_argument("--vmin", type=float, default=-50.0, help="Lower RM for colormap (rad/m^2).")
    p.add_argument("--vmax", type=float, default=50.0, help="Upper RM for colormap (rad/m^2).")
    p.add_argument("--cmap", type=str, default="RdBu_r", help="Matplotlib colormap name.")

    # Cutout options: when all three are given, do a cutout instead of all-sky
    p.add_argument("--ra", type=float, help="Center RA in degrees (ICRS).")
    p.add_argument("--dec", type=float, help="Center Dec in degrees (ICRS).")
    p.add_argument("--imwidth-deg", type=float, help="Cutout square width in degrees.")

    # Custom options
    p.add_argument('--objects_csv', default=None, help='Path to CSV file to overplot text', type=Path)

    # Output name (optional)
    p.add_argument("--outname", type=str, default=None, help="Optional output filename for the PNG.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Quick sanity for the stretch
    if args.vmin >= args.vmax:
        raise ValueError(f"--vmin ({args.vmin}) must be < --vmax ({args.vmax}).")

    # Choose mode: cutout if RA, Dec, and width are all provided
    do_cutout = (args.ra is not None) and (args.dec is not None) and (args.imwidth_deg is not None)

    if do_cutout:
        outfile = plot_cutout(
            staps_path=args.staps_fits,
            possum_path=args.possum_catalog,
            center_ra=args.ra,
            center_dec=args.dec,
            width_deg=args.imwidth_deg,
            vmin=args.vmin,
            vmax=args.vmax,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            rm_col=args.rm_col,
            objects_csv=args.objects_csv,
            outname=args.outname,
        )
        print(f"Saved cutout plot to: {outfile}")
    else:
        outfile = plot_allsky(
            staps_path=args.staps_fits,
            possum_path=args.possum_catalog,
            vmin=args.vmin,
            vmax=args.vmax,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            rm_col=args.rm_col,
            outname=args.outname,
        )
        print(f"Saved all-sky plot to: {outfile}")


if __name__ == "__main__":
    main()
