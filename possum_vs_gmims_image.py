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


L_DORADO = 	264.3103
B_DORADO = -43.3928
RA_DORADO = 65.0025000
DEC_DORADO = -54.9380556


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
              cmap: str = "RdBu_r",
              no_cbar: bool = False
) -> None:
    """Display the STAPS RM image with WCSAxes."""
    im = ax.imshow(img, origin="lower", vmin=vmin, vmax=vmax, cmap=cmap,
                #    transform=ax.get_transform(wcs), # type: ignore
                   interpolation="nearest"
    )
    ax.set_xlabel("RA")
    ax.set_ylabel("Dec")
    if no_cbar:
        return
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("RM (rad/m^2)")
    return


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

def sample_staps_rm_at_possum(
    img: np.ndarray,
    wcs: WCS,
    possum: Table,
    ra_col: str,
    dec_col: str,
    rm_col: str,
) -> tuple[np.ndarray, np.ndarray, SkyCoord]:
    """
    For each POSSUM source, sample the STAPS RM map at the nearest pixel.

    Returns:
      rm_possum: RM from POSSUM (float array)
      rm_staps:  RM from STAPS at same positions (float array, NaN if out-of-bounds)
      sky_icrs:  SkyCoord of POSSUM positions (ICRS)
    """
    ra = np.asarray(possum[ra_col], dtype=float)
    dec = np.asarray(possum[dec_col], dtype=float)
    rm_possum = np.asarray(possum[rm_col], dtype=float)

    sky_icrs = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    x, y = wcs.world_to_pixel(sky_icrs)
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)

    ny, nx = img.shape
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)

    rm_staps = np.full_like(rm_possum, np.nan, dtype=float)
    rm_staps[inside] = img[yi[inside], xi[inside]]

    return rm_possum, rm_staps, sky_icrs


from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt

from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.wcs import WCS
from astropy.table import Table


def sample_staps_rm_at_possum(
    img: np.ndarray,
    wcs: WCS,
    possum: Table,
    ra_col: str,
    dec_col: str,
    rm_col: str,
) -> tuple[np.ndarray, np.ndarray, SkyCoord]:
    """
    For each POSSUM source, sample the STAPS RM map at the nearest pixel.

    Returns:
      rm_possum: RM from POSSUM (float array)
      rm_staps:  RM from STAPS at same positions (float array, NaN if out-of-bounds)
      sky_icrs:  SkyCoord of POSSUM positions (ICRS)
    """
    ra = np.asarray(possum[ra_col], dtype=float)
    dec = np.asarray(possum[dec_col], dtype=float)
    rm_possum = np.asarray(possum[rm_col], dtype=float)

    sky_icrs = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs")

    x, y = wcs.world_to_pixel(sky_icrs)
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)

    ny, nx = img.shape
    inside = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)

    rm_staps = np.full_like(rm_possum, np.nan, dtype=float)
    rm_staps[inside] = img[yi[inside], xi[inside]]

    return rm_possum, rm_staps, sky_icrs


def plot_allsky(staps_path: Path,
                possum_path: Path,
                vmin: float,
                vmax: float,
                ra_col: str,
                dec_col: str,
                rm_col: str,
                outname: Optional[str] = None) -> Path:
    """
    Create two all-sky figures.

    Figure 1 (RA/Dec coordinates):
      Left:  STAPS diffuse RM with POSSUM points colored by RM_possum.
      Right: XRM = RM_possum / RM_staps (nearest STAPS pixel) in RA/Dec.

    Figure 2 (Galactic coordinates):
      Left:  POSSUM RM_possum in Galactic (l, b).
      Right: XRM = RM_possum / RM_staps in Galactic (l, b).

    Returns the path to the RA/Dec figure. The Galactic figure is saved
    with a '_galactic' suffix in the same directory.
    """
    img, wcs, _ = load_staps_map(staps_path)
    img_gal, wcs_gal, _ = load_staps_map(staps_path.parent / (staps_path.stem + "_galactic.fits"))
    possum = load_possum_catalog(possum_path, ra_col, dec_col, rm_col)

    # Sample STAPS RM at POSSUM locations and compute XRM
    rm_possum, rm_staps, sky_icrs = sample_staps_rm_at_possum(
        img=img,
        wcs=wcs,
        possum=possum,
        ra_col=ra_col,
        dec_col=dec_col,
        rm_col=rm_col,
    )

    # Avoid divide-by-zero and NaNs for the ratio
    good_ratio = np.isfinite(rm_possum) & np.isfinite(rm_staps) & (rm_staps != 0.0)
    xrm = np.full_like(rm_possum, np.nan, dtype=float)
    xrm[good_ratio] = rm_possum[good_ratio] / rm_staps[good_ratio]

    # RA/Dec figure (both panels in RA/Dec / native WCS)
    outdir = make_plots_dir()
    if outname is None:
        base_name = "staps_possum_allsky"
    else:
        base_name = outname.rsplit(".png", 1)[0]

    fig1 = plt.figure(figsize=(12, 5))

    ############## MARKER SIZE AND ALPHA #########################################################################################
    markersize = 0.1
    markersize = 10
    alpha = 1.0
    ############## MARKER SIZE AND ALPHA #########################################################################################

    # Left panel: original STAPS + POSSUM overlay in native WCS (RA/Dec)
    ax1 = fig1.add_subplot(1, 2, 1, projection=wcs)
    imshow_rm(ax1, img, wcs, vmin=vmin, vmax=vmax)
    scatter_possum(ax1, possum, vmin=vmin, vmax=vmax,
                   ra_col=ra_col, dec_col=dec_col, rm_col=rm_col,
                   size=markersize, alpha=alpha
    )
    ax1.set_title("STAPS RM + POSSUM RM (RA-Dec)")
    # ax1.scatter(RA_DORADO, DEC_DORADO, marker='x', s=5, color='k',alpha=0.5,
    #             transform=ax1.get_transform("world"), # type:ignore
    # )
    circle = plt.Circle((RA_DORADO, DEC_DORADO), 3.0, fill=False, edgecolor='k', 
                        linewidth=1, linestyle='--', alpha=1.0,
                        transform=ax1.get_transform("world"))
    ax1.add_patch(circle)
    print(f"Plotting Dorado at {RA_DORADO=}, {DEC_DORADO=}")

    # Right panel: XRM in RA/Dec using the same WCS projection
    ax2 = fig1.add_subplot(1, 2, 2, projection=wcs, sharex=ax1, sharey=ax1)
    ax2.set_aspect('equal')

    m_xrm = np.isfinite(xrm)
    if np.count_nonzero(m_xrm) > 0:
        xrm_good = xrm[m_xrm]
        # Robust color stretch for XRM
        xrm_lo, xrm_hi = np.nanpercentile(xrm_good, [5, 95])
        xrm_lo, xrm_hi = -1, 3
        sc2 = ax2.scatter(
            sky_icrs.ra.deg[m_xrm],
            sky_icrs.dec.deg[m_xrm],
            s=markersize,
            c=xrm_good,
            cmap="rainbow",
            vmin=xrm_lo,
            vmax=xrm_hi,
            alpha=alpha,
            transform=ax2.get_transform("world"),
        )
        cb2 = plt.colorbar(sc2, ax=ax2, fraction=0.046, pad=0.04)
        cb2.set_label("XRM = RM_possum / RM_staps")

        # also plot dorado
        circle = plt.Circle((RA_DORADO, DEC_DORADO), 3.0, fill=False, edgecolor='k', 
                            linewidth=1, linestyle='--', alpha=1.0,
                            transform=ax2.get_transform("world"))
        ax2.add_patch(circle)

    else:
        ax2.text(0.5, 0.5, "No valid XRM values",
                 ha="center", va="center", transform=ax2.transAxes)

    ax2.set_xlabel("RA")
    ax2.set_ylabel("Dec")
    ax2.set_title("XRM (RA-Dec)")
    ax2.grid(True, linestyle=":", alpha=0.5)



    fig1.tight_layout()
    outfile_radec = outdir / f"{base_name}_radec.png"
    fig1.savefig(outfile_radec, dpi=200)

    if False: # for interactive plotting
        # bottom right corner
        xlim_low, ylim_low = 195.00, -55.00
        # top left corner
        xlim_high, ylim_high = 15*15 , -20.00 # 15d in an hour 

        xlims_world = [xlim_high*u.deg, xlim_low*u.deg] # RA inverted
        ylims_world = [ylim_low*u.deg, ylim_high*u.deg]

        print(f"Setting limits {xlims_world=}")
        print(f"Setting limits {ylims_world=}")
        world_coords = SkyCoord(xlims_world, ylims_world)#, frame=)
        pixel_coords_x, pixel_coords_y = wcs.world_to_pixel(world_coords)

        ax1.set_xlim(pixel_coords_x)
        ax1.set_ylim(pixel_coords_y)
        # ax2.set_xlim()
        plt.show()
    # plt.show()


    plt.close(fig1)

    # Galactic figure (both panels in Galactic l, b, Mollweide)
    sky_gal = sky_icrs.galactic
    l_rad = sky_gal.l.wrap_at(180.0 * u.deg).radian
    b_rad = sky_gal.b.radian

    fig2 = plt.figure(figsize=(12, 5))

    # Left: POSSUM RM in Galactic coordinates
    ax1g = fig2.add_subplot(1, 2, 1, projection="mollweide")
    # imshow_rm(ax1g, img_gal, wcs_gal, vmin=vmin, vmax=vmax)

    m_rm = np.isfinite(rm_possum)
    if np.count_nonzero(m_rm) > 0:
        rm_good = rm_possum[m_rm]
        sc1g = ax1g.scatter(
            l_rad[m_rm],
            b_rad[m_rm],
            s=markersize,
            c=rm_good,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            alpha=alpha,
        )
        cb1g = plt.colorbar(sc1g, ax=ax1g, orientation="horizontal",
                            pad=0.1, fraction=0.05)
        cb1g.set_label("RM_possum (rad/m^2)")
    else:
        ax1g.text(0.0, 0.0, "No POSSUM RM values",
                  ha="center", va="center", transform=ax1g.transAxes)

    ax1g.grid(True, linestyle=":", alpha=0.5)
    ax1g.set_xlabel("Galactic longitude l")
    ax1g.set_ylabel("Galactic latitude b")
    ax1g.set_title("POSSUM RM (Galactic)")

    # Right: XRM in Galactic coordinates
    ax2g = fig2.add_subplot(1, 2, 2, projection="mollweide")
    m_xrm = np.isfinite(xrm)
    if np.count_nonzero(m_xrm) > 0:
        xrm_good = xrm[m_xrm]
        xrm_lo, xrm_hi = np.nanpercentile(xrm_good, [5, 95])
        xrm_lo, xrm_hi = -2, 2
        sc2g = ax2g.scatter(
            l_rad[m_xrm],
            b_rad[m_xrm],
            s=markersize,
            c=xrm_good,
            cmap="rainbow",
            vmin=xrm_lo,
            vmax=xrm_hi,
            alpha=alpha,
        )
        cb2g = plt.colorbar(sc2g, ax=ax2g, orientation="horizontal",
                            pad=0.1, fraction=0.05)
        cb2g.set_label("XRM = RM_possum / RM_staps")
    else:
        ax2g.text(0.0, 0.0, "No valid XRM values",
                  ha="center", va="center", transform=ax2g.transAxes)

    ax2g.grid(True, linestyle=":", alpha=0.5)
    ax2g.set_xlabel("Galactic longitude l")
    ax2g.set_ylabel("Galactic latitude b")
    ax2g.set_title("XRM (Galactic)")

    print(f"Plotting Dorado at {L_DORADO-360=}, {B_DORADO=}")
    ax1g.scatter(np.radians(L_DORADO-360), np.radians(B_DORADO), marker='x', color='k', s=10)
    ax2g.scatter(np.radians(L_DORADO-360), np.radians(B_DORADO), marker='x', color='k', s=10,alpha=0.5)

    fig2.tight_layout()
    outfile_gal = outdir / f"{base_name}_galactic.png"
    fig2.savefig(outfile_gal, dpi=200)
    plt.close(fig2)

    print(f"Saved RA/Dec all-sky figure to: {outfile_radec}")
    print(f"Saved Galactic all-sky figure to: {outfile_gal}")

    # Preserve original return type: return the RA/Dec figure path
    return outfile_radec


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

    print("Background statistics for cutout:")
    print(f"min={np.nanmin(cutout.data):.2f}, max={np.nanmax(cutout.data):.2f}, mean={np.nanmean(cutout.data):.2f}, median={np.nanmedian(cutout.data):.2f} rad/m^2")
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


    # -------------------------
    # Figure 2: XRM overlay on same STAPS cutout
    # -------------------------
    # -------------------------
    # Select POSSUM sources in cutout and compute STAPS RM at those positions
    # -------------------------
    sky_all = SkyCoord(
        ra=np.asarray(possum[ra_col]) * u.deg,
        dec=np.asarray(possum[dec_col]) * u.deg,
        frame="icrs",
    )

    w_cut = cutout.wcs
    x_all, y_all = w_cut.world_to_pixel(sky_all)
    xi_all = np.rint(x_all).astype(int)
    yi_all = np.rint(y_all).astype(int)

    ny, nx = cutout.data.shape
    inside = (
        (xi_all >= 0) & (xi_all < nx) &
        (yi_all >= 0) & (yi_all < ny)
    )

    possum_in = possum[inside]
    xi_in = xi_all[inside]
    yi_in = yi_all[inside]

    # STAPS RM at POSSUM positions inside cutout
    rm_staps_in = cutout.data[yi_in, xi_in]
    rm_possum_in = np.asarray(possum_in[rm_col], dtype=float)

    # XRM = RM_possum / RM_staps, with safe handling of zeros and NaNs
    xrm = np.full_like(rm_possum_in, np.nan, dtype=float)
    good_xrm = np.isfinite(rm_possum_in) & np.isfinite(rm_staps_in) & (rm_staps_in != 0.0)
    xrm[good_xrm] = rm_possum_in[good_xrm] / rm_staps_in[good_xrm]

    fig2 = plt.figure(figsize=(6.5, 6.5))
    ax2 = plt.subplot(projection=cutout.wcs)

    imshow_rm(ax2, cutout.data, cutout.wcs, vmin=vmin, vmax=vmax, no_cbar=True)

    if np.any(np.isfinite(xrm)):
        # Robust stretch for XRM
        xrm_good = xrm[np.isfinite(xrm)]
        xrm_lo, xrm_hi = np.nanpercentile(xrm_good, [5, 95])
        xrm_lo, xrm_hi = -1, 3

        # Plot only sources with finite XRM
        sky_in = SkyCoord(
            ra=np.asarray(possum_in[ra_col]) * u.deg,
            dec=np.asarray(possum_in[dec_col]) * u.deg,
            frame="icrs",
        )
        mask_plot = np.isfinite(xrm)
        sc = ax2.scatter(
            sky_in.ra.deg[mask_plot],
            sky_in.dec.deg[mask_plot],
            s=10.0,
            c=xrm[mask_plot],
            cmap="rainbow",
            vmin=xrm_lo,
            vmax=xrm_hi,
            alpha=1.0,
            transform=ax2.get_transform("world"),
            edgecolors="k",
            linewidths=0.3,
        )
        cb = plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
        cb.set_label("XRM = RM_possum / RM_staps")
    else:
        ax2.text(
            0.5,
            0.5,
            "No valid XRM values",
            transform=ax2.transAxes,
            ha="center",
            va="center",
        )

    ax2.set_title(f"XRM on STAPS RM (cutout {width_deg:.2f} deg)")
    ax2.grid(color="k", alpha=0.2, linestyle=":", linewidth=0.5)

    if objects_csv is not None:
        futil.overlay_groupmembers(ax2, objects_csv, cutout.wcs, header, box_size_arcmin=7.57, no_text=True)

    if outname.endswith(".png"):
        xrm_name = outname[:-4] + "_xrm.png"
    else:
        xrm_name = outname + "_xrm.png"
    outfile2 = outdir / xrm_name

    fig2.tight_layout()
    fig2.savefig(outfile2, dpi=220)
    plt.close(fig2)

    print(f"Saved cutout XRM figure to: {outfile2}")

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
