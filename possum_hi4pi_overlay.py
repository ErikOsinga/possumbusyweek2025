#!/usr/bin/env python3
"""
Overlay POSSUM rotation measures on HI4PI cutouts and sample HI at source positions.

Outputs:
- <possum_catalog_basename>.overlay.png          : HI cutout with POSSUM RMs overplotted
- <possum_catalog_basename>.rm_vs_hi.png         : Scatter of RM vs HI
- <possum_catalog>.withHI.fits                   : Input table plus 'hi_nearest' column

Notes:
- HI4PI file is a HEALPix BINTABLE (NESTED) with a single column containing column density.
- POSSUM table must have columns: 'ra', 'dec' (deg, ICRS) and 'rm'.
- Uses nearest native HEALPix pixel for per-source sampling (no interpolation).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import healpy as hp # type:ignore
import matplotlib.pyplot as plt

from astropy.io import fits # type:ignore
from astropy.table import Table, Column # type:ignore
from astropy.wcs import WCS # type:ignore
from astropy.wcs.utils import skycoord_to_pixel # type:ignore
from astropy.visualization.wcsaxes import Quadrangle # type:ignore
from astropy.coordinates import SkyCoord # type:ignore
import astropy.units as u # type:ignore
import fgutils as futil


def load_objects_csv(path: Path) -> Table:
    """
    Load a CSV of group members and return a table with columns: 'name','ra','dec' (deg).
    Accepts either (radeg, decdeg) or (RA_J2000, Dec_J2000).
    """
    tbl = Table.read(str(path), format="ascii.csv")

    # Resolve RA/Dec in degrees
    if "radeg" in tbl.colnames and "decdeg" in tbl.colnames:
        ra = np.asarray(tbl["radeg"], dtype=float)
        dec = np.asarray(tbl["decdeg"], dtype=float)
    elif "RA_J2000" in tbl.colnames and "Dec_J2000" in tbl.colnames:
        sc = SkyCoord(tbl["RA_J2000"], tbl["Dec_J2000"],
                      unit=(u.hourangle, u.deg), frame="icrs")
        ra = sc.ra.deg
        dec = sc.dec.deg
    else:
        raise ValueError("objects_csv must contain either (radeg,decdeg) or (RA_J2000,Dec_J2000).")

    # Resolve a name column
    if "Name" in tbl.colnames:
        names = [str(x) for x in tbl["Name"]]
    elif "Ref" in tbl.colnames:
        names = [str(x) for x in tbl["Ref"]]
    else:
        # Fallback: synthetic names
        names = [f"Obj{i}" for i in range(len(ra))]

    return Table({"name": names, "ra": ra, "dec": dec})


def select_objects_in_cutout(objects: Table, w: WCS, npix: int) -> Table:
    """
    Return only objects whose world coords fall inside the npix x npix WCS image.
    """
    sc = SkyCoord(objects["ra"] * u.deg, objects["dec"] * u.deg, frame="icrs")
    x, y = w.world_to_pixel(sc)
    mask = (x >= 0) & (x < npix) & (y >= 0) & (y < npix)
    return objects[mask]


def read_hi4pi_healpix_vector(hi4pi_fits: str) -> Tuple[np.ndarray, fits.Header]:
    """
    Read the HI4PI BINTABLE HEALPix file into a 1D numpy array of length 12*nside^2.
    The file has TFIELDS=1, TFORM1 like '1024E', rows chunked, so total length rows*vector_length.
    Replace BAD_DATA sentinel with NaN when present.
    """
    with fits.open(hi4pi_fits, memmap=True) as hdul:
        if len(hdul) < 2 or not isinstance(hdul[1], fits.BinTableHDU):
            raise ValueError("Expected a BINTABLE in extension 1 for HI4PI.")
        hdu = hdul[1]
        hdr = hdu.header
        colname = hdu.columns[0].name
        tbl = hdu.data
        if tbl is None:
            raise ValueError("Empty HI4PI table.")
        data_chunks = tbl[colname]
        vec = np.asarray(data_chunks).reshape(-1).astype(np.float64)
        bad = hdr.get("BAD_DATA")
        if bad is not None:
            vec = np.where(vec == float(bad), np.nan, vec)
        return vec, hdr


def make_tan_wcs(ra_deg: float, dec_deg: float, width_deg: float,
                 npix: int | None = None,
                 pixscale_arcmin: Optional[float] = None) -> Tuple[WCS, fits.Header, int | None]:
    """
    Create a TAN WCS centered at (ra, dec) with a given width.
    Choose resolution via either npix or pixscale_arcmin. If both None, use sensible defaults.
    Returns WCS (celestial only), header, and resolved npix.
    """
    if npix is None and pixscale_arcmin is None:
        npix = 600  # sensible default

    if npix is None and pixscale_arcmin is not None:
        # compute npix from width / pixscale
        pixscale_deg = (pixscale_arcmin / 60.0)
        npix = int(np.clip(np.round(width_deg / pixscale_deg), 64, 4096))

    if npix is not None and pixscale_arcmin is None:
        pixscale_deg = width_deg / float(npix)
    elif npix is not None and pixscale_arcmin is not None:
        # trust npix and recompute pixel scale from it
        pixscale_deg = width_deg / float(npix)

    # Build a minimal FITS header for TAN projection
    hdr = fits.Header()
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = npix
    hdr["NAXIS2"] = npix
    hdr["CTYPE1"] = "RA---TAN"
    hdr["CTYPE2"] = "DEC--TAN"
    hdr["CUNIT1"] = "deg"
    hdr["CUNIT2"] = "deg"
    hdr["CRVAL1"] = float(ra_deg)
    hdr["CRVAL2"] = float(dec_deg)
    hdr["CRPIX1"] = (npix + 1) / 2.0 # type:ignore
    hdr["CRPIX2"] = (npix + 1) / 2.0 # type:ignore
    hdr["CD1_1"] = -pixscale_deg  # RA increases to the left by convention for images
    hdr["CD1_2"] = 0.0
    hdr["CD2_1"] = 0.0
    hdr["CD2_2"] = pixscale_deg

    # Important: use .celestial for RA/Dec work to be safe with redundant axes
    w = WCS(hdr).celestial
    return w, hdr, npix


def world_grid_from_wcs(w: WCS, npix: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized world coordinate grid (RA, Dec in deg) from a square WCS of size npix x npix.
    """
    # pixel grid
    y, x = np.mgrid[0:npix, 0:npix]  # note order: [row, col]
    # world coords
    world = w.pixel_to_world(x, y)
    ra = world.ra.deg
    dec = world.dec.deg
    return ra, dec


def healpix_project_to_tan(vec: np.ndarray, header: fits.Header, w: WCS, npix: int) -> np.ndarray:
    """
    Sample the HEALPix vector onto the TAN grid defined by WCS w and size npix.
    Assumes ORDERING=NESTED per provided header.
    """
    ordering = (header.get("ORDERING", "") or "").strip().upper()
    if ordering not in {"NESTED"}:
        raise ValueError(f"Expected ORDERING=NESTED, got '{ordering}'.")

    npix_hp = vec.size
    nside = hp.npix2nside(npix_hp)

    # Grid of RA/Dec
    ra_deg, dec_deg = world_grid_from_wcs(w, npix)
    # Convert to healpy angles
    theta = np.radians(90.0 - dec_deg)  # colatitude
    phi = np.radians(ra_deg)            # longitude
    # Map to indices
    hp_idx = hp.ang2pix(nside, theta, phi, nest=True)
    cutout = vec[hp_idx].reshape(npix, npix)
    return cutout


def sample_hi_nearest(vec: np.ndarray, header: fits.Header, ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    """
    Sample nearest native HEALPix pixel values at arrays of RA/Dec in degrees.
    """
    ordering = (header.get("ORDERING", "") or "").strip().upper()
    if ordering not in {"NESTED"}:
        raise ValueError(f"Expected ORDERING=NESTED, got '{ordering}'.")
    nside = hp.npix2nside(vec.size)
    theta = np.radians(90.0 - dec_deg)
    phi = np.radians(ra_deg)
    idx = hp.ang2pix(nside, theta, phi, nest=True)
    return vec[idx]


def load_possum_table(path: Path) -> Table:
    tbl = Table.read(str(path))
    for col in ("ra", "dec", "rm"):
        if col not in tbl.colnames:
            raise ValueError(f"Column '{col}' missing from POSSUM table.")
    return tbl


def add_hi_column_and_save(tbl: Table, hi_vals: np.ndarray, out_path: Path) -> None:
    colname = "hi_nearest"
    if colname in tbl.colnames:
        tbl.remove_column(colname)
    tbl.add_column(Column(data=hi_vals.astype(np.float64), name=colname))
    tbl.write(str(out_path), overwrite=True)


def make_overlay_plot(cutout: np.ndarray, w: WCS, tbl: Table, hi_hdr: fits.Header,
                      hi_vmin: Optional[float], hi_vmax: Optional[float],
                      rm_vmin: Optional[float], rm_vmax: Optional[float],
                      out_png: Path,
                      objects_in_field: Path | None = None,
                      box_size_arcmin: float = 7.57
) -> None:
    """
    Plot HI cutout with POSSUM RM overlay. Two colorbars: one for HI, one for RM.
    """
    # Extract units for HI if available
    hi_unit = hi_hdr.get("BUNIT", "HI units")

    fig = plt.figure(figsize=(7.2, 6.6))
    ax = fig.add_subplot(111, projection=w)

    log10 = True

    if log10:
        im = ax.imshow(np.log10(cutout), origin="lower", vmin=np.log10(hi_vmin), vmax=np.log10(hi_vmax), cmap="gray_r")
        cbar_hi = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar_hi.set_label(f"log10( HI ({hi_unit}))")
    else:
        im = ax.imshow(cutout, origin="lower", vmin=hi_vmin, vmax=hi_vmax, cmap="gray_r")
        cbar_hi = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar_hi.set_label(f"HI ({hi_unit})")

    # Overlay POSSUM points, colored by RM
    sc = ax.scatter(tbl["ra"], tbl["dec"], c=tbl["rm"], s=18, cmap="coolwarm",
                    vmin=rm_vmin, vmax=rm_vmax, edgecolor="k", linewidths=0.3,
                    transform=ax.get_transform("world")) # type:ignore
    cbar_rm = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar_rm.set_label("RM (rad m^-2)")

    ax.set_xlabel("RA (ICRS)")
    ax.set_ylabel("Dec (ICRS)")
    ax.set_title("HI4PI cutout with POSSUM RM overlay")

    # Add a quadrangle showing cutout footprint (optional visual cue)
    # (Not strictly necessary, but helps when the field is sparse.)
    # width/height from WCS CD matrix
    # ax.add_patch(Quadrangle((w.wcs.crval[0], w.wcs.crval[1]),
    #                         width=u.deg*np.hypot(w.pixel_scale_matrix[0,0]*cutout.shape[1],
    #                                              w.pixel_scale_matrix[0,1]*cutout.shape[0]),
    #                         height=u.deg*np.hypot(w.pixel_scale_matrix[1,0]*cutout.shape[1],
    #                                               w.pixel_scale_matrix[1,1]*cutout.shape[0]),
    #                         edgecolor="yellow",
    #                         transform=ax.get_transform("world"), fill=False, lw=0.8))

    # Add a compact text box listing group members inside the cutout, if provided
    # Draw group member rectangles if provided
    if objects_in_field is not None and len(objects_in_field) > 0: # type: ignore
        draw_object_rectangles(ax, w, objects_in_field, box_size_arcmin)


    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    # plt.show()
    plt.close(fig)


def make_scatter_plot(tbl: Table, out_png: Path, vmin=None, vmax=None) -> None:
    """
    Scatter of RM vs HI (nearest native pixel).
    """
    mask = np.isfinite(tbl["rm"]) & np.isfinite(tbl["hi_nearest"])
    x = tbl["rm"][mask]
    y = tbl["hi_nearest"][mask]

    fig = plt.figure(figsize=(6.4, 5.6))
    ax = fig.add_subplot(111)
    ax.scatter(x, y, s=16, alpha=0.75)
    ax.set_xlabel("RM (rad m^-2)")
    ax.set_ylabel("HI (nearest native pixel)")
    ax.set_title("POSSUM RM vs HI")
    if vmin is not None: 
        ax.set_xlim(left=vmin)
    if vmax is not None:
        ax.set_xlim(right=vmax)
    
    plt.yscale('log')
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def resolve_output_paths(possum_catalog: Path) -> Tuple[Path, Path, Path]:
    stem = possum_catalog.name
    overlay_png = possum_catalog.with_suffix("").with_name(f"{stem}.overlay.png")
    scatter_png = possum_catalog.with_suffix("").with_name(f"{stem}.rm_vs_hi.png")
    out_fits = possum_catalog.with_suffix(".withHI.fits")
    return overlay_png, scatter_png, out_fits


from matplotlib.patches import Patch  # for legend proxy


def draw_object_rectangles(ax, w: WCS, objects_in_field: Table, box_size_arcmin: float) -> None:
    """
    Draw sky-aligned square boxes centered on each object's RA/Dec.
    Uses WCSAxes. Squares are specified in world-angle units and rendered by WCSAxes.
    """
    if objects_in_field is None or len(objects_in_field) == 0:
        return

    box_w = (box_size_arcmin / 60.0) * u.deg
    box_h = (box_size_arcmin / 60.0) * u.deg
    trans = ax.get_transform("world")

    for ra0, dec0 in zip(objects_in_field["ra"], objects_in_field["dec"]):
        # Lower-left corner of the square in world coords
        ll_ra = (ra0 - 0.5 * box_w.to_value(u.deg)) * u.deg
        ll_dec = (dec0 - 0.5 * box_h.to_value(u.deg)) * u.deg

        rect = Quadrangle((ll_ra, ll_dec), width=box_w, height=box_h,
                          transform=trans, edgecolor="lime", facecolor="none",
                          lw=1.2, zorder=3)
        ax.add_patch(rect)

    # Optional: add a legend proxy so it's clear what the lime boxes are
    proxy = Patch(facecolor="none", edgecolor="lime", lw=1.2)
    leg = ax.legend([proxy], ["Group members"], loc="lower right", frameon=True)
    leg.get_frame().set_alpha(0.7)



def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay POSSUM RMs on HI4PI cutout and sample HI at source positions.")
    p.add_argument("--hi4pi-fits", required=True, help="Path to HI4PI HEALPix BINTABLE FITS (ORDERING=NESTED).")
    p.add_argument("--possum-catalog", required=True, help="Path to POSSUM catalog FITS with 'ra','dec','rm'.")
    p.add_argument("--ra", type=float, required=True, help="Center RA in degrees (ICRS).")
    p.add_argument("--dec", type=float, required=True, help="Center Dec in degrees (ICRS).")
    p.add_argument("--imwidth-deg", type=float, required=True, help="Cutout square width in degrees.")
    p.add_argument("--npix", type=int, default=None, help="Cutout image size in pixels (square). Default 600 if not set and no pixscale is given.")
    p.add_argument("--pixscale-arcmin", type=float, default=None, help="Pixel scale in arcmin. If set without --npix, npix is derived from width.")
    # Color scaling controls
    p.add_argument("--hi-vmin", type=float, default=None, help="vmin for HI color scale.")
    p.add_argument("--hi-vmax", type=float, default=None, help="vmax for HI color scale.")
    p.add_argument("--rm-vmin", type=float, default=None, help="vmin for RM color scale.")
    p.add_argument("--rm-vmax", type=float, default=None, help="vmax for RM color scale.")
    # Output override (optional)
    p.add_argument("--overlay-png", default=None, help="Override output path for overlay PNG.")
    p.add_argument("--scatter-png", default=None, help="Override output path for scatter PNG.")
    p.add_argument("--out-fits", default=None, help="Override output path for augmented POSSUM FITS.")

    # custom
    p.add_argument("--objects-csv", default=None,
                help="Optional CSV of group members (columns radeg,decdeg or RA_J2000,Dec_J2000; Name or Ref for labels).")
    p.add_argument("--box-size-arcmin", type=float, default=7.57,
                help="Side length of the square marker for objects, in arcmin. Default 7.57.")


    return p.parse_args(argv)


def main() -> None:
    args = parse_args()

    hi_vec, hi_hdr = read_hi4pi_healpix_vector(args.hi4pi_fits)

    # Build TAN WCS and project HI to cutout
    w, tan_hdr, npix = make_tan_wcs(args.ra, args.dec, args.imwidth_deg,
                                    npix=args.npix, pixscale_arcmin=args.pixscale_arcmin)
    cutout = healpix_project_to_tan(hi_vec, hi_hdr, w, npix) # type:ignore

    # Load POSSUM table
    possum_path = Path(args.possum_catalog)
    tbl = load_possum_table(possum_path)

    # Sample nearest native HI pixel at each source
    hi_at_sources = sample_hi_nearest(hi_vec, hi_hdr, tbl["ra"].astype(float), tbl["dec"].astype(float))

    # Save augmented table
    overlay_png_default, scatter_png_default, out_fits_default = resolve_output_paths(possum_path)
    out_fits = Path(args.out_fits) if args.out_fits else out_fits_default
    add_hi_column_and_save(tbl.copy(), hi_at_sources, out_fits)

    # Plots
    overlay_png = Path(args.overlay_png) if args.overlay_png else overlay_png_default
    scatter_png = Path(args.scatter_png) if args.scatter_png else scatter_png_default

    objects_in_field = None
    if args.objects_csv:
        objects_all = load_objects_csv(Path(args.objects_csv))
        objects_in_field = select_objects_in_cutout(objects_all, w, npix)


        # Make overlay (two colorbars: HI and RM)
    make_overlay_plot(cutout, w, tbl, hi_hdr,
                      hi_vmin=args.hi_vmin, hi_vmax=args.hi_vmax,
                      rm_vmin=args.rm_vmin, rm_vmax=args.rm_vmax,
                      out_png=overlay_png,
                      objects_in_field=objects_in_field,
                    )

    # The scatter uses the saved 'hi_nearest' to match the sampling policy
    tbl_scatter = tbl.copy()
    if "hi_nearest" not in tbl_scatter.colnames:
        tbl_scatter.add_column(Column(hi_at_sources, name="hi_nearest"))
    make_scatter_plot(tbl_scatter, scatter_png, vmin=args.rm_vmin, vmax=args.rm_vmax)

    print(f"Saved overlay plot: {overlay_png}")
    print(f"Saved scatter plot: {scatter_png}")
    print(f"Saved augmented catalog: {out_fits}")


if __name__ == "__main__":
    main()
