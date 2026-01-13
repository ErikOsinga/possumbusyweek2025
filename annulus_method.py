#!/usr/bin/env python3
"""
annulus_method.py, idea from Craig Anderson + 2024

Implementation of an aperture-based (annulus) method to estimate the
foreground Faraday rotation signal for each source in a POSSUM catalogue.

For each source:
- Exclude an inner circular region (exclusion_radius_deg) to avoid subtracting
  coherent / local signal.
- In the remaining sky, find the n_nearest neighboring RM sources.
- Take the median of their RMs as the "foreground RM" for that source.
- Record the outer radius of that neighbor set (distance to the furthest of
  the selected neighbors).

Typical usage (inside another script):

    from astropy.table import Table
    from annulus_method import compute_annulus_foreground

    possum = Table.read("possum.fits", format="fits")
    fg_rm, annulus_r = compute_annulus_foreground(possum,
                                                  ra_col="RA",
                                                  dec_col="Dec",
                                                  rm_col="rm")

    possum["rm_fg_annulus"] = fg_rm
    possum["rm_fg_radius_deg"] = annulus_r
"""

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _build_coord_array(possum_table: Table, ra_col: str, dec_col: str) -> SkyCoord:
    """
    Build a SkyCoord array from the POSSUM table.

    Parameters
    ----------
    possum_table : Table
        Input catalogue with RA and Dec columns.
    ra_col : str
        Name of the right ascension column, in degrees.
    dec_col : str
        Name of the declination column, in degrees.

    Returns
    -------
    coords : SkyCoord
        Sky positions of all sources.
    """
    ra = np.array(possum_table[ra_col], dtype=float)
    dec = np.array(possum_table[dec_col], dtype=float)
    return SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")


def _compute_separation_matrix(coords: SkyCoord) -> np.ndarray:
    """
    Compute the full pairwise separation matrix in degrees.

    Parameters
    ----------
    coords : SkyCoord
        Sky positions of all sources.

    Returns
    -------
    sep_deg : ndarray, shape (N, N)
        Angular separations in degrees between all pairs.
    """
    # coords[:, None] has shape (N, 1)
    # coords[None, :] has shape (1, N)
    # separation() broadcasts to (N, N)
    sep = coords[:, None].separation(coords[None, :])
    return sep.to(u.deg).value


def compute_annulus_foreground(
    possum_table: Table,
    ra_col: str = "RA",
    dec_col: str = "Dec",
    rm_col: str = "rm",
    exclusion_radius_deg: float = 0.4,
    n_nearest: int = 40,
):
    """
    Compute the annulus-based foreground RM estimate for each source.

    For each source i:
    - Consider all other sources j.
    - Exclude those with angular separation < exclusion_radius_deg.
    - Among the remaining sources, select the n_nearest neighbors by distance.
    - Take the median of their RM values as the foreground RM for source i.
    - Define the outer radius as the separation to the furthest of those
      selected neighbors.

    If fewer than n_nearest neighbors exist outside the exclusion zone,
    use all available neighbors. If no neighbor survives, the foreground
    RM and outer radius are set to NaN for that source.

    Parameters
    ----------
    possum_table : Table
        Input POSSUM catalogue with at least RA, Dec, and RM columns.
    ra_col : str, optional
        Name of the right ascension column in degrees. Default "RA".
    dec_col : str, optional
        Name of the declination column in degrees. Default "Dec".
    rm_col : str, optional
        Name of the RM column. Default "rm".
    exclusion_radius_deg : float, optional
        Inner exclusion radius around each source, in degrees, inside which
        neighbors are ignored. Default 0.4 deg.
    n_nearest : int, optional
        Number of nearest neighbors to use for the foreground RM, outside
        the exclusion radius. Default 40.

    Returns
    -------
    fg_rm : ndarray, shape (N,)
        Foreground RM estimate for each source (median of neighbors).
    annulus_outer_radius_deg : ndarray, shape (N,)
        Outer radius of the annulus for each source = distance to the
        furthest neighbor used in the median, in degrees.
    """
    if n_nearest <= 0:
        raise ValueError("n_nearest must be a positive integer.")

    # Extract RM values as a plain numpy array
    rm_all = np.array(possum_table[rm_col], dtype=float)
    n_sources = rm_all.size

    # Build coordinate array and separation matrix
    coords = _build_coord_array(possum_table, ra_col=ra_col, dec_col=dec_col)
    sep_deg = _compute_separation_matrix(coords)

    # Prepare outputs
    fg_rm = np.full(n_sources, np.nan, dtype=float)
    annulus_outer_radius_deg = np.full(n_sources, np.nan, dtype=float)

    # For each source, find neighbors outside exclusion radius and
    # then select up to n_nearest closest neighbors.
    for i in range(n_sources):
        # Distances from source i to all sources j
        dists = sep_deg[i]

        # Exclude self and all sources within the exclusion radius
        mask_valid = dists >= exclusion_radius_deg

        # If nothing survives, we leave NaNs
        if not np.any(mask_valid):
            continue

        valid_dists = dists[mask_valid]
        valid_rms = rm_all[mask_valid]

        # Sort neighbors by distance
        order = np.argsort(valid_dists)
        k = min(n_nearest, valid_dists.size)

        nearest_dists = valid_dists[order[:k]]
        nearest_rms = valid_rms[order[:k]]

        # Compute foreground RM as median of neighbor RMs
        fg_rm[i] = np.median(nearest_rms)

        # Outer radius of this annulus: max distance among selected neighbors
        annulus_outer_radius_deg[i] = np.max(nearest_dists)

    return fg_rm, annulus_outer_radius_deg


def add_annulus_columns(
    possum_table: Table,
    ra_col: str = "ra",
    dec_col: str = "dec",
    rm_col: str = "rm",
    exclusion_radius_deg: float = 0.4,
    n_nearest: int = 40,
    fg_col_name: str = "rm_fg_annulus",
    radius_col_name: str = "rm_fg_radius_deg",
    exclusion_radius_col_name: str = "exclusion_radius_deg",
) -> Table:
    """
    Convenience wrapper that computes the annulus foreground RM and adds
    the results as new columns to the input table.

    Parameters
    ----------
    possum_table : Table
        Input POSSUM catalogue.
    ra_col, dec_col, rm_col : str, optional
        Column names for RA, Dec (deg), and RM.
    exclusion_radius_deg : float, optional
        Inner exclusion radius in degrees.
    n_nearest : int, optional
        Number of neighbors to use in the annulus.
    fg_col_name : str, optional
        Name of the new column to store the foreground RM.
    radius_col_name : str, optional
        Name of the new column to store the outer radius of the annulus.

    Returns
    -------
    possum_table : Table
        The same Table, with two extra columns added.
    """
    fg_rm, annulus_r = compute_annulus_foreground(
        possum_table,
        ra_col=ra_col,
        dec_col=dec_col,
        rm_col=rm_col,
        exclusion_radius_deg=exclusion_radius_deg,
        n_nearest=n_nearest,
    )

    possum_table[fg_col_name] = fg_rm
    possum_table[radius_col_name] = annulus_r
    possum_table[exclusion_radius_col_name] = exclusion_radius_deg

    return possum_table

def _plot_diagnostics(
    possum_table: Table,
    exclusion_radius_deg: float,
    mean_outer_radius: float,
    n_nearest: int,
    ra_col: str = "ra",
    dec_col: str = "dec",
    fg_col_name: str = "rm_fg_annulus",
):
    """
    Diagnostic plots:
    - RA/Dec scatter coloured by annulus foreground RM.
    - Spherical circle (small circle on the sky) at median RA/Dec
      with radius equal to exclusion_radius_deg.
    - Histogram of foreground RMs.
    """
    ra = np.array(possum_table[ra_col], dtype=float)
    dec = np.array(possum_table[dec_col], dtype=float)
    fg = np.array(possum_table[fg_col_name], dtype=float)

    # First figure: RA/Dec coloured by foreground RM
    plt.figure()
    plt.title(
        f"Exclusion radius = {exclusion_radius_deg} deg, n_nearest = {n_nearest}"
    )
    sc = plt.scatter(
        ra,
        dec,
        c=fg,
        cmap="RdBu_r",
        vmin=-40,
        vmax=40,
        s=10,
        alpha=0.8,
    )
    cb = plt.colorbar(sc, label="Rotation Measure (rad/m^2)")
    cb.ax.set_ylabel("Rotation Measure (rad/m^2)")

    # Spherical circle around the median RA/Dec
    median_ra = np.nanmedian(ra)
    median_dec = np.nanmedian(dec)

    center = SkyCoord(
        ra=median_ra * u.deg,
        dec=median_dec * u.deg,
        frame="icrs",
    )
    radius = exclusion_radius_deg * u.deg

    pa = np.linspace(0.0, 2.0 * np.pi, 361) * u.rad
    circle_coords = center.directional_offset_by(pa, radius)

    plt.plot(circle_coords.ra.deg, circle_coords.dec.deg, "k--", linewidth=1.2)

    # also the mean outer radius
    circle_coords = center.directional_offset_by(pa, mean_outer_radius*u.deg)
    plt.plot(circle_coords.ra.deg, circle_coords.dec.deg, "k-", linewidth=1.2)


    plt.xlabel("RA (deg)")
    plt.ylabel("Dec (deg)")
    plt.gca().invert_xaxis()  # optional: usual astronomical convention
    plt.tight_layout()
    plt.show()

    # Second figure: histogram of foreground RMs
    plt.figure()
    mask_finite = np.isfinite(fg)
    plt.hist(fg[mask_finite], bins=51)
    plt.xlabel("Rotation measure [rad/m^2]")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Simple CLI / sanity-check usage:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compute annulus-based foreground RM estimates for a POSSUM RM catalogue."
        )
    )
    parser.add_argument("possum_fits", type=str, help="Input POSSUM FITS catalogue.")
    parser.add_argument("--snrcut", type=float, default=8.0)
    parser.add_argument(
        "--ra-col", type=str, default="ra", help="RA column name in degrees."
    )
    parser.add_argument(
        "--dec-col", type=str, default="dec", help="Dec column name in degrees."
    )
    parser.add_argument(
        "--rm-col", type=str, default="rm", help="RM column name."
    )
    parser.add_argument(
        "--exclusion-radius",
        type=float,
        default=0.4,
        help="Exclusion radius in degrees around each source.",
    )
    parser.add_argument(
        "--n-nearest",
        type=int,
        default=40,
        help="Number of nearest neighbors outside exclusion radius.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output FITS file with added columns.",
    )
    parser.add_argument(
        "--doplot", action="store_true", help="Plot result"
    )

    args = parser.parse_args()

    # Read input table
    possum_cat = Table.read(args.possum_fits, format="fits")
    print(f"Loaded table with {len(possum_cat)} sources")

    possum_cat = possum_cat[possum_cat['SNR_PI'] > args.snrcut]
    print(f"Found {len(possum_cat)} sources with snr > {args.snrcut}")

    # Add annulus columns
    possum_cat = add_annulus_columns(
        possum_cat,
        ra_col=args.ra_col,
        dec_col=args.dec_col,
        rm_col=args.rm_col,
        exclusion_radius_deg=args.exclusion_radius,
        n_nearest=args.n_nearest,
    )

    # Print some basic diagnostics
    mean_radius = np.nanmean(possum_cat["rm_fg_radius_deg"])
    print(f"Computed annulus foreground RM for {len(possum_cat)} sources.")
    print(f"Mean outer radius of annuli: {mean_radius:.3f} deg = {mean_radius*60:.1f} arcmin")

    # Optionally write output
    if args.output is not None:
        possum_cat.write(args.output, format="fits", overwrite=True)
        print(f"Wrote output table with annulus columns to {args.output}")

    if args.doplot:
        _plot_diagnostics(
            possum_cat,
            exclusion_radius_deg=args.exclusion_radius,
            mean_outer_radius=mean_radius,
            n_nearest=args.n_nearest,
            ra_col=args.ra_col,
            dec_col=args.dec_col,
            fg_col_name="rm_fg_annulus",
        )