#!/usr/bin/env python3
"""
Compute local source density (sources per square degree) on user-defined
angular scales for a POSSUM-like catalog.

The script:
- Reads a catalog with at least columns 'ra' and 'dec' in degrees (ICRS).
- For each source and each specified angular scale, counts neighbours within
  that angular radius and converts to a surface density.
- Writes a small output table with RA, DEC, and one column per scale.
- Produces a full-sky Mollweide scatter plot of the local density for each scale.

Example:
    python compute_local_density.py \
        --input-table possum_catalog.ecsv \
        --input-scales-deg 0.5 1.0 1.5 \
        --output-table possum_local_density.ecsv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
from astropy.table import Table # type: ignore
from astropy.coordinates import SkyCoord, search_around_sky     # type: ignore
import astropy.units as u # type: ignore
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute local source density per source on given angular scales."
    )
    parser.add_argument(
        "--input-table",
        type=Path,
        required=True,
        help="Input catalog with at least 'ra' and 'dec' columns in degrees (ICRS).",
    )
    parser.add_argument(
        "--input-scales-deg",
        "--scales-deg",
        nargs="+",
        type=float,
        required=True,
        help="Angular scales in degrees, e.g. --input-scales-deg 0.5 1.0 1.5",
    )
    parser.add_argument(
        "--output-table",
        type=Path,
        default=Path("local_source_density.ecsv"),
        help="Output table filename (format inferred from extension).",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help="Prefix for output plot filenames. "
             "Default: output-table name without extension.",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        help="If set, include the source itself in the neighbour count.",
    )
    return parser.parse_args()


def read_catalog(path: Path) -> Table:
    """
    Read the input catalog using astropy.table.Table.

    The file format is inferred from the extension (fits, ecsv, csv, etc.).
    """
    table = Table.read(path)
    if "ra" not in table.colnames or "dec" not in table.colnames:
        raise ValueError("Input table must contain 'ra' and 'dec' columns (in degrees).")
    return table


def compute_local_densities(
    coords: SkyCoord,
    scales_deg: list[float],
    include_self: bool = False,
) -> Dict[float, np.ndarray]:
    """
    Compute local source density for each source and scale.

    Parameters
    ----------
    coords : SkyCoord
        Source positions in ICRS.
    scales_deg : list of float
        Angular scales in degrees.
    include_self : bool
        If False, the central source is excluded from its own neighbour count.

    Returns
    -------
    densities : dict
        Mapping scale (deg) -> array of densities (sources / deg^2) of length Nsrc.
    """
    n_src = len(coords)
    densities: Dict[float, np.ndarray] = {}

    # Pre-compute conversion factor from steradian to square degree.
    deg2_per_sr = (180.0 / np.pi) ** 2

    for scale in scales_deg:
        max_sep = scale * u.deg

        # Find all pairs of sources within max_sep.
        # idx1: indices of "primary" sources, idx2: indices of neighbours.
        idx1, idx2, d2d, d3d = search_around_sky(coords, coords, max_sep)

        # Optionally drop self-pairs.
        mask = np.ones(len(idx1), dtype=bool)
        if not include_self:
            mask &= idx1 != idx2

        idx1 = idx1[mask]
        # idx2 and d2d not needed for counts, only idx1 matters.

        counts = np.zeros(n_src, dtype=int)
        np.add.at(counts, idx1, 1)

        # Exact spherical cap area on the sphere with radius = scale (in radians).
        theta_rad = np.deg2rad(scale)
        area_sr = 2.0 * np.pi * (1.0 - np.cos(theta_rad))
        area_deg2 = area_sr * deg2_per_sr

        densities[scale] = counts.astype(float) / area_deg2

    return densities


def build_output_table(
    base_table: Table,
    densities: Dict[float, np.ndarray],
    scales_deg: list[float],
) -> Table:
    """
    Construct a compact output table with RA, DEC and density columns.
    """
    out = Table()
    out["ra"] = base_table["ra"]
    out["dec"] = base_table["dec"]

    for scale in scales_deg:
        colname = f"rho_deg2_r{scale:g}"
        out[colname] = densities[scale]

    return out


def make_mollweide_plots(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    densities: Dict[float, np.ndarray],
    scales_deg: list[float],
    output_prefix: str,
) -> None:
    """
    Make a full-sky Mollweide scatter plot of local density for each scale.
    """
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    # Convert RA to the [-pi, +pi] range and flip longitude for Mollweide.
    ra_wrap = np.remainder(ra_rad + 2.0 * np.pi, 2.0 * np.pi)
    ra_plot = -(ra_wrap - np.pi)

    for scale in scales_deg:
        vals = densities[scale]

        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111, projection="mollweide")

        sc = ax.scatter(ra_plot, dec_rad, c=vals, s=5, alpha=0.8)
        ax.grid(True)

        cbar = fig.colorbar(sc, ax=ax, pad=0.05)
        cbar.set_label(f"Local source density (sources / deg^2) within {scale} deg")

        ax.set_title(f"Local source density within {scale} deg")
        fig.tight_layout()

        out_name = f"{output_prefix}_mollweide_r{scale:g}deg.png"
        fig.savefig(out_name, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()

    table = read_catalog(args.input_table)
    coords = SkyCoord(
        ra=table["ra"].astype(float),## * u.deg,
        dec=table["dec"].astype(float),## * u.deg,
        frame="icrs",
    )

    scales_deg = list(args.input_scales_deg)

    densities = compute_local_densities(
        coords=coords,
        scales_deg=scales_deg,
        include_self=args.include_self,
    )

    out_table = build_output_table(table, densities, scales_deg)
    out_table.write(args.output_table, overwrite=True)

    if args.output_prefix is not None:
        prefix = args.output_prefix
    else:
        prefix = str(args.output_table.with_suffix(""))

    make_mollweide_plots(
        ra_deg=np.array(table["ra"], dtype=float),
        dec_deg=np.array(table["dec"], dtype=float),
        densities=densities,
        scales_deg=scales_deg,
        output_prefix=prefix,
    )


if __name__ == "__main__":
    main()
