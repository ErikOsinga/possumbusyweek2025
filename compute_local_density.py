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
from scipy.spatial import cKDTree # type: ignore

from collections import defaultdict


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

    parser.add_argument(
        "--deduplicate",
        action="store_true",
        help="Filter and de-duplicate catalogue before computing densities.",
    )
    parser.add_argument(
        "--dedup-snr-thresh",
        type=float,
        default=6.5,
        help="SNR_PI threshold used during de-duplication (default: 6.5).",
    )
    parser.add_argument(
        "--dedup-fracpol-thresh",
        type=float,
        default=0.005,
        help="Fractional polarisation threshold used during de-duplication (default: 0.005).",
    )
    parser.add_argument(
        "--dedup-arcsec",
        type=float,
        default=3.0,
        help="Maximum separation in arcsec within which sources are considered duplicates (default: 3.0).",
    )
    parser.add_argument(
        "--dedup-keep-rule",
        type=str,
        default="max_snr",
        choices=["max_snr", "min_rm_err", "first"],
        help='Rule for which source to keep in each duplicate cluster '
             '(default: "max_snr").',
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

def filter_and_dedup_fast(
    table: Table,
    snr_thresh: float = 6.5,
    fracpol_thresh: float = 0.005,
    dup_arcsec: float = 3.0,
    keep_rule: str = "max_snr",
) -> tuple[Table, dict]:
    """
    Filter and de-duplicate a table of polarised sources based on angular proximity.

    Parameters
    ----------
    table : astropy Table
        Input catalogue with at least 'ra', 'dec', 'SNR_PI', 'fracpol',
        and 'rm_err' columns.

    snr_thresh : float
        Minimum required signal-to-noise ratio in polarised intensity.

    fracpol_thresh : float
        Minimum required fractional polarisation.

    dup_arcsec : float
        Maximum separation (arcseconds) within which sources are considered duplicates.

    keep_rule : str
        Rule to select which entry to keep within each duplicate cluster:
          - "max_snr": Keep the source with the highest SNR_PI.
          - "min_rm_err": Keep the source with the lowest RM uncertainty.
          - "first": Keep the first source encountered in each cluster.

    Returns
    -------
    tbl : astropy Table
        The filtered and de-duplicated source catalogue.

    summary : dict
        Summary statistics with counts before and after filtering and de-duplication.
    """
    required_cols = ["ra", "dec", "SNR_PI", "fracpol", "rm_err"]
    missing = [c for c in required_cols if c not in table.colnames]
    if missing:
        raise ValueError(
            f"--deduplicate requested but required columns are missing: {missing}"
        )

    summary: dict = {}
    summary["initial"] = len(table)

    print("De-duplicating...")

    # Work on a copy to avoid surprising side effects
    tbl = table.copy()

    # 1. Filter by SNR and fractional polarisation
    sel = (tbl["SNR_PI"] > snr_thresh) & (tbl["fracpol"] > fracpol_thresh)
    tbl = tbl[sel]
    summary["after_snr_fracpol"] = len(tbl)

    if len(tbl) == 0:
        summary["after_dedup"] = 0
        print("No sources pass SNR/fracpol cuts; nothing to de-duplicate.")
        return tbl, summary

    # 2. Find all nearby source pairs
    coords = SkyCoord(
        ra=tbl["ra"].astype(float),# * u.deg,
        dec=tbl["dec"].astype(float),# * u.deg,
        frame="icrs",
    )

    idx1, idx2, sep2d, _ = search_around_sky(coords, coords, dup_arcsec * u.arcsec)

    # Remove self-matches and keep i < j to avoid duplicate edges
    m = idx1 < idx2
    i1 = idx1[m]
    i2 = idx2[m]

    # If no close pairs, just return the filtered table
    if len(i1) == 0:
        summary["after_dedup"] = len(tbl)
        print("No duplicates found within "
              f"{dup_arcsec} arcsec; returning filtered catalogue.")
        return tbl, summary

    # 3. Group nearby sources using Union-Find
    parent = np.arange(len(tbl), dtype=int)
    rank = np.zeros(len(tbl), dtype=int)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra_root, rb_root = find(a), find(b)
        if ra_root == rb_root:
            return
        if rank[ra_root] < rank[rb_root]:
            parent[ra_root] = rb_root
        elif rank[ra_root] > rank[rb_root]:
            parent[rb_root] = ra_root
        else:
            parent[rb_root] = ra_root
            rank[ra_root] += 1

    for a, b in zip(i1, i2):
        union(a, b)

    roots = np.array([find(k) for k in range(len(tbl))])

    groups: dict[int, list[int]] = defaultdict(list)
    for k, r in enumerate(roots):
        groups[r].append(k)

    keep_idx: list[int] = []

    if keep_rule == "max_snr":
        snr = np.asarray(tbl["SNR_PI"])
        for g in groups.values():
            keep_idx.append(g[int(np.argmax(snr[g]))])

    elif keep_rule == "min_rm_err":
        sig = np.asarray(tbl["rm_err"])
        for g in groups.values():
            gfinite = [i for i in g if np.isfinite(sig[i])]
            if gfinite:
                # argmin over the finite subset
                keep_idx.append(gfinite[int(np.argmin(sig[gfinite]))])
            else:
                keep_idx.append(g[0])

    else:  # "first"
        for g in groups.values():
            keep_idx.append(g[0])

    keep_idx = np.unique(keep_idx) # type: ignore
    tbl = tbl[keep_idx]
    summary["after_dedup"] = len(tbl)

    print("De-duplicated: "
          f"{summary['after_snr_fracpol']} -> {summary['after_dedup']} sources.")

    return tbl, summary


def compute_local_densities_old(
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

def compute_local_densities(
    coords: SkyCoord,
    scales_deg: list[float],
    include_self: bool = False,
) -> dict[float, np.ndarray]:
    """
    Compute local source density for each source and scale using a cKDTree
    on 3D unit vectors (memory efficient).

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
    densities: dict[float, np.ndarray] = {}

    # Convert to 3D Cartesian unit vectors on the unit sphere
    # (using astropy to keep the spherical geometry correct).
    cart = coords.cartesian
    xyz = np.vstack((cart.x.value, cart.y.value, cart.z.value)).T  # shape (N, 3)

    # Build KD-tree once
    tree = cKDTree(xyz)

    # Pre-compute conversion factor from steradian to square degree.
    deg2_per_sr = (180.0 / np.pi) ** 2

    for scale in scales_deg:
        theta_rad = np.deg2rad(scale)

        # Euclidean chord distance corresponding to angular separation theta:
        # d = 2 * sin(theta / 2) for unit vectors.
        r_chord = 2.0 * np.sin(0.5 * theta_rad)

        # Query neighbours for all points at once.
        # This returns a list of arrays: neighbours[i] are indices around source i.
        neighbours = tree.query_ball_point(xyz, r=r_chord)

        # Count neighbours for each source.
        counts = np.fromiter(
            (
                len(idx_list) if include_self else max(len(idx_list) - 1, 0)
                for idx_list in neighbours
            ),
            dtype=int,
            count=n_src,
        )

        # Exact spherical cap area in steradians and then in deg^2.
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
    coords: SkyCoord,
    densities: dict[float, np.ndarray],
    scales_deg: list[float],
    output_prefix: str,
) -> None:
    """
    Make full-sky Mollweide scatter plots of local density for each scale,
    in both ICRS (RA/Dec) and Galactic (l, b).
    """
    # Prepare coordinate transforms once
    icrs = coords.icrs
    gal = coords.galactic

    # Equatorial (ICRS)
    ra_wrap = icrs.ra.wrap_at(180.0 * u.deg).radian
    dec_rad = icrs.dec.radian
    x_icrs = -ra_wrap  # flip longitude for Mollweide (RA increasing to the left)
    y_icrs = dec_rad

    # Galactic
    l_wrap = gal.l.wrap_at(180.0 * u.deg).radian
    b_rad = gal.b.radian
    x_gal = -l_wrap
    y_gal = b_rad

    cmap = plt.get_cmap("viridis")
    cmap = plt.get_cmap("rainbow")
    # cmap = plt.get_cmap("inferno")

    markersize=3

    for scale in scales_deg:
        vals = densities[scale]

        # ---------- ICRS plot ----------
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111, projection="mollweide")

        sc = ax.scatter(x_icrs, y_icrs, c=vals, s=markersize, alpha=0.8, cmap=cmap)
        ax.grid(True)

        cbar = fig.colorbar(sc, ax=ax, pad=0.05)
        cbar.set_label(f"Local source density (sources / deg^2) within {scale} deg")

        ax.set_title(f"Local source density within {scale} deg (ICRS)")
        fig.tight_layout()

        out_name = f"{output_prefix}_mollweide_icrs_r{scale:g}deg.png"
        fig.savefig(out_name, dpi=200)
        plt.close(fig)

        # ---------- Galactic plot ----------
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111, projection="mollweide")

        sc = ax.scatter(x_gal, y_gal, c=vals, s=markersize, alpha=0.8, cmap=cmap)
        ax.grid(True)

        cbar = fig.colorbar(sc, ax=ax, pad=0.05)
        cbar.set_label(f"Local source density (sources / deg^2) within {scale} deg")

        ax.set_title(f"Local source density within {scale} deg (Galactic)")
        fig.tight_layout()

        out_name = f"{output_prefix}_mollweide_galactic_r{scale:g}deg.png"
        fig.savefig(out_name, dpi=200)
        plt.close(fig)


def main() -> None:
    args = parse_args()

    table = read_catalog(args.input_table)

    if args.deduplicate:
        table, summary = filter_and_dedup_fast(
            table,
            snr_thresh=args.dedup_snr_thresh,
            fracpol_thresh=args.dedup_fracpol_thresh,
            dup_arcsec=args.dedup_arcsec,
            keep_rule=args.dedup_keep_rule,
        )
        print("De-duplication summary:", summary)

    coords = SkyCoord(
        ra=table["ra"].astype(float), #* u.deg,
        dec=table["dec"].astype(float),# * u.deg,
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
        coords=coords,
        densities=densities,
        scales_deg=scales_deg,
        output_prefix=prefix,
    )


if __name__ == "__main__":
    main()
