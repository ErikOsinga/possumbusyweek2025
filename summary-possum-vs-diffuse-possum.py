#!/usr/bin/env python

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from astropy import units as u
from astropy import constants as const


def load_image(image_path: str):
    """
    Load the Stokes I MFS image and return (data, celestial WCS, header).
    """
    hdul = fits.open(image_path)
    try:
        # Find the first HDU that looks like an image
        image_hdu = None
        for hdu in hdul:
            if hasattr(hdu, "data") and hdu.data is not None and hdu.data.ndim >= 2:
                image_hdu = hdu
                break
        if image_hdu is None:
            raise RuntimeError("No image HDU with data found in image FITS file.")

        data = np.squeeze(image_hdu.data)
        header = image_hdu.header
        wcs = WCS(header).celestial  # important for RA/Dec only
    finally:
        hdul.close()

    return data, wcs, header


def load_tables(spectra_path: str, fdf_path: str, catalog_path: str):
    """
    Load the PolSpectra table, FDF table, and catalog table.
    """
    spectra = Table.read(spectra_path)
    fdf = Table.read(fdf_path)
    catalog = Table.read(catalog_path)

    if not (len(spectra) == len(fdf) == len(catalog)):
        raise RuntimeError(
            "Spectra, FDF, and catalog tables do not all have the same length."
        )

    return spectra, fdf, catalog


def make_cutout(
    image_data: np.ndarray,
    image_wcs: WCS,
    ra_deg: float,
    dec_deg: float,
    size_arcsec: float,
):
    """
    Create a Cutout2D centered on (ra_deg, dec_deg) with given size in arcseconds.
    """
    position = SkyCoord(ra_deg, dec_deg, unit="deg", frame="icrs")
    size = (size_arcsec * u.arcsec, size_arcsec * u.arcsec)
    cutout = Cutout2D(
        image_data,
        position,
        size,
        wcs=image_wcs,
        mode="trim",
        copy=True,
    )
    return cutout.data, cutout.wcs


def pixel_scale_arcsec(wcs: WCS) -> float:
    """
    Return a representative pixel scale in arcseconds/pixel.
    Uses the average of the |proj_plane_pixel_scales|.
    """
    scales_deg = proj_plane_pixel_scales(wcs) * u.deg
    # Convert to arcsec and average the absolute values (for slightly non-square pixels)
    scales_arcsec = np.abs(scales_deg.to(u.arcsec).value)
    return float(np.mean(scales_arcsec))


def add_annuli_and_box(
    ax,
    cutout_wcs: WCS,
    inner_rad_as: float,
    outer_rad_as: float,
    box_size_as: float,
):
    """
    Overplot inner/outer annuli and a central box on the cutout image.
    """
    scale_as_per_pix = pixel_scale_arcsec(cutout_wcs)

    inner_radius_pix = inner_rad_as / scale_as_per_pix
    outer_radius_pix = outer_rad_as / scale_as_per_pix
    half_box_pix = (box_size_as / scale_as_per_pix) / 2.0

    # In the cutout frame, the source is at the center of the image:
    ny, nx = cutout_wcs.array_shape
    x_center = nx / 2.0
    y_center = ny / 2.0

    # Circles for annuli
    inner_circle = plt.Circle(
        (x_center, y_center), inner_radius_pix, fill=False, linestyle="-", linewidth=1.0,
        color='white'
    )
    outer_circle = plt.Circle(
        (x_center, y_center), outer_radius_pix, fill=False, linestyle="-", linewidth=1.0,
        color='white'
    )
    ax.add_patch(inner_circle)
    ax.add_patch(outer_circle)

    # Central box
    from matplotlib.patches import Rectangle

    rect = Rectangle(
        (x_center - half_box_pix, y_center - half_box_pix),
        2 * half_box_pix,
        2 * half_box_pix,
        fill=False,
        linestyle="-",
        linewidth=1.0,
        color='green',
    )
    ax.add_patch(rect)


def compute_polarized_intensity(
    q: np.ndarray,
    u: np.ndarray,
    q_err: np.ndarray | None = None,
    u_err: np.ndarray | None = None,
):
    """
    Compute polarized intensity P = sqrt(Q^2 + U^2) and (optionally) its uncertainty.

    If q_err and u_err are provided, propagate errors assuming they are
    independent. Returns (P, P_err) with P_err=None if no errors provided.
    """
    p = np.sqrt(q**2 + u**2)

    if q_err is None or u_err is None:
        return p, None

    # Error propagation: sigma_P^2 = (Q^2 * sigma_Q^2 + U^2 * sigma_U^2) / (Q^2 + U^2)
    denom = q**2 + u**2
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        p_var = (q**2 * q_err**2 + u**2 * u_err**2) / denom
        p_var[denom == 0.0] = 0.0
    p_err = np.sqrt(p_var)

    return p, p_err


def spectra_row_to_arrays(row):
    """
    Convert a table row (with array-valued columns) to plain numpy arrays.

    Assumes that each of the relevant columns is a 1D array per row.
    """
    freq_hz = np.array(row["freq"])

    # On-source values and errors
    i = np.array(row["stokesI"])
    i_err = np.array(row["stokesI_error"])

    q = np.array(row["stokesQ"])
    q_err = np.array(row["stokesQ_error"])

    u = np.array(row["stokesU"])
    u_err = np.array(row["stokesU_error"])

    # Diffuse / off-source components; assume column naming follows the same pattern
    i_off = np.array(row["stokesI_offsource_contaminated"])
    q_off = np.array(row["stokesQ_offsource_contaminated"])
    u_off = np.array(row["stokesU_offsource_contaminated"])

    # No explicit uncertainties for off-source component given
    return freq_hz, i, i_err, q, q_err, u, u_err, i_off, q_off, u_off


def fdf_row_to_arrays(row):
    """
    Convert a FDF table row (with array-valued columns) to numpy arrays.
    """
    phi = np.array(row["phi_fdf"])  # rad/m^2
    fdf = np.abs(np.array(row["fdf"])) # Jy/beam/RMSF
    return phi, fdf


def plot_one_source(
    index: int,
    image_data: np.ndarray,
    image_wcs: WCS,
    peakpi_data: np.ndarray,
    peakpi_wcs: WCS,
    spectra: Table,
    fdf_table: Table,
    catalog: Table,
    diffuse_fdf_table: Table,
    diffuse_catalog: Table,
    output_dir: Path,
    inner_rad_as: float,
    outer_rad_as: float,
    source_box_size_as: float,
):


    """
    Produce and save the multi-panel plot for a single source index.
    """
    cat_row = catalog[index]
    spec_row = spectra[index]
    fdf_row = fdf_table[index]

    ra = float(cat_row["ra"])
    dec = float(cat_row["dec"])
    rm = float(cat_row["rm"])  # catalog RM in rad/m^2

    # Diffuse (off-source) RM and FDF for this source
    diff_cat_row = diffuse_catalog[index]
    diff_rm = float(diff_cat_row["rm"])  # diffuse RM in rad/m^2

    diff_fdf_row = diffuse_fdf_table[index]
    phi_diff, fdf_diff_vals = fdf_row_to_arrays(diff_fdf_row)


    # Use a cutout slightly larger than the outer annulus
    cutout_size_as = outer_rad_as * 2.2

    # Stokes I cutout
    cutout_I, cutout_wcs_I = make_cutout(
        image_data, image_wcs, ra, dec, cutout_size_as
    )

    # Peak polarized intensity cutout
    cutout_P, cutout_wcs_P = make_cutout(
        peakpi_data, peakpi_wcs, ra, dec, cutout_size_as
    )

    # Get spectra arrays
    (
        freq_hz,
        i_src,
        i_src_err,
        q_src,
        q_src_err,
        u_src,
        u_src_err,
        i_diff,
        q_diff,
        u_diff,
    ) = spectra_row_to_arrays(spec_row)

    # Faraday dispersion function
    phi, fdf_vals = fdf_row_to_arrays(fdf_row)

    # Convert frequencies to lambda^2
    c = const.c.value  # m/s
    lambda_sq = (c / freq_hz) ** 2  # m^2

    # Convert intensities to mJy/beam for plotting
    i_src_mjy = i_src * 1000.0
    i_src_err_mjy = i_src_err * 1000.0
    i_diff_mjy = i_diff * 1000.0

    q_src_mjy = q_src * 1000.0
    q_src_err_mjy = q_src_err * 1000.0
    u_src_mjy = u_src * 1000.0
    u_src_err_mjy = u_src_err * 1000.0

    q_diff_mjy = q_diff * 1000.0
    u_diff_mjy = u_diff * 1000.0

    p_src_mjy, p_src_err_mjy = compute_polarized_intensity(
        q_src_mjy, u_src_mjy, q_src_err_mjy, u_src_err_mjy
    )
    p_diff_mjy, _ = compute_polarized_intensity(q_diff_mjy, u_diff_mjy)

    # FDF to mJy/beam/RMSF
    fdf_mjy = fdf_vals * 1000.0
    fdf_diff_mjy = fdf_diff_vals * 1000.0


    # Set up the figure layout
    fig = plt.figure(figsize=(10, 6))
    from matplotlib.gridspec import GridSpec

    gs = GridSpec(
        3,
        3,
        width_ratios=[1.7, 1.0, 1.0],
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.6,
        wspace=0.6,
    )

    # Left column: top = Stokes I, middle = peak P, bottom left unused
    ax_img_I = fig.add_subplot(gs[0, 0], projection=cutout_wcs_I)
    ax_img_P = fig.add_subplot(gs[1, 0], projection=cutout_wcs_P)


    vmin, vmax = np.percentile(cutout_I, [5, 95])
    # Stokes I image
    im_I = ax_img_I.imshow(
        cutout_I,
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax
    )
    ax_img_I.set_ylabel("Dec (J2000)")
    ax_img_I.set_title(f"Source index {index} – Stokes I")

    add_annuli_and_box(
        ax_img_I,
        cutout_wcs_I,
        inner_rad_as=inner_rad_as,
        outer_rad_as=outer_rad_as,
        box_size_as=source_box_size_as,
    )

    cbar_I = fig.colorbar(im_I, ax=ax_img_I, pad=0.01)
    cbar_I.set_label("Stokes I [Jy/beam]")

    # Peak polarized intensity image
    vminp, vmaxp = np.nanpercentile(cutout_P, [1, 99])
    im_P = ax_img_P.imshow(
        cutout_P,
        origin="lower",
        cmap="inferno",
        interpolation="nearest",
        vmin=vminp,
        vmax=vmaxp
    )
    ax_img_P.set_xlabel("RA (J2000)")
    ax_img_P.set_ylabel("Dec (J2000)")
    ax_img_P.set_title("Peak polarized intensity")

    add_annuli_and_box(
        ax_img_P,
        cutout_wcs_P,
        inner_rad_as=inner_rad_as,
        outer_rad_as=outer_rad_as,
        box_size_as=source_box_size_as,
    )
    cbar_P = fig.colorbar(im_P, ax=ax_img_P, pad=0.01)
    cbar_P.set_label("Peak P [Jy/beam]")

    # Right-top row: Stokes I spectra (source and diffuse)
    ax_I_src = fig.add_subplot(gs[0, 1])
    ax_I_diff = fig.add_subplot(gs[0, 2], sharex=ax_I_src, sharey=ax_I_src)

    ax_I_src.errorbar(
        lambda_sq, i_src_mjy, yerr=i_src_err_mjy, fmt=".", ms=3, capsize=2
        ,color='C7'
    )
    ax_I_src.set_ylabel("Intensity [mJy/beam]")
    ax_I_src.set_title("Source I (from box)")
    ax_I_src.grid(True, linestyle=":", linewidth=0.5)

    ax_I_diff.plot(lambda_sq, i_diff_mjy, ".", ms=3, color='C7')
    ax_I_diff.set_title("Diffuse I (median in annulus)")
    ax_I_diff.grid(True, linestyle=":", linewidth=0.5)

    # Right-middle row: Q, U, P spectra (source and diffuse)
    ax_QU_src = fig.add_subplot(gs[1, 1], sharex=ax_I_src)
    ax_QU_diff = fig.add_subplot(gs[1, 2], sharex=ax_I_src, sharey=ax_QU_src)

    ax_QU_src.errorbar(
        lambda_sq,
        q_src_mjy,
        yerr=q_src_err_mjy,
        fmt=".",
        ms=3,
        capsize=2,
        label="Q",
    )
    ax_QU_src.errorbar(
        lambda_sq,
        u_src_mjy,
        yerr=u_src_err_mjy,
        fmt=".",
        ms=3,
        capsize=2,
        label="U",
    )
    if p_src_err_mjy is not None:
        ax_QU_src.errorbar(
            lambda_sq,
            p_src_mjy,
            yerr=p_src_err_mjy,
            fmt=".",
            ms=3,
            capsize=2,
            label="P",
            color='k'
        )
    else:
        ax_QU_src.plot(lambda_sq, p_src_mjy, ".", ms=3, label="P", color='k')

    ax_QU_src.set_ylabel("Intensity [mJy/beam]")
    ax_QU_src.set_title("Source Q, U, P")
    ax_QU_src.grid(True, linestyle=":", linewidth=0.5)
    ax_QU_src.legend(fontsize="small", loc="best")

    # Diffuse Q, U, P (no errors assumed)
    ax_QU_diff.plot(lambda_sq, q_diff_mjy, ".", ms=3, label="Q")
    ax_QU_diff.plot(lambda_sq, u_diff_mjy, ".", ms=3, label="U")
    ax_QU_diff.plot(lambda_sq, p_diff_mjy, ".", ms=3, label="P", color='k')
    ax_QU_diff.set_title("Diffuse Q, U, P")
    ax_QU_diff.grid(True, linestyle=":", linewidth=0.5)
    ax_QU_diff.legend(fontsize="small", loc="best")

    # Right-bottom row: FDFs (source and diffuse)
    ax_FDF_src = fig.add_subplot(gs[2, 1])
    ax_FDF_diff = fig.add_subplot(gs[2, 2], sharey=ax_FDF_src)

    fdflim = 400 # rad/m2 for plotting
    ax_FDF_src.plot(phi, fdf_mjy, "-")
    ax_FDF_src.axvline(rm, linestyle="--", linewidth=1.0, color='k')
    ax_FDF_src.set_xlabel("Faraday depth phi [rad/m^2]")
    ax_FDF_src.set_ylabel("Intensity [mJy/RMSF]")
    ax_FDF_src.set_title("Source FDF")
    ax_FDF_src.grid(True, linestyle=":", linewidth=0.5)
    ax_FDF_src.set_xlim(-fdflim, fdflim)

    # Annotate RM near top right of the panel
    y_max = np.nanmax(fdf_mjy)
    ax_FDF_src.text(
        0.98,
        0.9,
        f"RM = {rm:.1f}",
        transform=ax_FDF_src.transAxes,
        ha="right",
        va="top",
        fontsize="small",
    )

    # Diffuse FDF
    ax_FDF_diff.plot(phi_diff, fdf_diff_mjy, "-")
    ax_FDF_diff.axvline(diff_rm, linestyle="--", linewidth=1.0)
    ax_FDF_diff.set_xlabel("Faraday depth phi [rad/m^2]")
    ax_FDF_diff.set_title("Diffuse FDF")
    ax_FDF_diff.grid(True, linestyle=":", linewidth=0.5)
    ax_FDF_diff.set_xlim(-fdflim, fdflim)

    y_max_diff = np.nanmax(fdf_diff_mjy)
    if np.isfinite(y_max_diff) and y_max_diff > 0.0:
        ax_FDF_diff.text(
            0.98,
            0.9,
            f"RM = {diff_rm:.1f}",
            transform=ax_FDF_diff.transAxes,
            ha="right",
            va="top",
            fontsize="small",
        )


    # Common x-axis label for lambda^2
    ax_QU_src.set_xlabel("lambda^2 [m^2]")
    ax_QU_diff.set_xlabel("lambda^2 [m^2]")

    # Tight layout and save
    # Try to get a useful identifier if present, otherwise fallback to index
    if "source_id" in catalog.colnames:
        source_id = str(cat_row["source_id"])
    elif "name" in catalog.colnames:
        source_id = str(cat_row["name"])
    else:
        source_id = f"{index:04d}"

    output_path = output_dir / f"polspectra_source_{source_id}.png"
    # fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def select_sources_by_snr(catalog: Table, snr_threshold: float):
    """
    Return indices of sources with SNR_PI >= snr_threshold.
    """
    if "SNR_PI" not in catalog.colnames:
        raise RuntimeError("Catalog does not contain column 'SNR_PI'.")

    mask = catalog["SNR_PI"] >= snr_threshold
    indices = np.where(mask)[0]
    return indices


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Produce multi-panel plots of Stokes I images, spectra, and FDFs "
            "for polarized sources."
        )
    )
    parser.add_argument(
        "--image",
        default=(
            "/home/osingae/OneDrive/postdoc/projects/DoradoGroup/"
            "flint_files_niagara/flint_deep/final_linmos_images/"
            "band1_combined_and_band2_combined_separatelty/cubes_freqavg/"
            "combined/NGC1566.round4.i.linmos.corrected.freqavg.fits"
        ),
        help="Path to Stokes I MFS image FITS file.",
    )
    parser.add_argument(
        "--peakpi-image",
        default=(
            "/home/osingae/Documents/postdoc-no-onedrive/doradogroup/"
            "rmsynth3d/fullres/NGC1566.round4.pipeline_3d_ampPeakPIfit.fits"
        ),
        help="Path to peak polarized intensity image FITS file.",
    )
    parser.add_argument(
        "--spectra",
        default=(
            "/home/osingae/OneDrive/postdoc/projects/DoradoGroup/"
            "exploration/after_rmsynth/data_combined/dorado_combined.spectra.fits"
        ),
        help="Path to PolSpectra FITS table with I/Q/U spectra.",
    )
    parser.add_argument(
        "--fdf",
        default=(
            "/home/osingae/OneDrive/postdoc/projects/DoradoGroup/"
            "exploration/after_rmsynth/data_combined/dorado_combined.FDF.fits"
        ),
        help="Path to FDF FITS table.",
    )
    parser.add_argument(
        "--catalog",
        default=(
            "/home/osingae/OneDrive/postdoc/projects/DoradoGroup/"
            "exploration/after_rmsynth/data_combined/dorado_combined.catalog.fits"
        ),
        help="Path to catalog FITS table.",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=30.0,
        help="SNR_PI threshold for selecting sources.",
    )
    parser.add_argument(
        "--outer-rad-as",
        type=float,
        default=109.0,
        help="Outer radius of annulus in arcseconds.",
    )
    parser.add_argument(
        "--inner-rad-as",
        type=float,
        default=35.0,
        help="Inner radius of annulus in arcseconds.",
    )
    parser.add_argument(
        "--source-box-size",
        type=float,
        default=17.0,
        help="Width of central source box in arcseconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="polspectra_plots",
        help="Directory to write output PNG files.",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=None,
        help="Optional maximum number of sources to plot (after SNR cut).",
    )
    parser.add_argument(
        "--diffuse-catalog",
        default=(
            "/home/osingae/OneDrive/postdoc/projects/DoradoGroup/"
            "exploration/after_rmsynth/data_combined/diffuse_highres_catalog.fits"
        ),
        help="Path to diffuse-emission catalog FITS table (for diffuse RM).",
    )
    parser.add_argument(
        "--diffuse-fdf",
        default=(
            "/home/osingae/OneDrive/postdoc/projects/DoradoGroup/"
            "exploration/after_rmsynth/data_combined/diffuse_highres_fdf.fits"
        ),
        help="Path to diffuse-emission FDF FITS table.",
    )


    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image data
    i_image_data, i_image_wcs, _ = load_image(args.image)
    peakpi_data, peakpi_wcs, _ = load_image(args.peakpi_image)
    
    # Load tables
    spectra, fdf_table, catalog = load_tables(
        args.spectra, args.fdf, args.catalog
    )

    # Load diffuse catalog as well
    diffuse_catalog = Table.read(args.diffuse_catalog)
    diffuse_fdf_table = Table.read(args.diffuse_fdf)
    if not (len(diffuse_catalog) == len(diffuse_fdf_table) == len(catalog)):
        raise RuntimeError(
            "Diffuse catalog/FDF tables do not match length of main catalog."
        )



    indices = select_sources_by_snr(catalog, args.snr_threshold)

    if len(indices) == 0:
        print(
            f"No sources found with SNR_PI >= {args.snr_threshold}. "
            "Nothing to do."
        )
        return

    if args.max_sources is not None:
        indices = indices[: args.max_sources]

    print(
        f"Will make plots for {len(indices)} sources "
        f"(SNR_PI >= {args.snr_threshold})."
    )

    for idx in indices:
        print(f"Plotting source index {idx}...")
    for idx in indices:
        print(f"Plotting source index {idx}...")
        plot_one_source(
            index=idx,
            image_data=i_image_data,
            image_wcs=i_image_wcs,
            peakpi_data=peakpi_data,
            peakpi_wcs=peakpi_wcs,
            spectra=spectra,
            fdf_table=fdf_table,
            catalog=catalog,
            diffuse_fdf_table=diffuse_fdf_table,
            diffuse_catalog=diffuse_catalog,
            output_dir=output_dir,
            inner_rad_as=args.inner_rad_as,
            outer_rad_as=args.outer_rad_as,
            source_box_size_as=args.source_box_size,
        )


    print(f"Finished. Plots saved in {output_dir}")


if __name__ == "__main__":
    main()
