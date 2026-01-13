#!/usr/bin/env python3
import argparse
import numpy as np
from astropy.table import Table
from scipy import odr
import matplotlib.pyplot as plt
from annulus_method import compute_annulus_foreground


def load_catalogue(path: str) -> Table:
    """
    Load a FITS catalogue using astropy.Table (handles endian conversion automatically).
    Expected columns: rm, rm_err, SNR_PI   (POSSUM) or rm, rm_err   (STAPS).
    """
    return Table.read(path, format="fits")


def percentile_clip(x: np.ndarray, y: np.ndarray, lo: float = 1.0, hi: float = 99.0):
    x_lo, x_hi = np.nanpercentile(np.array(x), [lo, hi])
    y_lo, y_hi = np.nanpercentile(np.array(y), [lo, hi])

    print(f"Found limits {x_lo=:.0f}, {x_hi=:.0f} for x-axis")
    print(f"Found limits {y_lo=:.0f}, {y_hi=:.0f} for y-axis")
    return (x >= x_lo) & (x <= x_hi) & (y >= y_lo) & (y <= y_hi)


def linear_model(B, x):
    return B[0] * x + B[1]


def fit_with_uncertainties(x, y, xerr, yerr):
    model = odr.Model(linear_model)
    data = odr.RealData(x, y, sx=xerr, sy=yerr)
    odr_instance = odr.ODR(data, model, beta0=[0.0, 0.0])
    out = odr_instance.run()
    return out.beta, out.sd_beta, out.res_var


def make_plot(
    possum: Table,
    staps: Table,
    snr_cut: float = 8.0,
    snr_on_staps: bool = False,
    snr_on_both: bool = False,
    rmerrcut: float = None,
    inlier_lo: float = 5.0,
    inlier_hi: float = 95.0
):
    # ----------- SNR selection or RM error selection-----------
    if rmerrcut is not None:
        largest_rm_err = np.max([staps["rm_err"], possum["rm_err"]], axis=0)
        mask_snr = largest_rm_err <= rmerrcut

        title = f"POSSUM vs STAPS RM with uncertainties (RMerr <= {rmerrcut})"

        print(f"{np.sum(mask_snr)} out of {len(mask_snr)} pass the {rmerrcut=} threshold")
    
    else:
        if snr_on_both:
            snr = np.min([staps["SNR_PI"], possum["SNR_PI"]], axis=0)
        elif snr_on_staps:
            snr = staps["SNR_PI"]
        else:
            snr = possum["SNR_PI"]

        mask_snr = snr >= snr_cut

        title = f"POSSUM vs STAPS RM with uncertainties (SNR_PI >= {snr_cut})"

        print(f"{np.sum(mask_snr)} out of {len(mask_snr)} pass the {snr_cut=} threshold")


    if not np.any(mask_snr):
        raise ValueError("No sources pass SNR cut.")

    rm_possum = possum["rm"][mask_snr]
    rm_possum_err = possum["rm_err"][mask_snr]

    rm_staps = staps["rm"][mask_snr]
    rm_staps_err = staps["rm_err"][mask_snr]


    # ----------- Inlier clipping -----------
    mask_in = percentile_clip(rm_staps, rm_possum, inlier_lo, inlier_hi)
    mask_out = ~mask_in

    print(f"Found {np.sum(mask_in)} inlier ({inlier_lo}-{inlier_hi} pct) sources and {np.sum(mask_out)} outlier sources")

    # ----------- Fit with uncertainties -----------
    beta, beta_err, red_chi2 = fit_with_uncertainties(
        rm_staps[mask_in],
        rm_possum[mask_in],
        rm_staps_err[mask_in],
        rm_possum_err[mask_in],
    )
    slope, intercept = beta

    print(f"{slope=}, {intercept=}")

    # Fit without uncertainties
    a, b = np.polyfit(rm_staps[mask_in], rm_possum[mask_in], 1)
    print(f"{a=}, {b=}")
    # slope, intercept = a, b

    # ----------- Residuals -----------
    fitted = slope * rm_staps[mask_in] + intercept
    residuals = rm_possum[mask_in] - fitted

    # residual uncertainty
    residual_err = np.sqrt(
        rm_possum_err[mask_in] ** 2 + (slope**2) * rm_staps_err[mask_in] ** 2
    )

    # ----------- Plotting -----------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9, 12), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Scatter
    ax1.scatter(rm_staps[mask_in], rm_possum[mask_in], s=10, alpha=0.6, label="Inliers")
    ax1.scatter(
        rm_staps[mask_out],
        rm_possum[mask_out],
        s=10,
        alpha=0.3,
        color="grey",
        label="Outliers",
    )

    # Errorbars — all points passing SNR cut
    ax1.errorbar(
        rm_staps,
        rm_possum,
        xerr=rm_staps_err,
        yerr=rm_possum_err,
        fmt="none",
        ecolor="k",
        alpha=0.12,
        linewidth=0.7,
    )

    # Fit and reference lines
    xgrid = np.linspace(np.min(rm_staps), np.max(rm_staps), 300)
    ax1.plot(xgrid, xgrid, "k--", label="1:1")
    ax1.plot(xgrid, 2 * xgrid, "r--", label="2:1")
    ax1.plot(
        xgrid,
        slope * xgrid + intercept,
        "g-",
        linewidth=2,
        label=f"Fit: y = {slope:.3f} x + {intercept:.3f}",
    )

    ax1.set_xlabel("rm_staps (rad/m^2)")
    ax1.set_ylabel("rm_possum (rad/m^2)")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # Residuals
    ax2.scatter(rm_staps[mask_in], residuals, s=10, alpha=0.6)

    ax2.errorbar(
        rm_staps[mask_in],
        residuals,
        yerr=residual_err,
        fmt="none",
        ecolor="k",
        alpha=0.15,
        linewidth=0.7,
    )

    ax2.axhline(0, color="k", ls="--")
    ax2.set_xlabel("rm_staps (rad/m^2)")
    ax2.set_ylabel("Residual (rad/m^2)")
    ax2.set_title(f"Residuals (inliers only) | Reduced chi^2 = {red_chi2:.3f}")
    ax2.grid(True)

    # if False:
    if True:
        rmlim = 300
        ax1.set_xlim(-rmlim, rmlim)
        ax1.set_ylim(-rmlim, rmlim)
        ax2.set_xlim(-rmlim, rmlim)

    plt.tight_layout()
    plt.savefig("./plots/possum_vs_staps_with_uncertainties.png")
    plt.show()
    plt.close()

    # staps_snr = staps["SNR_PI"][mask_snr]
    # possum_snr = possum["SNR_PI"][mask_snr]
    # bins = np.linspace(0,100,101)
    # plt.hist(staps_snr, label='staps',alpha=1.0,bins=bins)
    # plt.hist(possum_snr, label='possum',alpha=0.5,bins=bins)
    # plt.legend()
    # plt.xlabel('snr')
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot POSSUM vs STAPS RM with uncertainties."
    )
    parser.add_argument("possum_fits", type=str)
    parser.add_argument("staps_fits", type=str)
    parser.add_argument("--snrcut", type=float, default=8.0)
    parser.add_argument(
        "--snr-on-staps",
        action="store_true",
        help="Apply SNR cut to STAPS instead of POSSUM",
    )
    parser.add_argument(
        "--snr-on-both", action="store_true", help="Apply SNR cut to both"
    )
    parser.add_argument("--rmerrcut", type=float, default=None)
    parser.add_argument("--inlier_lo", type=float, default=5)
    parser.add_argument("--inlier_hi", type=float, default=95)

    args = parser.parse_args()

    possum = load_catalogue(args.possum_fits)
    staps = load_catalogue(args.staps_fits)

    if len(possum) != len(staps):
        raise ValueError("Catalogues must have equal length and correspond row-by-row.")

    make_plot(
        possum,
        staps,
        snr_cut=args.snrcut,
        snr_on_staps=args.snr_on_staps,
        snr_on_both=args.snr_on_both,
        rmerrcut=args.rmerrcut,
        inlier_lo=args.inlier_lo,
        inlier_hi=args.inlier_hi
    )
