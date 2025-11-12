#!/usr/bin/env python3
"""
Reproject a Galactic CAR FITS image to Equatorial (ICRS) CAR.
"""

from __future__ import annotations
import argparse
from astropy.io import fits # type: ignore
from astropy.wcs import WCS # type: ignore
from reproject import reproject_interp # type: ignore
from pathlib import Path


def build_icrs_car_wcs(nx: int, ny: int, cdelt: tuple[float, float]) -> WCS:
    """
    Full-sky ICRS plate-carree (CAR) grid, centered at RA=0, Dec=0.
    Preserves handedness: if cdelt1 < 0, RA increases to the left (common in sky images).
    """
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---CAR", "DEC--CAR"]
    w.wcs.cunit = ["deg", "deg"]

    # pixel scales (deg/pix), preserve sign from input to keep orientation consistent
    w.wcs.cdelt = [cdelt[0], cdelt[1]]

    # center of the image is the reference pixel
    w.wcs.crpix = [nx / 2.0 + 0.5, ny / 2.0 + 0.5]

    # reference world coords at the image center: RA=0 deg, Dec=0 deg
    w.wcs.crval = [0.0, 0.0]

    w.wcs.radesys = "ICRS"
    w.wcs.equinox = 2000.0

    # Use celestial to be robust to redundant axes
    return w.celestial


def fix_crpix_header(in_fits: str) -> str:
    """
    Read input FITS, set:
      CRPIX1 = NAXIS1//2 - 0.5
      CRPIX2 = NAXIS2//2 - 0.5
      CRVAL1 = 0.0
      CRVAL2 = 0.0
    Save to "<input>.fixed.fits" and return that path.
    """
    in_path = Path(in_fits)
    out_path = in_path.with_suffix(in_path.suffix + ".fixed.fits")

    with fits.open(str(in_path), mode="readonly", memmap=True) as hdul:
        hdr = hdul[0].header.copy()
        data = hdul[0].data

        nx = int(hdr["NAXIS1"])
        ny = int(hdr["NAXIS2"])

        # Set requested CRPIX/CRVAL values
        hdr["CRPIX1"] = float(nx // 2) - 0.5
        hdr["CRPIX2"] = float(ny // 2) - 0.5
        hdr["CRVAL1"] = 0.0
        hdr["CRVAL2"] = 0.0

        hdr.add_history("CRPIX corrected: CRPIX1=NAXIS1//2-0.5, CRPIX2=NAXIS2//2-0.5")
        hdr.add_history("CRVAL set to 0.0, 0.0 in native frame (here: Galactic).")

        fits.HDUList([fits.PrimaryHDU(data=data, header=hdr)]).writeto(str(out_path), overwrite=True)

    return str(out_path)


def reproject_gal_to_icrs(in_fits: str, out_fits: str) -> None:
    with fits.open(in_fits, memmap=True) as hdul:
        hdu = hdul[0]  # image in primary
        data = hdu.data
        hdr = hdu.header

        # Original WCS (Galactic CAR). Use .celestial to guard against redundant axes.
        src_wcs = WCS(hdr).celestial

        ny = int(hdr["NAXIS2"])
        nx = int(hdr["NAXIS1"])
        cdelt1 = float(hdr["CDELT1"])
        cdelt2 = float(hdr["CDELT2"])

        tgt_wcs = build_icrs_car_wcs(nx, ny, (cdelt1, cdelt2))

        # Reproject
        reproj, footprint = reproject_interp((data, src_wcs), tgt_wcs, shape_out=(ny, nx), return_footprint=True)

        # Build output header: start from target WCS then carry over useful cards
        out_hdr = tgt_wcs.to_header()
        for key in ("BUNIT", "LAMSQ0"):
            if key in hdr:
                out_hdr[key] = hdr[key]
        out_hdr["HISTORY"] = "Reprojected from Galactic (GLON/GLAT, CAR) to ICRS (RA/Dec, CAR) with reproject_interp."

        # Write both image and footprint as an ImageHDU
        prim = fits.PrimaryHDU(reproj.astype("float32"), header=out_hdr)
        fp_hdu = fits.ImageHDU(footprint.astype("float32"), name="FOOTPRINT")

        fits.HDUList([prim, fp_hdu]).writeto(out_fits, overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproject a Galactic CAR FITS map to Equatorial (ICRS) CAR.")
    ap.add_argument("input_fits", help="Input FITS (GLON/GLAT in CAR projection)")
    ap.add_argument("output_fits", help="Output FITS (ICRS RA/Dec in CAR projection)")
    ap.add_argument("--fix-crpix", action="store_true",
                    help="First save a corrected '<input>.fixed.fits' with CRPIX set to NAXIS//2 - 0.5, then reproject from it.")
    args = ap.parse_args()

    src_path = args.input_fits
    if args.fix_crpix:
        src_path = fix_crpix_header(args.input_fits)  # writes <input>.fixed.fits and returns its path

    reproject_gal_to_icrs(src_path, args.output_fits)



if __name__ == "__main__":
    main()
