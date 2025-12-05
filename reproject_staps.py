from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from reproject import reproject_interp


def reproject_staps_to_galactic(
    staps_path: Path,
    out_path: Path | None = None,
    nlon: int = 3600,
    nlat: int = 1800,
) -> Path:
    """
    Reproject the STAPS RM map to a Galactic CAR (plate carrée) projection using reproject.

    Parameters
    ----------
    staps_path : Path
        Input STAPS RM FITS file.
    out_path : Path or None
        Output FITS path. If None, use "<stem>_galactic.fits" next to the input file.
    nlon : int
        Number of pixels in Galactic longitude (l). Covers 360 deg.
    nlat : int
        Number of pixels in Galactic latitude (b). Covers -90 to +90 deg.

    Returns
    -------
    Path
        Path to the reprojected Galactic FITS file.
    """
    # Load original STAPS image and WCS. If you already have load_staps_map, reuse it:
    # data_in, wcs_in, header_in = load_staps_map(staps_path)
    with fits.open(staps_path) as hdul:
        # grab first non-empty 2D image
        hdu = None
        for h in hdul:
            if hasattr(h, "data") and h.data is not None and h.data.ndim >= 2:
                hdu = h
                break
        if hdu is None:
            raise ValueError("No image HDU with data found in STAPS FITS file.")
        data_in = np.squeeze(hdu.data)
        header_in = hdu.header
        wcs_in = WCS(header_in).celestial  # ensure pure celestial 2D WCS

    if data_in.ndim != 2:
        raise ValueError("STAPS image is not 2D after squeeze; check FITS file.")

    # Build target Galactic CAR WCS
    #
    # Full-sky, center at l=0, b=0
    # l runs [-180, +180] deg; b runs [-90, +90] deg
    wcs_out = WCS(naxis=2)
    wcs_out.wcs.ctype = ["GLON-CAR", "GLAT-CAR"]
    wcs_out.wcs.crval = [0.0, 0.0]  # reference at l=0, b=0
    wcs_out.wcs.crpix = [nlon / 2.0 + 0.5, nlat / 2.0 + 0.5]
    wcs_out.wcs.cdelt = [-360.0 / nlon, 180.0 / nlat]  # negative in l for usual increasing-to-left
    wcs_out.wcs.cunit = ["deg", "deg"]

    shape_out = (nlat, nlon)

    # Reproject using interpolation
    data_out, footprint = reproject_interp(
        (data_in, wcs_in),
        wcs_out,
        shape_out=shape_out,
        order="bilinear",
    )

    # Build output header
    header_out = wcs_out.to_header()
    # Propagate some basic metadata if present
    for key in ("BUNIT", "BTYPE", "OBJECT", "TELESCOP", "INSTRUME"):
        if key in header_in:
            header_out[key] = header_in[key]
    header_out["COMMENT"] = "Reprojected to Galactic CAR using reproject_interp"

    # Primary HDU: reprojected data
    hdu_primary = fits.PrimaryHDU(data=data_out.astype(np.float32), header=header_out)

    # Second HDU: footprint (0..1), showing how much of each pixel is covered
    hdu_foot = fits.ImageHDU(data=footprint.astype(np.float32), name="FOOTPRINT")

    hdul_out = fits.HDUList([hdu_primary, hdu_foot])

    if out_path is None:
        out_path = staps_path.with_name(staps_path.stem + "_galactic.fits")

    hdul_out.writeto(out_path, overwrite=True)
    return out_path


# Optional small CLI so you can run it as a script
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reproject a STAPS RM map to Galactic CAR coordinates."
    )
    parser.add_argument("staps_fits", type=Path,
                        help="Input STAPS RM FITS file.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Output Galactic FITS file (default: <stem>_galactic.fits).")
    parser.add_argument("--nlon", type=int, default=3600,
                        help="Number of pixels in Galactic longitude (default: 3600).")
    parser.add_argument("--nlat", type=int, default=1800,
                        help="Number of pixels in Galactic latitude (default: 1800).")

    args = parser.parse_args()

    out = reproject_staps_to_galactic(
        staps_path=args.staps_fits,
        out_path=args.out,
        nlon=args.nlon,
        nlat=args.nlat,
    )
    print(f"Saved Galactic reprojection to: {out}")
