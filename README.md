# possumbusyweek2025
Scripts for checking RM 'foreground'. Also known as the Milky Way. 


# Plotting STAPS vs POSSUM

Assuming you've downloaded the POSSUM 1D catalogue, you can visualise your favourite region of the sky as follows
```
python possum_vs_gmims_image.py --possum-catalog ./possum_1dpipeline_full_goodpol.fits --ra 65.005 --dec -54.9379 --imwidth-deg 6
```

or if you omit the `ra`, `dec`, `imwidth-deg` parameters, you can visualise the full sky. 


You can also create RM vs RM scatter plots with 

```
python gmims_vs_possum_scatter.py --staps-fits /home/osingae/Documents/postdoc-no-onedrive/doradogroup/other_wavelengths/RMp.fits --possum-catalog ./possum_1dpipeline_full_goodpol.fits --ra 65.005 --dec -54.9379 --radius-deg 3 --rmlim 100
```


# Plotting GMIMS low vs POSSUM

Assuming you've downloaded the GMIMS low RM map (or any other map, see e.g. https://www.canfar.net/storage/arc/list/projects/CIRADA/osinga/gmims_low), first fix the header coordinate system and project from galactic to equatorial coordinates with

```
python reproject_galactic_to_icrs.py \
  gmims_low/gmims_low.pipeline_3d_phiPeakPIfit_rm2.fits \
  gmims_low/gmims_low.pipeline_3d_phiPeakPIfit_rm2_icrs_car.fits \
  --fix-crpix
```

Then plot the comparison with

```
python possum_vs_gmims_image.py --staps-fits ./gmims_low/gmims_low.pipeline_3d_phiPeakPIfit_rm2_icrs_car.fits --possum-catalog ./possum_1dpipeline_full_goodpol.fits --ra 65.005 --dec -54.9379 --imwidth-deg 6 --objects_csv dorado_table7_cleanedv2.csv --outname gmimslow_vs_possum.png
```

or the scatter plot equivalently as explain in STAPS vs POSSUM.


# Plotting HI vs POSSUM 
Assuming you've downloaded the HI4PI column density map and the POSSUM catalogue of your liking

```
hi4pi=/path/to/HI4PI_NH_1_1024.fits
mycatalog=/path/to/possum.catalog.filtered.snr6.fits

mkdir ./plotsHI/

python possum_hi4pi_overlay.py \
  --hi4pi-fits $hi4pi \
  --possum-catalog $mycatalog\
  --ra 65.005 --dec -54.9379 --imwidth-deg 9 \
  --npix 800 \
  --hi-vmin 6e19 --hi-vmax 3e20 \
  --rm-vmin -50 --rm-vmax 50 \
  --overlay-png ./plotsHI/possum_hi4pi_overlay.png \
  --scatter-png ./plotsHI/possum_hi4pi_scatter.png \
  --out-fits possum_hi4pi_combined.fits \
```

# Computing local source density for an RM grid catalogue

For a POSSUM catalogue (important to include --deduplicate)
```
python compute_local_density.py --input-table possum_1dpipeline_full_goodpol.fits \
        --input-scales-deg 0.5 1.0 1.5 \
        --output-prefix ./plots_local_density/possum_local_dens_molw_ \
        --output-table possum_local_density.fits \
        --deduplicate 
```

For the SPICERACS catalogue
```
# shouldnt have to deduplicate the SPICERACS catalogue
python compute_local_density.py --input-table ./SPICERACS_polcat_snr8_fracpol001.fits \
        --input-scales-deg 0.5 1.0 1.5 \
        --output-prefix ./plots_local_density/spiceracs_local_dens_molw_ \
```

