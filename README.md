# possumbusyweek2025
Scripts for checking RM 'foreground'. Also known as the Milky Way. 


# Plotting STAPS vs POSSUM

Assuming you've downloaded the POSSUM 1D catalogue, you can visualise your favourite region of the sky as follows
```
python rmcat_vs_staps.py --possum-catalog ./possum_1dpipeline_full_goodpol.fits --ra 65.005 --dec -54.9379 --imwidth-deg 6
```

or if you omit the `ra`, `dec`, `imwidth-deg` parameters, you can visualise the full sky. 


You can also create RM vs RM scatter plots with 

```
python staps_possum_compare.py --staps-fits /home/osingae/Documents/postdoc-no-onedrive/doradogroup/other_wavelengths/RMp.fits --possum-catalog ./possum_1dpipeline_full_goodpol.fits --ra 65.005 --dec -54.9379 --radius-deg 3 --rmlim 100
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
python rmcat_vs_staps.py --staps-fits ./gmims_low/gmims_low.pipeline_3d_phiPeakPIfit_rm2_icrs_car.fits --possum-catalog ./possum_1dpipeline_full_goodpol.fits --ra 65.005 --dec -54.9379 --imwidth-deg 6 --objects_csv dorado_table7_cleanedv2.csv --outname gmimslow_vs_possum.png
```

