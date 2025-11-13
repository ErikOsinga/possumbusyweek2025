hi4pi=/home/osingae/Documents/postdoc-no-onedrive/doradogroup/other_wavelengths/HI4PI_NH_1_1024.fits
mycatalog=/home/osingae/OneDrive/postdoc/projects/DoradoGroup/exploration/after_rmsynth/dorado.catalog.filtered.snr6.fits


python possum_hi4pi_overlay.py \
  --hi4pi-fits $hi4pi \
  --possum-catalog $mycatalog\
  --ra 65.005 --dec -54.9379 --imwidth-deg 9 \
  --npix 800 \
  --hi-vmin 6.2e19 --hi-vmax 1.2e20 \
  --rm-vmin -50 --rm-vmax 50 \
  --overlay-png ./plotsHI/possum_hi4pi_overlay.png \
  --scatter-png ./plotsHI/possum_hi4pi_scatter.png \
  --out-fits possum_hi4pi_combined.fits \
  --objects-csv dorado_table7_cleanedv2.csv