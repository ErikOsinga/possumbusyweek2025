# possumbusyweek2025
Scripts for checking RM 'foreground'. Also known as the Milky Way. 


# Plotting STAPS vs POSSUM

Assuming you've downloaded the POSSUM 1D catalogue, you can visualise your favourite region of the sky as follows
```
python rmcat_vs_staps.py --possum-catalog ./possum_1dpipeline_full_goodpol.fits --ra 65.005 --dec -54.9379 --imwidth-deg 6
```

or if you omit the `ra`, `dec`, `imwidth-deg` parameters, you can visualise the full sky. 

