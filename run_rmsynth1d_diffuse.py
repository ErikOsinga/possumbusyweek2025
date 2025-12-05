#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Grab a set of off-source median spectra and perform 1D RM synthesis on them.

"""

from RMtools_1D.do_RMsynth_1D import run_rmsynth
import numpy as np
from astropy.table import Table
import tqdm
from typing import Any



def parse_results_into_astropy_tables(results: list[tuple[dict,dict]]) -> tuple[Table,Table]:
    """
    parses the cursed format of the results into a table "catalog" and a "FDF" spectra table.

    Convert a list of results (each with 'synth_dict' and 'synth_array_dict')
    into two astropy Tables:

    - catalog: one column per key in 'synth_dict'; one row per entry in results
    - FDF:     one column per key in 'synth_array_dict'; one row per entry

    Parameters
    ----------
    results : list of dict
        Each element should be a mapping with keys 'synth_dict' and
        'synth_array_dict'.

    Returns
    -------
    catalog : astropy.table.Table
        Table built from the 'synth_dict' entries.

    FDF : astropy.table.Table
        Table built from the 'synth_array_dict' entries. Entries can be arrays.
    """
    if not results:
        raise ValueError("Input 'results' is empty.")

    # Collect all keys across entries for robustness
    synth_keys = set()
    synth_array_keys = set()

    for entry in results:
        synth = entry.get("synth_dict", {})
        synth_array = entry.get("synth_array_dict", {})
        synth_keys.update(synth.keys())
        synth_array_keys.update(synth_array.keys())

    # Preserve the order from the first entry when possible
    first_synth_keys = list(results[0]["synth_dict"].keys())
    catalog_keys = first_synth_keys + [
        k for k in sorted(synth_keys) if k not in first_synth_keys
    ]

    first_synth_array_keys = list(results[0]["synth_array_dict"].keys())
    FDF_keys = first_synth_array_keys + [
        k for k in sorted(synth_array_keys) if k not in first_synth_array_keys
    ]

    # Build data dicts for astropy Tables
    catalog_data: dict[str, list[Any]] = {}
    for key in catalog_keys:
        values = [
            entry.get("synth_dict", {}).get(key) for entry in results
        ]
        # deal with failed sources
        values = np.array([np.nan if v is None else v for v in values])
        # attach to catalog
        catalog_data[key] = values

    FDF_data: dict[str, list[Any]] = {}

    # find a good reference row to deal with failed sources
    for goodrow in range(len(results)):
        if results[goodrow]['synth_array_dict']['dirtyFDF'] is not None:
            break

    for key in FDF_keys:
        values = [
            entry.get("synth_array_dict", {}).get(key) for entry in results
        ]

        # deal with failed sources by setting appropriate shape NaN array
        values = np.array([np.ones(len(results[goodrow]['synth_array_dict'][key])) if v is None else v for v in values])

        # assign values
        FDF_data[key] = values

    catalog = Table(catalog_data)
    FDF = Table(FDF_data)

    return catalog, FDF


def main(data, spectra, outdir, log=print):
    
    phimax=6000
    nSamples=10
    polyOrd=-2
    weightType='variance'
    fit_function='log'

    # list of dictionaries, cursed format.
    results = [{} for i in range(len(data))]
    
    for i in tqdm.trange(len(data)):
        try:
            rmsynth1d_input = np.vstack((data[i]['freq'],
                                         data[i]['i_values'],
                                         data[i]['q_values'],
                                         data[i]['u_values'],
                                         data[i]['di_values'],
                                         data[i]['dq_values'],
                                         data[i]['du_values']))
            
            synth_dict,synth_array_dict = \
                run_rmsynth(rmsynth1d_input, verbose = False, fitRMSF=True,
                  phiMax_radm2 = phimax, dPhi_radm2=None,
                  nSamples=nSamples,
                  fit_function=fit_function,
                  polyOrd=polyOrd,
                  weightType=weightType,
                  log=log)
            results[i]['synth_dict']=synth_dict
            results[i]['synth_array_dict']=synth_array_dict
            
        except Exception as error:
            log(f'Failed to perform rmsynth1D for source {i} (ra {spectra[i]["ra"]}, dec {spectra[i]["dec"]}). No output dicts produced.')
            log(f"An exception occured: {error}")

    catalog, fdf = parse_results_into_astropy_tables(results)

    # add classic named columns
    catalog['ra'] = spectra["ra"]
    catalog['dec'] = spectra["dec"]
    catalog['rm'] = catalog['phiPeakPIfit_rm2']
    
    fdf['ra'] = spectra['ra']
    fdf['dec'] = spectra["dec"]
    fdf.rename_column('phiArr_radm2', 'phi_fdf')
    fdf.rename_column('dirtyFDF', 'fdf')
    fdf.rename_column('phi2Arr_radm2', 'phi_rmsf')
    fdf.rename_column('RMSFArr', 'rmsf')

    # catalog.write(outdir+'diffuse_highres_catalog.fits', overwrite=True)
    catalog.write(outdir+'diffuse_catalog.fits', overwrite=True)
    # fdf.write(outdir+'diffuse_highres_fdf.fits', overwrite=True)
    fdf.write(outdir+'diffuse_fdf.fits', overwrite=True)

    return results, catalog, fdf


def get_data_possum_diffuse():
    """
    Get the data table for POSSUM diffuse spectra
    """

    # read iqu spectra file
    spectra = Table.read("/home/osingae/OneDrive/postdoc/projects/DoradoGroup/exploration/after_rmsynth/data_combined/dorado_combined.spectra.fits")
    data = Table()
    # grab the off-source diffuse median spectra
    data['freq'] = spectra['freq']
    data['i_values'] = spectra['stokesI_offsource']
    data['q_values'] = spectra['stokesQ_offsource']
    data['u_values'] = spectra['stokesU_offsource']
    # since noise is only used for inverse variance weighting, use the noise determined for the source
    data['di_values'] = spectra['stokesI_error']
    data['dq_values'] = spectra['stokesQ_error']
    data['du_values'] = spectra['stokesU_error']

    outdir = "/home/osingae/OneDrive/postdoc/projects/DoradoGroup/exploration/after_rmsynth/data_combined/"
    
    return data, spectra, outdir

if __name__ == "__main__":

    ## get possum diffuse data, high-res
    data, spectra, outdir = get_data_possum_diffuse()
    ## possum low res data has been run on lofar6 /data1/osinga/dorado/rmsynth1d/lowres
    ## see results at /home/osingae/OneDrive/postdoc/projects/DoradoGroup/exploration/after_rmsynth/data_combined/rmsynth_lowres/diffuse_catalog.fits


    # get GMIMS diffuse data


    results, catalog, fdf = main(data, spectra, outdir)