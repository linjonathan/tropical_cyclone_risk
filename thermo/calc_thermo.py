#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""

import dask
import datetime
import os
import namelist
import numpy as np
import xarray as xr

from util import input, mat
from thermo import thermo

def get_fn_thermo():
    fn_th = '%s/thermo_%s_%d%02d_%d%02d.nc' % (namelist.output_directory, namelist.exp_prefix,
                                               namelist.start_year, namelist.start_month,
                                               namelist.end_year, namelist.end_month)
    return(fn_th)


def compute_thermo(dt_start, dt_end):
    ds_sst = input.load_sst(dt_start, dt_end).load()
    ds_psl = input.load_mslp(dt_start, dt_end).load()
    ds_ta = input.load_temp(dt_start, dt_end).load()
    ds_hus = input.load_sp_hum(dt_start, dt_end).load()
    lon_ky = input.get_lon_key()
    lat_ky = input.get_lat_key()
    sst_ky = input.get_sst_key()

    nTime = len(ds_sst['time'])
    vmax = np.zeros(ds_psl[input.get_mslp_key()].shape)
    chi = np.zeros(ds_psl[input.get_mslp_key()].shape)
    rh_mid = np.zeros(ds_psl[input.get_mslp_key()].shape)
    for i in range(nTime):
        # Convert all variables to the atmospheric grid.
        sst_interp = mat.interp_2d_grid(ds_sst[lon_ky], ds_sst[lat_ky],
                                        np.nan_to_num(ds_sst[sst_ky][i, :, :].data),
                                        ds_ta[lon_ky], ds_ta[lat_ky])
        if 'C' in ds_sst[sst_ky].units:
            sst_interp = sst_interp + 273.15

        psl = ds_psl[input.get_mslp_key()][i, :, :]
        ta = ds_ta[input.get_temp_key()][i, :, :, :]
        hus = ds_hus[input.get_sp_hum_key()][i, :, :, :]
        lvl = ds_ta[input.get_lvl_key()]
        lvl_d = np.copy(ds_ta[input.get_lvl_key()].data)

        # Ensure lowest model level is first.
        # Here we assume the model levels are in pressure.
        if (lvl[0] - lvl[1]) < 0:
            ta = ta.reindex({input.get_lvl_key(): lvl[::-1]})
            hus = hus.reindex({input.get_lvl_key(): lvl[::-1]})
            lvl_d = lvl_d[::-1]
    
        p_midlevel = namelist.p_midlevel                    # Pa
        if lvl.units in ['millibars', 'hPa']:
            lvl_d *= 100                                    # needs to be in Pa
            p_midlevel = namelist.p_midlevel / 100          # hPa
            lvl_mid = lvl.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest')

        # TODO: Check units of psl, ta, and hus
        vmax_args = (sst_interp, psl.data, lvl_d, ta.data, hus.data)
        vmax[i, :, :] = thermo.CAPE_PI_vectorized(*vmax_args)
        ta_midlevel = ta.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest').data
        hus_midlevel = hus.sel({input.get_lvl_key(): p_midlevel}, method = 'nearest').data

        p_midlevel_Pa = float(lvl_mid) * 100 if lvl_mid.units in ['millibars', 'hPa'] else float(lvl_mid)
        chi_args = (sst_interp, psl.data, ta_midlevel,
                    p_midlevel_Pa, hus_midlevel)
        chi[i, :, :] = np.minimum(np.maximum(thermo.sat_deficit(*chi_args), 0), 10)
        rh_mid[i, :, :] = thermo.conv_q_to_rh(ta_midlevel, hus_midlevel, p_midlevel_Pa)

    return (vmax, chi, rh_mid)

def gen_thermo():
    # TODO: Assert all of the datasets have the same length in time.
    if os.path.exists(get_fn_thermo()):
        return

    # Load datasets metadata. Since SST is split into multiple files and can
    # cause parallel reads with open_mfdataset to hang, save as a single file.
    dt_start, dt_end = input.get_bounding_times()
    ds = input.load_mslp()

    ct_bounds = [dt_start, dt_end]
    ds_times = input.convert_from_datetime(ds,
                   np.array([x for x in input.convert_to_datetime(ds, ds['time'].values)
                             if x >= ct_bounds[0] and x <= ct_bounds[1]]))

    n_chunks = namelist.n_procs
    chunks = np.array_split(ds_times, np.minimum(n_chunks, np.floor(len(ds_times) / 2)))
    lazy_results = []
    for i in range(len(chunks)):
        lazy_result = dask.delayed(compute_thermo)(chunks[i][0], chunks[i][-1])
        lazy_results.append(lazy_result)
    out = dask.compute(*lazy_results, scheduler = 'processes', num_workers = n_chunks)

    # Clean up and process output.
    # Ensure monthly timestamps have middle-of-the-month days.
    ds_times = input.convert_from_datetime(ds,
                  np.array([datetime.datetime(x.year, x.month, 15) for x in
                           [x for x in input.convert_to_datetime(ds, ds['time'].values)
                            if x >= ct_bounds[0] and x <= ct_bounds[1]]]))
    vmax = np.concatenate([x[0] for x in out], axis = 0)
    chi = np.concatenate([x[1] for x in out], axis = 0)
    rh_mid = np.concatenate([x[2] for x in out], axis = 0)
    ds_thermo = xr.Dataset(data_vars = dict(vmax = (['time', 'lat', 'lon'], vmax),
                                            chi = (['time', 'lat', 'lon'], chi),
                                            rh_mid = (['time', 'lat', 'lon'], rh_mid)),
                           coords = dict(lon = ("lon", ds[input.get_lon_key()].data),
                                         lat = ("lat", ds[input.get_lat_key()].data),
                                         time = ("time", ds_times)))
    ds_thermo.to_netcdf(get_fn_thermo())
    print('Saved %s' % get_fn_thermo())
