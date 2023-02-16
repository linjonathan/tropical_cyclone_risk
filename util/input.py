#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Jonathan Lin
Utility library for dealing with CMIP6 and reanalysis data files.
"""

import calendar
import cftime
import datetime
import glob
import numpy as np
import xarray as xr
import namelist

def _open_fns(fns):
    if len(fns) == 1:
        ds = xr.open_dataset(fns[0])
    else:
        ds = xr.open_mfdataset(fns, concat_dim = "time", combine = "nested", data_vars="minimal")
    return ds

def _glob_prefix(var_prefix):
    fns = glob.glob('%s/**/*%s*.nc' % (namelist.base_directory, namelist.exp_prefix), recursive = True)
    fns_var = sorted([x for x in fns if '_%s_' % var_prefix in x])
    if len(fns_var) == 0:
        fns_var = sorted([x for x in fns if '%s_' % var_prefix in x])
    return(fns_var)

def _find_in_timerange(fns, ct_start, ct_end = None):
    fns_multi = []
    for fn in fns:
        ds = xr.open_dataset(fn)
        time = ds['time']                      # time
        if ct_start is not None and ct_end is None:
            if ((ct_start >= time[0]) & (ct_start <= time[-1])):
                fns_multi.append(fn)
        else:
            if ((time >= ct_start) & (time <= ct_end)).any():
                fns_multi.append(fn)
        ds.close()

    return(fns_multi)

"""
Opens files described by "var", bounded by times ct_start and ct_end.
If only ct_start, then opens the files at time ct_start.
If both ct_start and ct_end are None, opens all files by var.
"""
def _load_var(var, ct_start, ct_end):
    if ct_start == None and ct_end == None:
        ds = _open_fns(_glob_prefix(var))
    elif ct_start is not None and ct_end == None:
        ds = _open_fns(_find_in_timerange(_glob_prefix(var), ct_start)).sel(time = ct_start)
    else:
        fns = _find_in_timerange(_glob_prefix(var), ct_start, ct_end)
        ds = _open_fns(fns).sel(time=slice(ct_start, ct_end))
    return ds

def get_sst_key():
    return namelist.var_keys[namelist.dataset_type]['sst']

def get_mslp_key():
    return namelist.var_keys[namelist.dataset_type]['mslp']

def get_temp_key():
    return namelist.var_keys[namelist.dataset_type]['temp']

def get_sp_hum_key():
    return namelist.var_keys[namelist.dataset_type]['sp_hum']

def get_u_key():
    return namelist.var_keys[namelist.dataset_type]['u']

def get_v_key():
    return namelist.var_keys[namelist.dataset_type]['v']

def get_w_key():
    return namelist.var_keys[namelist.dataset_type]['w']

def get_lvl_key():
    return namelist.var_keys[namelist.dataset_type]['lvl']

def get_lon_key():
    return namelist.var_keys[namelist.dataset_type]['lon']

def get_lat_key():
    return namelist.var_keys[namelist.dataset_type]['lat']

def load_sst(ct_start = None, ct_end = None):
    return _load_var(get_sst_key(), ct_start, ct_end)

def load_mslp(ct_start = None, ct_end = None):
    return _load_var(get_mslp_key(), ct_start, ct_end)

def load_w(ct_start = None, ct_end = None):
    return _load_var(get_w_key(), ct_start, ct_end)

def load_temp(ct_start = None, ct_end = None):
    return _load_var(get_temp_key(), ct_start, ct_end)

def load_sp_hum(ct_start = None, ct_end = None):
    return _load_var(get_sp_hum_key(), ct_start, ct_end)

def _load_var_daily(fn):
    # Daily variables are large cannot be loaded into memory as easily.
    # So this is an internal function that loads a file directly.
    ds = xr.open_dataset(fn)
    return ds

def convert_from_datetime(ds, dts):
    # Convert the datetime array dts to the timestamps used by ds.
    # Only supports np.datetime64 or cftime.DatetimeNoLeap to datetimes.
    # Necessary to convert between non-standard calendars (like no leap).
    if isinstance(np.array(ds['time'])[0], np.datetime64):
        adt = np.array([np.datetime64(str(x)) for x in np.array(dts)])
    elif isinstance(np.array(ds['time'])[0], cftime.DatetimeNoLeap):
        adt = np.array([cftime.DatetimeNoLeap(x.year, x.month, x.day, x.hour) for x in np.array(dts)])
    else:
        raise Exception("Did not understand type of time.")
    return adt

def convert_to_datetime(ds, dts):
    # Convert the timestamps types of ds to datetime timestamps.
    # Only supports np.datetime64 or cftime.DatetimeNoLeap to datetimes.
    # Necessary to convert between non-standard calendars (like no leap).
    if isinstance(np.array(ds['time'])[0], np.datetime64):
        adt = np.array(dts.astype('datetime64[s]').tolist())
    elif isinstance(np.array(ds['time'])[0], cftime.DatetimeNoLeap):
        adt = np.array([datetime.datetime(x.year, x.month, x.day, x.hour) for x in np.array(dts)])
    else:
        raise Exception("Did not understand type of time.")
    return adt

def get_bounding_times():
    s_dt = datetime.datetime(namelist.start_year, namelist.start_month, 1)
    N_day = calendar.monthrange(namelist.end_year, namelist.end_month)[1]
    e_dt = datetime.datetime(namelist.end_year, namelist.end_month, N_day)
    return (s_dt, e_dt)
