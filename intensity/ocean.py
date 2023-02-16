import numpy as np
import os
import xarray as xr
from datetime import datetime

"""
Returns climatological mixed layer depth for a given year.
Returns (lat, lon, t) and mixed layer depth (m),
with dimensions (lat, lon, t). dt_months is type datetimes.
"""
def mld_climatology(year, basin):
    f_dir = os.path.dirname(os.path.abspath(__file__))
    ds = xr.open_dataset('%s/data/mld_climatology.nc' % f_dir)
    mld = np.asarray(ds['mixed_layer'])
    mld = np.concatenate((mld, np.expand_dims(mld[:, :, 0], 2)), axis=2)
    lon = np.asarray(ds['lon'])
    lat = np.asarray(ds['lat'])
    month = np.asarray(ds['month'])

    # Define the time spacing between months (in seconds).
    # Each monthly mean climatology is taken as the 15th of the month.
    dt_months = [0] * 13
    mld_b = [0] * 13
    for i in range(0, 12):
        dt_months[i] = datetime(year, int(month[i]), 15)
        lon_b, lat_b, mld_b[i] = basin.transform_global_field(lon[0:-1], lat, mld[:, 0:-1, i])
    dt_months[12] = datetime(year + 1, 1, 15) # wrap around
    mld_b[12] = np.copy(mld_b[0])
    mld_clim = np.dstack(mld_b)
    ds.close()
    da_mld = xr.DataArray(data = mld_clim, dims = ['lat', 'lon', 'time'],
                          coords = dict(lon = ("lon", lon_b), lat = ("lat", lat_b),
                                        time = ("time", np.asarray(dt_months))))    
    return da_mld

"""
Returns climatological sub-mixed layer thermal stratification
for a given year. Returns (lat, lon, t) and thermal stratification
(K/100m), with dimensions (lat, lon, t). dt_months is type datetimes.
"""
def strat_climatology(year, basin):
    f_dir = os.path.dirname(os.path.abspath(__file__))
    ds = xr.open_dataset('%s/data/strat_climatology.nc' % f_dir)
    strat = np.asarray(ds['strat'])
    strat = np.concatenate((strat, np.expand_dims(strat[:, :, 0], 2)), axis=2)
    lon = np.asarray(ds['lon'])
    lat = np.asarray(ds['lat'])
    month = np.asarray(ds['month'])

    # Define the time spacing between months (in seconds).
    # Each monthly mean climatology is taken as the 15th of the month.
    dt_months = [0] * 13
    strat_b = [0] * 13
    for i in range(0, 12):
        dt_months[i] = datetime(year, int(month[i]), 15)
        lon_b, lat_b, strat_b[i] = basin.transform_global_field(lon[0:-1], lat, strat[:, 0:-1, i])
    dt_months[12] = datetime(year + 1, 1, 15) # wrap around
    strat_b[12] = np.copy(strat_b[0])
    strat_clim = np.dstack(strat_b)
    ds.close()
    da_strat = xr.DataArray(data = strat_clim, dims = ['lat', 'lon', 'time'],
                          coords = dict(lon = ("lon", lon_b), lat = ("lat", lat_b),
                                        time = ("time", np.asarray(dt_months))))        
    return da_strat
