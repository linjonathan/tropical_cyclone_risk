import numpy as np
import xarray as xr

from scipy.interpolate import RectBivariateSpline as interp2d

import namelist

# Reads in bathymetry file.
def read_bathy(basin):
    fdir = namelist.src_directory
    fn = '%s/intensity/data/bathymetry.nc' % fdir
    ds = xr.open_dataset(fn)
    lon = ds['lon'].data
    lat = ds['lat'].data
    bathy = ds['bathymetry'].data
    ds.close()

    lon_b, lat_b, bathy_b = basin.transform_global_field(lon, lat, bathy)
    f_bath = interp2d(lon_b, lat_b, bathy_b.T, kx=1, ky=1)
    return(f_bath)

# Reads in land mask file.
def read_land(basin):
    fdir = namelist.src_directory
    fn = '%s/intensity/data/land.nc' % fdir
    ds = xr.open_dataset(fn)
    lon = ds['lon'].data
    lat = ds['lat'].data
    land = ds['land'].data
    ds.close()

    lon_b, lat_b, land_b = basin.transform_global_field(lon, lat, land)
    f_land = interp2d(lon_b, lat_b, land_b.T, kx=1, ky=1)
    return(f_land)
