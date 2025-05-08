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

# Reads in drag coefficient file.
# Returns a normalized drag coefficient, which represents the multiplicative
# value of the drag coefficient as compared to the neutral over-ocean Cd.
def read_drag(basin):
    fdir = namelist.src_directory
    fn = '%s/intensity/data/Cd.nc' % fdir
    ds = xr.open_dataset(fn)
    lon = ds['longitude'].data
    lat = ds['latitude'].data
    Cd = ds['Cd'].data
    ds.close()

    # Converts from a 10m altitude drag coefficient to a
    # drag coefficient more appropriate to the gradient wind, and
    # normalizes with respect to over-ocean Cd.
    Cd_gradient = Cd / (1 + 50.0 * Cd)
    Cd_norm = Cd_gradient / np.min(Cd_gradient)
    Cd = namelist.Cd * Cd_norm

    lon_b, lat_b, Cd_b = basin.transform_global_field(lon, lat, Cd)
    f_Cd = interp2d(lon_b, lat_b, Cd_b.T, kx=1, ky=1)
    return(f_Cd)
