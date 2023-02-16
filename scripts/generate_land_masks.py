from global_land_mask import globe
import numpy as np
from util import basins
import xarray as xr
import os

"""
Generates land masks across basins from the global_land_mask module.
Saves output in the "land" folder in master directory.
"""
def generate_land_masks():
    print('Generating land masks...')
    os.makedirs('land', exist_ok = True)

    fns = ['land.nc', 'NA.nc', 'EP.nc', 'NI.nc', 'SI.nc'
           'AU.nc', 'SP.nc', 'WP.nc', 'GL.nc']
    fn_exists = np.array([False]*len(fns))
    for i in range(len(fns)):
        fn_exists[i] = os.path.exists('land/%s' % fns[i])
    if np.all(fn_exists):
        return

    # Generate global land mask
    lat = np.linspace(-90, 90, 721)
    lon = np.linspace(-180, 180, 1441)[:-1]
    lon_grid, lat_grid = np.meshgrid(lon,lat)
    globe_land_mask = globe.is_land(lat_grid, lon_grid)

    # Change to 0-360 coordinates for a easier global mask.
    b_GL = basins.TC_Basin('GL')
    lon_GL, land_GL = b_GL.transform_lon_r(lon, globe_land_mask)
    lat_GL = lat
    lon_GL_grid, lat_GL_grid = np.meshgrid(lon_GL, lat_GL)
    land = xr.DataArray(data = land_GL, dims = ["lat", "lon"],
                        coords = dict(lon=lon, lat=lat))
    ds_land = xr.Dataset(data_vars = dict(land = land))
    ds_land.to_netcdf('land/land.nc')

    # Generate basin masks.
    # Atlantic basin
    lat_box_NA = [0, 9, 10, 14, 18]
    lon_box_NA = np.array([285, 278, 276, 271, 262])
    NA_mask = (lon_GL_grid >= 255) & (lon_GL_grid <= 360) & (lat_GL_grid >= 0) & (lat_GL_grid <= 60)
    NA_box_mask = np.full(NA_mask.shape, False)
    for i in range(len(lat_box_NA)):
        box_mask = (lat_GL_grid >= lat_box_NA[i]) & (lon_GL_grid >= lon_box_NA[i]) & (~land)
        NA_box_mask[box_mask] = True

    NA_mask = xr.DataArray(data = NA_mask & NA_box_mask, dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_NA = xr.Dataset(data_vars = dict(basin = NA_mask))
    ds_NA.to_netcdf('land/NA.nc')

    # Eastern Pacific basin
    lat_box_EP = [7.5, 8.8,   9,    10,  15,  18,  60]
    lon_box_EP = [295, 282, 277, 276.5, 276, 271, 262]
    EP_mask = (lon_GL_grid >= 180) & (lon_GL_grid <= 290) & (lat_GL_grid >= 0) & (lat_GL_grid <= 60)
    EP_box_mask = np.full(EP_mask.shape, False)
    for i in range(len(lon_box_EP)):
        # EP defined as everything to the west of the Atlantic mask
        box_mask = (lat_GL_grid <= lat_box_EP[i]) & (lon_GL_grid <= lon_box_EP[i]) & (~land_GL)
        EP_box_mask[box_mask] = True

    EP_mask = xr.DataArray(data = EP_mask & EP_box_mask, dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_EP = xr.Dataset(data_vars = dict(basin = EP_mask))
    ds_EP.to_netcdf('land/EP.nc')

    # Western Pacific basin
    WP_mask = (lon_GL_grid >= 100) & (lon_GL_grid <= 180) & (lat_GL_grid >= 0) & (lat_GL_grid <= 60)
    WP_mask = xr.DataArray(data = WP_mask & (~land_GL), dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_WP = xr.Dataset(data_vars = dict(basin = WP_mask))
    ds_WP.to_netcdf('land/WP.nc')

    # Indian Ocean basin
    NI_mask = (lon_GL_grid >= 30) & (lon_GL_grid <= 100) & (lat_GL_grid >= 0) & (lat_GL_grid <= 49)
    NI_mask = xr.DataArray(data = NI_mask & (~land_GL), dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_NI = xr.Dataset(data_vars = dict(basin = NI_mask))
    ds_NI.to_netcdf('land/NI.nc')

    # Southern Indian basin
    SI_mask = (lon_GL_grid >= 10) & (lon_GL_grid <= 100) & (lat_GL_grid >= -45) & (lat_GL_grid <= 0)
    SI_mask = xr.DataArray(data = SI_mask & (~land_GL), dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_SI = xr.Dataset(data_vars = dict(basin = SI_mask))
    ds_SI.to_netcdf('land/SI.nc')

    # Australia basin
    AU_mask = (lon_GL_grid >= 100) & (lon_GL_grid <= 170) & (lat_GL_grid >= -45) & (lat_GL_grid <= 0)
    AU_mask = xr.DataArray(data = AU_mask & (~land_GL), dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_AU = xr.Dataset(data_vars = dict(basin = AU_mask))
    ds_AU.to_netcdf('land/AU.nc')

    # Southern Pacific basin
    SP_mask = (lon_GL_grid >= 170) & (lon_GL_grid <= 260) & (lat_GL_grid >= -45) & (lat_GL_grid <= 0)
    SP_mask = xr.DataArray(data = SP_mask & (~land_GL), dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_SP = xr.Dataset(data_vars = dict(basin = SP_mask))
    ds_SP.to_netcdf('land/SP.nc')

    # Global
    GL_mask = ~land.data
    GL_mask[np.abs(lat_GL_grid) > 50] = 0
    GL_mask = xr.DataArray(data = GL_mask, dims = ["lat", "lon"],
                        coords = dict(lon=lon_GL, lat=lat_GL))
    ds_GL = xr.Dataset(data_vars = dict(basin = GL_mask))
    ds_GL.to_netcdf('land/GL.nc')
