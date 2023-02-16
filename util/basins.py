#!/usr/bin/env python
"""
Author: Jonathan Lin
Implementation of class that holds basin information of a tropical cyclone.
Done in order to keep track of longitude transformations.
"""

import namelist
import numpy as np

class TC_Basin:
    """
    Base class that stores information of a basin.
    """

    def __init__(self, basin_id):
        if basin_id.upper() not in namelist.basin_bounds.keys():
            raise ValueError('Basin ID is not valid. See list of valid basins.')
        else:
            self.basin_id = basin_id
            self.basin_bounds = namelist.basin_bounds[self.basin_id]

    def _adj_bnd(self, bound):
        xd = float(bound[:-1])
        if bound[-1] in ['W', 'S']:
            xd *= -1
        return xd

    """
    Returns true if the position is within dx degrees of the basin bounds.
    """
    def in_basin(self, clon, clat, dx):
        lon_min, lat_min, lon_max, lat_max = self.get_bounds()

        is_in_basin = ((lon_min + dx) < clon < (lon_max - dx) and
                       (lat_min + dx) < clat < (lat_max - dx))
        return(is_in_basin)
    """
    Returns the lower left, and upper right coordinates of the longitude
    and latitude of the particular basin.
    """
    def get_bounds(self):
        bounds = self.basin_bounds

        ll_lon = self._adj_bnd(bounds[0])
        ll_lat = self._adj_bnd(bounds[1])
        ul_lon = self._adj_bnd(bounds[2])
        ul_lat = self._adj_bnd(bounds[3])

        return(ll_lon, ll_lat, ul_lon, ul_lat)

    """
    Reduces a global field to the basin boundaries.
    The global field has dimensions [latitude, longitude],
    which are described by lon, lat.
    """
    def transform_global_field(self, lon, lat, field):
        lon_min, lat_min, lon_max, lat_max = self.get_bounds()

        if lon[0] >= -1e-5 and (lon_min < 0 or lon_max < 0):
            # If the basin bounds are phrased in negative longitude
            # coordinates, transform the field into those coordinates.
            lon_t, X_t = self.transform_lon(lon, field)
        elif (lon < 0).any() and lon_min >= 0:
            # If the grid is phrased in negative longitude coordinates,
            # transform the grid into those coordinates.
            lon_t, X_t = self.transform_lon_r(lon, field)
        else:
            lon_t = lon
            X_t = field

        lon_mask = np.logical_and(lon_t <= (lon_max+1e-5), lon_t >= (lon_min - 1e-5))
        lat_mask = np.logical_and(lat >= (lat_min-1e-5), lat <= (lat_max + 1e-5))
        X_c = X_t[lat_mask, :]
        return (lon_t[lon_mask], lat[lat_mask], X_c[:, lon_mask])

    """
    Returns the basin array size.
    """
    def get_basin_size(self, lon, lat):
        lon_min, lat_min, lon_max, lat_max = self.get_bounds()
        if lon_min < 0 or lon_max < 0:
            lon_t, _ = self.transform_lon(lon, np.zeros((lat.size, lon.size)))
        else:
            lon_t = lon
        lon_mask = np.logical_and(lon_t <= lon_max, lon_t >= lon_min)
        lat_mask = np.logical_and(lat >= lat_min, lat <= lat_max)
        return (lat[lat_mask].size, lon_t[lon_mask].size)

    """
    Transform a field with longitude from 0-360E to -180-180.
    """
    def transform_lon(self, lon, X):
        lon_mask = lon >= (180 - 1e-5)
        X_t = np.concatenate((X[:, lon_mask],
                              X[:, np.logical_not(lon_mask)]), axis=1)
        lon_t = np.hstack((lon[lon_mask] - 360, lon[np.logical_not(lon_mask)]))
        return (lon_t, X_t)

    """
    Transform a field with longitude from -180-180 to 0-360E.
    """
    def transform_lon_r(self, lon, X):
        lon_mask = lon < -1e-5  # 0
        X_t = np.concatenate((X[:, np.logical_not(lon_mask)], X[:, lon_mask]), axis=1)
        lon_t = np.hstack((lon[np.logical_not(lon_mask)], lon[lon_mask] + 360))
        return (lon_t, X_t)
