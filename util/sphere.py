#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Jonathan Lin
Utility library that implements common functions on the sphere.
"""

import numpy as np
from util import constants

"""
Implements the Haversine algorithm to calculate the great circle distance
between two points. Returns distance in kilometers.
"""
def haversine(lon1, lat1, lon2, lat2):
    # Convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lat1 = np.deg2rad(lat1)
    lon2 = np.deg2rad(lon2)
    lat2 = np.deg2rad(lat2)

    # Use the Haversine formula.
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.square(np.sin(dlat/2)) + np.cos(lat1) *
         np.cos(lat2) * np.square(np.sin(dlon/2)))
    c = 2 * np.arcsin(np.sqrt(a))

    km = (constants.earth_R / 1000.) * c
    return(km)

"""
Returns the cartesian angles from a center (lonc, latc)
"""
def sphere_theta(lonc, latc, lon_grid, lat_grid):
    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    lon_dist = haversine(LON, latc, lonc, latc)
    lat_dist = haversine(lonc, LAT, lonc, latc)

    with np.errstate(invalid='ignore'):
        lon_dist[LON < lonc] *= -1
        lat_dist[LAT < latc] *= -1
    theta = np.arctan2(lat_dist, lon_dist)

    return(theta)

""" Transforms x-y distances to lon-lat distances (Cartesian approximation). """
def to_sphere_dist(clon, clat, dx, dy):
    p_lat = clat + (dy / constants.earth_R) * (180. / np.pi)
    p_lon = clon + (dx / constants.earth_R) * (180. / np.pi) / np.cos(clat * np.pi / 180.)
    return(p_lon, p_lat)

"""
Calculates the translational speed given a vector of lon/lat, and dt_s which is
the discrete time displacement in the lon/lat vectors. Returns in m/s.
dt_s can be either a constant or a vector.
"""
def calc_translational_speed(lon, lat, dt_s):
    if len(lon) <= 1:
        return(np.full(1, np.nan), np.full(1, np.nan))
    elif len(lon.shape) == 1:
        lon = np.expand_dims(lon, 0)
        lat = np.expand_dims(lat, 0)

    fa = lambda x, idx: np.expand_dims(x[:, idx], 1)
    e_lon = np.hstack((2*fa(lon,0) - fa(lon, 1), lon[:],
                       2*fa(lon,-1) - fa(lon,-2)))
    e_lat = np.hstack((2*fa(lat,0) - fa(lat, 1), lat[:],
                       2*fa(lat,-1) - fa(lat,-2)))

    dlon = 0.5 * (np.sign(e_lon[:, 2:] - e_lon[:, 0:-2]) *
                   haversine(e_lon[:,2:], e_lat[:,1:-1],
                             e_lon[:,0:-2], e_lat[:,1:-1]))
    dlat = 0.5 * (np.sign(e_lat[:,2:] - e_lat[:,0:-2]) *
                   haversine(e_lon[:,1:-1], e_lat[:,2:],
                             e_lon[:,1:-1], e_lat[:,0:-2]))

    ut = dlon * 1000. / dt_s
    vt = dlat * 1000. / dt_s
    if len(lon.shape) == 1:
        ut = ut.flatten(); vt = vt.flatten();

    return(ut, vt)
