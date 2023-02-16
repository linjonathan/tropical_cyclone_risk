#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Author: Jonathan Lin
Utility library for dealing with lon/lat matrices.
"""

import numpy as np
from numpy import linalg as la
from scipy.interpolate import RectBivariateSpline

"""
Returns the longitude and latitude vector given lonn/latn spacings in
the below range.
Longitude range is from [0, 360), and latitude range is from [90, -90).
"""
def lon_lat(lonn, latn):
    lon = np.linspace(0, 360, lonn+1)[0:-1]
    lat = np.linspace(90, -90, latn+1)[0:-1]
    return(lon, lat)

"""
Transform longitude from 0-360 to -180-180.
"""
def transform_lon(lon, X):
    lon_mask = lon >= 180
    X_t = np.concatenate((X[:, lon_mask],
                          X[:, np.logical_not(lon_mask)]), axis=1)
    lon_t = np.hstack((lon[lon_mask] - 360, lon[np.logical_not(lon_mask)]))
    return(lon_t, X_t)

def lon_lat_mask(lon, lat, lon_mask, lat_mask, X):
    X_c = X[lat_mask, :]; X_c = X_c[:, lon_mask]
    lon_c = lon[lon_mask]
    lat_c = lat[lat_mask]
    return(lon_c, lat_c, X_c)


"""
Returns the rectangule composed by (lat_idx1, lat_idx2, lon_idx1, lon_idx2)
such that:
    lat[lat_idx1] < clat < lat[lat_idx2]
    lon[lon_idx1] < clon < lon[lon_idx2]
"""
def find_neighbors(lon, lat, clon, clat):
    lat_idx = np.abs(lat - clat).argmin()
    lon_idx = np.abs(lon - clon).argmin()

    dlat = lat[lat_idx] - clat
    dlon = lon[lon_idx] - clon

    # Offsets for the enclosing indices.
    lat1_off = 0; lat2_off = 0;
    lon1_off = 0; lon2_off = 0;

    if dlat <= 0:
        lat2_off = int(np.sign(lat[1] - lat[0]) * 1)
    elif dlat > 0:
        lat1_off = int(np.sign(lat[1] - lat[0]) * -1)

    if dlon <= 0:
        lon2_off = int(np.sign(lon[1] - lon[0]) * 1)
    elif dlon > 0:
        lon1_off = int(np.sign(lon[1] - lon[0]) * -1)
    lat_idxs = [lat_idx + lat1_off, lat_idx + lat2_off]
    lon_idxs = [lon_idx + lon1_off, lon_idx + lon2_off]

    points = []
    for lat_idx in lat_idxs:
        for lon_idx in lon_idxs:
            points.append([lon_idx % lon.size, lat_idx])

    return(points)

"""
Interpolate (clon,clat) from values associated with four points.

The four points are a list of four triplets:  (lon, lat, value).
The four points can be in any order.  They should form a rectangle.

    >>> bilinear_interpolation(12, 5.5,
    ...                        [(10, 4, 100),
    ...                         (20, 4, 200),
    ...                         (10, 6, 150),
    ...                         (20, 6, 300)])
    165.0
StackOverflow: Q-8661537.
"""
def bilinear_interpolation(clon, clat, points):
    points = sorted(points)               # order points by x, then by y
    (x1, y1, q11), (_x1, y2, q12), (x2, _y1, q21), (_x2, _y2, q22) = points

    if np.isnan(clon):
        return np.nan

    if (x1 != _x1 or x2 != _x2 or y1 != _y1 or y2 != _y2):
        print('(%f, %f, %f), (%f, %f, %f)' % (x1, clon, x2, y1, clat, y2))
        raise ValueError('points do not form a rectangle')
    if (not x1 <= clon <= x2 or not y1 <= clat <= y2):
        raise ValueError('(%d, %d) not within the rectangle' % (clon, clat))

    return (q11 * (x2 - clon) * (y2 - clat) +
            q21 * (clon - x1) * (y2 - clat) +
            q12 * (x2 - clon) * (clat - y1) +
            q22 * (clon - x1) * (clat - y1)) / ((x2 - x1) * (y2 - y1) + 0.0)

"""
Interpolates x in space time, where x has dimensions [lat, lon, time].
Interpolate linearly in time, using spacings t to time ct, and bilinearly
in space, using spacings lat/lon and location clat/clon.
"""
def interp_space_time(x, lon, lat, t, clon, clat, ct):
    rect = find_neighbors(lon, lat, clon, clat)
    if rect == None:
        return(None)
    else:
        points = []
        for p in rect:
            plon = lon[p[0]]
            plat = lat[p[1]]
            points.append((plon, plat, np.interp(ct, t, x[p[1], p[0], :])))
        return(bilinear_interpolation(clon, clat, points))

"""
Interpolates x in space time, where x has dimensions [lat, lon].
Interpolate bilinearly in space, using spacings lat/lon and location clat/clon.
"""
def interp_space(x, lon, lat, clon, clat):
    rect = find_neighbors(lon, lat, clon, clat)
    if rect == None:
        return(None)
    else:
        points = []
        for p in rect:
            plon = lon[p[0]]
            points.append((plon, lat[p[1]], x[p[1], p[0]]))
        return(bilinear_interpolation(clon, clat, points))

"""
Returns the function that is the interpolation of a 2-D field.
"""
def interp2_fx(lon, lat, X):
    if lat[1] - lat[0] < 0:
        # Reverse grid since x and y must be strictly increasing.
        r_lat = np.flip(lat, 0)
        r_X = np.flip(X, 0).T
    else:
        r_lat = lat
        r_X = X.T

    # Interpolate SST grid to the model grid.
    f_X = RectBivariateSpline(lon, r_lat, r_X, kx=1, ky=1)
    return(f_X)

"""
2-D interpolation. Interpolates a field X (dimensions [lat, lon]),
to a grid defined by [lat_grid, lon_grid].
"""
def interp_2d_grid(lon, lat, X, lon_grid, lat_grid):
    f_X = interp2_fx(lon, lat, X)

    LON, LAT = np.meshgrid(lon_grid, lat_grid)
    X_grid = f_X.ev(LON, LAT)
    return(X_grid)

"""
2-D interpolation. Interpolates a field X (dimensions [lat, lon]),
to a grid defined by [lat_grid, lon_grid].
"""
def interp_2d_points(lon, lat, X, interp_lons, interp_lats):
    f_X = interp2_fx(lon, lat, X)
    X_grid = f_X.ev(interp_lons, interp_lats)
    return(X_grid)

"""Find the nearest positive-definite matrix to input

A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
credits [2].

[1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

[2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
"""
def nearestPD(A):
    B = (A + A.T) / 2
    _, s, V = la.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(la.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False

def smooth_anomaly(lon_idx, lat_idx, X, dx):
    X_smooth = np.copy(X)
    X_temp = np.copy(X)

    for niter in range(15):
        for i in range(-dx, dx):
            for j in range(-dx, dx):
                clon = int(lon_idx + i)
                clat = int(lat_idx + j)
                X_box = X_smooth[(clat-1):(clat+2), (clon-1):(clon+2)]
                X_temp[clat, clon] = np.nanmean(X_box)

        X_smooth = X_temp
    return(X_smooth)
