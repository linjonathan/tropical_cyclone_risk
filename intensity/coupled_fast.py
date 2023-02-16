#!/usr/bin/env python
"""
Author: Jonathan Lin
Implementation of FAST (Emanuel, 2017) that is coupled to the track.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d, RectBivariateSpline
import warnings

import namelist
from intensity import geo
from thermo import thermo
from track import bam_track, env_wind
from util import constants

class Coupled_FAST(bam_track.BetaAdvectionTrack):
    def __init__(self, fn_wnd_stat, basin, dt_start, dt_s, total_time_s):
        super().__init__(fn_wnd_stat, basin, dt_start, dt_s, total_time_s)

        """ FAST Constants """
        self.Ck = namelist.Ck                       # surface enthalpy coefficient
        self.h_bl = namelist.atm_bl_depth           # boundary layer depth (m)
        self.epsilon = 0.33                         # thermodynamic efficiency
        self.kappa = 0.1                            # dimensionless parameter
        self.beta = 1 - self.epsilon - self.kappa   # dimensionless parameter

        # Read in high-resolution bathymetry and land masks.
        self.f_bath = geo.read_bathy(basin)
        self.f_land = geo.read_land(basin)
        self.debug = False

    """ Return if over land (True) or ocean (False) """
    def _get_over_land(self, clon, clat):
        # 9/2/2020: Changed to 1 (and not a rounded number), since PI
        # is changed to be reduced by the area of the core over land.
        return(self.f_land.ev(clon, clat).flatten()[0] == 1)

    """ Return the current bathymetry/topography at a position."""
    def _get_current_bathymetry(self, clon, clat):
        return self.f_bath.ev(clon, clat).flatten()[0]

    """ Return the current mixed layer depth at a position. """
    def _get_current_mixed(self, clon, clat):
        return self.f_mld.ev(clon, clat).flatten()[0]

    """ Return the current sub-mixed layer thermal stratification, (K / 100 m)
        at time t_s."""
    def _get_current_strat(self, clon, clat):
        return self.f_strat.ev(clon, clat).flatten()[0]

    """ Return the current potential intensity at a position"""
    def _get_current_vpot(self, clon, clat):
        if self._get_over_land(clon, clat):
            return(0)
        else:
            return self.f_vpot.ev(clon, clat).flatten()[0]

    """ Calculate, alpha (Equation 4), the ocean feedback parameter.
    Ocean mixing is turned off when the mixed layer depth equals or
    exceeds the local ocean depth (alpha = 1), or when the hurricane
    is over land.
    """
    def _calc_alpha(self, clon, clat, v_trans, v):
        h_m = self._get_current_mixed(clon, clat)
        t_strat = self._get_current_strat(clon, clat)
        v_pot = self._get_current_vpot(clon, clat)
        u_T = np.linalg.norm(v_trans)
        bathymetry = self._get_current_bathymetry(clon, clat)

        # When over land, h_m = NaN, so we make alpha = 1.
        if (bathymetry >= 0 or -h_m <= bathymetry or t_strat == 0):
            return(1)
        else:
            z = self._calc_z(v, h_m, v_pot, u_T, t_strat)
            try:
                fac = np.exp(-np.clip(z, 0, 100))
            except OverflowError:
                fac = np.nan
            return(1 - 0.87 * fac)

    """ Calculate z (Equation 5) at a time t.
    v is the current intensity in m/s.
    h_m is the ocean mixed layer depth in meters.
    v_pot is the potential intensity in m/s.
    u_T is the storm translation speed in m/s.
    t_strat is the sub-mixed layer thermal stratification in K per 100 m.
    """
    def _calc_z(self, v, h_m, v_pot, u_T, t_strat):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            z = 0.01 * (t_strat ** -0.4) * h_m * u_T * v_pot / v
        return z

    """ Calculate the dimensionless parameter beta (Equation 6). """
    def _calc_beta(self):
        return self.beta

    """ Calcuate the dimensionless parameter gamma (Equation 7).
        gamma varies with v, since it is a function of alpha."""
    def _calc_gamma(self, alpha):
        return self._calc_epsilon() + alpha * self._calc_kappa()

    """ Calculate the dimensionless parameter epsilon (Equation 8). """
    def _calc_epsilon(self):
        return self.epsilon

    """ Calculate the dimensionless parameter kappa (Equation 9). """
    def _calc_kappa(self):
        return self.kappa

    """ Calculate the shear vector from a vector of environmental winds.
        The shear vector is [zonal shear, meridional shear]."""
    def _calc_env_wnd_to_shr(self, env_wnds):
        u250, v250, u850, v850 = env_wind.deep_layer_winds(np.expand_dims(env_wnds, axis = 0))
        shr = np.array([u250 - u850, v250 - v850]).flatten()
        return shr

    """ Calculate the magnitude of the 250-850 hPa environmental wind shear."""
    def _calc_S(self, env_wnds):
        return np.linalg.norm(self._calc_env_wnd_to_shr(env_wnds))

    """ Calculate the normalized mid-level saturation entropy deficit."""
    def _calc_chi(self, clon, clat):
        return self.f_chi.ev(clon, clat).flatten()[0]

    """ Calculate the ventilation. """
    def _calc_venti(self, t, clon, clat, env_wnds):
        chi = self._calc_chi(clon, clat)
        return (self._calc_S(env_wnds) * chi)

    """ Define the first ODE in the coupled set, Equation 2.
    For now, use a drag coefficient, constant boundary layer depth,
    constant thermodynamic efficiency (Equation 8), and constant
    kappa (Equation 9). This means beta is constant (Equation 6).
    However, alpha still varies (Equation 4, 5) with time, which means
    gamma must vary with time as well (Equation 7).
    Potential intensity (v_p) also varies with time.
    """
    def _dvdt(self, clon, clat, v, m, v_trans, env_wnds, t):
        v_pot = self._get_current_vpot(clon, clat)
        alpha = self._calc_alpha(clon, clat, v_trans, v)
        gamma = self._calc_gamma(alpha)
        beta = self._calc_beta()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dvdt = (0.5 * self.Ck / self.h_bl * (alpha * beta * (v_pot ** 2) * (m ** 3) -
                                                      (1 - gamma * (m ** 3)) * (v ** 2)))
        return dvdt if ~np.isnan(dvdt) else 0

    """ Initializes m if no m has been given. Assumes an initial dvdt. """
    def _init_m(self, y, dvdt):
        steering_coefs = self._calc_steering_coefs(y[2])
        v_bam, env_wnds = self._step_bam_track(y[0], y[1], 0., steering_coefs)
        v_pot = np.max([self._get_current_vpot(y[0], y[1]),
                        self._get_current_vpot(y[0]-0.25, y[1]-0.25),
                        self._get_current_vpot(y[0]-0.25, y[1]+0.25),
                        self._get_current_vpot(y[0]+0.25, y[1]-0.25),
                        self._get_current_vpot(y[0]+0.25, y[1]+0.25)])
        alpha = self._calc_alpha(y[0], y[1], v_bam, y[2])
        gamma = self._calc_gamma(alpha)
        beta = self._calc_beta()

        numer = 2 * self.h_bl / self.Ck * dvdt + (y[2] ** 2)
        denom = alpha * beta * (v_pot ** 2) + gamma * (y[2] ** 2)
        return(np.maximum(np.minimum(np.cbrt(numer / denom), 1), 0))

    """ Define the second ODE in the coupled set, Equation 3.
    For now, use a drag coefficient, constant boundary layer depth,
    constant thermodynamic efficiency (Equatton 8), and constant
    kappa (Equation 9).
    However, environmental wind shear (S) still varies with time.
    """
    def _dmdt(self,  clon, clat, v, m, env_wnds, t):
        venti = self._calc_venti(t, clon, clat, env_wnds)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            dmdt = 0.5 * self.Ck / self.h_bl * ((1 - m) * v - venti * m)
        return(dmdt)

    """ Calculate the steering coefficients. """
    def _calc_steering_coefs(self, v):
        assert len(namelist.steering_coefs) == len(namelist.steering_levels)
        if namelist.coupled_track:
            alpha_fx = (v*1.94384)*np.array(namelist.m_alpha) + np.array(namelist.y_alpha)
            steering_coefs = np.maximum(np.minimum(alpha_fx, namelist.alpha_max), namelist.alpha_min)
            if np.any(np.isnan(steering_coefs)):
                steering_coefs = np.array(namelist.y_alpha)
        else:
            steering_coefs = np.array(namelist.steering_coefs)
        return steering_coefs

    """ Time-derivative of the state vector, y.
    y[0] is longitude, y[1] is latitude, y[2] is v, and y[3] is m."""
    def dydt(self, t, y):
        steering_coefs = self._calc_steering_coefs(y[2])
        v_bam, env_wnds = self._step_bam_track(y[0], y[1], t, steering_coefs)
        dLondt = v_bam[0] / constants.earth_R * 180. / np.pi / (np.cos(y[1] * np.pi / 180.))
        dLatdt = v_bam[1] / constants.earth_R * 180. / np.pi

        dvdt = self._dvdt(*y, v_bam, env_wnds, t)
        dmdt = self._dmdt(*y, env_wnds, t)
        if self.debug:
            return(np.array([0, 0, dvdt, dmdt]))
        else:
            return(np.array([dLondt, dLatdt, dvdt, dmdt]))

    """
    lon is a 1-D array describing the longitude of the fields: [lon]
    lat is a 1-D array describing the latitude of the fields: [lat]
    chi is a 2-D matrix of chi in space: [lat, lon]
    vpot is a 2-D matrix of potential intensity in space: [lat, lon]
    mld is a 2-D matrix of mixed-layer depth in space: [lat, lon]
    strat is a 2-D matrix of sub-mixed layer thermal stratification in space: [lat, lon]
    """
    def init_fields(self, lon, lat, chi, vpot, mld, strat):
        lon_b, lat_b, chi_b = self.basin.transform_global_field(lon, lat, chi)
        self.f_chi = RectBivariateSpline(lon_b, lat_b, chi_b.T, kx=1, ky=1)
        _, _, vpot_b = self.basin.transform_global_field(lon, lat, vpot)
        self.f_vpot = RectBivariateSpline(lon_b, lat_b, vpot_b.T, kx=1, ky=1)
        _, _, mld_b = self.basin.transform_global_field(lon, lat, mld)
        self.f_mld = RectBivariateSpline(lon_b, lat_b, mld_b.T, kx=1, ky=1)
        _, _, strat_b = self.basin.transform_global_field(lon, lat, strat)
        self.f_strat = RectBivariateSpline(lon_b, lat_b, strat_b.T, kx=1, ky=1)

    """ Generate a track with an initial position of (clon, clat),
        an initial intensity of v, and initial inner core moisture m """
    def gen_track(self, clon, clat, v, m = None):
        # Make sure that tracks are sufficiently randomized.
        bam_track.random_seed()

        # Create the weights for the beta-advection model (across time).
        self.Fs = self.gen_synthetic_f()
        self.Fs_i = interp1d(self.t_s, self.Fs, axis = 1)

        # If the ventilation index is above some threshold, do not integrate.
        S = self._calc_S(self._env_winds(clon, clat, 0))
        vpot = self._get_current_vpot(clon, clat)
        chi = self._calc_chi(clon, clat)
        if vpot > 0:
            vent_index = S * chi / vpot
            if vent_index >= 1:
                return None

        def tc_dissipates(t, y):
            if not self.basin.in_basin(y[0], y[1], 1):
                # Do not let the track wander outside the basin.
                return 0
            elif np.abs(y[1]) <= 2:
                # Do not let the track wander too equatorward.
                return 0
            else:
                # stopping point when TC reaches 4 m/s
                return np.maximum(0, y[2] - 4)
        tc_dissipates.terminal = True

        # Solve for the intensity.
        if m is None:
            # This means no m has been provided. Initialize with dvdt = 0.
            m_init = self._init_m(np.asarray([clon, clat, v]), 0)
        else:
            m_init = m
        res = solve_ivp(self.dydt, (0, self.total_time), np.asarray([clon, clat, v, m_init]),
                        t_eval = np.linspace(0, self.total_time, self.total_steps),
                        events = tc_dissipates, max_step = 86400)
        return res
