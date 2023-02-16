#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jzlin@mit.edu
"""

earth_R = 6.3781 * (10**6)        # mean radius of the earth (m)
T_trip = 273.16  # temperature at the triple point (K)
e_trip = 611.65  # pressure at the triple point (Pa)
Rd = 287.04      # gas constant for dry air (J/kg/K)
Rv = 461.5       # gas constant for water vapor (J/kg/K)
cv = 718         # specific heat of dry air at constant pressure (J/kg/K)
cp = cv + Rd     # specific heat of dry air at constant pressure (J/kg/K)
cpv = 1870       # specific heat of water vapor at constant pressure (J/kg/K)
cl = 4190        # specific heat of liquid water (J/kg/K)
eps = Rd/Rv      # ratio of dry air and water vapor gas constants (-)
Lv = 2.5e6       # latent heat of vaporization (J/kg/K)
L0 = 2.555e6     # Constant latent heat of vaporization for pseudoadoabatic computations (J/kg/K) (Bryan 2008)
C_to_K = 273.15  # factor to convert celsius to kelvin
