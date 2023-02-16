#!/usr/bin/env python
"""
Author: Raphael
Utility library for thermodynamics and PI computations.
"""

# Import packages
import numpy as np
# For inverting entropy
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interpn
from scipy.optimize import minimize
from scipy.special import lambertw
# Import parameters
import namelist
from util import constants as pr

""" Saturation mixing ratio and saturation vapor pressure computation. """
def sat_thermo_pog(T,p):
    # Reference, for formula assuming no ice and constant Lv
    # Paul O'Gorman simple formulation
    # Saturation vapor pressure
    es   = pr.e_trip * np.exp(pr.Lv/pr.Rv*(1.0/pr.T_trip - 1.0/T))
    # Saturation mixing ratio
    rs   = pr.Rd/pr.Rv*es/(p - es)
    return es, rs

""" Saturation mixing ratio and saturation vapor pressure computation, Bolton. """
def sat_thermo(T,p):
    # Saturation vapor pressure
    T_c = T - 273  # Celsius
    es = np.zeros(T_c.shape)
    mask = np.logical_not(np.isnan(T_c))
    es[mask] = 610.94 * np.exp(np.minimum(np.divide(17.625 * T_c[mask],
                                                     T_c[mask] + 243.04), 10))
    # Saturation mixing ratio
    rs   = pr.Rd/pr.Rv*es/(p - es)                                                     
    return es, rs

""" Convert specific humidity to relative humidity. """
def conv_q_to_rh(T, q, p_Pa):
    # Convert specific humidity to relative humidity.
    es, rs = sat_thermo(T, p_Pa)
    qs = rs / (1 + rs)
    rh = np.minimum(np.maximum(np.divide(q, qs), 1e-5), 1)
    return rh

""" Moist entropy """
def s_unsat(T,p,r,r_t,select_thermo):
    # Saturation thermodynamics
    es, rs = sat_thermo(T,p)
    rh = np.maximum(r/rs*(1+rs/pr.eps)/(1+r/pr.eps), 0)
    if select_thermo == 1:
        # Pseudoadiabatic computation
        s = pr.cp*np.log(T) -pr.Rd*np.log(p-es*rh) + pr.L0*r/T - r*pr.Rv*np.log(rh)
    elif select_thermo == 2:
        # Reversible computation
        L = pr.Lv - (pr.cpv-pr.cl)*(273.15-T)
        s = (pr.cp+pr.cl*r_t)*np.log(T) -pr.Rd*np.log(p-es*rh) + L*r/T - r*pr.Rv*np.log(rh)
    return s


""" Saturation entropy """
def s_sat(T,p,r_t,select_thermo):
    # Saturation thermodynamics
    es, rs = sat_thermo(T,p)
    T = np.maximum(T, 1e-4)
    if select_thermo == 1:
        # Pseudoadiabatic computation
        ss = pr.cp*np.log(T) -pr.Rd*np.log(np.maximum(p-es, 1e-4)) + pr.L0*rs/T
    elif select_thermo == 2:
        # Reversible computation
        L = pr.Lv - (pr.cpv-pr.cl)*(273.15-T)
        ss = (pr.cp+r_t*pr.cl)*np.log(T) -pr.Rd*np.log(np.maximum(p-es, 1e-4)) + L*rs/T
    return ss

""" Analytical temperature derivative of saturation entropy """
def s_sat_der(T,p,r_t,select_thermo):
    # Saturation thermodynamics
    es, rs = sat_thermo(T,p)
    if select_thermo == 1:
        # Temperature derivative of entropy guess
        ss_deriv = 1/T*(pr.cp + pr.L0**2*rs/pr.Rv/T**2 * (1 - rs/pr.eps))
    elif select_thermo == 2:
        # Non-constant heat of condensation
        L = pr.Lv - (pr.cpv-pr.cl)*(273.15-T)
        # Temperature derivative of entropy guess
        ss_deriv = 1/T*(pr.cp + pr.cpv*rs + pr.cl*(r_t-rs) + L**2*rs/pr.Rv/T**2 * (1 - rs/pr.eps))
    return ss_deriv

""" Saturation deficit computation """
def sat_deficit(sst,ps,T,pm,rv):
    # sst and ps are surface temperature [K] and pressure [Pa]
    # T, pm and rv and mid-level temperature [K], pressure [Pa] and mixing ratio [-]
    # Midlevel entropy
    select_thermo = namelist.select_thermo
    sp = s_unsat(T,pm,rv,rv,select_thermo)
    # Midlevel saturation entropy
    sps = s_sat(T,pm,rv,select_thermo)
    # Surface saturation entropy
    spss = s_sat(sst,ps,rv,select_thermo)
    # Saturation deficit
    chi = (sps-sp)/(spss-sps);
    return chi

"""LCL computations Based on "Exact expression for the lifting condensation level", Romps 2017 """
def get_LCL(p,T,r,rh):
    # Internal parameters
    E0v   = 2.3740e6   # J/kg
    cvv   = 1418       # J/kg/K
    cvl   = 4119       # J/kg/K
    cpv   = cvv + pr.Rv
    # Convert mixing ratio to specific humidity
    q = r/(1+r)
    # Compute moist gas constant and heat capacity
    Rm = (1-q)*pr.Rd + q*pr.Rv
    cpm = (1-q)*pr.cp + q*cpv
    # Compute intermediate variables
    a = cpm/Rm + (cvl - cpv)/pr.Rv
    b = -(E0v - (cvv-cvl)*pr.T_trip)/(pr.Rv*T)
    c = b/a
    # Compute T_LCL
    T_LCL = c*T/(lambertw(rh**(1/a)*c*np.exp(c),-1).real)
    # Compute p_LCL
    p_LCL = p*(T_LCL/T)**(cpm/Rm)
    return p_LCL

"""Density temperature computation"""
def calc_T_rho(T,rv,rt):
    if namelist.select_thermo == 1:
        T_rho = T*(1+rv/pr.eps)/(1+rv)
    elif namelist.select_thermo == 2:
        T_rho = T*(1+rv/pr.eps)/(1+rt)
    return T_rho

""" Function to compute potential intensity
Inputs:
2D sst array [K]
2D near-surface pressure array [Pa]
1D Environmental pressure sounding [Pa]
3D Environmental arrays of temperature [K] and water vapor mixing ratio
Output:
Potential intensity [m/s]"""
def CAPE_PI(sst,p_surf,p_env,T_env,r_env):
    # Ratio of the exchange coefficients
    cecd = namelist.Ck / namelist.Cd;
    # Select thermodynamics: Set to 1 for pseudoadiabatic, 2 for reversible
    select_thermo = namelist.select_thermo
    # Select inversion method: Set to 1 for computation, 2 for interpolation
    select_interp = namelist.select_interp
    # If interpolation, load temperature table and coordinates
    if namelist.select_interp == 2:
        if select_thermo == 1:
            with np.load('%s/thermo/entropy_table.npz' % namelist.src_directory) as entropy_table:
                p_look = entropy_table['p']
                s_look = entropy_table['s']
                T_lookup = entropy_table['T']
        elif select_thermo == 2:
            with np.load('%s/thermo/entropy_table_reversible.npz' % namelist.src_directory) as entropy_table:
                p_look = entropy_table['p']
                s_look = entropy_table['s']
                rt_look = entropy_table['rt']
                T_lookup = entropy_table['T']
    # Dimensions of the input
    dim_in = T_env.shape
    # Near surface conditions: For now, assume that parcels at the first level
    # are representative of near surface parcels
    T_ns = T_env[0,:,:]
    r_ns = r_env[0,:,:]
    p_ns = p_env[0]
    # Initialize saturated parcels at SST
    ess, rs = sat_thermo(sst,p_surf)
    # Compute RH in near-surface parcels
    rh = r_ns/rs*(1+rs/pr.eps)/(1+r_ns/pr.eps)
    # Compute reference entropies of:
    # BL parcel
    s_ns = s_unsat(T_ns,p_ns,r_ns,r_ns,select_thermo)
    # Saturated parcels at SST
    ss = s_sat(sst,p_surf,rs,select_thermo)
    # Initialize arrays for the CAPE computation
    lnp = np.log(p_env)
    dlnp = np.diff(lnp,n=1,append = (2*lnp[-1] - lnp[-2]))
    T_rho_env = calc_T_rho(T_env,r_env,r_env)
    # Initialize ascent profiles
    Ta_prof = np.zeros(dim_in)
    T_rho_a = np.zeros(dim_in)
    ra_prof = np.zeros(dim_in)
    Ts_prof = np.zeros(dim_in)
    T_rho_s = np.zeros(dim_in)
    rs_prof = np.zeros(dim_in)
    # Initialize PI
    PI = np.zeros(sst.shape)
    # Compute LCL pressure and index
    pLCL = get_LCL(p_ns,T_ns,r_ns,rh)
    if select_thermo == 1:
        # Create interpolation function for solutions of entropy inversion.
        f_lookup = RectBivariateSpline(p_look, s_look, T_lookup, kx=1, ky=1)
    for hh in np.arange(dim_in[1]):
        for gg in np.arange(dim_in[2]):
            # Invert for temperature, using dry adiabat under pLCL and moist adiabat above
            Icond_idxs = np.where(pLCL[hh,gg]>p_env)[0]
            if len(Icond_idxs) == 0:
                continue
            Icond = Icond_idxs[0]       # first index where there is condensation
            # For lower levels (higher pressure), compute T based on a dry adiabat
            # NOTE THAT THIS NOTATION EXCLUDES ICOND
            Ta_prof[0:Icond,hh,gg] = T_ns[hh,gg]*(p_env[0:Icond]/p_ns)**(pr.Rd/pr.cp) # Factor *(1-0.24*r) neglected because very very small
            # Maintain constant vapor mixing ratio
            ra_prof[0:Icond,hh,gg] = r_ns[hh,gg]
            if select_interp == 1:
                # Then, compute T by inverting moist adiabats
                jj = 0
                for p_i in p_env[Icond:]:
                    Ta_prof[Icond+jj,hh,gg] = minimize(s_diff, 230.0, args = (p_i,r_ns[hh,gg],select_thermo,s_ns[hh,gg]),
                                                       method='BFGS', jac=s_diff_der,  options={'gtol': 1e-02}).x
                    jj += 1
                # For the saturated parcel, pLCL = p_ns, so only invert moist adiabat
                jj = 0
                for p_i in p_env:
                    Ts_prof[jj,hh,gg] = minimize(s_diff, 230.0, args = (p_i,rs[hh,gg],select_thermo,ss[hh,gg]),
                                                 method='BFGS', jac=s_diff_der, options={'gtol': 1e-02}).x
                    jj += 1
            elif select_interp == 2:
                if select_thermo == 1:
                    # Then, compute T by inverting moist adiabats
                    Ta_prof[Icond:,hh,gg] = f_lookup.ev(p_env[Icond:], np.full((len(p_env[Icond:]),), s_ns[hh,gg]))
                    # For the saturated parcel, pLCL = p_ns, so only invert moist adiabat
                    Ts_prof[:,hh,gg] = f_lookup.ev(p_env, np.full((len(p_env),), ss[hh,gg]))
                elif select_thermo == 2:
                    Ta_prof[Icond:,hh,gg] = interpn((p_look,s_look,rt_look),T_lookup,(p_env[Icond:],s_ns[hh,gg],r_ns[hh,gg]), method='linear', bounds_error = False, fill_value=np.nan)
                    Ts_prof[:,hh,gg] = interpn((p_look,s_look,rt_look),T_lookup,(p_env,ss[hh,gg],rs[hh,gg]), method='linear',  bounds_error = False, fill_value=np.nan)
            # Compute mixing ratio of ascent profiles
            tmp, ra_prof[Icond:,hh,gg] = sat_thermo(Ta_prof[Icond:,hh,gg],p_env[Icond:])
            tmp, rs_prof[:,hh,gg] = sat_thermo(Ts_prof[:,hh,gg],p_env)
            # Compute density temperature. Assume that there is no liquid or ice water at lower level.
            T_rho_a[:,hh,gg] = calc_T_rho(Ta_prof[:,hh,gg],ra_prof[:,hh,gg],r_ns[hh,gg])
            T_rho_s[:,hh,gg] = calc_T_rho(Ts_prof[:,hh,gg],rs_prof[:,hh,gg],rs[hh,gg])
            # Compute CAPE differences
            # Find LNB for saturated and BL profiles
            a_out_I_idxs = np.where(T_rho_a[:,hh,gg]>=T_rho_env[:,hh,gg])[0]
            s_out_I_idxs = np.where(T_rho_s[:,hh,gg]>=T_rho_env[:,hh,gg])[0]
            if (len(a_out_I_idxs) > 0) and (len(s_out_I_idxs) > 0):
                a_out_I = a_out_I_idxs[-1]
                s_out_I = s_out_I_idxs[-1]
                # Get outflow temperature
                T_out = Ts_prof[s_out_I,hh,gg];
                # Compute CAPE
                CAPE = np.sum(pr.Rd*(T_rho_a[0:a_out_I+1,hh,gg]-T_rho_env[0:a_out_I+1,hh,gg])*-dlnp[0:a_out_I+1], axis=0)
                # Saturated CAPE
                CAPEs = np.sum(pr.Rd*(T_rho_s[0:s_out_I+1,hh,gg]-T_rho_env[0:s_out_I+1,hh,gg])*-dlnp[0:s_out_I+1], axis=0)
                # Difference of the CAPEs
                CAPE = np.maximum(CAPE,0)
                CAPE[np.isnan(CAPE)] = 0
                cape_diff = CAPEs-CAPE
                PI[hh,gg] = (cecd*sst[hh,gg]/T_out*cape_diff)**0.5
    return PI

""" Vectorized version of a function to compute potential intensity
Inputs:
2D sst array [K]
2D near-surface pressure array [Pa]
1D Environmental pressure sounding [Pa]
3D Environmental arrays of temperature [K] and water vapor mixing ratio
Output:
Potential intensity [m/s]"""
def CAPE_PI_vectorized(sst,p_surf,p_env,T_env,r_env):
    # Ratio of the exchange coefficients
    cecd = namelist.Ck / namelist.Cd;
    # Select thermodynamics: Set to 1 for pseudoadiabatic, 2 for reversible
    select_thermo = namelist.select_thermo
    # Load temperature table, with pressure and entropy coordinates
    if namelist.select_interp == 2:
        if select_thermo == 1:
            with np.load('%s/thermo/entropy_table.npz' % namelist.src_directory) as entropy_table:
                p_look = entropy_table['p']
                s_look = entropy_table['s']
                T_lookup = entropy_table['T']
        elif select_thermo == 2:
            with np.load('%s/thermo/entropy_table_reversible.npz' % namelist.src_directory) as entropy_table:
                p_look = entropy_table['p']
                s_look = entropy_table['s']
                rt_look = entropy_table['rt']
                T_lookup = entropy_table['T']
    # Dimensions of the input
    dim_in = T_env.shape
    # Near surface conditions: For now, assume that parcels at the first level
    # are representative of near surface parcels
    T_ns = T_env[0,:,:]
    r_ns = r_env[0,:,:]
    p_ns = p_env[0]
    # Initialize saturated parcels at SST
    ess, rs = sat_thermo(sst,p_surf)
    # Compute RH in near-surface parcels
    rh = r_ns/rs*(1+rs/pr.eps)/(1+r_ns/pr.eps)
    # Compute reference entropies of:
    # BL parcel
    s_ns = s_unsat(T_ns,p_ns,r_ns,r_ns,select_thermo)
    # Saturated parcels at SST
    ss = s_sat(sst,p_surf,rs,select_thermo)
    # Initialize arrays for the CAPE computation
    lnp = np.log(p_env)
    dlnp = np.diff(lnp,n=1,append = (2*lnp[-1] - lnp[-2]))
    T_rho_env = calc_T_rho(T_env,r_env,r_env)
    # Initialize ascent profiles
    Ta_prof = np.zeros(dim_in)
    T_rho_a = np.zeros(dim_in)
    ra_prof = np.zeros(dim_in)
    Ts_prof = np.zeros(dim_in)
    T_rho_s = np.zeros(dim_in)
    rs_prof = np.zeros(dim_in)
    # Initialize PI
    PI = np.zeros(sst.shape)
    # Compute LCL pressure and index
    pLCL = get_LCL(p_ns,T_ns,r_ns,rh)
    if select_thermo == 1:
        # Create interpolation function for solutions of entropy inversion.
        f_lookup = RectBivariateSpline(p_look, s_look, T_lookup, kx=1, ky=1)
    # Find pressure level for condensation.
    nPLev = len(p_env)
    p_env_mat = np.moveaxis(np.tile(p_env, pLCL.shape + (1,)), 2, 0)
    Icond = np.tile(pLCL, (len(p_env), 1, 1)) > p_env_mat
    Icond[-1, :, :] = True                                                  # if no condensation, set it to the highest level
    Icond_idxs = np.apply_along_axis(np.argmax, 0, Icond)                   # find first index of condensation
    is_cond = np.full(Icond_idxs.shape, True)
    is_cond[Icond_idxs == (len(p_env)-1)] = False                           # if no condensation, store for later
    # For lower levels (higher pressure), compute T based on a dry adiabat
    Ta_prof = np.multiply(np.tile(T_ns, (nPLev, 1, 1)), np.power(p_env_mat / p_ns, pr.Rd / pr.cp))
    # Maintain constant vapor mixing ratio
    ra_prof = np.tile(r_ns, (nPLev, 1, 1))
    # Then, compute T by inverting moist adiabats
    if select_thermo == 1:
        for p_idx in range(nPLev):
            lvl_mask = Icond_idxs == p_idx
            nLvl = np.sum(lvl_mask)
            Ta_prof[p_idx, :, :][lvl_mask] = f_lookup.ev(np.full((nLvl,), p_env[p_idx]), np.full((nLvl,), s_ns[lvl_mask]))
            tmp, ra_prof[p_idx, :, :][lvl_mask] = sat_thermo(Ta_prof[p_idx, :, :][lvl_mask],np.full((nLvl,), p_env[p_idx]))
            for p2_idx in range(p_idx, nPLev):
                Ta_prof[p2_idx, :, :][lvl_mask] = f_lookup.ev(np.full((nLvl,), p_env[p2_idx]), np.full((nLvl,), s_ns[lvl_mask]))
                tmp, ra_prof[p2_idx, :, :][lvl_mask]  = sat_thermo(Ta_prof[p2_idx, :, :][lvl_mask],np.full((nLvl,), p_env[p2_idx]))
        # For the saturated parcel, pLCL = p_ns, so only invert moist adiabat
        Ts_prof = f_lookup.ev(p_env_mat, np.full(p_env_mat.shape, ss))
    elif select_thermo == 2:
        for p_idx in range(nPLev):
            lvl_mask = Icond_idxs == p_idx
            nLvl = np.sum(lvl_mask)
            Ta_prof[p_idx, :, :][lvl_mask] = interpn((p_look,s_look,rt_look),T_lookup,(np.full((nLvl,), p_env[p_idx]), np.full((nLvl,), s_ns[lvl_mask]),np.full((nLvl,), r_ns[lvl_mask])), method='linear', bounds_error = False, fill_value=np.nan)
            tmp, ra_prof[p_idx, :, :][lvl_mask] = sat_thermo(Ta_prof[p_idx, :, :][lvl_mask],np.full((nLvl,), p_env[p_idx]))
            for p2_idx in range(p_idx, nPLev):
                Ta_prof[p2_idx, :, :][lvl_mask] = interpn((p_look,s_look,rt_look),T_lookup,(np.full((nLvl,), p_env[p2_idx]), np.full((nLvl,), s_ns[lvl_mask]),np.full((nLvl,), r_ns[lvl_mask])), method='linear', bounds_error = False, fill_value=np.nan)
                tmp, ra_prof[p2_idx, :, :][lvl_mask]  = sat_thermo(Ta_prof[p2_idx, :, :][lvl_mask],np.full((nLvl,), p_env[p2_idx]))
        # For the saturated parcel, pLCL = p_ns, so only invert moist adiabat
        Ts_prof = interpn((p_look,s_look,rt_look),T_lookup,(p_env_mat, np.full(p_env_mat.shape, ss), np.full(p_env_mat.shape, rs)), method='linear',  bounds_error = False, fill_value=np.nan)
    # Compute mixing ratio of saturated ascent profiles
    tmp, rs_prof = sat_thermo(Ts_prof,p_env_mat)
    # Compute density temperature. Assume that there is no liquid or ice water at lower level.
    T_rho_a = calc_T_rho(Ta_prof,ra_prof,r_ns)
    T_rho_s = calc_T_rho(Ts_prof,rs_prof,rs)
    # Compute CAPE differences
    # Find LNB for saturated and BL profiles
    a_out_I = (nPLev-1) - np.apply_along_axis(np.argmax, 0, np.flip(T_rho_a >= T_rho_env, axis = 0))
    s_out_I = (nPLev-1) - np.apply_along_axis(np.argmax, 0, np.flip(T_rho_s >= T_rho_env, axis = 0))
    # Get outflow properties
    T_out_a = np.zeros(pLCL.shape)
    p_out_a = np.zeros(pLCL.shape)
    add_area_a = np.zeros(pLCL.shape)
    T_out_s = np.full(pLCL.shape, np.nan)
    p_out_s = np.zeros(pLCL.shape)
    add_area_s = np.zeros(pLCL.shape)
    for p_idx in range(nPLev-1):
        # For the surface-saturated parcel
        mask = s_out_I == p_idx
        Te1 = T_env[p_idx, :, :][mask]
        Te2 = T_env[p_idx+1, :, :][mask]
        Tre1 = T_rho_env[p_idx, :, :][mask]
        Tre2 = T_rho_env[p_idx+1, :, :][mask]
        Trs1 = T_rho_s[p_idx, :, :][mask]
        Trs2 = T_rho_s[p_idx+1, :, :][mask]
        dT1 = Trs1-Tre1
        dT2 = Trs2-Tre2
        p_out_s[mask] = (p_env[p_idx]*dT2 - p_env[p_idx+1]*dT1)/(dT2-dT1)
        T_out_s[mask] = (Te1*(p_out_s[mask]-p_env[p_idx+1])+Te2*(p_env[p_idx]-p_out_s[mask]))/(p_env[p_idx] - p_env[p_idx+1])
        add_area_s[mask]=pr.Rd *dT1*(p_env[p_idx]-p_out_s[mask])/(p_env[p_idx]+p_out_s[mask])
        # For the boundary layer parcel
        mask = a_out_I == p_idx
        Te1 = T_env[p_idx, :, :][mask]
        Te2 = T_env[p_idx+1, :, :][mask]
        Tre1 = T_rho_env[p_idx, :, :][mask]
        Tre2 = T_rho_env[p_idx+1, :, :][mask]
        Tra1 = T_rho_a[p_idx, :, :][mask]
        Tra2 = T_rho_a[p_idx+1, :, :][mask]
        dT1 = Tra1-Tre1
        dT2 = Tra2-Tre2
        p_out_a[mask] = (p_env[p_idx]*dT2 - p_env[p_idx+1]*dT1)/(dT2-dT1)
        T_out_a[mask] = (Te1*(p_out_a[mask]-p_env[p_idx+1])+Te2*(p_env[p_idx]-p_out_a[mask]))/(p_env[p_idx] - p_env[p_idx+1])
        add_area_a[mask]=pr.Rd *dT1*(p_env[p_idx]-p_out_a[mask])/(p_env[p_idx]+p_out_a[mask])
    # Compute CAPE
    CAPE = np.zeros(pLCL.shape)
    CAPEs = np.zeros(pLCL.shape)
    for p_idx in range(nPLev):
        mask = p_idx <= a_out_I
        CAPE[mask] += pr.Rd * (T_rho_a[p_idx, :, :][mask] - T_rho_env[p_idx, :, :][mask]) * -dlnp[p_idx]
        mask = p_idx <= s_out_I
        CAPEs[mask] += pr.Rd * (T_rho_s[p_idx, :, :][mask] - T_rho_env[p_idx, :, :][mask]) * -dlnp[p_idx]
    CAPE += add_area_a
    CAPEs += add_area_s
    # Difference of the CAPEs
    CAPE = np.maximum(CAPE,0)
    CAPE[np.isnan(CAPE)] = 0
    cape_diff = CAPEs-CAPE
    PI = np.sqrt(np.maximum(cecd*np.divide(sst, T_out_s)*cape_diff,0))
    PI[np.isnan(PI)] = 0
    return PI

""" Emanuel GPI function """
def gpi(PI,chi,vort,S):
    # Set GPI to zero when PI falls under the 35 m/s threshold
    PI_abs = np.maximum((PI-35),0)
    GPI = (abs(vort))**3*chi**(-4/3)*PI_abs**2/(S+25)**4
    return GPI

""" Emanuel GPI function """
def gpi_en04(PI,rh,vort,S):
    # Set GPI to zero when PI falls under the 35 m/s threshold
    GPI = (1e5*abs(vort))**(rh / 50)**3*(PI/70)**3/(1+0.1*S)**2
    return GPI

""" Misc functions for inverting entropy """
def s_diff(T,p,r_t,select_thermo,s_ref):
    diff = (s_sat(T,p,r_t,select_thermo)-s_ref)**2
    return diff

def s_diff_der(T,p,r_t,select_thermo,s_ref):
    diff_der = 2*(s_sat(T,p,r_t,select_thermo)-s_ref)*s_sat_der(T,p,r_t,select_thermo)
    return diff_der

"""Function to generate a Temperature lookup table with pressure and entropy coordinates.
The temperature table is obtained by inverting the entropy function at constant pressure.
Inputs:
pmin: minimum of the pressure axis [Pa]
pmax: maximum of the pressure axis [Pa]
nprs: number of pressure points
smin: minimum of the entropy axis [J/kg/K]
smax: maximum of the entropy axis [J/kg/K]
ns: number of entropy points
rtmin: minimum total water content (only for reversible case)
rtmax: maximum total water content (only for reversible case)
nrt: number of total water points
Values of about 100 for nprs and ns are recommended for a earth-like ranges of climate.
smin and smax can be generated using s_unsat and s_sat, and selecting appropriate T, P and rv"""

def generate_entropy_table(pmin,pmax,nprs,smin,smax,ns,rtmin,rtmax,nrt,select_thermo):
    # Create the entropy and pressure axes
    s_look = np.linspace(smin,smax,ns)
    p_look = 100*np.linspace(pmin,pmax,nprs)
    # Initial temperature guess (250 K works pretty well)
    T_approx = 250.0

    if select_thermo == 1:
        # Create the inversion table, by iterating through pressure and entropy
        T_lookup = np.zeros((nprs,ns), dtype=float)
        for ii in np.arange(0,nprs):
            for jj in np.arange(0,ns):
                T_lookup[ii,jj] = minimize(s_diff, T_approx, args = (p_look[ii],0.0,select_thermo,s_look[jj]),
                                           method='Nelder-Mead', jac=None).x
        # Save entropy inversion table
        np.savez('entropy_table.npz', p=p_look, s=s_look, T=T_lookup)

    elif select_thermo == 2:
        # Create an additional dimension for total water content
        rt_look = np.linspace(rtmin,rtmax,nrt)
        # Create the inversion table, by iterating through pressure, entropy and total water
        T_lookup = np.zeros((nprs,ns,nrt), dtype=float)
        for ii in np.arange(0,nprs):
            for jj in np.arange(0,ns):
                for kk in np.arange(0,nrt):
                    T_lookup[ii,jj,kk] = minimize(s_diff, T_approx, args = (p_look[ii],rt_look[kk],select_thermo,s_look[jj]),
                                               method='Nelder-Mead', jac=None).x
        # Save entropy inversion table
        np.savez('entropy_table_reversible.npz', p=p_look, s=s_look, rt=rt_look, T=T_lookup)

    return
