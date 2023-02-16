# %%
import datetime
import glob
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import namelist

from thermo import calc_thermo
from util import basins, constants, input, mat, sphere

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from copy import copy

# %%
def plot_genesis():
    # %%
    basin_ids = ['AL', 'EP', 'WP', 'SH', 'IO']
    fns = ['']*len(basin_ids)
    seed_frequency = np.zeros(len(basin_ids))

    # Bounds for genesis boxes
    lon_min = 0; lon_max = 360;
    lat_min = -50; lat_max = 50;
    x_binEdges = np.arange(lon_min, lon_max + 0.1, 2.5)
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 2.5)
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2
    gen_pdf = np.zeros((len(x_binCenter), len(y_binCenter)))
    den_pdf = np.zeros((len(x_binCenter), len(y_binCenter)))

    for i in range(len(basin_ids)):
        fns[i] = '/data0/jlin/era5/tracks_%s_era5_197901_202112.nc' % basin_ids[i]
        ds = xr.open_dataset(fns[i])
        seed_frequency[i] = np.nanmean(ds['seeds_per_months'])

    # Calculate seed normalization factor for each basin.
    basin_area = np.zeros(len(basin_ids))
    for i in range(len(basin_ids)):
        basin_id = basin_ids[i]
        ds_b = xr.open_dataset('land/%s.nc' % basin_id)
        basin_mask = ds_b['basin']
        b = basins.TC_Basin(basin_id)
        bounds = b.get_bounds()
        LON, LAT = np.meshgrid(basin_mask['lon'], basin_mask['lat'])
        bound_mask = ((LON >= bounds[0]) & (LON <= bounds[2]) &
                      (LAT >= bounds[1]) & (LAT <= bounds[3]))
        # Because the basin mask is high resolution, we can get away with
        # one-off indexing in longitude.
        x_clat = (LAT[:, 0:-1] + LAT[:, 1:]) / 2
        xdist = sphere.haversine(LON[:, 1:], x_clat, LON[:, 0:-1], x_clat)
        ydist = float(sphere.haversine(0, 0, 0, basin_mask['lat'][1] - basin_mask['lat'][0]))
        basin_area[i] = np.sum(xdist[bound_mask[:, 0:-1]] * ydist)

    # Create normalization factors
    norm_factor = np.min(seed_frequency) / seed_frequency * basin_area / np.max(basin_area)
    norm_factor /= np.max(norm_factor)

    for j in range(len(basin_ids)):
        print(fns[j])

        # Filter for surviving tracks.
        #fns = sorted(fns)
        #ds = xr.open_mfdataset(fns, concat_dim = "n_trk", combine = "nested")
        ds = xr.open_dataset(fns[j])
        vmax = ds['vmax_trks'].load().data
        lon_trks = ds['lon_trks'].load().data
        lat_trks = ds['lat_trks'].load().data
        m_trks = ds['m_trks'].load().data
        lon_filt = np.full(ds['lon_trks'].shape, np.nan) #ds['lon_trks'][mask, :].data
        lat_filt = np.full(ds['lat_trks'].shape, np.nan) #ds['lat_trks'][mask, :].data
        vmax_filt = np.full(ds['vmax_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
        m_filt = np.full(ds['m_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data

        # Here, we only consider a TC from the first point where it exceeds
        # the threshold, to the point it decays to 10 m/s (after it has
        # reached its peak intensity).
        lon_genesis = np.full(lon_filt.shape[0], np.nan)
        lat_genesis = np.full(lon_filt.shape[0], np.nan)

        for i in range(lon_filt.shape[0]):
            if len(np.argwhere(vmax[i, :] >= 15).flatten()) > 0:
                # Genesis occurs when the TC first achieves 30 knots (15 m/s).
                gen_idxs = np.argwhere(vmax[i, :] < 15).flatten()
                idx_gen = np.argwhere(vmax[i, :] >= 15).flatten()[0]
                lon_genesis[i] = ds['lon_trks'][i, idx_gen]
                lat_genesis[i] = ds['lat_trks'][i, idx_gen]

                # TC decays after it has reached 15 m/s
                decay_idxs = np.argwhere(vmax[i, :] < 15).flatten()
                idxs_lmi = np.argwhere(decay_idxs >= np.nanargmax(vmax[i, :])).flatten()
                if len(decay_idxs) > 0 and len(idxs_lmi) > 0:
                    idx_decay = decay_idxs[idxs_lmi[0]]
                else:
                    idx_decay = vmax.shape[1]

                nt = idx_decay - idx_gen
                vmax_filt[i, 0:nt] = vmax[i, idx_gen:idx_decay]
                lon_filt[i, 0:nt] = lon_trks[i, idx_gen:idx_decay]
                lat_filt[i, 0:nt] = lat_trks[i, idx_gen:idx_decay]
                m_filt[i, 0:nt] = m_trks[i, idx_gen:idx_decay]

        gen_pdf += np.histogram2d(lon_genesis, lat_genesis, bins = [x_binEdges, y_binEdges])[0] * norm_factor[j]
        den_pdf += np.histogram2d(lon_filt.flatten(), lat_filt.flatten(), bins = [x_binEdges, y_binEdges])[0] * norm_factor[j]

    gen_pdf /= len(basin_ids)
    den_pdf /= len(basin_ids)

    # %%
    plt.rcParams.update({'font.size': 14})
    lon_cen = 180
    dlon_label = 20
    lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
    xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
    xlocs_shift = np.copy(xlocs)
    xlocs_shift[xlocs > lon_cen] -= 360
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(15, 8)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    levels = np.arange(3, 30, 3)

    from copy import copy
    palette = copy(plt.get_cmap('jet'))
    palette.set_under('white', 1.0)
    #gen_pdf[gen_pdf == 0] = 1e-6                # so we can take the log
    #cmin = np.quantile(np.log(gen_pdf[gen_pdf > 1]), 0.1)
    #cmax = np.quantile(np.log(gen_pdf), 1)
    cmax = 20 #np.quantile(gen_pdf, 0.999)
    cmin = 1 # cmax / 10
    levels = np.linspace(cmin, cmax, 11)
    ax = plt.contourf(x_binCenter, y_binCenter, gen_pdf.T, levels = levels, extend = 'max', cmap=palette, transform = ccrs.PlateCarree());
    plt.colorbar(orientation = 'horizontal'); plt.title('Genesis PDF');
    plt.xlim([-180, 180])

    # %% TC Density PDF.
    plt.rcParams.update({'font.size': 14})
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(15, 7)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    cmax = np.quantile(den_pdf[den_pdf > 0], 1)
    cmin = cmax / 10
    levels = np.linspace(cmin, cmax, 11)
    plt.contourf(x_binCenter, y_binCenter, den_pdf.T, levels, cmap=palette, transform = ccrs.PlateCarree());
    plt.colorbar(orientation = 'horizontal')
    plt.xlim([-180, 180])
    # %%

def compare_simulations():
    # %%
    sim_name = 'track_random_seeding'
    fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/%s/*GL*.nc' % sim_name));
    #fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/%s/*.nc' % sim_name));

    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                           data_vars="minimal", drop_variables = "seeds_per_months")

    drop_vars = ["lon_trks", "lat_trks", "u250_trks", "v250_trks", "u850_trks", "v850_trks",
                 "v_trks", "m_trks", "vmax_trks", "tc_month", "tc_years"]
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                                 data_vars="minimal", drop_variables = drop_vars)

    plt.figure(figsize = (10,6))
    total_seeds = np.sum(np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0))
    plt.plot(range(1, 13), np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0), 'b')
    counts, bins = np.histogram(ds['tc_month'], bins = np.arange(0.5, 12.6, 1))
    cbins = np.arange(1, 12.1, 1)
    counts = counts / (len(fn_tracks) * 20)
    n_tc_control = np.sum(counts)
    seed_survival_rate_control = counts / np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0)

    ax = plt.gca(); ax2 = ax.twinx()
    ax2.bar(cbins, counts, fc = ([0,0,1,0.4]))
    plt.xlabel('Month'); ax.set_ylabel('Number of Seeds per Month')
    ax2.set_ylabel('Number of TCs per Month')

    fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_plus2K_tigercpu_intelmpi_18_540PE/%s/*GL*.nc' % sim_name));
    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                           data_vars="minimal", drop_variables = "seeds_per_months")
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                                 data_vars="minimal", drop_variables = drop_vars)
    seeds = np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0)
    seed_ratio = np.sum(seeds) / total_seeds

    ax.plot(range(1, 13), seeds / seed_ratio, 'r')
    counts, bins = np.histogram(ds['tc_month'], bins = np.arange(0.5, 12.6, 1))
    counts = counts / (len(fn_tracks) * 20 * seed_ratio)
    ax2.bar(cbins, counts, fc = ([1,0,0,0.4]))
    ax.set_ylim([200, 550]); ax2.set_ylim([0, 22])
    plt.xticks(range(1, 13, 1)); ax.set_xlabel("Month"); ax.set_xlim([0.5, 12.5])
    plt.legend(['Control', '2K'], loc = 'best')
    plt.title('Random Seeding: %.2f / year (control); %.2f / year (+2K)' % (n_tc_control, np.sum(counts)))
    plt.savefig('%s/random_seeding_count.png' % '/data0/jlin/gfdl/annual_cycle/HIRAM/')

    seed_survival_rate_2K = counts / (seeds / seed_ratio)

    # %% compare SPI between plus 2K
    fn_spi = '/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/SPI_monthly_HIRAM_11101_13012.nc';
    fn_spi_2K = '/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_plus2K_tigercpu_intelmpi_18_540PE/SPI_monthly_HIRAM_11101_13012.nc';

    ds_spi = xr.open_dataset(fn_spi)
    ds_spi_2K = xr.open_dataset(fn_spi_2K)

    ds_b = xr.open_dataset('land/GL.nc')
    basin_mask = ds_b['basin']
    f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
    basin_gen = mat.interp_2d_grid(basin_mask['lon'], basin_mask['lat'], basin_mask, ds_spi['lon'].data, ds_spi['lat'].data)
    LON, LAT = np.meshgrid(ds_spi['lon'].data, ds_spi['lat'].data)
    spi_clim = ds_spi['spi'].data
    spi_clim[np.tile(basin_gen, (spi_clim.shape[0], 1, 1)) < 0.5] = 0
    spi_clim[np.tile(np.abs(LAT), (spi_clim.shape[0], 1, 1)) > 30] = 0
    spi_clim[spi_clim < 0] = 0

    spi_clim_2K = ds_spi_2K['spi'].data
    spi_clim_2K[np.tile(basin_gen, (spi_clim.shape[0], 1, 1)) < 0.5] = 0
    spi_clim_2K[np.tile(np.abs(LAT), (spi_clim.shape[0], 1, 1)) > 30] = 0
    spi_clim_2K[spi_clim_2K < 0] = 0

    pct_change_2K = (np.sum(spi_clim_2K[0:12, :, :]) - np.sum(spi_clim[0:12, :, :])) / np.sum(spi_clim[0:12, :, :])

    sim_name = 'track_spi'
    fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/%s/*GL*.nc' % sim_name));

    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                           data_vars="minimal", drop_variables = "seeds_per_months")
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                                 data_vars="minimal", drop_variables = drop_vars)
    N_per_year = 90

    plt.figure(figsize = (10,6))
    total_seeds = np.sum(np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0))

    plt.plot(range(1, 13), np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0), 'b')
    counts, bins = np.histogram(ds['tc_month'], bins = np.arange(0.5, 12.6, 1))
    cbins = np.arange(1, 12.1, 1)
    counts = counts / (len(fn_tracks) * 20)
    seed_scale_ratio = np.sum(counts) / N_per_year
    t_str = 'SPI Seeds: %d / yr (control)' % (np.sum(counts / seed_scale_ratio))
    ax = plt.gca(); ax2 = ax.twinx()
    ax2.bar(cbins, counts / seed_scale_ratio, fc = ([0,0,1,0.4]))
    plt.xlabel('Month'); ax.set_ylabel('Number of Seeds per Month')
    ax2.set_ylabel('Number of TCs per Month')
    n_tc_control = np.sum(counts / seed_scale_ratio)
    seed_survival_rate_spi_control = counts / np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0)

    #plt.plot(counts / np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0))

    fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_plus2K_tigercpu_intelmpi_18_540PE/%s/*GL*.nc' % sim_name));
    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                           data_vars="minimal", drop_variables = "seeds_per_months")
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                                 data_vars="minimal", drop_variables = drop_vars)
    seeds = np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0)
    seed_ratio = 1 + pct_change_2K

    ax.plot(range(1, 13), seeds * seed_ratio , 'r')
    counts, bins = np.histogram(ds['tc_month'], bins = np.arange(0.5, 12.6, 1))
    counts = counts / (len(fn_tracks) * 20 * seed_scale_ratio) * seed_ratio
    ax2.bar(cbins, counts, fc = ([1,0,0,0.4]))
    ax.set_ylim([0, 50]); ax2.set_ylim([0, 16])
    plt.xticks(range(1, 13, 1)); ax.set_xlabel("Month"); ax.set_xlim([0.5, 12.5])
    plt.legend(['Control', '2K'], loc = 'best')
    plt.title('SPI Seeding: %.2f / year (control); %.2f / year (+2K)' % (n_tc_control, np.sum(counts)))
    plt.savefig('%s/spi_seeding_count.png' % ('/data0/jlin/gfdl/annual_cycle/HIRAM'))

    seed_survival_rate_spi_2K = counts / (seeds * seed_ratio)

    # %%
    plt.figure(figsize = (10,6)); plt.grid();
    h1, = plt.plot(range(1, 13), seed_survival_rate_control, 'k')
    ax = plt.gca(); ax2 = ax.twinx();
    ax.set_ylabel('Seed Survival Rate')
    h2, = ax2.plot(range(1, 13), seed_survival_rate_spi_control, 'b')
    h3, = ax2.plot(range(1, 13), seed_survival_rate_spi_2K, 'r')
    ax2.set_ylabel('Seed Survival Rate')
    ax.set_xlabel("Month"); ax.set_xticks(range(1, 13)); ax.set_xlim([1, 12])
    plt.legend([h1, h2, h3], ['Random', 'SPI', 'SPI +2K'])

    # %%

def plot_tracks():
    ds.close()
    ds_seeds.close()

    # %%
    import glob
    from copy import copy

    #fn_tracks = '/data0/jlin/cmip6/tracks/spi/tracks_AL_GFDL-CM4_ssp585_r1i1p1f1_201501_210012_e*.nc'
    sim_name = 'test'
    #sim_name = 'track_control_GL_genesis_EMAN086_MINITLOG_FIX5_AU9'
    #sim_name = 'track_coupled_control_test11_WP_PI'
    fn_tracks = sorted(glob.glob('/nfs/emanuellab001/jzlin/tc_risk/%s/tracks_GL_era5_197901_202112*.nc' % sim_name));
    #sim_name = 'track_random_seeding'
    #fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_plus2K_tigercpu_intelmpi_18_540PE/%s/*GL*.nc' % sim_name));
    #fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/%s/*.nc' % sim_name));

    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                           data_vars="minimal", drop_variables = "seeds_per_month")

    drop_vars = ["lon_trks", "lat_trks", "u250_trks", "v250_trks", "u850_trks", "v850_trks",
                 "v_trks", "m_trks", "vmax_trks", "tc_month", "tc_years", "tc_basins"]
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                                 data_vars="minimal", drop_variables = drop_vars)

    yearS = 1979
    yearE = 2021
    n_sim = len(fn_tracks)
    ntrks_per_year = 90
    ntrks_per_sim = ntrks_per_year * (yearE - yearS + 1)
    seeds_per_month_basin = ds_seeds['seeds_per_month'].load()
    seeds_per_month = np.nansum(seeds_per_month_basin, axis = 1)
    spi_seeding = False
    print('Number of Simulations: %d' % n_sim)
    print(np.nanmean(seeds_per_month, axis = 0))

    plt.figure(figsize = (10,6))
    plt.plot(range(1, 13), np.nansum(seeds_per_month, axis = 0), 'k')
    ax = plt.gca(); ax2 = ax.twinx();
    ax2.hist(ds['tc_month'], bins = np.arange(0.5, 12.5, 1))
    plt.xlabel('Month'); ax.set_ylabel('Number of Seeds')
    ax2.set_ylabel('Number of Storms')

    if spi_seeding:
        ds_spi = xr.open_dataset('%s/SPI_monthly_era5_197901_202112.nc' % namelist.base_directory).load()
        spi = ds_spi['spi']
        lon_spi = ds_spi['lon'].data
        lat_spi = ds_spi['lat'].data
        spi_clim = spi.data
        spi_clim[spi_clim < 0] = 0
        dt_spi = np.array([datetime.datetime.utcfromtimestamp(int(x) / 1e9) for x in spi['time']])
        dt_yr = np.array([x.year for x in dt_spi])
        yr_spi = np.zeros((yearE - yearS + 1,))
        spi_monthly = np.sum(spi_clim, axis = (1, 2))
        for yr in range(yearS, yearE + 1):
            yr_spi[yr - yearS] = np.sum(spi_monthly[dt_yr == yr])
        yr_spi /= np.sum(yr_spi) / (yearE - yearS + 1)

    # Filter for surviving tracks.
    vmax = ds['vmax_trks'].load().data
    lon_filt = np.full(ds['lon_trks'].shape, np.nan) #ds['lon_trks'][mask, :].data
    lat_filt = np.full(ds['lat_trks'].shape, np.nan) #ds['lat_trks'][mask, :].data
    vmax_filt = np.full(ds['vmax_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    m_filt = np.full(ds['m_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    lon_trks = ds['lon_trks'].load().data
    lat_trks = ds['lat_trks'].load().data
    m_trks = ds['m_trks'].load().data
    basin_trks = ds['tc_basins'].load().data
    yr_trks = ds['tc_years'].load().data
    mnth_trks = ds['tc_month'].load().data

    # Here, we only consider a TC from the first point where it exceeds
    # the threshold, to the point it decays to 10 m/s (after it has
    # reached its peak intensity).
    lon_genesis = np.full(lon_filt.shape[0], np.nan)
    lat_genesis = np.full(lon_filt.shape[0], np.nan)
    for i in range(lon_filt.shape[0]):
        if len(np.argwhere(vmax[i, :] >= 15).flatten()) > 0:
            # Genesis occurs when the TC first achieves 30 knots (15 m/s).
            gen_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idx_gen = np.argwhere(vmax[i, :] >= 15).flatten()[0]
            lon_genesis[i] = lon_trks[i, idx_gen]
            lat_genesis[i] = lat_trks[i, idx_gen]

            # TC decays after it has reached 15 m/s
            decay_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idxs_lmi = np.argwhere(decay_idxs >= np.nanargmax(vmax[i, :])).flatten()
            if len(decay_idxs) > 0 and len(idxs_lmi) > 0:
                idx_decay = decay_idxs[idxs_lmi[0]]
            else:
                idx_decay = vmax.shape[1]

            nt = idx_decay - idx_gen
            vmax_filt[i, 0:nt] = vmax[i, idx_gen:idx_decay]
            lon_filt[i, 0:nt] = lon_trks[i, idx_gen:idx_decay]
            lat_filt[i, 0:nt] = lat_trks[i, idx_gen:idx_decay]
            m_filt[i, 0:nt] = m_trks[i, idx_gen:idx_decay]

    # %% Read in IBTrACS data
    n_tc_emanuel = np.array([2.3995, 3.7472, 4.7565, 3.3178,
                             3.8894, 3.1360, 4.4686, 2.2770,
                             3.3814, 3.8279, 2.8226, 6.3410,
                             2.8146, 3.6206, 4.1867, 2.7198,
                             5.0066, 2.8506, 3.8354, 8.1853,
                             7.1715, 6.5825, 4.8782, 5.6885,
                             4.9672, 6.9847, 13.1994, 7.7164,
                             6.6171, 7.5774, 3.8121, 8.5860,
                             8.5695, 5.3999, 6.5196, 5.3201,
                             4.9189, 5.8032, 6.1446, 3.1375,
                             6.2809, 7.4246])

    lat_min = 2; lat_max = 50;
    if 'EP' in fn_tracks[0]:
        lon_min = 180; lon_max = 280;
    elif 'WP' in fn_tracks[0]:
        lon_min = 100; lon_max = 180;
    elif 'SH' in fn_tracks[0]:
        lon_min = 40; lon_max = 220;
        lat_min = -60; lat_max = 0;
    elif 'AS' in fn_tracks[0]:
        lon_min = 90; lon_max = 180;
        lat_min = -45; lat_max = 0;
    elif 'AL' in fn_tracks[0]:
        lon_min = 260.0; lon_max = 350;
    else:
        lon_min = 0; lon_max = 359.99;
        lat_min = -60;

    fn_ib = '/nfs/emanuellab001/jzlin/tc_risk/IBTrACS.ALL.v04r00.nc'
    yearS_ib = 1979; yearE_ib = 2021
    ds_ib = xr.open_dataset(fn_ib)
    dt_ib = np.array([datetime.datetime.utcfromtimestamp(int(x)/1e9) for x in np.array(ds_ib['time'][:, 0])])
    date_mask = np.logical_and(dt_ib >= datetime.datetime(yearS_ib, 1, 1), dt_ib <= datetime.datetime(yearE_ib+1, 1, 1))
    ib_lon = ds_ib['lon'].data
    ib_lon[ib_lon < 0] += 360
    ib_lat = ds_ib['lat'].load()
    usa_wind = ds_ib['usa_wind'].load()
    n_tc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
    n_mtc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
    gen_lon_hist = np.full(dt_ib.shape, np.nan)
    gen_lat_hist = np.full(dt_ib.shape, np.nan)
    track_lon_hist = np.full(ib_lon.shape, np.nan)
    track_lat_hist = np.full(ib_lon.shape, np.nan)
    basin_hist = ds_ib['basin'].load().data[:, 0].astype('U2')
    basin_names = np.array(sorted(np.unique(basin_hist)))
    n_tc_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    pdi_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    mpd_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    vmax_tc = np.nanmax(usa_wind, axis = 1)
    tc_lmi = np.full(dt_ib.shape, np.nan)
    tc_lat_lmi = np.full(dt_ib.shape, np.nan)
    for i in range(yearS_ib, yearE_ib + 1, 1):
        mask = ((dt_ib >= datetime.datetime(i, 1, 1)) &
                (dt_ib <= datetime.datetime(i, 12, 31)) &
                (~np.all(np.isnan(usa_wind.data), axis = 1)) &
                (ib_lon[:, 0] >= lon_min) &
                (ib_lon[:, 0] <= lon_max))
        mask_f = np.argwhere(mask).flatten()

        n_tc_per_year[i - yearS_ib] = np.sum(mask)
        for (b_idx, b_name) in enumerate(basin_names):
            if b_name[0] == 'S':
                b_mask = ((dt_ib >= datetime.datetime(i-1, 6, 1)) &
                        (dt_ib <= datetime.datetime(i, 6, 30)) &
                        (~np.all(np.isnan(usa_wind.data), axis = 1)))                
            else:
                b_mask = ((dt_ib >= datetime.datetime(i, 1, 1)) &
                        (dt_ib <= datetime.datetime(i, 12, 31)) &
                        (~np.all(np.isnan(usa_wind.data), axis = 1)))
            n_tc_per_year_basin[b_idx, i - yearS_ib] = np.sum(b_mask & (vmax_tc >= 34) & (basin_hist == b_name))
            pdi_mask = b_mask & (basin_hist == b_name)
            mpd_per_year_basin[b_idx, i - yearS_ib] = np.sum(np.power(vmax_tc[pdi_mask] / 1.94384, 3))

        # n_mtc_per_year[i - yearS_ib] = np.sum(vmax_tc_yr >= 35)
        vmax_time = usa_wind[mask, :]['time'].load().data
        int_obs = usa_wind[mask, :].data / 1.94384

        for j in range(int(np.sum(mask))):
            if not bool(np.all(np.isnan(usa_wind[mask][j]))):
                lmi_idx = np.nanargmax(usa_wind[mask][j], axis = 0)
                tc_lat_lmi[mask_f[j]] = float(ib_lat[mask][j,lmi_idx])
                tc_lmi[mask_f[j]] = float(usa_wind[mask][j, lmi_idx])
 
                gen_idx = np.nanargmin(np.abs(usa_wind[mask][j, :] - 35), axis = 0)
                gen_lon_hist[mask_f[j]] = ib_lon[mask][j, gen_idx]
                gen_lat_hist[mask_f[j]] = ib_lat[mask][j, gen_idx]
                track_lon_hist[mask_f[j], gen_idx:] = ib_lon[mask][j, gen_idx:]
                track_lat_hist[mask_f[j], gen_idx:] = ib_lat[mask][j, gen_idx:]
    gen_lon_hist[gen_lon_hist < 0] += 360

    # %% AUS mask
    AUS_mask = (gen_lon_hist >= 100) & (gen_lon_hist <= 180) & (gen_lat_hist <= 0)
    basin_hist[AUS_mask] = 'AU'
    basin_names = np.array(sorted(np.unique(basin_hist)))
    n_tc_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    pdi_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    mpd_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))    
    for i in range(yearS_ib, yearE_ib + 1, 1):
        mask = ((dt_ib >= datetime.datetime(i, 1, 1)) &
                (dt_ib <= datetime.datetime(i, 12, 31)) &
                (~np.all(np.isnan(usa_wind.data), axis = 1)) &
                (ib_lon[:, 0] >= lon_min) &
                (ib_lon[:, 0] <= lon_max))

        n_tc_per_year[i - yearS_ib] = np.sum(mask)
        for (b_idx, b_name) in enumerate(basin_names):
            if b_name[0] == 'S' or (b_name == 'AU'):
                b_mask = ((dt_ib >= datetime.datetime(i-1, 6, 1)) &
                        (dt_ib <= datetime.datetime(i, 5, 31)) &
                        (~np.all(np.isnan(usa_wind.data), axis = 1)))                
            else:
                b_mask = ((dt_ib >= datetime.datetime(i, 1, 1)) &
                        (dt_ib <= datetime.datetime(i, 12, 31)) &
                        (~np.all(np.isnan(usa_wind.data), axis = 1)))
            n_tc_per_year_basin[b_idx, i - yearS_ib] = np.sum(b_mask & (vmax_tc >= 34) & (basin_hist == b_name))
            pdi_mask = b_mask & (basin_hist == b_name)
            mpd_per_year_basin[b_idx, i - yearS_ib] = np.sum(np.power(vmax_tc[pdi_mask] / 1.94384, 3))


    # %% Plot seed genesis probability
    plt.rcParams.update({'font.size': 16})
    basin_ids = np.array([k for k in namelist.basin_bounds if k != 'GL'])
    plt.figure(figsize=(8, 6))
    for (i, basin_id) in enumerate(basin_ids):
        seeds_per_month = ds_seeds['seeds_per_month'][:, ds['basin'] == basin_id, :].squeeze()
        # Compute probability of seed to TCs
        nTCs_per_month = np.zeros(12)
        nSeeds_per_month = np.zeros(12)
        for m_idx in range(12):
            mask = np.sum((mnth_trks == (m_idx + 1)) & (basin_trks == basin_id))
            nTCs_per_month[m_idx] = np.sum(mask)
            nSeeds_per_month[m_idx] = np.sum(seeds_per_month[:, m_idx])
        plt.plot(range(1, 13), nTCs_per_month / nSeeds_per_month)
    plt.legend(basin_ids); plt.grid();
    plt.xlabel('Month'); plt.ylabel('Seed Genesis Probability')
    plt.xticks(range(1, 13, 1)); plt.xlim([1, 12]); plt.ylim([0, 0.07])
    plt.savefig('%s/%s/seed_genesis_probability.png' % (namelist.base_directory, sim_name))

    # %% Plot interannual variability
    basin_ids = np.array([k for k in namelist.basin_bounds if k != 'GL'])
    n_tc_per_year_downscaling_basin = np.zeros((len(basin_ids), yearE - yearS + 1))
    mpd_per_year_downscaling_basin = np.zeros((len(basin_ids), yearE - yearS + 1))
    dt_seeds = np.array([[datetime.datetime(x, y, 1) for y in range(1, 13)] for x in seeds_per_month['year']])
    dt_trks = np.array([datetime.datetime(int(yr_trks[i]), int(mnth_trks[i]), 1) for i in range(len(yr_trks))])
    fig, axs = plt.subplots(figsize=(15, 18), ncols = 2, nrows = 4);
    for (i, basin_id) in enumerate(basin_ids):
        seeds_per_month = ds_seeds['seeds_per_month'][:, ds['basin'] == basin_id, :].squeeze()

        # Compute probability of seed to TCs
        nTCs_per_year = np.zeros((yearE - yearS + 1))
        nSeeds_per_year = np.zeros((yearE - yearS + 1))
        for yr in np.unique(ds_seeds['year']):
            if basin_id[0] == 'X':
                mask = ((dt_trks >= datetime.datetime(yr-1, 6, 1)) &
                        (dt_trks <= datetime.datetime(yr, 5, 31)) &
                        (basin_trks == basin_id))
                seed_mask = ((dt_seeds >= datetime.datetime(yr-1, 6, 1)) &
                        (dt_seeds <= datetime.datetime(yr, 5, 31)))
            else:
                mask = np.sum((yr_trks == yr) & (basin_trks == basin_id))
                seed_mask = ((dt_seeds >= datetime.datetime(yr, 1, 1)) &
                             (dt_seeds <= datetime.datetime(yr, 12, 31)))
            nTCs_per_year[yr - yearS] = np.sum(mask)
            nSeeds_per_year[yr - yearS] = np.sum(seeds_per_month.data[seed_mask])

        n_tc_per_year = np.sum(n_tc_per_year_basin[(basin_names == basin_id), :], axis = 0).flatten()
        dt_year = ds['year'].sel(year = slice(yearS, yearE))
        n_tc_per_year_downscaling = nTCs_per_year / nSeeds_per_year
        seed_ratio = np.nanmean(n_tc_per_year_downscaling) / np.nanmean(n_tc_per_year)
        n_tc_per_year_downscaling /= seed_ratio
        n_tc_per_year_downscaling_basin[i, :] = n_tc_per_year_downscaling

        n_ss = 1000
        pdi_per_year_downscaling = np.zeros((n_ss, yearE-yearS+1))
        max_pdi_per_year_downscaling_ss = np.zeros((n_ss, yearE-yearS+1))

        for yr in np.unique(ds_seeds['year']):
            mask = (yr_trks == yr) & (basin_trks == basin_id)
            max_pdi_trks = np.power(np.nanmax(vmax_filt[mask, :], axis = 1), 3)
            #vmax_pdi_trks = np.power(vmax_filt[mask, :], 3)
            #vmax_pdi_trks[np.isnan(vmax_pdi_trks)] = 0
            for n_idx in range(n_ss):
                mpd_subsample = np.random.choice(max_pdi_trks, int(np.round(n_tc_per_year_downscaling[yr - yearS])))
                max_pdi_per_year_downscaling_ss[n_idx, yr-yearS] = np.nansum(mpd_subsample)
                #for j in range(vmax_pdi_trks.shape[0]):
                #    pdi_per_year_downscaling[yr-yearS] += np.trapz(vmax_pdi_trks[j, :], x=ds['time'].data)

        max_pdi_per_year_downscaling = np.nanmean(max_pdi_per_year_downscaling_ss, axis = 0)
        mpd_per_year = mpd_per_year_basin[basin_names == basin_id, :].squeeze()
        mpd_per_year_downscaling_basin[i, :] = max_pdi_per_year_downscaling
        #print(np.corrcoef(mpd_per_year_downscaling_basin[i, :], mpd_per_year)[0,1])

        if basin_id[0] == 'S':
            ax = axs.flatten()[i]
            h1, = ax.plot(dt_year[1:], n_tc_per_year[1:], 'k', linewidth=4);
            h2, = ax.plot(dt_year[1:], n_tc_per_year_downscaling[1:], color='r', linewidth=3)
            mask_yrs = (ds['year'] >= yearS) & (ds['year'] <= yearE)
            ax.grid(); ax.set_xlim([yearS+1, yearE]); 
            ax.set_xlabel('Year'); ax.set_ylabel('Number of TCs');
            ax.text(0.02, 0.97, basin_id, transform = ax.transAxes, verticalalignment = 'top', weight="bold", fontsize = 24)
            ax.text(0.78, 0.9, 'r = %0.2f' % float(np.corrcoef(n_tc_per_year[1:], n_tc_per_year_downscaling[1:])[0, 1]), transform = ax.transAxes,
                    bbox=dict(facecolor='gray', edgecolor='black'))
            ax.set_xticks(range(yearS+1, yearE+1, 5));
            ax.set_xticklabels(range(yearS+1, yearE+1, 5));
            ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year)*1.15 / 5) * 5])
            yLim = ax.get_ylim(); ax.set_yticks(np.arange(0, yLim[1]+1, 5))
        else:
            ax = axs.flatten()[i]
            h1, = ax.plot(dt_year, n_tc_per_year, 'k', linewidth=4);
            h2, = ax.plot(dt_year, n_tc_per_year_downscaling, color='r', linewidth=3)
            mask_yrs = (ds['year'] >= yearS) & (ds['year'] <= yearE)
            ax.grid(); ax.set_xlim([yearS, yearE]);
            ax.set_xticks(range(yearS+1, yearE+1, 5));
            ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year)*1.15 / 5) * 5])
            ax.set_xlabel('Year'); ax.set_ylabel('Number of TCs');
            ax.text(0.02, 0.97, basin_id, transform = ax.transAxes, verticalalignment = 'top', weight="bold", fontsize = 24)
            ax.text(0.78, 0.9, 'r = %0.2f' % float(np.corrcoef(n_tc_per_year, n_tc_per_year_downscaling)[0, 1]), transform = ax.transAxes,
                    bbox=dict(facecolor='gray', edgecolor='black'))
            yLim = ax.get_ylim(); ax.set_yticks(np.arange(0, yLim[1]+1, 5))
            ax.set_xticks(range(yearS, yearE+1, 5));
            ax.set_xticklabels(range(yearS, yearE+1, 5));
    
    ax = axs.flatten()[-1]
    gl_hist_tc = np.sum(n_tc_per_year_basin, axis = 0)
    gl_downscaling_tc = np.sum(n_tc_per_year_downscaling_basin, axis = 0)
    h1, = ax.plot(dt_year, gl_hist_tc, 'k', linewidth=4);
    h2, = ax.plot(dt_year, gl_downscaling_tc, color='r', linewidth=3)
    mask_yrs = (ds['year'] >= yearS) & (ds['year'] <= yearE)
    ax.grid(); ax.set_xlim([yearS+1, yearE]); ax.set_ylim([60, 115])
    ax.set_xlabel('Year'); ax.set_ylabel('Number of TCs');
    ax.text(0.02, 0.98, 'GL', transform = ax.transAxes, verticalalignment = 'top', weight="bold", fontsize = 24)
    ax.text(0.78, 0.9, 'r = %0.2f' % float(np.corrcoef(gl_hist_tc[1:], gl_downscaling_tc[1:])[0, 1]), transform = ax.transAxes,
            bbox=dict(facecolor='gray', edgecolor='black'))
    ax.set_xticks(range(yearS+1, yearE+1, 5));
    ax.set_xticklabels(range(yearS+1, yearE+1, 5));

    plt.savefig('%s/%s/interannual_tc_count.png' % (namelist.base_directory, sim_name), bbox_inches = 'tight')
    # %%
    fig, axs = plt.subplots(figsize=(15, 18), ncols = 2, nrows = 4);
    for (i, basin_id) in enumerate(basin_ids):
        idx = np.argwhere(basin_names == basin_id).flatten()[0]  
        print(i)      
        if basin_id[0] == 'S':
            ax = axs.flatten()[i]
            h1, = ax.plot(dt_year[1:], mpd_per_year_basin[idx, 1:], 'k', linewidth=4);
            h2, = ax.plot(dt_year[1:], mpd_per_year_downscaling_basin[i, 1:], color='r', linewidth=3)
            mask_yrs = (ds['year'] >= yearS) & (ds['year'] <= yearE)
            ax.grid(); #ax.set_xlim([yearS+1, yearE]); 
            ax.set_xlabel('Year'); ax.set_ylabel('Storm Max. PDI ($m^3 / s^3$)');
            ax.text(0.02, 0.97, basin_id, transform = ax.transAxes, verticalalignment = 'top', weight="bold", fontsize = 24)
            ax.text(0.78, 0.9, 'r = %0.2f' % float(np.corrcoef(mpd_per_year_basin[idx, 1:], mpd_per_year_downscaling_basin[i, 1:])[0, 1]), transform = ax.transAxes,
                    bbox=dict(facecolor='gray', edgecolor='black'))
            ax.set_xticks(range(yearS+1, yearE+1, 5));
            ax.set_xticklabels(range(yearS+1, yearE+1, 5));
            ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(6,6))
            #ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year)*1.15 / 5) * 5])
            #yLim = ax.get_ylim(); ax.set_yticks(np.arange(0, yLim[1]+1, 5))
        else:
            ax = axs.flatten()[i]
            h1, = ax.plot(dt_year, mpd_per_year_basin[idx, :], 'k', linewidth=4);
            h2, = ax.plot(dt_year, mpd_per_year_downscaling_basin[i, :], color='r', linewidth=3)
            mask_yrs = (ds['year'] >= yearS) & (ds['year'] <= yearE)
            ax.grid(); ax.set_xlim([yearS, yearE]);
            ax.set_xticks(range(yearS+1, yearE+1, 5));
            #ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year)*1.15 / 5) * 5])
            ax.set_xlabel('Year'); ax.set_ylabel('Storm Max. PDI ($m^3 / s^3$)');
            ax.text(0.02, 0.97, basin_id, transform = ax.transAxes, verticalalignment = 'top', weight="bold", fontsize = 24)
            ax.text(0.78, 0.9, 'r = %0.2f' % float(np.corrcoef(mpd_per_year_basin[idx, :], mpd_per_year_downscaling_basin[i, :])[0, 1]), transform = ax.transAxes,
                    bbox=dict(facecolor='gray', edgecolor='black'))
            ax.ticklabel_format(axis = 'y', style = 'sci', scilimits=(6,6))
            #yLim = ax.get_ylim(); ax.set_yticks(np.arange(0, yLim[1]+1, 5))
            #ax.set_xticks(range(yearS, yearE+1, 5));
            #ax.set_xticklabels(range(yearS, yearE+1, 5));

    ax = axs.flatten()[-1]
    gl_hist_mpdi = np.sum(mpd_per_year_basin, axis = 0)
    gl_downscaling_mpdi = np.sum(mpd_per_year_downscaling_basin, axis = 0)
    ax = plt.gca(); ax.set_axisbelow(True)
    ax.bar(dt_year-0.2, gl_hist_mpdi, 0.4)
    ax.bar(dt_year+0.2, gl_downscaling_mpdi, 0.4, fc = [1, 0, 0, 0.7])
    plt.ylim([0, 1.5e7]); plt.grid(); plt.xlim([1980 - 0.4, 2021 + 0.4])
    ax.set_xticks(np.arange(yearS+1, 2022, 5));
    ax.text(0.02, 0.93, 'r = %0.2f' % float(np.corrcoef(gl_hist_mpdi[1:], gl_downscaling_mpdi[1:])[0, 1]), transform = ax.transAxes,
            bbox=dict(facecolor='gray', edgecolor='black'))
    #yr_labels = np.array([str(x) for x in np.arange(1979, 2022, 1)])
    #yr_labels[0::5] = ''; yr_labels[2::5] = ''; yr_labels[3::5] = ''; yr_labels[4::5] = ''
    #ax.set_xticklabels(yr_labels)
    plt.legend(['Observed', 'Downscaling'], ncol = 1, loc = 'upper center')
    plt.ylabel('Storm Max. PDI ($m^3 / s^3$)'); plt.xlabel('Year')
    plt.savefig('%s/%s/storm_max_pdi.png' % (namelist.base_directory, sim_name), bbox_inches = 'tight')

    # %%
    plt.figure(figsize=(8, 6))
    gl_hist_mpdi = np.sum(mpd_per_year_basin, axis = 0)
    gl_downscaling_mpdi = np.sum(mpd_per_year_downscaling_basin, axis = 0)
    ax = plt.gca(); ax.set_axisbelow(True)
    ax.bar(dt_year-0.2, gl_hist_mpdi, 0.4)
    ax.bar(dt_year+0.2, gl_downscaling_mpdi, 0.4, fc = [1, 0, 0, 0.7])
    plt.ylim([0, 1.5e7]); plt.grid(); plt.xlim([1980 - 0.4, 2021 + 0.4])
    ax.set_xticks(np.arange(yearS+1, 2022, 5));
    ax.text(0.02, 0.93, 'r = %0.2f' % float(np.corrcoef(gl_hist_mpdi[1:], gl_downscaling_mpdi[1:])[0, 1]), transform = ax.transAxes,
            bbox=dict(facecolor='gray', edgecolor='black'))
    #yr_labels = np.array([str(x) for x in np.arange(1979, 2022, 1)])
    #yr_labels[0::5] = ''; yr_labels[2::5] = ''; yr_labels[3::5] = ''; yr_labels[4::5] = ''
    #ax.set_xticklabels(yr_labels)
    plt.legend(['Observed', 'Downscaling'], ncol = 1, loc = 'upper center')
    plt.ylabel('Storm Max. PDI ($m^3 / s^3$)'); plt.xlabel('Year')
    plt.savefig('%s/%s/global_storm_max_pdi.png' % (namelist.base_directory, sim_name), bbox_inches = 'tight')

    # %%
    # tc_month_ensemble = np.zeros((len(np.unique(ds['n_trk'])), len(fn_tracks)))
    # tc_months = ds['tc_month'].load().data
    # for nt in np.unique(ds['n_trk']):
    #     mask = ds['n_trk'] == nt
    #     tc_month_ensemble[nt, :] = ds['tc_month'][mask]
    # m_bins = np.arange(0.5, 12.6, 1)
    # h_season = np.apply_along_axis(lambda a: np.histogram(a, bins = m_bins, density = 'pdf')[0], 0, tc_month_ensemble)
    basin_id = 'GL'
    basin_mask = basin_hist == basin_id if (basin_id != 'GL') else (basin_hist != 'GL')
    basin_trks_mask = basin_trks == basin_id if (basin_id != 'GL') else (basin_trks != 'GL')
    tc_sample_mask = (dt_ib >= datetime.datetime(1979, 1, 1)) & basin_mask
    mask = (np.nanmax(ds_ib['usa_wind'][tc_sample_mask, :], axis = 1) >= 34)
    seasonal_hist_counts = np.array([dt_ib[i].month for i in range(dt_ib.shape[0]) if tc_sample_mask[i]])
    hist_season = np.histogram(seasonal_hist_counts, bins = np.arange(0.5, 12.6, 1), density = 'pdf')
    N_per_year = len(seasonal_hist_counts) / (yearE - yearS + 1)
    sept_count = np.sum(hist_season[0][0:]) * N_per_year

    tc_months = ds['tc_month'].load().data
    n_ss = 300 # int(np.sum(n_tc_per_year))
    n_hist = int(N_per_year * (yearE - yearS + 1))
    tc_months_subsample = np.zeros((n_hist, n_ss))
    for i in range(n_ss):
        tc_months_subsample[:, i] = np.random.choice(tc_months[basin_trks_mask], n_hist)
    m_bins = np.arange(0.5, 12.6, 1)
    h_season = np.apply_along_axis(lambda x: np.histogram(x, bins = m_bins, density = 'pdf')[0], 0, tc_months_subsample)
    downscaling_fac = sept_count / np.sum(np.nanmean(h_season, axis = 1)[0:])
    mn_season_pdf = np.nanmean(h_season, axis = 1)*downscaling_fac
    mn_season_errP = np.nanquantile(downscaling_fac*h_season, 0.975, axis = 1) - mn_season_pdf
    mn_season_errN = mn_season_pdf - np.nanquantile(downscaling_fac*h_season, 0.025, axis = 1)

    plt.figure(figsize=(8, 6));
    plt.bar(range(1, 13, 1), mn_season_pdf,
            yerr = np.stack((mn_season_errN, mn_season_errP)), error_kw = {'elinewidth': 2, 'capsize': 6});
    plt.ylabel('Storms per Month'); plt.xlabel("Month");
    #plt.bar(range(1, 13, 1), hist_season[0]*np.nanmean(n_tc_per_year), fc=(1, 0, 0, 0.5));
    plt.plot(range(1, 13, 1), hist_season[0]*N_per_year, 'r*', linewidth = 5)
    plt.ylabel('Storms per Month'); plt.xlabel("Month"); plt.legend(['Observed', 'Downscaling'])
    plt.xlim([0.5, 12.5]); plt.xticks(range(1, 13, 1)); plt.grid();
    plt.gca().set_axisbelow(True)
    plt.savefig('%s/%s/seasonal_cycle.png' % (namelist.base_directory, sim_name), bbox_inches='tight')

    # %% Plot distribution of genesis
    dlon_label = 30
    lon_cen = 0
    lat_min = -50; lat_max = 50;
    lon_min = 0; lon_max = 359.999
    x_binEdges = np.arange(lon_min, lon_max + 0.1, 3)
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 3)
    gen_pdf_hist = np.histogram2d(gen_lon_hist[~np.isnan(gen_lat_hist)], gen_lat_hist[~np.isnan(gen_lat_hist)], bins = [x_binEdges, y_binEdges])[0]
    gen_pdf = np.histogram2d(lon_genesis, lat_genesis, bins = [x_binEdges, y_binEdges])[0]
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2

    plt.rcParams.update({'font.size': 20})

    lon_cen = 180
    lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
    lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label
    xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
    xlocs_shift = np.copy(xlocs)
    xlocs_shift[xlocs > lon_cen] -= 360
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(14, 12)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(211, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([30, 350, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    levels = np.arange(3, 30, 3)

    palette = copy(plt.get_cmap('jet'))
    palette.set_under('white', 1.0)
    #gen_pdf[gen_pdf == 0] = 1e-6                # so we can take the log
    #cmin = np.quantile(np.log(gen_pdf[gen_pdf > 1]), 0.1)
    #cmax = np.quantile(np.log(gen_pdf), 1)
    gen_pdf = gen_pdf / (n_sim * len(ds['year']))   # normalize by number of simulations
    gen_pdf_hist /= len(ds['year'])    # normalize by number of years
    gen_pdf = np.log(gen_pdf / (np.sum(gen_pdf) / np.sum(gen_pdf_hist)))

    levels = np.log(np.array([0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 1.5]))
    ax = plt.contourf(x_binCenter, y_binCenter, gen_pdf.T, levels = levels, extend = 'max', cmap=palette, transform = ccrs.PlateCarree());
    #plt.colorbar(orientation = 'horizontal');
    #plt.title('Number of Genesis Events per Year');
    #plt.savefig('%s/%s/genesis_pdf.png' % (namelist.base_directory, sim_name))

    #fig = plt.figure(facecolor='w', edgecolor='k');
    #fig.set_size_inches(12, 9)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(212, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([30, 350, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax = plt.contourf(x_binCenter, y_binCenter, np.log(gen_pdf_hist.T), levels = levels, extend = 'max', cmap=palette, transform = ccrs.PlateCarree());
    #plt.colorbar(orientation = 'horizontal', anchor = (0, 0.5));
    cbar_ax = fig.add_axes([0.15, 0.495, 0.7, 0.015])
    cbar = plt.colorbar(cax=cbar_ax, orientation = 'horizontal')
    cbar.ax.set_xticklabels([0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 1, 1.5])
    #plt.title('Number of Genesis Events per Year (Historical)');
    plt.gca().set_axisbelow(True);
    plt.savefig('%s/%s/genesis_pdf.png' % (namelist.base_directory, sim_name), bbox_inches='tight')
    #if lon_cen == 0:
        #plt.xlim([lon_min-360, lon_max-360])

    # %%
    basin_hist_ib = basin_hist[date_mask].astype('U2')
    n_count = np.zeros((n_ss, len(basin_ids)))
    n_count_hist = np.zeros(basin_ids.shape)
    for n_idx in range(n_ss):
        basin_subsample = np.random.choice(basin_trks, int(np.sum(n_tc_per_year, axis = 0)))
        for (i, basin_id) in enumerate(basin_ids):
            n_count[n_idx, i] = np.sum(basin_subsample == basin_id)
        n_count[n_idx, :] /= np.sum(n_count[n_idx, :])
    for (i, basin_id) in enumerate(basin_ids):
        n_count_hist[i] = np.sum(basin_hist_ib == basin_id) / basin_hist_ib.shape[0]
    
    mn_count_pdf = np.nanmean(n_count, axis = 0)
    mn_count_errP = np.nanquantile(n_count, 0.975, axis = 0) - mn_count_pdf
    mn_count_errN = mn_count_pdf - np.nanquantile(n_count, 0.025, axis = 0)

    plt.figure(figsize=(8, 6))
    plt.bar(basin_ids, mn_count_pdf, yerr = np.stack((mn_count_errN, mn_count_errP)), fc = (0, 0, 1, 0.5),
            error_kw = {'elinewidth': 2, 'capsize': 6});
    plt.scatter(basin_ids, n_count_hist, c = 'r')
    print(np.nanmean(n_count, axis = 0))
    plt.xlabel('Basin'); plt.ylabel('Fraction of TCs')
    plt.legend(['Obs', 'Downscaling']); plt.ylim([0, 0.4])
    plt.tight_layout()
    plt.savefig('%s/%s/basin_count.png' % (namelist.base_directory, sim_name))

    # %% Genesis latitudes
    basin_id = 'GL'
    basin_mask = basin_trks == basin_id if basin_id != 'GL' else basin_trks != 'GL'
    basin_hist_mask = basin_hist == basin_id if basin_id != 'GL' else basin_hist != 'GL'
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 2.5)
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2

    hist_gen_lat = gen_lat_hist[~np.isnan(gen_lat_hist) & date_mask & basin_hist_mask]
    tc_lat_subsample = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    for i in range(n_ss):
        tc_lat_subsample[:, i] = np.random.choice(lat_genesis[basin_mask], int(np.sum(n_tc_per_year)))
    h_lat = np.apply_along_axis(lambda x: np.histogram(x, bins = y_binEdges)[0], 0, tc_lat_subsample)
    h_lat_hist = np.histogram(hist_gen_lat, bins = y_binEdges)
    # NH Hemisphere
    #idx_et = np.argwhere(y_binCenter >= 30).flatten()[0]
    #downscaling_fac = np.sum(h_lat_hist[0][0:idx_et]) / np.sum(np.nanmean(h_lat, axis = 1)[0:idx_et])
    # SH Hemisphere
    idx_et = np.argwhere(y_binCenter <= 90).flatten()[0]
    downscaling_fac = np.sum(h_lat_hist[0][idx_et:]) / np.sum(np.nanmean(h_lat, axis = 1)[idx_et:])

    mn_lat_pdf = np.nanmean(h_lat, axis = 1)*downscaling_fac
    mn_lat_errP = np.nanquantile(downscaling_fac*h_lat, 0.975, axis = 1) - mn_lat_pdf
    mn_lat_errN = mn_lat_pdf - np.nanquantile(downscaling_fac*h_lat, 0.025, axis = 1)

    plt.figure(figsize=(8, 6));
    plt.bar(y_binCenter, mn_lat_pdf, width = 2.,
            yerr = np.stack((mn_lat_errN, mn_lat_errP)), error_kw = {'elinewidth': 2, 'capsize': 6});
    plt.plot(y_binCenter, h_lat_hist[0], 'r*', linewidth = 5)
    plt.xlabel('Latitude'); plt.ylabel('Number of Events')
    plt.legend(['Observations', 'Downscaling']); plt.grid();
    plt.xticks(np.arange(-40, 41, 10)); plt.xlim([-45, 45])
    plt.gca().set_axisbelow(True)
    #plt.title(basin_id)
    plt.savefig('%s/%s/genesis_latitudes.png' % (namelist.base_directory, sim_name))

    # plt.figure(figsize=(9,5));
    # spi_pdf = np.sum(spi_clim, axis = (0, 2)) / np.sum(spi_clim);
    # y_min = np.sin(np.pi / 180 * 3)
    # y_max = np.sin(np.pi / 180 * 45)
    # gen_lat = np.arcsin(np.random.uniform(y_min, y_max, 1000000)) * 180 / np.pi
    # plt.plot(lat_spi, spi_pdf);
    # plt.hist(gen_lat, bins = np.arange(0, 45.1, 1), density = True);
    # plt.xlim([0, 40]); plt.xlabel('Latitude'); plt.ylabel('Genesis Probability'); plt.grid()
    # plt.legend(['SPI ERA5', 'Random Seeding'])

    gen_lat = np.linspace(0, 20, 100)
    prob_lowlat = np.power(np.minimum(np.maximum((np.abs(gen_lat) - 2) / 12.0, 0), 1), 10)
    prob_lowlat2 = np.power(np.minimum(np.maximum((np.abs(gen_lat) - 2) / 12.0, 0), 1), 3)
    #plt.plot(gen_lat, prob_lowlat, gen_lat, prob_lowlat2)

    # %% TC Density PDF.
    den_pdf = np.histogram2d(lon_filt.flatten(), lat_filt.flatten(), bins = [x_binEdges, y_binEdges])[0]
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2

    # Plot distribution of density
    plt.rcParams.update({'font.size': 20})
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(16, 12)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(211, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    palette = copy(plt.get_cmap('jet'))
    palette.set_under('white', 1.0)

    den_pdf /= np.sum(den_pdf)
    den_pdf = np.log(den_pdf)
    cmax = -5
    cmin = -9
    levels = np.linspace(cmin, cmax, 9)
    plt.contourf(x_binCenter, y_binCenter, den_pdf.T, levels, cmap=palette, transform = ccrs.PlateCarree());

    den_pdf = np.histogram2d(track_lon_hist.flatten(), track_lat_hist.flatten(), bins = [x_binEdges, y_binEdges])[0]
    den_pdf /= np.sum(den_pdf)
    den_pdf = np.log(den_pdf)

    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(212, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    ax = plt.contourf(x_binCenter, y_binCenter, den_pdf.T, levels, cmap=palette, transform = ccrs.PlateCarree());
    cbar_ax = fig.add_axes([0.15, 0.49, 0.7, 0.02])
    plt.colorbar(cax=cbar_ax, orientation = 'horizontal')
    plt.savefig('%s/%s/track_density.png' % (namelist.base_directory, sim_name), bbox_inches = 'tight')

    # %% Plot a random set of tracks
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(8, 8)
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.gca(projection = proj)
    #ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=np.arange(120, 160, 20),
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #r_idxs = np.random.randint(0, lon_filt.shape[0], size = 20)
    r_idxs = np.random.choice(np.argwhere((lat_genesis >= -20) & (lat_genesis <= -10) &
                                          (lon_genesis >= 135) & (lon_genesis <= 145)).flatten(), size = 30)
    #r_idxs = np.random.choice(np.argwhere((basin_hist == 'SP') & date_mask).flatten(), size = 10)

    plt.plot(track_lon_hist[r_idxs, :].T, track_lat_hist[r_idxs, :].T, transform=ccrs.PlateCarree());
    plt.scatter(track_lon_hist[r_idxs, 0], track_lat_hist[r_idxs, 0], color='k', transform=ccrs.PlateCarree());
    plt.xlim([-70, 70])

    # %% Plot a random set of tracks
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(12, 10)
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.gca(projection = proj)
    #ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=np.arange(120, 241, 20),
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #r_idxs = np.random.randint(0, lon_filt.shape[0], size = 20)
    #r_idxs = np.random.choice(np.argwhere((lat_genesis >= -20) & (lat_genesis <= 5) &
    #                                      (lon_genesis >= 135) & (lon_genesis <= 145)).flatten(), size = 30)
    r_idxs = np.random.choice(np.argwhere(basin_trks == 'SP').flatten(), size = 25)

    plt.plot(lon_filt[r_idxs, :].T, lat_filt[r_idxs, :].T, transform=ccrs.PlateCarree());
    plt.scatter(lon_filt[r_idxs, 0].data, lat_filt[r_idxs, 0].data, color='k', transform=ccrs.PlateCarree());
    plt.xlim([-60, 60]); plt.ylim([-30, 0])

    # %% Plot lifetime maximum intensity distribution
    basin_id = 'GL'
    basin_mask = (basin_trks == basin_id) if (basin_id != 'GL') else (basin_trks != 'GL')
    lmi_bins = np.arange(35, 206, 10)
    lmi_cbins = np.arange(40, 201, 10)
    vmax_lmi = np.nanmax(vmax_filt * 1.94384, axis = 1)
    tc_lmi_subsample = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    for i in range(n_ss):
        tc_lmi_subsample[:, i] = np.random.choice(vmax_lmi[basin_mask], int(np.sum(n_tc_per_year)))
 
    h_lmi = np.apply_along_axis(lambda a: np.histogram(a, bins = lmi_bins, density = 'pdf')[0], 1, tc_lmi_subsample)
    mn_lmi_errP = np.nanquantile(h_lmi, 0.95, axis = 0) - np.nanmean(h_lmi, axis = 0)
    mn_lmi_errN = np.nanmean(h_lmi, axis = 0) - np.nanquantile(h_lmi, 0.05, axis = 0)
    h_lmi_obs = np.histogram(tc_lmi[date_mask & (basin_hist == basin_id) if (basin_id != 'GL') else (basin_hist != 'GL')],
                             density = True, bins = lmi_bins)[0];

    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(8,6));
    plt.plot(lmi_cbins, h_lmi_obs, 'r*', markersize = 10)
    plt.bar(lmi_cbins, np.nanmean(h_lmi, axis = 0), width = 8,
            yerr = np.stack((mn_lmi_errN, mn_lmi_errP)), error_kw = {'elinewidth': 2, 'capsize': 6});
    #plt.bar(lmi_cbins, h_lmi_obs, fc = (1, 0, 0, 0.5), width = 8)
    #plt.hist(tc_lmi, density=True, bins = np.arange(35, 186, 10), fc=(1, 0, 0, 0.5));
    plt.xlabel('Lifetime Maximum Intensity (kts)'); plt.ylabel('Density'); plt.grid();
    plt.legend(['Observed', 'Downscaling']); plt.xlim([35, 205])
    plt.tight_layout()
    plt.savefig('%s/%s/lmi_pdf.png' % (namelist.base_directory, sim_name))

    # %%
    x_binEdges = np.arange(lon_min, lon_max + 0.1, 1)
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 1)
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2
    nX = len(x_binEdges)
    nY = len(y_binEdges)

    track_arr_hist = np.zeros((180 * track_lon_hist.shape[0], 3))
    track_arr_hist[:, 0] = track_lon_hist[:, ::2].flatten()
    track_arr_hist[:, 1] = track_lat_hist[:, ::2].flatten()
    track_arr_hist[:, 2] = np.tile(np.array(range(0, 180)), (track_lon_hist.shape[0], 1)).flatten()
    H_td, _ = np.histogramdd(track_arr_hist, bins = [x_binEdges, y_binEdges, range(0, 180)])

    # %%
    H_count = np.zeros((nX-1, nY-1))
    H_count_hist = np.zeros((nX-1, nY-1))
    dt_mask = (dt_ib >= datetime.datetime(1979, 1, 1, 0)) & (dt_ib <= datetime.datetime(2021, 12, 31, 23))
    for i in range(track_lon_hist.shape[0]):
        if ~np.all(np.isnan(track_lon_hist[i, :])) and dt_mask[i]:
            H, _, _ = np.histogram2d(track_lon_hist[i, usa_wind[i, :] >= 64], 
                                     track_lat_hist[i, usa_wind[i, :] >= 64],
                                     bins = [x_binEdges, y_binEdges])
            H_count_hist += np.minimum(H, 1)

    for i in range(lon_filt.shape[0]):
        H, _, _ = np.histogram2d(lon_filt[i, (vmax_filt[i, :] * 1.94384) >= 64], 
                                 lat_filt[i, (vmax_filt[i, :] * 1.94384) >= 64],
                                 bins = [x_binEdges, y_binEdges])
        H_count += np.minimum(H, 1)

    #plt.pcolormesh(x_binEdges, y_binEdges, np.log10(43 / H_td.T))
    #plt.colorbar()
    # %%
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(20, 13)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(311, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    return_period_fac = np.nanmean(np.sum(n_tc_per_year_basin, axis = 0))
    return_period = (43 * n_sim) / (H_count.T * 90 / return_period_fac)
    return_period_hist = 43 / gaussian_filter(H_count_hist.T, sigma = 1)

    levels = np.linspace(0, 1.7, 101)
    palette = copy(plt.get_cmap('jet_r'))
    palette.set_over('white', 1.0)
    ax.contourf(x_binCenter, y_binCenter, np.log10(return_period), levels = levels, extend = 'both', cmap=palette, transform = ccrs.PlateCarree());
    ax.text(0.015, 0.88, 'Downscaling', transform = ax.transAxes, bbox=dict(facecolor='gray', edgecolor='black'))

    ax = fig.add_subplot(312, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    im = ax.contourf(x_binCenter, y_binCenter, np.log10(return_period_hist),  levels = levels, extend = 'both', cmap=palette, transform = ccrs.PlateCarree());
    ax.text(0.015, 0.88, 'Observations', transform = ax.transAxes, bbox=dict(facecolor='gray', edgecolor='black'))

    cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks = np.log10(np.array([1, 2, 5, 10, 25, 50, 100])), orientation = 'horizontal')
    cbar.ax.set_xticklabels([1, 2, 5, 10, 25, 50, 100]);
    cbar.ax.set_xlabel('Return Period (Years)')

    import matplotlib
    diff_ret_pd = return_period_hist - return_period
    diff_ret_pd[np.isneginf(diff_ret_pd) | np.isposinf(diff_ret_pd)] = np.nan
    log_diff_ret_pd = np.zeros(diff_ret_pd.shape)
    lin_thresh = 2
    lin_scale = 0.5
    log_diff_ret_pd[diff_ret_pd > lin_thresh] = np.log10(diff_ret_pd[diff_ret_pd > lin_thresh]) + lin_scale - np.log10(lin_thresh)
    log_diff_ret_pd[diff_ret_pd < -lin_thresh] = -np.log10(-diff_ret_pd[diff_ret_pd < -lin_thresh]) - lin_scale + np.log10(lin_thresh)
    log_diff_ret_pd[np.abs(diff_ret_pd) <= lin_thresh] = diff_ret_pd[np.abs(diff_ret_pd) <= lin_thresh] * (lin_scale * 2) / lin_thresh
    log_diff_ret_pd[(return_period >= 1e5) | (return_period_hist >= 1e5)] = np.nan
    log_diff_ret_pd[log_diff_ret_pd <= -4.25] = np.nan
    log_diff_ret_pd[log_diff_ret_pd >= 4.25] = np.nan
    log_diff_ret_pd[return_period_hist > 43] = np.nan

    ax = fig.add_subplot(313, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    palette = copy(plt.get_cmap('jet'))
    palette.set_under('white', 1.0)
    levels = np.linspace(-1.7 - lin_scale / 2, 1.7 + lin_scale / 2, 101)

    im = ax.contourf(x_binCenter, y_binCenter, log_diff_ret_pd,
                 levels = levels, cmap='RdBu_r', transform = ccrs.PlateCarree());
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
    yr_labels = np.array([5, 10, 25, 50])
    lin_labels = np.array([-2, -1, 0, 1, 2])
    lin_ticks = lin_labels * lin_scale / lin_thresh
    yr_ticks = np.concatenate((np.flip(-np.log10(yr_labels) - lin_scale + np.log10(lin_thresh)),
                               lin_ticks, np.log10(yr_labels) + lin_scale - np.log10(lin_thresh)))
    yr_tick_labels = np.concatenate((np.flip(-yr_labels), lin_labels, yr_labels))
    cbar = fig.colorbar(im, cax=cbar_ax, ticks = yr_ticks, orientation = 'horizontal')
    cbar.ax.set_xticklabels(yr_tick_labels);
    cbar.ax.set_xlabel('Return Period Difference (Years)');

    # %% return period
    from scipy.ndimage import gaussian_filter
    x_binEdges = np.arange(lon_min, lon_max + 0.1, 5)
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 5)
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2
    cat1_mask = (vmax_filt * 1.94384) >= 64
    den_pdf = np.histogram2d(lon_filt[cat1_mask].flatten(), lat_filt[cat1_mask].flatten(), bins = [x_binEdges, y_binEdges])[0]
    den_pdf /= np.sum(~np.isnan(lon_filt)) #np.sum(den_pdf)
    den_pdf = gaussian_filter(den_pdf, sigma = 0.75)

    cat1_hist_mask = usa_wind >= 64
    hist_den_pdf = np.histogram2d(track_lon_hist[cat1_hist_mask].flatten(), track_lat_hist[cat1_hist_mask].flatten(), bins = [x_binEdges, y_binEdges])[0]
    hist_den_pdf /= np.sum(np.sum(~np.isnan(track_lon_hist)))
    hist_den_pdf = gaussian_filter(hist_den_pdf, sigma = 0.5)

    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(20, 13)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(311, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    return_period_fac = np.nanmean(np.sum(n_tc_per_year_basin, axis = 0))
    return_period = 1 / (den_pdf * return_period_fac)
    return_period_hist = 1 / (hist_den_pdf * return_period_fac)    
    levels = np.linspace(0, 3, 101)
    palette = copy(plt.get_cmap('jet_r'))
    palette.set_over('white', 1.0)
    ax.contourf(x_binCenter, y_binCenter, np.log10(return_period.T), levels = levels, extend = 'both', cmap=palette, transform = ccrs.PlateCarree());
    ax.text(0.015, 0.88, 'Downscaling', transform = ax.transAxes, bbox=dict(facecolor='gray', edgecolor='black'))

    ax = fig.add_subplot(312, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}
    im = ax.contourf(x_binCenter, y_binCenter, np.log10(return_period_hist.T),  levels = levels, extend = 'both', cmap=palette, transform = ccrs.PlateCarree());
    ax.text(0.015, 0.88, 'Observations', transform = ax.transAxes, bbox=dict(facecolor='gray', edgecolor='black'))

    cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.015])
    cbar = fig.colorbar(im, cax=cbar_ax, ticks = np.log10(np.array([1, 2, 5, 10, 25, 50, 100, 500, 1000])), orientation = 'horizontal')
    #cbar.ax.set_xticks([0, 1, 2, 3]);
    cbar.ax.set_xticklabels([1, 2, 5, 10, 25, 50, 100, 500, 1000]);
    cbar.ax.set_xlabel('Return Period (Years)')
    #cbar.ax.xaxis.set_label_position('top')

    import matplotlib
    diff_ret_pd = return_period_hist - return_period
    diff_ret_pd[np.isneginf(diff_ret_pd) | np.isposinf(diff_ret_pd)] = np.nan
    log_diff_ret_pd = np.zeros(diff_ret_pd.shape)
    lin_thresh = 2
    lin_scale = 0.5
    log_diff_ret_pd[diff_ret_pd > lin_thresh] = np.log10(diff_ret_pd[diff_ret_pd > lin_thresh]) + lin_scale - np.log10(lin_thresh)
    log_diff_ret_pd[diff_ret_pd < -lin_thresh] = -np.log10(-diff_ret_pd[diff_ret_pd < -lin_thresh]) - lin_scale + np.log10(lin_thresh)
    log_diff_ret_pd[np.abs(diff_ret_pd) <= lin_thresh] = diff_ret_pd[np.abs(diff_ret_pd) <= lin_thresh] * (lin_scale * 2) / lin_thresh
    log_diff_ret_pd[(return_period >= 1e5) | (return_period_hist >= 1e5)] = np.nan
    log_diff_ret_pd[log_diff_ret_pd <= -4.25] = np.nan
    log_diff_ret_pd[log_diff_ret_pd >= 4.25] = np.nan

    ax = fig.add_subplot(313, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 18}
    gl.ylabel_style = {'size': 18}

    palette = copy(plt.get_cmap('jet'))
    palette.set_under('white', 1.0)
    levels = np.linspace(-2 - lin_scale / 2, 2 + lin_scale / 2, 101)

    im = ax.contourf(x_binCenter, y_binCenter, log_diff_ret_pd.T, extend = 'both',
                 levels = levels, cmap='RdBu_r', transform = ccrs.PlateCarree());
    cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
    yr_labels = np.array([5, 10, 25, 50, 100])
    lin_labels = np.array([-2, -1, 0, 1, 2])
    lin_ticks = lin_labels * lin_scale / lin_thresh
    yr_ticks = np.concatenate((np.flip(-np.log10(yr_labels) - lin_scale + np.log10(lin_thresh)),
                               lin_ticks, np.log10(yr_labels) + lin_scale - np.log10(lin_thresh)))
    yr_tick_labels = np.concatenate((np.flip(-yr_labels), lin_labels, yr_labels))
    cbar = fig.colorbar(im, cax=cbar_ax, ticks = yr_ticks, orientation = 'horizontal')
    cbar.ax.set_xticklabels(yr_tick_labels);
    cbar.ax.set_xlabel('Return Period Difference (Years)');
    plt.savefig('%s/%s/return_period.png' % (namelist.base_directory, sim_name))

    # %% Plot 6-hour zonal and meridional displacements in region defined by basin_mask
    basin_dr = [0, 360, -30, 30]   # AL
    #basin_dr = [130, 160, 5, 30]   # WP
    #basin_dr = [200, 260, 10, 30]   # EP
    #basin_dr = [170, 220, -30, -5]   # SP

    ds_ib_lon = ds_ib['lon'].data
    ds_ib_lon[ds_ib_lon < 0] = ds_ib_lon[ds_ib_lon < 0] + 360
    ib_basin_mask = ((ds_ib['lat'][:, 1:-1] >= basin_dr[2]) & (ds_ib['lat'][:, 1:-1] <= basin_dr[3]) &
                     (ds_ib_lon[:, 1:-1] >= basin_dr[0]) & (ds_ib_lon[:, 1:-1] <= basin_dr[1]) &
                     (ds_ib['usa_wind'][:, 1:-1].data >= 35))
    basin_mask = ((lat_filt >= basin_dr[2]) & (lat_filt <= basin_dr[3]) &
                  (lon_filt >= basin_dr[0]) & (lon_filt <= basin_dr[1]))
    dlon_6h = (lon_filt[:, ::6][:, 1:] - lon_filt[:, ::6][:, 0:-1])[basin_mask[:, ::6][:, 0:-1]]
    dlat_6h = (lat_filt[:, ::6][:, 1:] - lat_filt[:, ::6][:, 0:-1])[basin_mask[:, ::6][:, 0:-1]]
    ib_dlon_6h = (ds_ib_lon[:, 2:] - ds_ib_lon[:, 0:-2])[ib_basin_mask]
    ib_dlat_6h = (ds_ib['lat'][:, 2:] - ds_ib['lat'][:, 0:-2]).data[ib_basin_mask]

    plt.figure(figsize=(10,8)); plt.hist(dlon_6h, density=True, bins = np.arange(-3.5, 3.51, 0.25));
    plt.hist(ib_dlon_6h, density=True, bins = np.arange(-3.5, 3.51, 0.25), fc = (1, 0, 0, 0.5));
    plt.xlabel('6-Hour Zonal Displacement (deg)'); plt.ylabel('Density'); plt.grid();
    plt.legend(['Downscaling', 'Obs'])
    plt.savefig('%s/%s/zonal_track_displacement.png' % (namelist.base_directory, sim_name))    
    plt.figure(figsize=(10,8)); plt.hist(dlat_6h, density=True, bins = np.arange(-2.5, 2.51, 0.25));
    plt.hist(ib_dlat_6h, density=True, bins = np.arange(-2.5, 2.51, 0.25), fc = (1, 0, 0, 0.5));
    plt.xlabel('6-Hour Meridional Displacement (deg)'); plt.ylabel('Density'); plt.grid();
    plt.legend(['Downscaling', 'Obs'])
    plt.savefig('%s/%s/meridional_track_displacement.png' % (namelist.base_directory, sim_name))
    # %%% BLAH BLAH
    def autocorr(x):
        result = np.correlate(x, x, mode='same')
        return result[int(result.size/2):]

    def gen_f(N, T, t, num):
        fs = np.zeros((num, np.size(t)))
        for i in range(0, num):
            n = np.linspace(1, N, N)
            #xln = np.tile(np.random.rand(N, 1) / 4, (1, np.size(t)))   # Zero phase correlation
            xln = np.tile(np.random.rand(N, 1), (1, np.size(t)))   # Zero phase correlation
            fs[i, :] = np.sqrt(2 / np.sum(np.power(n, -3))) * \
                    np.sum(np.multiply(np.tile(np.power(n, -1.5), (np.size(t), 1)).T,
                                        np.sin(2. * np.pi * (np.outer(n, t) / T + xln))), axis=0)
        return(fs)

    t = np.linspace(0, 15*24*60*60, 3600)
    Fs = gen_f(15, 15*24*60*60, t, 100)
    acf_Fs = np.zeros((100, 1800))

    for i in range(100):
        acf_Fs[i, :] = autocorr(Fs[i, :])
    # %% Plot 24h intensity change distribution
    basin_id = 'GL'
    ds_b = xr.open_dataset('/home/jlin/src/tc_risk/land/land.nc')
    land_mask = ds_b['land']
    f_land = mat.interp2_fx(land_mask['lon'], land_mask['lat'], land_mask)

    mask = ((dt_ib >= datetime.datetime(yearS, 1, 1)) &
            (dt_ib <= datetime.datetime(yearE, 12, 31)) &
            (basin_hist == basin_id if (basin_id != 'GL') else (basin_hist != 'GL')) &
            (np.nanmax(ds_ib['usa_wind'], axis = 1) >= 35))
    int_time = ds_ib['usa_wind'][mask, ::2]['time'].load().data
    int_obs = ds_ib['usa_wind'][mask, ::2].load().data
    int_landfall = ds_ib['landfall'][mask, ::2].load().data
    trk_24h = np.arange(0, 45.1, 0.25) * 24 * 60 * 60
    int_24h_obs = np.zeros((int_obs.shape[0],)+(177,))
    for idx in range(int_24h_obs.shape[0]):
        t_trk = ((int_time[idx, :] - int_time[idx, 0]) / 1e9).astype(np.float64)     # seconds
        t_trk[t_trk < 0] = np.nan
        int_24h = np.interp(trk_24h, t_trk, int_obs[idx, :])
        int_24h_landfall = np.interp(trk_24h, t_trk, int_landfall[idx, :])
        int_rate_24h = int_24h[4:] - int_24h[0:-4]
        int_rate_24h[int_24h[0:-4] < 30] = np.nan
        int_rate_24h[int_24h_landfall[0:-4] <= 0] = np.nan
        int_24h_obs[idx, :] = int_rate_24h
    int_24h_obs = int_24h_obs[~np.isnan(int_24h_obs)] 

    idx_24h = int((24 * 60 * 60) / (ds['time'][1] - ds['time'][0]))
    downscaling_mask = (basin_trks == basin_id if (basin_id != 'GL') else (basin_trks != 'GL'))
    tc_int_mask = np.logical_and(f_land.ev(lon_filt[downscaling_mask, idx_24h::idx_24h],
                                 lat_filt[downscaling_mask, idx_24h::idx_24h]).flatten() >= 1,
                                 vmax_filt[downscaling_mask, idx_24h::idx_24h].flatten() >= 15,
                                 lat_filt[downscaling_mask, idx_24h::idx_24h].flatten() <= 45);
    int_24h = (np.round(vmax_filt[downscaling_mask, idx_24h::idx_24h] * 1.94384 / 5) * 5 -
               np.round(vmax_filt[downscaling_mask, 0:-idx_24h:idx_24h] * 1.94384 / 5) * 5).flatten()
    int_24h = int_24h[tc_int_mask]

    bin_width = 5
    bin_edges = np.arange(-65.0, 65.1, bin_width)
    bin_center = (bin_edges[1:] + bin_edges[0:-1]) / 2

    n_ss = 300
    h_int_24h_downscaling_ss = np.zeros((n_ss, bin_center.shape[0]))
    for i in range(n_ss):
        int_24h_ss = np.random.choice(int_24h, int_24h_obs.shape[0])
        h_int_24h_downscaling_ss[i, :] = np.histogram(int_24h_ss, bins = bin_edges, density=True)[0]

    h_int_24h_obs = np.histogram(int_24h_obs, bins = bin_edges, density=True)
    h_int_24h_downscaling = np.nanmean(h_int_24h_downscaling_ss, axis = 0)
    h_int_24h_errP = np.nanquantile(h_int_24h_downscaling_ss, 0.975, axis = 0) - h_int_24h_downscaling
    h_int_24h_errN = h_int_24h_downscaling - np.nanquantile(h_int_24h_downscaling_ss, 0.025, axis = 0)    
    ri_ints = np.linspace(20, 40, 1000)
    #int_95 = ri_ints[np.argmin(np.abs(np.interp(ri_ints, bin_center, np.cumsum(h_int_24h_downscaling[0] * bin_width)) - 0.95))]

    plt.figure(figsize=(8,6)); ax = plt.gca();
    #plt.bar(bin_center, h_int_24h_downscaling, yerr = np.stack((h_int_24h_errN, h_int_24h_errP)), fc = (0, 0, 1, 0.5),
    #        error_kw = {'elinewidth': 2, 'capsize': 6});    
    ax.bar(bin_center, h_int_24h_downscaling, 4)
    plt.plot(bin_center, h_int_24h_obs[0], 'r*', markersize = 6)
    plt.gca().set_xticks(bin_center); plt.xlabel('24h Intensity Change (kts)');
    plt.ylabel('PDF'); plt.grid(); plt.xticks(np.arange(-60, 61, 10))
    #plt.title('95th Percentile: %f kts / 24h' % int_95)
    plt.legend(['Observed', 'Downscaling'])
    plt.gca().set_axisbelow(True); plt.xlim([-62.5, 62.5])
    plt.savefig('%s/%s/24h_intensity_distribution.png' % (namelist.base_directory, sim_name))

    # %% Plot latitude of LMI
    basin_id = 'GL'
    basin_mask = basin_trks == basin_id if basin_id != 'GL' else basin_trks != 'GL'
    basin_hist_mask = basin_hist == basin_id if basin_id != 'GL' else basin_hist != 'GL'
    lat_lmi = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    lat_filt_basin = lat_filt[basin_mask, :]
    vmax_filt_basin = vmax_filt[basin_mask, :]
    for i in range(n_ss):
        ss_idxs = np.random.choice(range(np.sum(basin_mask)), int(np.sum(n_tc_per_year)))
        lat_filt_ens = lat_filt_basin[ss_idxs, :]
        vmax_filt_ens = vmax_filt_basin[ss_idxs, :]
        vmax_filt_ens_lmi_idx = np.nanargmax(vmax_filt_ens, axis = 1)

        for j in range(vmax_filt_ens_lmi_idx.shape[0]):
            lat_lmi[j, i] = lat_filt_ens[j, vmax_filt_ens_lmi_idx[j]]

    lat_lmi_width = 2.5
    lat_lmi_bins = np.arange(-50.0, 50.1, lat_lmi_width)
    lat_lmi_cbins = np.arange(-50.0 + lat_lmi_width / 2, 50.1 - lat_lmi_width / 2, lat_lmi_width)
    h_lat_lmi = np.apply_along_axis(lambda a: np.histogram(a, bins = lat_lmi_bins, density = 'pdf')[0], 1, lat_lmi)
    hist_lat_lmi = np.histogram(tc_lat_lmi[basin_hist_mask], bins = lat_lmi_bins, density = 'pdf')[0] * lat_lmi_width
    mn_lat_lmi_pdf = lat_lmi_width*np.nanmean(h_lat_lmi, axis = 0)
    mn_lat_lmi_errP = lat_lmi_width*np.nanquantile(h_lat_lmi, 0.95, axis = 0) - mn_lat_lmi_pdf
    mn_lat_lmi_errN = mn_lat_lmi_pdf - lat_lmi_width*np.nanquantile(h_lat_lmi, 0.05, axis = 0)

    plt.figure(figsize=(8,6));
    plt.plot(lat_lmi_cbins, hist_lat_lmi, 'r*', markersize = 10)
    plt.bar(lat_lmi_cbins, mn_lat_lmi_pdf, width = 2.,
            yerr = np.stack((mn_lat_lmi_errN, mn_lat_lmi_errP)), error_kw = {'elinewidth': 2, 'capsize': 6});
    #plt.bar(lat_lmi_cbins, hist_lat_lmi, fc = (1, 0, 0, 0.5), width = 4)
    plt.xlabel('Lifetime Maximum Intensity Latitude ($\degree$)'); plt.ylabel('Density'); plt.grid();
    plt.legend(['Observed', 'Downscaling'])
    plt.tight_layout(); plt.xlim([-45, 45])
    plt.gca().set_axisbelow(True)
    plt.savefig('%s/%s/lat_lmi.png' % (namelist.base_directory, sim_name))

    # %%
    rh = np.linspace(0, 1, 100)
    #rh_1 = np.minimum(np.maximum(np.power((rh + 0.7) / 2, 3), 0.15), 0.35)
    rh_1 = 0.25 / (1 + np.exp(-(rh - 0.55) * 10)) + 0.1
    rh_2 = np.minimum(np.maximum(rh - 0.35, 0.1), 0.35)
    rh_3 = np.minimum(np.maximum(rh / 2, 0.1), 0.35)
    plt.plot(rh, rh_1); plt.plot(rh, rh_2); plt.plot(rh, rh_3); plt.grid(); plt.ylim([0.05, 0.4])
    plt.legend(['New', 'rh-0.35', 'rh / 2'])

    # %% LMI trends
    yearS = 1979; yearE = 2021;
    tc_years = np.tile(np.array([[x]*90 for x in range(yearS, yearE + 1)]).flatten(), len(fn_tracks))
    yr_mask = tc_years == yearS
    lat_lmi = np.full((yearE - yearS + 1, np.sum(yr_mask)), np.nan)
    for i in range(lat_lmi.shape[0]):
        yr_mask = tc_years == (i + yearS)
        lat_filt_ens = lat_filt[yr_mask, :]
        vmax_filt_ens = vmax_filt[yr_mask, :]
        vmax_filt_ens_lmi_idx = np.nanargmax(vmax_filt_ens, axis = 1)
        lmi_ens = np.nanmax(vmax_filt_ens, axis = 1) * 1.94384

        for j in range(lat_filt_ens.shape[0]):
            if lmi_ens[j] > 0:
                lat_lmi[i, j] = lat_filt_ens[j, vmax_filt_ens_lmi_idx[j]]

    from scipy.stats import linregress
    lin_m = linregress(range(yearS, yearE + 1), np.nanmean(np.abs(lat_lmi), axis = 1))
    #lin_m2 = linregress(range(yearS, yearE + 1), np.nanmean(np.abs(lat_lmi_coupled), axis = 1))
    plt.figure(figsize=(10,6))
    plt.plot(range(yearS, yearE + 1), np.nanmean(np.abs(lat_lmi), axis = 1))
    #plt.plot(range(yearS, yearE + 1), np.nanmean(np.abs(lat_lmi_coupled), axis = 1))
    plt.plot(range(yearS, yearE + 1), lin_m.slope * np.array(range(yearS, yearE + 1)) + lin_m.intercept)
    #plt.plot(range(yearS, yearE + 1), lin_m2.slope * np.array(range(yearS, yearE + 1)) + lin_m2.intercept)
    plt.xlabel("Year"); plt.ylabel('Latitude of LMI'); plt.grid();

    # %%
    yr_ib = np.array([x.year for x in dt_ib])
    tc_lat_lmi_yr = np.zeros(43)
    mask = ds_ib['lat'][:, 0].data < 0
    for y in range(1979, 2022):
        tc_lat_lmi_yr[y - 1979] = np.nanmean(np.abs(tc_lat_lmi[(yr_ib == y) & mask]))

    plt.plot(range(1979, 2022), tc_lat_lmi_yr)
    plt.grid();
    # %%
    b = basins.TC_Basin('NA')
    idx_month = 8
    fn_th = calc_thermo.get_fn_thermo()
    ds = xr.open_dataset(fn_th)
    lon = ds['lon'].data
    lat = ds['lat'].data

    if 'EP' in b.basin_id:
        lon_min = 180; lon_max = 280;
        lat_min = 0; lat_max = 45;         
    elif 'NA' in b.basin_id:
        lon_min = 260; lon_max = 350;    
        lat_min = 0; lat_max = 45;         
    elif 'WP' in b.basin_id:
        lon_min = 100; lon_max = 180;
        lat_min = 0; lat_max = 45;         
    elif 'AU' in b.basin_id:        
        lon_min = 100; lon_max = 180; 
        lat_min = -45; lat_max = 0; 
    elif 'SI' in b.basin_id:
        lon_min = 10; lon_max = 100;
        lat_min = -45; lat_max = 0;         
    else:
        lon_min = 260; lon_max = 360;
    dlon_label = 20
    lon_cen = 0
    lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
    lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label
    xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
    xlocs_shift = np.copy(xlocs)
    xlocs_shift[xlocs > lon_cen] -= 360

    #dt_s = input.convert_from_datetime(ds, [datetime.datetime(2005, 8, 15)])[0]
    #dt_e = input.convert_from_datetime(ds, [datetime.datetime(2005, 9, 15)])[0]
    rh_to_m = lambda x: x #- 0.3 # np.minimum(x ** 2 + 0.1, 0.5)
    #rhm = np.nanmean(ds['rh_mid'][8::12], axis = 0)# - ds['rh_mid'][500, :, :]
    rhm = np.nanmean(ds['rh_mid'][idx_month::12], axis = 0) #- np.nanmean(ds['rh_mid'][7::12], axis = 0)
    rh_mid = rh_to_m(rhm) #- rh_to_m(np.nanmean(ds['rh_mid'][7::12], axis = 0))
    vpot = np.nanmean(ds['vmax'][idx_month::12], axis = 0) #- np.nanmean(ds['vmax'][7::12], axis = 0)
    chi = np.nanmean(ds['chi'][idx_month::12], axis = 0)
    chi_to_CHI = lambda x: np.exp(np.log(x+0.01) + 1.6)
    CHI = chi_to_CHI(chi)
    CHI[np.isnan(CHI)] = 5

    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(10, 6)
    proj = ccrs.PlateCarree(central_longitude=0)
    ax = plt.gca(projection = proj)
    #ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    palette = copy(plt.get_cmap('RdBu_r'))
    palette.set_under('white', 1.0)
    levels = np.linspace(0.0, 0.35, 21)
    f_min_max = lambda x: np.maximum(np.minimum(x, 0.35), 0.1)
    #blah = np.minimum(np.maximum((rh_mid / 2 - 0.05, 0.1), 0.35)
    #plt.contourf(lon, lat, blah, cmap = 'jet', levels = levels, transform=ccrs.PlateCarree());
    #plt.contourf(lon, lat, f_min_max(np.power(rh_mid , 0.33) - 0.6), cmap = 'jet', levels = levels, transform=ccrs.PlateCarree());
    plt.contourf(lon, lat, f_min_max(0.20 / (1 + np.exp(-(rh_mid - 0.55) * 10)) + 0.125), cmap = 'jet', levels = levels, transform=ccrs.PlateCarree());
    #plt.contourf(lon, lat, f_min_max(rh_mid - 0.35), cmap = 'jet', levels = levels, transform=ccrs.PlateCarree());
    #plt.contourf(lon, lat, np.minimum(np.maximum(rh_mid - 0.35, 0.00), 0.4) - blah, cmap = 'RdBu_r', levels = np.linspace(-0.2, 0.2, 21), transform=ccrs.PlateCarree());

    plt.title('Relative Humidity')
    plt.colorbar();

    chi_month = np.maximum(np.minimum(np.exp(np.log(chi) + 1.0), 5), 1e-5) + 1.0
    chi_month[np.isnan(chi_month)] = 5
    # plt.figure(figsize=(10, 6)); plt.contourf(lon, lat, chi_month, cmap = 'jet', extend = 'both', levels = np.linspace(0, 3, 21));
    # plt.xlim([lon_min, lon_max]); plt.ylim([lat_min, lat_max])
    # plt.colorbar();

    chi_month2 = np.maximum(np.minimum(np.exp(np.log(chi) + 0.5), 5), 1e-5) + 1.3
    chi_month2[np.isnan(chi_month2)] = 5
    # plt.figure(figsize=(10, 6)); plt.contourf(lon, lat, chi_month2, cmap = 'jet', extend = 'both', levels = np.linspace(0, 3, 21));
    # plt.xlim([lon_min, lon_max]); plt.ylim([lat_min, lat_max])
    # plt.colorbar();

    plt.figure(figsize=(10, 6)); plt.pcolormesh(lon, lat, chi_month2 - chi_month, cmap = 'RdBu_r', vmin = -0.25, vmax = 0.25);
    plt.xlim([lon_min, lon_max]); plt.ylim([lat_min, lat_max])
    plt.colorbar();
    # %%

def plot_tracks_spi():
    ds.close()
    ds_seeds.close()

    import glob
    from copy import copy

    # %%
    sim_name = 'track_coupled_spi_GL'
    fn_tracks = sorted(glob.glob('/data0/jlin/era5/%s/tracks_GL_era5_197901_202112*.nc' % sim_name));
    #sim_name = 'track_spi'
    #fn_tracks = sorted(glob.glob('/data0/jlin/gfdl/annual_cycle/HIRAM/CTL1990s_v201910_tigercpu_intelmpi_18_540PE/%s/*GL*.nc' % sim_name));

    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                           data_vars="minimal", drop_variables = "seeds_per_months")

    drop_vars = ["lon_trks", "lat_trks", "u250_trks", "v250_trks", "u850_trks", "v850_trks",
                 "v_trks", "m_trks", "vmax_trks", "tc_month", "tc_years"]
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                                 data_vars="minimal", drop_variables = drop_vars)

    yearS = 1979
    yearE = 2021
    n_sim = len(fn_tracks)
    spi_seeding = True
    print('Number of Simulations: %d' % n_sim)
    print(np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0))
    plt.plot(range(1, 13), np.nanmean(ds_seeds['seeds_per_months'].load(), axis = 0))

    ds['vmax_trks'].shape[0] / np.nansum(ds_seeds['seeds_per_months'])

    ds_spi = xr.open_dataset('%s/SPI_monthly_era5_197901_202112.nc' % namelist.base_directory).load()
    #ds_spi = xr.open_dataset('%s/SPI_monthly_HIRAM_11101_13012.nc' % namelist.base_directory).load()
    spi = ds_spi['spi']
    lon_spi = ds_spi['lon'].data
    lat_spi = ds_spi['lat'].data
    spi_clim = spi.data
    spi_clim[spi_clim < 0] = 0
    dt_spi = np.array([datetime.datetime.utcfromtimestamp(int(x) / 1e9) for x in spi['time']])
    #dt_spi = np.array([datetime.datetime(x.year, x.month, x.day, x.hour) for x in ds_spi['time'].data])
    dt_yr = np.array([x.year for x in dt_spi])
    yr_spi = np.zeros((yearE - yearS + 1,))
    spi_monthly = np.sum(spi_clim, axis = (1, 2))
    for yr in range(yearS, yearE + 1):
        yr_spi[yr - yearS] = np.sum(spi_monthly[dt_yr == yr])
    yr_spi /= np.sum(yr_spi) / (yearE - yearS + 1)

    # Filter for surviving tracks.
    vmax = ds['vmax_trks'].load().data
    lon_filt = np.full(ds['lon_trks'].shape, np.nan) #ds['lon_trks'][mask, :].data
    lat_filt = np.full(ds['lat_trks'].shape, np.nan) #ds['lat_trks'][mask, :].data
    vmax_filt = np.full(ds['vmax_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    m_filt = np.full(ds['m_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    lon_trks = ds['lon_trks'].load().data
    lat_trks = ds['lat_trks'].load().data
    m_trks = ds['m_trks'].load().data

    # Here, we only consider a TC from the first point where it exceeds
    # the threshold, to the point it decays to 10 m/s (after it has
    # reached its peak intensity).
    lon_genesis = np.full(lon_filt.shape[0], np.nan)
    lat_genesis = np.full(lon_filt.shape[0], np.nan)
    for i in range(lon_filt.shape[0]):
        if len(np.argwhere(vmax[i, :] >= 15).flatten()) > 0:
            # Genesis occurs when the TC first achieves 30 knots (15 m/s).
            gen_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idx_gen = np.argwhere(vmax[i, :] >= 15).flatten()[0]
            lon_genesis[i] = lon_trks[i, idx_gen]
            lat_genesis[i] = lat_trks[i, idx_gen]

            # TC decays after it has reached 15 m/s
            decay_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idxs_lmi = np.argwhere(decay_idxs >= np.nanargmax(vmax[i, :])).flatten()
            if len(decay_idxs) > 0 and len(idxs_lmi) > 0:
                idx_decay = decay_idxs[idxs_lmi[0]]
            else:
                idx_decay = vmax.shape[1]

            nt = idx_decay - idx_gen
            vmax_filt[i, 0:nt] = vmax[i, idx_gen:idx_decay]
            lon_filt[i, 0:nt] = lon_trks[i, idx_gen:idx_decay]
            lat_filt[i, 0:nt] = lat_trks[i, idx_gen:idx_decay]
            m_filt[i, 0:nt] = m_trks[i, idx_gen:idx_decay]

    # %% Plot seasonal and interseasonal variability
    n_tc_emanuel = np.array([2.3995, 3.7472, 4.7565, 3.3178,
                             3.8894, 3.1360, 4.4686, 2.2770,
                             3.3814, 3.8279, 2.8226, 6.3410,
                             2.8146, 3.6206, 4.1867, 2.7198,
                             5.0066, 2.8506, 3.8354, 8.1853,
                             7.1715, 6.5825, 4.8782, 5.6885,
                             4.9672, 6.9847, 13.1994, 7.7164,
                             6.6171, 7.5774, 3.8121, 8.5860,
                             8.5695, 5.3999, 6.5196, 5.3201,
                             4.9189, 5.8032, 6.1446, 3.1375,
                             6.2809, 7.4246])

    lat_min = 2; lat_max = 60;
    if 'EP' in fn_tracks[0]:
        lon_min = 180; lon_max = 280;
    elif 'WP' in fn_tracks[0]:
        lon_min = 100; lon_max = 180;
    elif 'SH' in fn_tracks[0]:
        lon_min = 40; lon_max = 220;
        lat_min = -60; lat_max = 0;
    elif 'AL' in fn_tracks[0]:
        lon_min = 260.0; lon_max = 360;
    else:
        lon_min = 0; lon_max = 359.99;
        lat_min = -60;

    fn_ib = '/data0/jlin/ibtracs/IBTrACS.ALL.v04r00.nc'
    ds_ib = xr.open_dataset(fn_ib)
    dt_ib = np.array([datetime.datetime.utcfromtimestamp(int(x)/1e9) for x in np.array(ds_ib['time'][:, 0])])
    ib_lon = ds_ib['lon'].data
    ib_lon[ib_lon < 0] += 360
    ib_lat = ds_ib['lat'].load()
    usa_wind = ds_ib['usa_wind'].load()
    n_tc_per_year = np.zeros(yearE - yearS + 1)
    n_mtc_per_year = np.zeros(yearE - yearS + 1)
    pdi_per_year = np.zeros(yearE - yearS + 1)
    mpd_per_year = np.zeros(yearE - yearS + 1)
    gen_lon_hist = np.full(dt_ib.shape, np.nan)
    gen_lat_hist = np.full(dt_ib.shape, np.nan)
    tc_lmi = np.full(dt_ib.shape, np.nan)
    tc_lat_lmi = np.full(dt_ib.shape, np.nan)
    for i in range(yearS, yearE + 1, 1):
        mask = ((dt_ib >= datetime.datetime(i, 1, 1)) &
                (dt_ib <= datetime.datetime(i, 12, 31)) &
                (~np.all(np.isnan(usa_wind.data), axis = 1)) &
                (ib_lon[:, 0] >= lon_min) &
                (ib_lon[:, 0] <= lon_max))

        n_tc_per_year[i - yearS] = np.sum(mask)
        vmax_tc_yr = np.nanmax(usa_wind[mask, :], axis = 1)
        n_mtc_per_year[i - yearS] = np.sum(vmax_tc_yr >= 35)
        vmax_time = usa_wind[mask, :]['time'].load().data
        int_obs = usa_wind[mask, :].data / 1.94384

        for j in range(int(np.sum(mask))):
            if not bool(np.all(np.isnan(usa_wind[mask][j]))):
                lmi_idx = np.nanargmax(usa_wind[mask][j], axis = 0)
                tc_lat_lmi[np.argwhere(mask).flatten()[j]] = float(ib_lat[mask][j,lmi_idx])
                tc_lmi[np.argwhere(mask).flatten()[j]] = float(usa_wind[mask][j, lmi_idx])
                mpd = float(usa_wind[mask][j, lmi_idx] / 1.94384) ** 3
                mpd_per_year[i - yearS] += mpd

                # Calculate PDI
                t_trk = ((vmax_time[j, :] - vmax_time[j, 0]) / 1e9).astype(float)     # seconds
                t_trk[t_trk < 0] = np.nan
                pdi_per_year[i-yearS] += np.trapz(np.power(int_obs[j, ~np.isnan(int_obs[j, :])], 3), t_trk[~np.isnan(int_obs[j, :])])

                gen_idx = np.nanargmin(np.abs(usa_wind[mask][j, :] - 35), axis = 0)
                gen_lon_hist[np.argwhere(mask).flatten()[j]] = ib_lon[mask][j, gen_idx]
                gen_lat_hist[np.argwhere(mask).flatten()[j]] = ib_lat[mask][j, gen_idx]
    gen_lon_hist[gen_lon_hist < 0] += 360

    # %% Compute seeding variability for downscaling.
    n_tc_per_year = n_mtc_per_year
    tc_years = ds['tc_years'].load().data
    n_tc_per_year_downscaling = np.zeros((len(np.unique(tc_years))))
    for yr in np.unique(ds_seeds['year']):
        n_tc_per_year_downscaling[yr - yearS] = np.sum(tc_years == yr)
    n_tc_per_year_downscaling *= np.nanmean(n_tc_per_year) / np.nanmean(n_tc_per_year_downscaling)

    dt_year = range(yearS, yearE + 1)
    plt.figure(figsize=(12, 6));
    h1, = plt.plot(dt_year, n_tc_per_year, 'k', linewidth=4);
    h2, = plt.plot(dt_year, n_tc_per_year_downscaling, color='r', linewidth=3)
    #h3, = plt.plot(dt_year, n_tc_per_year_downscaling_AL, color='b', linewidth=3)

    #plt.boxplot(n_tc_per_year_ens_downscaling.T, sym = 'r.', positions=dt_year.data, whiskerprops={'linewidth': 2, 'color': 'b'},
    #            boxprops = {'linewidth': 2}, medianprops = {'linewidth': 2, 'color': 'b'}, flierprops = {'linewidth': 2, 'color': 'r'}, showfliers = True)

    #h2 = plt.errorbar(dt_year, n_tc_per_year_downscaling,
    #             yerr = np.stack((y_err_n, y_err_p)),
    #             color='r', linewidth=3, elinewidth=1, capsize = 5);
    emanuel_ratio = np.nanmean(n_tc_per_year) / np.nanmean(n_tc_emanuel)
    n_tc_per_year_emanuel = n_tc_emanuel * emanuel_ratio
    mask_yrs = (ds['year'] >= yearS) & (ds['year'] <= yearE)
    #plt.title('r = %f, %f (1979-2017)' % (float(np.corrcoef(n_tc_per_year, n_tc_per_year_downscaling)[0, 1]), np.corrcoef(n_tc_per_year[0:-4], n_tc_per_year_downscaling[0:-4])[0, 1]))
    plt.title('r = %f' % float(np.corrcoef(n_tc_per_year, n_tc_per_year_downscaling)[0, 1]))
    plt.grid(); plt.xlim([yearS-0.5, yearE+0.5]); plt.ylim([0, np.nanmax(n_tc_per_year)*1.15])
    plt.xticks(range(yearS, yearE+1, 5), range(yearS, yearE+1, 5));
    #h3, = plt.plot(list(range(1979, 2021)), n_tc_per_year_emanuel, 'b', linewidth = 4);
    plt.legend([h1, h2], ['Historical', 'SPI']); plt.ylabel('Number of TCs')
    #print(np.square(np.corrcoef(n_tc_per_year[0:], n_tc_per_year_downscaling[0:])[0, 1]))
    #print(np.corrcoef(n_tc_per_year[0:-4], n_tc_per_year_downscaling[0:-4])[0, 1])
    plt.savefig('%s/%s/interannual_tc_count.png' % (namelist.base_directory, sim_name))

    from scipy.stats import linregress
    lr_model = linregress(range(yearS, yearE+1), n_tc_per_year_downscaling)
    lr_ntc = lr_model.slope * range(yearS, yearE+1) + lr_model.intercept
    n_tc_per_year_downscaling_detrend = n_tc_per_year_downscaling - (lr_ntc - np.nanmean(lr_ntc))
    print(np.corrcoef(n_tc_per_year[0:], n_tc_per_year_downscaling[0:])[0, 1])

    from scipy.stats import levene
    #levene(n_tc_per_year_downscaling - n_tc_per_year, n_tc_per_year_downscaling_AL - n_tc_per_year, center = 'mean')


    # %%
    tc_months = ds['tc_month'].load().data
    n_ss = 3713
    tc_months_subsample = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    for i in range(n_ss):
        tc_months_subsample[:, i] = np.random.choice(tc_months, int(np.sum(n_tc_per_year)))
    m_bins = np.arange(0.5, 12.6, 1)
    h_season = np.apply_along_axis(lambda x: np.histogram(x, bins = m_bins, density = 'pdf')[0], 0, tc_months_subsample)
    mn_season_pdf = np.nanmean(h_season, axis = 1)*np.nanmean(n_tc_per_year)
    mn_season_errP = np.nanquantile(np.nanmean(n_tc_per_year)*h_season, 0.975, axis = 1) - mn_season_pdf
    mn_season_errN = mn_season_pdf - np.nanquantile(np.nanmean(n_tc_per_year)*h_season, 0.025, axis = 1)

    mask = np.nanmax(ds_ib['usa_wind'][dt_ib >= datetime.datetime(1979, 1, 1), :], axis = 1) >= 35
    seasonal_hist_counts = np.array([x.month for x in dt_ib if x.year >= 1979])[mask]
    hist_season = np.histogram(seasonal_hist_counts, bins = np.arange(0.5, 12.6, 1), density = 'pdf')
    plt.figure(figsize=(12, 6));
    plt.bar(range(1, 13, 1), mn_season_pdf,
            yerr = np.stack((mn_season_errN, mn_season_errP)), error_kw = {'elinewidth': 2, 'capsize': 6});
    plt.ylabel('Storms per Month'); plt.xlabel("Month");
    plt.bar(range(1, 13, 1), hist_season[0]*np.nanmean(n_tc_per_year), fc=(1, 0, 0, 0.5));
    plt.ylabel('Density'); plt.xlabel("Month"); plt.legend(['Downscaling', 'Observed'])
    plt.xlim([0.5, 12.5]); plt.xticks(range(1, 13, 1)); plt.grid();
    plt.savefig('%s/%s/seasonal_cycle.png' % (namelist.base_directory, sim_name))

    # %% Plot distribution of seeds
    ds_b = xr.open_dataset('land/GL.nc')
    basin_mask = ds_b['basin']
    f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
    #ds_spi = xr.open_dataset('%s/SPI_monthly_era5_197901_202112.nc' % namelist.base_directory).load()
    basin_gen = mat.interp_2d_grid(basin_mask['lon'], basin_mask['lat'], basin_mask, ds_spi['lon'].data, ds_spi['lat'].data)
    LON, LAT = np.meshgrid(ds_spi['lon'].data, ds_spi['lat'].data)
    spi_clim = ds_spi['spi'].data
    spi_clim[np.tile(basin_gen, (spi_clim.shape[0], 1, 1)) < 0.5] = 0
    spi_clim[np.tile(np.abs(LAT), (spi_clim.shape[0], 1, 1)) > 30] = 0
    spi_clim[spi_clim < 0] = 0

    dt_yr = np.array([x.year for x in dt_spi])
    yr_spi = np.zeros((yearE - yearS + 1,))
    spi_monthly = np.sum(spi_clim, axis = (1, 2))
    for yr in range(yearS, yearE + 1):
        yr_spi[yr - yearS] = np.sum(spi_monthly[dt_yr == yr])
    yr_spi /= np.sum(yr_spi) / (yearE - yearS + 1)
    spi_clim_monthly = np.zeros(12)
    for m in range(12):
        spi_clim_monthly[m] = np.sum(spi_clim[m::12])
    spi_clim_monthly /= np.sum(spi_clim_monthly)

    plt.figure(); plt.pcolormesh(ds_spi['lon'].data, ds_spi['lat'].data, np.nanmean(spi_clim[7::12, :, :], axis = 0), cmap = 'gist_heat_r'); plt.xlim([260, 350]); plt.ylim([0, 35])
    plt.figure(); plt.pcolormesh(ds_spi['lon'].data, ds_spi['lat'].data, np.nanmean(spi_clim[8::12, :, :], axis = 0), cmap = 'gist_heat_r'); plt.xlim([260, 350]); plt.ylim([0, 35])

    prob_lowlat = np.power(np.minimum(np.maximum((np.abs(LAT) - 2) / 12.0, 0), 1), 6)
    spi_clim *= np.tile(prob_lowlat, (spi_clim.shape[0], 1, 1))
    spi_clim_monthly_vortThresh = np.zeros(12)
    for m in range(12):
        spi_clim_monthly_vortThresh[m] = np.sum(spi_clim[m::12])
    spi_clim_monthly_vortThresh /= np.sum(spi_clim_monthly_vortThresh)

    plt.figure(figsize=(8,6)); plt.plot(range(1, 13), spi_clim_monthly); plt.plot(range(1, 13), spi_clim_monthly_vortThresh);
    plt.xlabel('Month'); plt.ylabel('Seed Probability (EP)'); plt.legend(['0-25N', 'Vorticity Threshold']); plt.grid();

    #plt.plot(range(1, 13), np.nanmean(ds_seeds['seeds_per_months'], axis = 0) / np.sum(np.nanmean(ds_seeds['seeds_per_months'], axis = 0)))

    plt.pcolormesh(ds_spi['lon'].data, ds_spi['lat'].data, np.nanmean(spi_clim[0::12, :, :], axis = 0), cmap = 'gist_heat_r'); plt.xlim([260, 350]); plt.ylim([0, 35])
    plt.pcolormesh(ds_spi['lon'].data, ds_spi['lat'].data, np.nanmean(spi_clim[7::12, :, :], axis = 0), cmap = 'gist_heat_r'); plt.xlim([260, 350]); plt.ylim([0, 35])
    plt.pcolormesh(ds_spi['lon'].data, ds_spi['lat'].data, np.nanmean(spi_clim[8::12, :, :], axis = 0), cmap = 'gist_heat_r'); plt.xlim([260, 350]); plt.ylim([0, 35])

    plt.figure(figsize=(10,6)); plt.plot(range(yearS, yearE+1), yr_spi); plt.grid()
    plt.xlabel('Year'); plt.ylabel('Seed Ratio')
    #plt.savefig('%s/%s/seed_ratio_year.png' % (namelist.base_directory, sim_name))

    seed_per_month = np.nanmean(ds_seeds['seeds_per_months'], axis = 0)
    plt.figure(figsize=(10,6));
    plt.plot(range(1, 13), spi_clim_monthly); plt.grid()
    plt.plot(range(1, 13), spi_clim_monthly_vortThresh); plt.grid()
    plt.plot(range(1, 13), seed_per_month / np.sum(seed_per_month))
    plt.xlabel('Month'); plt.ylabel('Seed Probability'); plt.ylim([0, 0.17])
    plt.legend(['0-25N', 'Vorticity Threshold', 'Seeding'])
    #plt.savefig('%s/%s/seed_seasonal_cycle.png' % (namelist.base_directory, sim_name))
    h_season = np.histogram(tc_months, bins = m_bins, density = 'pdf')[0] * len(tc_months)
    trans_prob = h_season / np.nansum(ds_seeds['seeds_per_months'], axis = 0)
    plt.figure(figsize=(10,6)); plt.plot(range(1, 13), trans_prob); plt.grid();
    plt.xlabel('Month'); plt.ylabel('Seed Transition Probability');

    tc_month_pdf = trans_prob / np.nansum(trans_prob)

#    plt.savefig('%s/%s/seed_transition_probability.png' % (namelist.base_directory, sim_name))

    # normalize??
    spi_clim_monthly_weights = spi_clim_monthly_vortThresh / spi_clim_monthly
    spi_clim_monthly_weights /= np.nansum(spi_clim_monthly_weights)

    #plt.plot(spi_clim_monthly); plt.plot(spi_clim_monthly_vortThresh)
    spi_monthly_adj = np.zeros(spi_clim.shape)
    for i in range(spi_monthly_adj.shape[0]):
        spi_monthly_adj[i, :, :] = spi_clim[i, :, :] / spi_clim_monthly_weights[i % 12]

    spi_clim_monthly_adj = np.zeros(12)
    for m in range(12):
        spi_clim_monthly_adj[m] = np.sum(spi_monthly_adj[m::12])
    spi_clim_monthly_adj /= np.sum(spi_clim_monthly_adj)

    plt.plot(spi_clim_monthly_vortThresh)
    plt.plot(spi_clim_monthly_adj)

    # %%
    spi_monthly_clim = np.zeros(12)
    for m in range(12):
        spi_monthly_clim[m] = np.nanmean(np.sum(spi_clim[m::12], axis = (1, 2)))
    spi_monthly_anom = np.zeros(spi_monthly.shape)
    for m in range(12):
        spi_monthly_anom[m::12] = spi_monthly[m::12] - spi_monthly_clim[m]
    spi_yearly_anom = np.zeros(43)
    for yr in range(yearS, yearE + 1):
        spi_yearly_anom[yr - yearS] = np.sum(spi_monthly_anom[dt_yr == yr])

    plt.figure(figsize=(10, 6))
    plt.plot(range(yearS, yearE+1), spi_yearly_anom, 'k')
    ax = plt.gca(); ax2 = ax.twinx()
    ax2.plot(range(yearS, yearE+1), n_tc_per_year, 'r')
    ax.set_xlabel('Year'); ax.set_ylabel('Basin SPI Anomaly'); ax2.set_ylabel('TC Count')
    ax2.spines['right'].set_color('red')
    ax2.yaxis.label.set_color('red')
    ax2.tick_params(axis='y', colors='red')

    # %% Plot distribution of genesis
    dlon_label = 30
    lon_cen = 0

    x_binEdges = np.arange(lon_min, lon_max + 0.1, 2)
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 2)
    gen_pdf_hist = np.histogram2d(gen_lon_hist[~np.isnan(gen_lat_hist)], gen_lat_hist[~np.isnan(gen_lat_hist)], bins = [x_binEdges, y_binEdges])[0]
    gen_pdf = np.histogram2d(lon_genesis, lat_genesis, bins = [x_binEdges, y_binEdges])[0]
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2

    plt.rcParams.update({'font.size': 14})

    if lon_max > 300:
        lon_cen = 0
    else:
        lon_cen = 180
    #lon_cen = 180
    lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
    lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label
    xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
    xlocs_shift = np.copy(xlocs)
    xlocs_shift[xlocs > lon_cen] -= 360
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(12, 9)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    levels = np.arange(3, 30, 3)

    palette = copy(plt.get_cmap('jet'))
    palette.set_under('white', 1.0)
    #gen_pdf[gen_pdf == 0] = 1e-6                # so we can take the log
    #cmin = np.quantile(np.log(gen_pdf[gen_pdf > 1]), 0.1)
    #cmax = np.quantile(np.log(gen_pdf), 1)
    gen_pdf = gen_pdf / (n_sim * len(ds['year']))   # normalize by number of simulations
    gen_pdf_hist /= len(ds['year'])   # normalize by number of years
    gen_pdf = gen_pdf / (np.sum(gen_pdf) / np.sum(gen_pdf_hist))

    cmax = 0.401 #1.001 #np.ceil(np.quantile(gen_pdf, 1)*25) / 25
    cmin = 0.025
    levels = np.arange(cmin, cmax, cmin)
    ax = plt.contourf(x_binCenter, y_binCenter, gen_pdf.T, levels = levels, extend = 'max', cmap=palette, transform = ccrs.PlateCarree());
    plt.colorbar(orientation = 'horizontal'); plt.title('Number of Genesis Events per Year (Downscaling)');
    plt.savefig('%s/%s/genesis_pdf.png' % (namelist.base_directory, sim_name))

    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(12, 9)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    levels = np.arange(3, 30, 3)

    palette = copy(plt.get_cmap('jet'))
    palette.set_under('white', 1.0)

    levels = np.arange(cmin, cmax, cmin)
    ax = plt.contourf(x_binCenter, y_binCenter, gen_pdf_hist.T, levels = levels, extend = 'max', cmap=palette, transform = ccrs.PlateCarree());
    plt.colorbar(orientation = 'horizontal'); plt.title('Number of Genesis Events per Year (Historical)');
    plt.savefig('%s/%s/genesis_obs_pdf.png' % (namelist.base_directory, sim_name))
    #if lon_cen == 0:
        #plt.xlim([lon_min-360, lon_max-360])

    # %% Genesis latitudes
    plt.figure(figsize=(9,5)); plt.plot(y_binCenter, np.sum(gen_pdf, axis = 0) * len(ds['year']), 'kx-');
    plt.hist(gen_lat_hist, bins = y_binEdges)
    plt.legend(['Downscaling', 'Observations'])
    plt.xlabel('Latitude'); plt.ylabel('Number of Events')
    plt.savefig('%s/%s/genesis_latitudes.png' % (namelist.base_directory, sim_name))

    # plt.figure(figsize=(9,5));
    # spi_pdf = np.sum(spi_clim, axis = (0, 2)) / np.sum(spi_clim);
    # y_min = np.sin(np.pi / 180 * 3)
    # y_max = np.sin(np.pi / 180 * 45)
    # gen_lat = np.arcsin(np.random.uniform(y_min, y_max, 1000000)) * 180 / np.pi
    # plt.plot(lat_spi, spi_pdf);
    # plt.hist(gen_lat, bins = np.arange(0, 45.1, 1), density = True);
    # plt.xlim([0, 40]); plt.xlabel('Latitude'); plt.ylabel('Genesis Probability'); plt.grid()
    # plt.legend(['SPI ERA5', 'Random Seeding'])

    gen_lat = np.linspace(0, 20, 100)
    prob_lowlat = np.power(np.minimum(np.maximum((np.abs(gen_lat) - 2) / 12.0, 0), 1), 10)
    prob_lowlat2 = np.power(np.minimum(np.maximum((np.abs(gen_lat) - 2) / 12.0, 0), 1), 3)
    #plt.plot(gen_lat, prob_lowlat, gen_lat, prob_lowlat2)

    # %%
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(20, 4)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, -30, 30], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_bottom = True
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    palette = copy(plt.get_cmap('RdBu_r'))
    palette.set_under('white', 1.0)
    spi_clim_annual = np.nanmean(spi_clim, axis = 0)
    spi_clim_annual[spi_clim_annual < 1e-3] = -1e-3
    ax = plt.contourf(lon_spi, lat_spi, spi_clim_annual, levels = np.linspace(0, 0.05, 21), cmap = 'gist_heat_r', transform = ccrs.PlateCarree());
    plt.colorbar(); plt.xlabel('Longitude'); plt.ylabel('Latitude')
    plt.title('SPI Annual Average')

    # %% TC Density PDF.
    den_pdf = np.histogram2d(lon_filt.flatten(), lat_filt.flatten(), bins = [x_binEdges, y_binEdges])
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2

    # Plot distribution of density
    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(8, 8)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    cmax = np.quantile(den_pdf[0][den_pdf[0] > 0], 1)
    cmin = cmax / 10
    levels = np.linspace(cmin, cmax, 11)
    plt.contourf(x_binCenter, y_binCenter, den_pdf[0].T, levels, cmap=palette, transform = ccrs.PlateCarree());
    plt.colorbar(orientation = 'horizontal')
    plt.savefig('%s/%s/track_density.png' % (namelist.base_directory, sim_name))

    # %% Plot a random set of tracks
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(18, 6)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = plt.gca(projection = proj)
    #ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min , lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    r_idxs = np.random.randint(0, lon_filt.shape[0], size = 20)
    #r_idxs = np.random.choice(np.argwhere((lat_filt[:, 0] >= 20) & (lat_filt[:, 0] <= 30)).flatten(), size = 10)
    #r_idxs = np.random.choice(np.argwhere(lat_lmi <= 10).flatten(), size = 10)

    plt.plot(lon_filt[r_idxs, :].T, lat_filt[r_idxs, :].T, transform=ccrs.PlateCarree());
    plt.scatter(lon_filt[r_idxs, 0].data, lat_filt[r_idxs, 0].data, color='k', transform=ccrs.PlateCarree());


    #r_idx = 7553; plt.plot(lon_filt[r_idx, :].T, lat_filt[r_idx, :].T, transform=ccrs.PlateCarree()); plt.xlim([-100, 0]); plt.ylim([3, 50])
    plt.figure(); plt.plot(ds['time'].data / (24 * 60 * 60), vmax_filt[r_idxs, :].T * 1.94384);
    plt.figure(); plt.plot(ds['time'].data / (24 * 60 * 60), m_filt[r_idxs, :].T);

    blah = ds['m_trks'][~np.isnan(vmax_filt[:, 0]), :]

    # %% Plot lifetime maximum intensity distribution
    vmax_lmi = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    for i in range(n_ss):
        ss_idxs = np.random.choice(vmax_filt.shape[0], int(np.sum(n_tc_per_year)))
        vmax_lmi[:, i] = np.nanmax(vmax_filt[ss_idxs, :], axis = 1) * 1.94384
    lmi_bins = np.arange(35, 196, 10)
    lmi_cbins = np.arange(40, 191, 10)

    h_lmi = np.apply_along_axis(lambda a: np.histogram(a, bins = lmi_bins, density = 'pdf')[0], 1, vmax_lmi)
    mn_lmi_errP = np.nanquantile(h_lmi, 0.95, axis = 0) - np.nanmean(h_lmi, axis = 0)
    mn_lmi_errN = np.nanmean(h_lmi, axis = 0) - np.nanquantile(h_lmi, 0.05, axis = 0)
    h_lmi_obs = np.histogram(tc_lmi, density=True, bins = np.arange(35, 196, 10))[0];

    plt.figure(figsize=(12,7));
    plt.bar(lmi_cbins, np.nanmean(h_lmi, axis = 0), width = 8,
            yerr = np.stack((mn_lmi_errN, mn_lmi_errP)), error_kw = {'elinewidth': 2, 'capsize': 6});
    plt.bar(lmi_cbins, h_lmi_obs, fc = (1, 0, 0, 0.5), width = 8)
    #plt.hist(tc_lmi, density=True, bins = np.arange(35, 186, 10), fc=(1, 0, 0, 0.5));
    plt.xlabel('Lifetime Maximum Intensity (kts)'); plt.ylabel('PDF'); plt.grid();
    plt.legend(['Model', 'Observed'])
    plt.savefig('%s/%s/lmi_pdf.png' % (namelist.base_directory, sim_name))

    # %% Plot 6-hour zonal and meridional displacements in region defined by basin_mask
    basin_dr = [265, 340, 10, 40]   # AL
    #basin_dr = [130, 160, 10, 30]   # WP
    #basin_dr = [200, 260, 10, 30]   # EP
    ds_ib_lon = ds_ib['lon'].data
    ds_ib_lon[ds_ib_lon < 0] = ds_ib_lon[ds_ib_lon < 0] + 360
    ib_basin_mask = ((ds_ib['lat'][:, 1:-1] >= basin_dr[2]) & (ds_ib['lat'][:, 1:-1] <= basin_dr[3]) &
                     (ds_ib_lon[:, 1:-1] >= basin_dr[0]) & (ds_ib_lon[:, 1:-1] <= basin_dr[1]) &
                     (ds_ib['usa_wind'][:, 1:-1].data >= 35))
    basin_mask = ((lat_filt >= basin_dr[2]) & (lat_filt <= basin_dr[3]) &
                  (lon_filt >= basin_dr[0]) & (lon_filt <= basin_dr[1]))
    dlon_6h = (lon_filt[:, ::6][:, 1:] - lon_filt[:, ::6][:, 0:-1])[basin_mask[:, ::6][:, 0:-1]]
    dlat_6h = (lat_filt[:, ::6][:, 1:] - lat_filt[:, ::6][:, 0:-1])[basin_mask[:, ::6][:, 0:-1]]
    ib_dlon_6h = (ds_ib_lon[:, 2:] - ds_ib_lon[:, 0:-2])[ib_basin_mask]
    ib_dlat_6h = (ds_ib['lat'][:, 2:] - ds_ib['lat'][:, 0:-2]).data[ib_basin_mask]

    plt.figure(figsize=(10,8)); plt.hist(dlon_6h, density=True, bins = np.arange(-3.5, 3.51, 0.25));
    plt.hist(ib_dlon_6h, density=True, bins = np.arange(-3.5, 3.51, 0.25), fc = (1, 0, 0, 0.5)); plt.xlabel('6-Hour Zonal Displacement (deg)'); plt.ylabel('Density'); plt.grid();
    plt.figure(figsize=(10,8)); plt.hist(dlat_6h, density=True, bins = np.arange(-1.5, 2.51, 0.25));
    plt.hist(ib_dlat_6h, density=True, bins = np.arange(-1.5, 2.51, 0.25), fc = (1, 0, 0, 0.5)); plt.xlabel('6-Hour Meridional Displacement (deg)'); plt.ylabel('Density'); plt.grid();

    # %% Plot 24h intensity change distribution
    mask = ((dt_ib >= datetime.datetime(yearS, 1, 1)) &
            (dt_ib <= datetime.datetime(yearE, 12, 31)))
    int_time = ds_ib['usa_wind'][mask, :]['time'].load().data
    int_obs = ds_ib['usa_wind'][mask, :].load().data
    int_24h_obs = np.zeros(int_obs.shape)
    for idx in range(int_24h_obs.shape[0]):
        t_trk = ((int_time[idx, :] - int_time[idx, 0]) / 1e9).astype(float)     # seconds
        t_trk[t_trk < 0] = np.nan
        int_24h_obs[idx, :] = np.interp(t_trk + 86400, t_trk, int_obs[idx, :]) - np.interp(t_trk, t_trk, int_obs[idx, :])
    int_24h_obs = int_24h_obs[~np.isnan(int_24h_obs)]

    ds_b = xr.open_dataset('land/land.nc')
    land_mask = ds_b['land']
    f_land = mat.interp2_fx(land_mask['lon'], land_mask['lat'], land_mask)
    idx_24h = int((24 * 60 * 60) / (ds['time'][1] - ds['time'][0]))
    tc_int_mask = np.logical_and(f_land.ev(lon_filt[:, idx_24h::idx_24h], lat_filt[:, idx_24h::idx_24h]).flatten() >= 1,
                                 lat_filt[:, idx_24h::idx_24h].flatten() <= 45);
    int_24h = (vmax_filt[:, idx_24h::idx_24h] - vmax_filt[:, 0:-idx_24h:idx_24h]).flatten() * 1.94384
    plt.figure(figsize=(13,7));  plt.hist(int_24h[tc_int_mask], bins = np.arange(-62.5, 62.6, 5), density=True)
    plt.hist(int_24h_obs, bins = np.arange(-62.5, 62.6, 5), density=True, fc=(1, 0, 0, 0.5))
    plt.gca().set_xticks(np.arange(-65., 65.1, 5)); plt.xlabel('24h Intensity Change (kts)'); plt.ylabel('PDF'); plt.grid();
    plt.title('95th Percentile: %f kts / 24h' % np.nanquantile(int_24h[tc_int_mask], 0.95))
    plt.legend(['Downscaling', 'Observed'])
    plt.savefig('%s/%s/24h_intensity_distribution.png' % (namelist.base_directory, sim_name))

    # %% Plot latitude of LMI
    lat_lmi = np.zeros((n_ss, int(np.sum(n_tc_per_year))))
    for i in range(n_ss):
        ss_idxs = np.random.choice(vmax_filt.shape[0], int(np.sum(n_tc_per_year)))
        lat_filt_ens = lat_filt[ss_idxs, :]
        vmax_filt_ens = vmax_filt[ss_idxs, :]
        vmax_filt_ens_lmi_idx = np.nanargmax(vmax_filt_ens, axis = 1)

        for j in range(lat_filt_ens.shape[0]):
            lat_lmi[i, j] = lat_filt_ens[j, vmax_filt_ens_lmi_idx[j]]

    lat_lmi_width = 5
    lat_lmi_bins = np.arange(-50.1, 50.1, lat_lmi_width)
    lat_lmi_cbins = np.arange(-47.5, 47.6, lat_lmi_width)
    h_lat_lmi = np.apply_along_axis(lambda a: np.histogram(a, bins = lat_lmi_bins, density = 'pdf')[0], 1, lat_lmi)
    hist_lat_lmi = np.histogram(tc_lat_lmi, bins = lat_lmi_bins, density = 'pdf')[0] * lat_lmi_width
    mn_lat_lmi_pdf = lat_lmi_width*np.nanmean(h_lat_lmi, axis = 0)
    mn_lat_lmi_errP = lat_lmi_width*np.nanquantile(h_lat_lmi, 0.95, axis = 0) - mn_lat_lmi_pdf
    mn_lat_lmi_errN = mn_lat_lmi_pdf - lat_lmi_width*np.nanquantile(h_lat_lmi, 0.05, axis = 0)

    plt.figure(figsize=(10,5));
    plt.bar(lat_lmi_cbins, mn_lat_lmi_pdf, width = 4,
            yerr = np.stack((mn_lat_lmi_errN, mn_lat_lmi_errP)), error_kw = {'elinewidth': 2, 'capsize': 6});
    plt.bar(lat_lmi_cbins, hist_lat_lmi, fc = (1, 0, 0, 0.5), width = 4)
    plt.xlabel('Lifetime Maximum Intensity Latitude (deg)'); plt.ylabel('PDF'); plt.grid();
    plt.legend(['Model', 'Observed'])
    plt.savefig('%s/%s/lat_lmi.png' % (namelist.base_directory, sim_name))
    # %%

    tc_years = ds['tc_years'].data
    lat_lmi = np.full((yearE - yearS + 1), np.nan)
    for i in range(lat_lmi.shape[0]):
        yr_mask = tc_years == (i + yearS)
        lat_filt_ens = lat_filt[yr_mask, :]
        vmax_filt_ens = vmax_filt[yr_mask, :]
        vmax_filt_ens_lmi_idx = np.nanargmax(vmax_filt_ens, axis = 1)
        lmi_ens = np.nanmax(vmax_filt_ens, axis = 1) * 1.94384
        lat_lmi_yr = np.full(lat_filt_ens.shape[0], np.nan)
        for j in range(lat_filt_ens.shape[0]):
            lat_lmi_yr[j] = lat_filt_ens[j, vmax_filt_ens_lmi_idx[j]]
        lat_lmi[i] = np.nanmean(np.abs(lat_lmi_yr))

    from scipy.stats import linregress
    lin_m = linregress(range(yearS, yearE + 1), lat_lmi)
    #lin_m2 = linregress(range(yearS, yearE + 1), np.nanmean(np.abs(lat_lmi_coupled), axis = 1))
    plt.figure(figsize=(10,6))
    plt.plot(range(yearS, yearE + 1), lat_lmi)
    #plt.plot(range(yearS, yearE + 1), np.nanmean(np.abs(lat_lmi_coupled), axis = 1))
    plt.plot(range(yearS, yearE + 1), lin_m.slope * np.array(range(yearS, yearE + 1)) + lin_m.intercept)
    #plt.plot(range(yearS, yearE + 1), lin_m2.slope * np.array(range(yearS, yearE + 1)) + lin_m2.intercept)
    plt.xlabel("Year"); plt.ylabel('Latitude of LMI'); plt.grid();
    # %%

def test_bam_track():
    #from importlib import reload
    #reload(namelist)
    # %% Instantiate the beta-advection model (loading wind fields).
    from track import env_wind, bam_track
    import datetime
    fn_wnd_stat = env_wind.get_env_wnd_fn()
    ct = datetime.datetime(2020, 9, 15, 0)
    b = basins.TC_Basin('AL')
    bam_lib = bam_track.BetaAdvectionTrack(fn_wnd_stat, ct, dt_track = 1800, total_time = 15*24*60*60)
    ts_bam = np.arange(0, bam_lib.total_time + 0.1, bam_lib.dt_track)
    n_steps_output = len(ts_bam)
    n_tracks = 1
    nt = 0

    lon_tracks = np.zeros((n_tracks, n_steps_output))
    lat_tracks = np.zeros((n_tracks, n_steps_output))
    vtrans_tracks = np.zeros((n_tracks, n_steps_output, 2))
    env_wnd_tracks = np.zeros((n_tracks, n_steps_output, 4))

    while nt < n_tracks:
        # Genesis location for the seed.
        gen_lon = [clon]; #np.random.uniform(-55, -20, 1)
        gen_lat = [clat]; #np.random.uniform(5, 10, 1)

        # Obtain environmental parameters along track.
        track, vtran, env_wnd = bam_lib.gen_track(b, ct, gen_lon[0], gen_lat[0])
        lon_tracks[nt, :] = track[:, 0]
        lat_tracks[nt, :] = track[:, 1]
        vtrans_tracks[nt, :, :] = vtran
        env_wnd_tracks[nt, :, :] = env_wnd
        nt += 1

    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(10, 6)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = plt.gca(projection = proj)
    #ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    r_idxs = np.random.randint(0, lon_tracks.shape[0], size = 10)
    plt.plot(lon_tracks[r_idxs, :].T, lat_tracks[r_idxs, :].T, transform=ccrs.PlateCarree()); plt.xlim([-100, 0]); plt.ylim([3, 50])

    # %%
    r_idx = 0
    plt.plot(lon_tracks[r_idx, :].T, lat_tracks[r_idx, :].T)
    plt.plot(vtrans_tracks[r_idx, :, :])
    plt.plot(env_wnd_tracks[r_idx, :, 0])
    plt.plot(env_wnd_tracks[r_idx, :, 3], 'kx-')

    #plt.hist(vtrans_tracks[r_idxs, 0:5, 0].flatten())
    #plt.hist(vtrans_tracks[r_idxs, 0:5, 1].flatten())
    plt.plot(bam_lib.Fs.T);

    # %%
    fn_ib = '/data0/jlin/ibtracs/IBTrACS.EP.v04r00.nc'
    ds_ib = xr.open_dataset(fn_ib)
    mask = ds_ib['lat'][:, 0] <= 10

    dir_cos = np.cos(np.deg2rad(90 - ds_ib['storm_dir'][mask, 0].data))
    dir_sin = np.sin(np.deg2rad(90 - ds_ib['storm_dir'][mask, 0].data))
    plt.hist(dir_cos * ds_ib['storm_speed'][mask, 0] / 1.94)
    plt.hist(dir_sin * ds_ib['storm_speed'][mask, 0] / 1.94)


    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(10, 6)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = plt.gca(projection = proj)
    #ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    plt.plot(ds_ib['lon'][mask, :].data.T, ds_ib['lat'][mask, :].data.T);
    plt.xlim([-100, 0]); plt.ylim([3, 50])

    # %%
    plt.figure(figsize=(10, 6)); plt.pcolormesh(lon_spi, lat_spi, spi[464, :, :], cmap = 'gist_heat_r'); plt.xlim([270, 360]); plt.ylim([0, 40]); plt.colorbar()
    sep_tc_lon = [ds_ib['lon'][x, :].data for x in range(2209, 2215)]
    sep_tc_lat = [ds_ib['lat'][x, :].data for x in range(2209, 2215)]
    for i in range(len(sep_tc_lon)):
        plt.plot(sep_tc_lon[i] + 360, sep_tc_lat[i], 'k')
    plt.xlabel('Longitude'); plt.ylabel('Latitude')

    # %%
    fn_ib = '/data0/jlin/ibtracs/IBTrACS.EP.v04r00.nc'
    ds_ib = xr.open_dataset(fn_ib)

    dt_ib = np.array([datetime.datetime.utcfromtimestamp(int(x)/1e9) for x in np.array(ds_ib['time'][:, 0])])
    yearS = 1979
    yearE = 2021
    n_tc_per_year = np.zeros(yearE - yearS + 1)
    n_mtc_per_year = np.zeros(yearE - yearS + 1)
    for i in range(yearS, yearE + 1, 1):
        mask = (dt_ib >= datetime.datetime(i, 1, 1)) & (dt_ib <= datetime.datetime(i, 12, 31))
        n_tc_per_year[i - yearS] = np.sum(mask)
        vmax_tc_yr = np.nanmax(ds_ib['usa_wind'], axis = 1)[mask]
        n_mtc_per_year[i - yearS] = np.sum(vmax_tc_yr >= 40)

    plt.figure(figsize=(12, 5)); plt.plot(range(yearS, yearE + 1), n_tc_per_year); plt.grid(); plt.xticks(range(1980, 2021, 5)); plt.ylim([0, 42])

    # %%
    u250_Mean = bam_lib.ua250_Mean
    u850_Mean = bam_lib.ua850_Mean
    v250_Mean = bam_lib.va250_Mean
    v850_Mean = bam_lib.va850_Mean
    lon = u250_Mean['lon']
    lat = u250_Mean['lat']
    LON, LAT = np.meshgrid(lon, lat)

    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(8, 8)
    proj = ccrs.PlateCarree(central_longitude=lon_cen)
    ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    #levels = np.arange(-20, 21, 5)
    levels = np.arange(-5, 6, 1)

    ax = plt.contourf(lon, lat, v250_Mean*0.2 + 0.8*v850_Mean + np.cos(np.deg2rad(LAT))*2.5, levels = levels, cmap = 'RdBu_r', transform = ccrs.PlateCarree());
    plt.colorbar(orientation = 'horizontal');
    #plt.xlim([270, 340]); plt.ylim([0, 30]); plt.colorbar()

    fn_wnd_stat = env_wind.get_env_wnd_fn()
    wnd_stats = env_wind.read_env_wnd_fn(fn_wnd_stat)

def calc_rh(dt_start, dt_end):
    ds_q = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_q_daily_*.nc')).sel(time=slice(dt_start, dt_end)).load()
    ds_ta = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_t_daily_*.nc')).sel(time=slice(dt_start, dt_end)).load()

    lon_ky = input.get_lon_key()
    lat_ky = input.get_lat_key()
    rh = np.zeros(ds_q[input.get_sp_hum_key()].shape)
    nTime = rh.shape[0]

    for i in range(nTime):
        ta = ds_ta[input.get_temp_key()][i, :, :]
        hus = ds_q[input.get_sp_hum_key()][i, :, :]
        rh[i, :, :] = thermo.conv_q_to_rh(ta.data, hus.data, 600 * 100)
    return rh

def calc_chi(dt_start, dt_end):
    ds_sst = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_sst_daily_*.nc')).sel(time=slice(dt_start, dt_end)).load()
    ds_psl = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_sp_daily_*.nc')).sel(time=slice(dt_start, dt_end)).load()
    ds_q = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_q_daily_*.nc')).sel(time=slice(dt_start, dt_end)).load()
    ds_ta = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_t_daily_*.nc')).sel(time=slice(dt_start, dt_end)).load()

    lon_ky = input.get_lon_key()
    lat_ky = input.get_lat_key()
    sst_ky = input.get_sst_key()

    chi = np.zeros(ds_q[input.get_sp_hum_key()].shape)
    nTime = chi.shape[0]

    for i in range(nTime):
        sst_i = np.nan_to_num(ds_sst[sst_ky].interp(time = ds_q['time'][i]).data)
        # Convert all variables to the atmospheric grid.
        sst_interp = mat.interp_2d_grid(ds_sst[lon_ky], ds_sst[lat_ky],
                                        sst_i, ds_ta[lon_ky], ds_ta[lat_ky])
        if 'C' in ds_sst[sst_ky].units:
            sst_interp = sst_interp + 273.15

        psl = ds_psl[input.get_mslp_key()].interp(time = ds_q['time'][i])
        ta = ds_ta[input.get_temp_key()][i, :, :]
        hus = ds_q[input.get_sp_hum_key()][i, :, :]

        chi_args = (sst_interp, psl.data, ta.data,
                    namelist.p_midlevel, hus.data)
        chi[i, :, :] = np.minimum(np.maximum(thermo.sat_deficit(*chi_args), 0), 10)
    chi[np.isnan(chi)] = 10
    chi_mean = np.nanmean(np.log(chi+0.01), axis = 0)
    chi_std = np.nanstd(np.log(chi+0.01), axis = 0)

    return (chi_mean, chi_std)

def calc_daily_rh():
    from thermo import thermo
    from dateutil import relativedelta
    import dask
    # %%
    dt_start, dt_end = input.get_bounding_times()
    ds_q = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_q_daily_*.nc')) #.sel(time=slice(dt_start, dt_end))

    lazy_results = []; f_args = [];
    for yr in range(namelist.start_year, namelist.end_year + 1):
        for mnth in range(1, 13):
            dt = datetime.datetime(yr, mnth, 1)
            dt_start = input.convert_from_datetime(ds_q, [dt])[0]
            dt = datetime.datetime(yr, mnth, 1) + relativedelta.relativedelta(months=+1) - relativedelta.relativedelta(hours = 1)
            dt_end = input.convert_from_datetime(ds_q, [dt])[0]
            lazy_result = dask.delayed(calc_rh)(dt_start, dt_end)
            lazy_results.append(lazy_result)
            f_args.append((dt_start, dt_end))
    out = dask.compute(*lazy_results, scheduler = 'processes', num_workers = 24)

    dt_q = np.array([datetime.datetime.utcfromtimestamp(int(x) / 1e9) for x in ds_q['time']])
    rh_daily = np.concatenate(out, axis = 0)
    da_rh = xr.DataArray(data = rh_daily, dims = ['time', 'lat', 'lon'],
                          coords = dict(lon = ("lon", ds_q[input.get_lon_key()].data),
                                        lat = ("lat", ds_q[input.get_lat_key()].data),
                                        time = ("time", dt_q)))

    rh_mn = np.zeros(12)
    m_std = np.zeros(12)
    for i in range(12):
        mask = np.array([x.month == (i+1) for x in dt_q])
        rh_mn[i] = np.nanmean(da_rh[mask, :, :].sel(lon = slice(270, 330), lat = slice(30, 10)).data.flatten())
        m_std[i] = np.nanvar(da_rh[mask, :, :].sel(lon = slice(270, 330), lat = slice(30, 10)).data.flatten() / 2)
    plt.plot(range(1, 13), rh_mn)
    plt.plot(range(1, 13), m_std)


    plt.figure(figsize=(18, 6)); plt.pcolormesh(ds_q['longitude'], ds_q['latitude'], np.nanvar(rh_daily[mask, :, :], axis = 0), vmin = 0, vmax = 0.1, cmap = 'gist_heat_r'); plt.colorbar(); plt.ylim([-40, 40])

def calc_daily_chi():
    from thermo import thermo
    from dateutil import relativedelta
    import dask
    # %%
    dt_start, dt_end = input.get_bounding_times()
    ds_q = xr.open_mfdataset(glob.glob('/data0/jlin/chi_daily/era5/era5_q_daily_*.nc')) #.sel(time=slice(dt_start, dt_end))

    lazy_results = []; f_args = [];
    for yr in range(namelist.start_year, namelist.end_year + 1):
        for mnth in range(1, 13):
            dt = datetime.datetime(yr, mnth, 1)
            dt_start = input.convert_from_datetime(ds_q, [dt])[0]
            dt = datetime.datetime(yr, mnth, 1) + relativedelta.relativedelta(months=+1) - relativedelta.relativedelta(hours = 1)
            dt_end = input.convert_from_datetime(ds_q, [dt])[0]
            lazy_result = dask.delayed(calc_chi)(dt_start, dt_end)
            lazy_results.append(lazy_result)
            f_args.append((dt_start, dt_end))
    out = dask.compute(*lazy_results, scheduler = 'processes', num_workers = 24)

    chi_mean = np.stack([x[0] for x in out])
    chi_std = np.stack([x[1] for x in out])
    dt_chi = np.array([x[0] for x in f_args])

    # Save the results using an intermediate file.
    da_chi = xr.DataArray(data = chi_mean, dims = ['time', 'lat', 'lon'],
                          coords = dict(lon = ("lon", ds_q[input.get_lon_key()].data),
                                        lat = ("lat", ds_q[input.get_lat_key()].data),
                                        time = ("time", dt_chi)))
    da_chi_std = xr.DataArray(data = chi_std, dims = ['time', 'lat', 'lon'],
                              coords = dict(lon = ("lon", ds_q[input.get_lon_key()].data),
                                            lat = ("lat", ds_q[input.get_lat_key()].data),
                                            time = ("time", dt_chi)))
    ds_chi = xr.Dataset(data_vars = {'chi_mean': da_chi, 'chi_std': da_chi_std})
    ds_chi.to_netcdf('/data0/jlin/era5/chi_era5_197901_202112.nc')

    ds_chi = xr.open_dataset('/data0/jlin/era5/chi_era5_197901_202112.nc')
    chi_mean = ds_chi['chi_mean'].data
    chi_std = ds_chi['chi_std'].data
    m_idx = 7
    chi_adj1 = np.nanmean(np.exp(chi_mean[m_idx::12] + 1.5*chi_std[m_idx::12] + 0.5), axis = 0)
    chi_adj2 = np.nanmean(np.exp(chi_mean[m_idx::12] + 1.5*chi_std[m_idx::12]) + 0.5, axis = 0)

    plt.figure(figsize=(18, 9)); plt.contourf(ds_chi['lon'], ds_chi['lat'], chi_adj1, levels = np.linspace(0, 3, 13), cmap = 'jet'); plt.colorbar(); #plt.ylim([0, 40]); plt.xlim([100, 180])

    plt.figure(figsize=(18, 9)); plt.contourf(ds_chi['lon'], ds_chi['lat'], chi_adj2, levels = np.linspace(0, 3, 13), cmap = 'jet'); plt.colorbar(); #plt.ylim([0, 40]); plt.xlim([100, 180])

    plt.figure(figsize=(18, 9)); plt.contourf(ds_chi['lon'], ds_chi['lat'], chi_adj1 - chi_adj2, levels = np.linspace(-0.5, 0.5, 13), cmap = 'RdBu_r'); plt.colorbar(); #plt.ylim([0, 40]); plt.xlim([100, 180])
    dt_chi[400]
    plt.figure(figsize=(12, 6)); plt.pcolormesh(da_chi['lon'], da_chi['lat'], np.nanstd(np.log(chi + 0.01), axis = 0), vmin = 0, vmax = 2, cmap = 'jet_r'); plt.colorbar(); plt.ylim([0, 40]); plt.xlim([270, 360])
    plt.figure(figsize=(18, 9)); plt.pcolormesh(da_chi['lon'], da_chi['lat'], np.nanmean(chi_std, axis = 0), vmin = 0, vmax = 2, cmap = 'jet'); plt.colorbar(); #plt.ylim([0, 40]); plt.xlim([100, 180])


    plt.figure(figsize=(12, 6)); plt.pcolormesh(da_chi['lon'], da_chi['lat'], np.nanmean(chi_adj, axis = 0), vmin = 0, vmax = 3, cmap = 'jet_r'); plt.colorbar(); plt.ylim([0, 40]); plt.xlim([100, 180])
    plt.figure(figsize=(12, 6)); plt.pcolormesh(da_chi['lon'], da_chi['lat'], np.nanmean(chi_adj, axis = 0) - np.nanmean(np.exp(np.log(chi + 0.01) + 1.6), axis = 0), vmin = -0.5, vmax = 0.5, cmap = 'RdBu_r'); plt.colorbar(); plt.ylim([0, 40]); plt.xlim([100, 180])

    plt.figure(figsize=(12, 6)); plt.pcolormesh(da_chi['lon'], da_chi['lat'], np.nanmean(chi_adj, axis = 0), vmin = 0, vmax = 3, cmap = 'jet_r'); plt.colorbar(); plt.ylim([0, 40]); plt.xlim([100, 180])





    # %%
    xlocs = np.arange(-180, 180 + 30, 30)
    xlocs_shift = np.copy(xlocs)
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(12, 6)
    proj = ccrs.PlateCarree(central_longitude=0)
    ax = plt.gca(projection = proj)
    #ax = fig.add_subplot(111, projection=proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([0, 359.99, -45, 45], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    #cs = plt.contourf(da_chi['lon'], da_chi['lat'], np.nanmean(np.exp(np.log(da_chi[:, :, :])), axis = 0), cmap = 'jet', extend = 'both',
    #                  levels = np.linspace(0, 1, 21), transform=ccrs.PlateCarree());
    #cs = plt.contourf(da_chi['lon'], da_chi['lat'], np.exp(np.log(np.nanmean(da_chi, axis = 0)+1.25)), cmap = 'jet', extend = 'both',
    #                  levels = np.linspace(0, 2.5, 21), transform=ccrs.PlateCarree());
    cs = plt.contourf(da_chi['lon'], da_chi['lat'], np.nanstd(np.log(da_chi+0.01), axis = 0), cmap = 'jet', extend = 'both',
                      levels = np.linspace(0, 1., 21), transform=ccrs.PlateCarree());
    #cs = plt.contourf(da_chi['lon'], da_chi['lat'], chi_ratio, cmap = 'jet', extend = 'both',
    #                  levels = np.linspace(1.5, 2.5, 21), transform=ccrs.PlateCarree());
    #plt.contourf(lon, lat, vpot * 0.8 * np.sqrt(0.9), cmap = palette, extend = 'both',
    #            levels = np.linspace(-16, 16, 11), transform=ccrs.PlateCarree());
    plt.colorbar(orientation = 'horizontal');
    np.nanmean(np.nanvar(np.log(da_chi.sel(lon=slice(200, 270), lat = slice(20, 10))+0.01), axis = 0))

    #chi_mdr = da_chi.sel(lon = slice(270, 330), lat = slice(35, 10)).data.flatten()

    # %%
    #chi_al = da_chi.sel(lon = slice(120, 170), lat = slice(20, 8)).data.flatten();
    chi_al = da_chi.sel(lon = slice(210, 270), lat = slice(20, 8)).data.flatten();
    #chi_al = da_chi.sel(lon = slice(280, 340), lat = slice(25, 8)).data.flatten();

    chi_al = chi_al[~np.isnan(chi_al) & (chi_al > 0.05)]
    plt.figure(figsize=(10,6)); plt.hist(np.log(chi_al), bins = np.linspace(-5, 3, 33), density=True)
    log_chi_mn = np.nanmean(np.log(chi_al))
    log_chi_std = np.nanstd(np.log(chi_al))
    x = np.linspace(-5, 3, 101)
    plt.plot(x, 1 / (log_chi_std * np.sqrt(2 * np.pi)) * np.exp(-0.5 * np.power((x - log_chi_mn) / log_chi_std, 2)))
    plt.xlabel('Log(Chi)'); plt.ylabel('Density')
    plt.title('Mean = %f, Std = %f' % (log_chi_mn, log_chi_std))
    np.nanquantile(chi_al, 0.9)
    np.exp(log_chi_mn+(log_chi_std*1.25))
    # %%


def test_w():
    from importlib import reload
    from util import sphere

    w = input.load_w().load()
    fn_wnd = env_wind.get_env_wnd_fn()
    ds_wnd = xr.open_dataset(fn_wnd)

    plt.pcolormesh(w['longitude'], w['latitude'], w['w'][8, :, :], vmin = -0.1, vmax = 0.1, cmap = 'RdBu_r'); plt.colorbar(); plt.xlim([270, 360]); plt.ylim([0, 40])

    ds_b = xr.open_dataset('land/%s.nc' % 'AL')
    basin_mask = ds_b['basin']
    f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
    basin_gen = mat.interp_2d_grid(basin_mask['lon'], basin_mask['lat'], basin_mask, ds_wnd['lon'].data, ds_wnd['lat'].data)

    # %%
    SPI_monthly = np.zeros((len(w['time']), len(ds_wnd['lat']), len(ds_wnd['lon'])))
    #SPI_basin = np.zeros(12)
    for m_idx in range(len(w['time'])):
        print(m_idx)
        omega_month = w['w'][m_idx, :, :].data
        omega_month[omega_month > 0] = 0
        #omega_mn = np.nanmean(omega_month, axis = 0)
        omega_mn = omega_month
        ua850 = ds_wnd['ua850_Mean'][m_idx, :, :] #.mean(dim = 'time')
        va850 = ds_wnd['va850_Mean'][m_idx, :, :] #.mean(dim = 'time')
        zeta850_mn = sphere.calc_rvor_sphere(ds_wnd['lon'].data, ds_wnd['lat'].data, ua850.data, va850.data)
        dlat = ds_wnd['lat'][1].data - ds_wnd['lat'][0].data
        LON, LAT = np.meshgrid(ds_wnd['lon'].data, ds_wnd['lat'].data)

        dy = sphere.haversine(LON[1:-1, :], LAT[2:, :], LON[1:-1, :], LAT[0:-2, :]) * 1000
        dy_zeta = np.divide(zeta850_mn[2:, :] - zeta850_mn[0:-2, :], 2 * dy)
        dy_zeta = np.vstack((dy_zeta[0, :], dy_zeta, dy_zeta[-1, :]))

        f = 2 * (2 * np.pi / (24 * 60 * 60)) * np.sin(np.deg2rad(LAT))
        beta = 2 * (2 * np.pi / (24 * 60 * 60)) * np.cos(np.deg2rad(LAT)) / 6378100

        Z = np.divide(np.abs(f + zeta850_mn), np.sqrt(np.abs(beta + dy_zeta) * 20))
        S = -omega_mn * np.divide(1, 1 + np.power(Z, -1 / 0.69))

        S[basin_gen < 0.5] = 0
        S[S<=1e-4] = -1e-4
        SPI_monthly[m_idx, :, :] = S

    SPI_monthly_tropics = np.copy(SPI_monthly)
    for m_idx in range(len(w['time'])):
        SPI_monthly_tropics[m_idx, :, :][LAT > 35] = 0
        #SPI_basin[m_idx] = np.sum(SPI_monthly[m_idx, :, :])

    plt.plot(range(1, 13), S_basin); plt.xlabel('Month'); plt.xlim([1, 12]); plt.ylabel('Basin-wide S'); plt.grid()

    # %%
    xlocs = np.arange(-180, 180 + 30, 30)
    xlocs_shift = np.copy(xlocs)
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(20, 4)
    proj = ccrs.PlateCarree(central_longitude=0)
    ax = plt.gca(projection = proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([0.001, 360, -40, 40], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    palette = copy(plt.get_cmap('gist_heat_r'))
    palette.set_under('white', 1.0)

    plt.contourf(w['longitude'], w['latitude'], np.nanmean(SPI_monthly[8:48:12, :, :], axis = 0), levels = np.linspace(0, 0.15, 21), cmap = palette);
    plt.colorbar(); plt.ylim([0, 45]); plt.xlim([-100, 0])

    # %%
    # Create a flat copy of the array
    SPI_monthly_tropics[SPI_monthly_tropics < 0] = 0
    SPI_monthly_tropics = SPI_monthly_tropics / np.sum(SPI_monthly_tropics)

    flat = SPI_monthly_tropics.flatten()

    # Then, sample an index from the 1D array with the
    # probability distribution from the original array
    sample_index = np.random.choice(a=flat.size, p=flat)

    # Take this index and adjust it so it matches the original array
    adjusted_index = np.unravel_index(sample_index, SPI_monthly_tropics.shape)

    lon = ds_wnd['lon'].data
    lat = ds_wnd['lat'].data
    dlon = np.abs(lon[1] - lon[0])
    dlat = np.abs(lat[1] - lat[0])

    lon[adjusted_index[2]] + dlon * (np.random.uniform() - 0.5)
    lat[adjusted_index[1]] + dlat * (np.random.uniform() - 0.5)


    da_spi = xr.DataArray(data = SPI_monthly_tropics,
                          dims = ['time', 'lat', 'lon'],
                          coords = dict(lon = ("lon", lon),
                                        lat = ("lat", lat),
                                        time = ("time", w['time'].data)))
    ds_spi = da_spi.to_dataset(name='spi')
    ds_spi.to_netcdf('/data0/jlin/era5/SPI_monthly_era5_197901_202112.nc')

    # GCM
    fn_omega = '/data0/jlin/cmip6/wap_Amon_GFDL-CM4_ssp585_r1i1p1f1_gr1_201501-210012.nc'
    ds_omega = xr.open_dataset(fn_omega)
    fn_wnd = '/data0/jlin/cmip6/env_wnd_GFDL-CM4_ssp585_r1i1p1f1_201501_210012.nc'
    ds_wnd = xr.open_dataset(fn_wnd)
    w = ds_omega['wap'].load()

    ds_b = xr.open_dataset('land/%s.nc' % 'AL')
    basin_mask = ds_b['basin']
    f_b = mat.interp2_fx(basin_mask['lon'], basin_mask['lat'], basin_mask)
    basin_gen = mat.interp_2d_grid(basin_mask['lon'], basin_mask['lat'], basin_mask, ds_wnd['lon'].data, ds_wnd['lat'].data)

    SPI_monthly = np.zeros((len(ds_wnd['time']), len(ds_wnd['lat']), len(ds_wnd['lon'])))
    for m_idx in range(len(ds_wnd['time'])):
        print(m_idx)
        omega_month = w[m_idx, 5, :, :].data
        omega_month[omega_month > 0] = 0
        #omega_mn = np.nanmean(omega_month, axis = 0)
        omega_mn = omega_month

        ua850 = ds_wnd['ua850_Mean'][m_idx, :, :] #.mean(dim = 'time')
        va850 = ds_wnd['va850_Mean'][m_idx, :, :] #.mean(dim = 'time')
        zeta850_mn = sphere.calc_rvor_sphere(ds_wnd['lon'].data, ds_wnd['lat'].data, ua850.data, va850.data)
        dlat = ds_wnd['lat'][1].data - ds_wnd['lat'][0].data
        LON, LAT = np.meshgrid(ds_wnd['lon'].data, ds_wnd['lat'].data)

        dy = sphere.haversine(LON[1:-1, :], LAT[2:, :], LON[1:-1, :], LAT[0:-2, :]) * 1000
        dy_zeta = np.divide(zeta850_mn[2:, :] - zeta850_mn[0:-2, :], 2 * dy)
        dy_zeta = np.vstack((dy_zeta[0, :], dy_zeta, dy_zeta[-1, :]))

        f = 2 * (2 * np.pi / (24 * 60 * 60)) * np.sin(np.deg2rad(LAT))
        beta = 2 * (2 * np.pi / (24 * 60 * 60)) * np.cos(np.deg2rad(LAT)) / 6378100

        Z = np.divide(np.abs(f + zeta850_mn), np.sqrt(np.abs(beta + dy_zeta) * 20))
        omega_mn_int = mat.interp_2d_grid(ds_omega['lon'].data, ds_omega['lat'].data, omega_mn, ds_wnd['lon'].data, ds_wnd['lat'].data)
        S = -omega_mn_int * np.divide(1, 1 + np.power(Z, -1 / 0.69))

        S[basin_gen < 0.5] = 0
        S[S<=1e-4] = -1e-4
        SPI_monthly[m_idx, :, :] = S

    da_spi = xr.DataArray(data = SPI_monthly,
                          dims = ['time', 'lat', 'lon'],
                          coords = dict(lon = ("lon", ds_wnd['lon'].data),
                                        lat = ("lat", ds_wnd['lat'].data),
                                        time = ("time", ds_wnd['time'].data)))
    ds_spi = da_spi.to_dataset(name='spi')
    ds_spi.to_netcdf('/data0/jlin/cmip6/SPI_monthly_GFDL-CM4_ssp585_r1i1p1f1_201501-210012.nc')

    # %%
    xlocs = np.arange(-180, 180 + 30, 30)
    xlocs_shift = np.copy(xlocs)
    fig = plt.figure(facecolor='w', edgecolor='k');
    fig.set_size_inches(20, 4)
    proj = ccrs.PlateCarree(central_longitude=0)
    ax = plt.gca(projection = proj)
    ax.coastlines(resolution='50m')
    ax.set_extent([0.001, 360, -40, 40], crs=ccrs.PlateCarree())
    ax.gridlines(draw_labels=False, crs=ccrs.PlateCarree(), xlocs=xlocs,
                 color='gray', alpha=0.3)
    gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), xlocs=xlocs_shift[1:-1],
                      color='gray', alpha=0.3)
    gl.xlabels_top = False
    gl.ylabels_right = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

    palette = copy(plt.get_cmap('gist_heat_r'))
    palette.set_under('white', 1.0)


    plt.contourf(ds_wnd['lon'], ds_wnd['lat'], np.nanmean(SPI_monthly[8:128:12, :, :], axis = 0), levels = np.linspace(0, 0.15, 21), cmap = palette);
    plt.colorbar(); plt.ylim([0, 45]); plt.xlim([-100, 0])

def plot_tracks_cmip():
    ds.close()
    ds_seeds.close()

    import glob
    from copy import copy

    # %%
    fn_tracks = glob.glob('/data0/jlin/cmip6/tracks/spi/tracks_AL_GFDL-CM4_ssp585_r1i1p1f1_201501_210012_e[0-9].nc')
    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                           data_vars="minimal", drop_variables = "seeds_per_months")

    drop_vars = ["lon_trks", "lat_trks", "u250_trks", "v250_trks", "u850_trks", "v850_trks",
                 "v_trks", "m_trks", "vmax_trks", "tc_month"]
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                                 data_vars="minimal", drop_variables = drop_vars)

    yearS = 2015
    yearE = 2100
    ntrks_per_year = 16
    ntrks_per_sim = (yearE - yearS + 1)*ntrks_per_year
    n_sim = len(fn_tracks)

    #np.sum(ds['seeds_per_months'], axis = 1).sel
    #tc_years = np.tile(np.array(range(1950, 2022)), (200, 1)).T.flatten()
    #plt.hist(ds['tc_month'][tc_years == 2020], bins = np.arange(0.5, 12.5, 1))

    # Filter for surviving tracks.
    vmax = ds['vmax_trks'].load().data
    lon_filt = np.full(ds['lon_trks'].shape, np.nan) #ds['lon_trks'][mask, :].data
    lat_filt = np.full(ds['lat_trks'].shape, np.nan) #ds['lat_trks'][mask, :].data
    vmax_filt = np.full(ds['vmax_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    m_filt = np.full(ds['m_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    lon_trks = ds['lon_trks'].load().data
    lat_trks = ds['lat_trks'].load().data
    m_trks = ds['m_trks'].load().data

    # Here, we only consider a TC from the first point where it exceeds
    # the threshold, to the point it decays to 10 m/s (after it has
    # reached its peak intensity).
    lon_genesis = np.full(lon_filt.shape[0], np.nan)
    lat_genesis = np.full(lon_filt.shape[0], np.nan)
    for i in range(lon_filt.shape[0]):
        if len(np.argwhere(vmax[i, :] >= 15).flatten()) > 0:
            # Genesis occurs when the TC first achieves 30 knots (15 m/s).
            gen_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idx_gen = np.argwhere(vmax[i, :] >= 15).flatten()[0]
            lon_genesis[i] = lon_trks[i, idx_gen]
            lat_genesis[i] = lat_trks[i, idx_gen]

            # TC decays after it has reached 15 m/s
            decay_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idxs_lmi = np.argwhere(decay_idxs >= np.nanargmax(vmax[i, :])).flatten()
            if len(decay_idxs) > 0 and len(idxs_lmi) > 0:
                idx_decay = decay_idxs[idxs_lmi[0]]
            else:
                idx_decay = vmax.shape[1]

            nt = idx_decay - idx_gen
            vmax_filt[i, 0:nt] = vmax[i, idx_gen:idx_decay]
            lon_filt[i, 0:nt] = lon_trks[i, idx_gen:idx_decay]
            lat_filt[i, 0:nt] = lat_trks[i, idx_gen:idx_decay]
            m_filt[i, 0:nt] = m_trks[i, idx_gen:idx_decay]

    # %% Plot seasonal and interseasonal variability
    # Compute seeding variability for downscaling.
    nSeeds_per_year_individual = np.zeros((yearE - yearS + 1, n_sim))
    for yr in np.unique(ds_seeds['year']):
        nSeeds_per_year_individual[yr - yearS, :] = np.sum(ds_seeds['seeds_per_months'][ds_seeds['year'] == yr, :].load(), axis = 1)
    nSeeds_per_year = np.nanmean(nSeeds_per_year_individual, axis = 1)
    pdi_per_year_downscaling = np.zeros((n_sim, yearE-yearS+1))
    YR, _ = np.meshgrid(range(yearS, yearE + 1), range(namelist.tracks_per_year))
    YR = YR.T.flatten()
    for i in range(n_sim):
        pdi_trks = np.power(vmax_filt[(i*ntrks_per_sim):((i+1)*ntrks_per_sim)], 3)
        for j in range(pdi_trks.shape[0]):
            nan_mask = ~np.isnan(pdi_trks[j, :])
            pdi_per_year_downscaling[i, YR[j]-yearS] += np.trapz(pdi_trks[j, nan_mask], ds['time'][nan_mask].data)

    dt_year = list(range(yearS, yearE+1))
    mean_ratio = np.nanmean(n_tc_per_year) / np.nanmean(np.min(nSeeds_per_year.data) / nSeeds_per_year)

    n_tc_per_year_downscaling = np.min(nSeeds_per_year.data) / nSeeds_per_year * mean_ratio
    n_tc_per_year_ens_downscaling = np.zeros(nSeeds_per_year_individual.shape)
    for e_idx in range(len(fn_tracks)):
        n_tc_per_year_ens_downscaling[:, e_idx] = np.min(nSeeds_per_year_individual[:, e_idx]) / nSeeds_per_year_individual[:, e_idx] * mean_ratio
    ens_ratio = np.nanmean(n_tc_per_year) / np.nanmean(np.nanmedian(n_tc_per_year_ens_downscaling, axis = 1))
    n_tc_per_year_ens_downscaling *= ens_ratio
    y_err_p = np.nanquantile(n_tc_per_year_ens_downscaling, 0.95, axis = 1) - np.nanmean(n_tc_per_year_ens_downscaling, axis = 1)
    y_err_n = np.nanmean(n_tc_per_year_ens_downscaling, axis = 1) - np.nanquantile(n_tc_per_year_ens_downscaling, 0.05, axis = 1)

    plt.figure(figsize=(12, 6));
    h1, = plt.plot(dt_year, n_tc_per_year_downscaling, color='r', linewidth=3)
    plt.boxplot(n_tc_per_year_ens_downscaling.T, sym = 'r.', positions=dt_year, whiskerprops={'linewidth': 2, 'color': 'b'},
                boxprops = {'linewidth': 2}, medianprops = {'linewidth': 2, 'color': 'b'}, flierprops = {'linewidth': 2, 'color': 'r'}, showfliers = True)
    plt.xticks(range(yearS, yearE+1, 5), range(yearS, yearE+1, 5));
    plt.grid();
    #h3, = plt.plot(list(range(1979, 2021)), n_tc_per_year_emanuel, 'b', linewidth = 4);
    plt.legend([h1], ['Downscaling']); plt.ylabel('Number of TCs')

    # %%
    from scipy.ndimage import uniform_filter1d
    plt.figure(figsize=(10, 6))
    h1, = plt.plot(dt_year, n_tc_per_year_downscaling);
    h2, = plt.plot(dt_year, n_tc_per_year_downscaling_random);
    plt.plot(dt_year, uniform_filter1d(n_tc_per_year_downscaling, 10), color = 'b', linewidth = 2)
    plt.plot(dt_year, uniform_filter1d(n_tc_per_year_downscaling_random, 10), color = 'r', linewidth = 2)

    plt.xlim([yearS - 0.05, yearE + 0.05]); plt.xlabel('Year'); plt.ylabel('Number of TCs (AL)')
    plt.grid(); plt.legend(['SPI', 'Random Seeding'])
