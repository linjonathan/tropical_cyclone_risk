# Script for downloading ERA5 re-analysis data necessary to run the
# tropical cyclone downscaling risk model.
# Requires cdsapi to be installed.

import cdsapi
import os

from multiprocessing import Pool

# In order to read namelist.py
import sys
cwd = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(cwd)
sys.path.append(parent)

import namelist

year_start = int(namelist.start_year)
year_end = int(namelist.end_year)
fn_base = namelist.base_directory
os.makedirs(fn_base, exist_ok = True)

def request_file(fn, req_type, req):
    c = cdsapi.Client()
    if not os.path.isfile(fn):
        try:
            print('Requesting %s...' % fn)
            c.retrieve(req_type, req, fn)
            print('Downloaded %s...' % fn)
        except:
            print("Error downloading the data...")
            pass
    else:
        print('Found file %s...' % fn)

def f_request(year):
    fn_dir = '%s/%d' % (fn_base, year)
    if not os.path.exists(fn_dir):
        os.makedirs(fn_dir)
    
    sst_fn = '%s/era5_sst_monthly_%d.nc' % (fn_dir, year)
    sp_fn = '%s/era5_sp_monthly_%d.nc' % (fn_dir, year)
    q_fn = '%s/era5_q_monthly_%d.nc' % (fn_dir, year)
    t_fn = '%s/era5_t_monthly_%d.nc' % (fn_dir, year)
    u_fn = '%s/era5_u_daily_%d.nc' % (fn_dir, year)
    v_fn = '%s/era5_v_daily_%d.nc' % (fn_dir, year)

    req_sst = {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'sea_surface_temperature',
        'year': str(year),
        'month': ['01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
              ],
        'grid': '1.0/1.0',
        'time': '00:00'
    }

    req_sp = {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'surface_pressure',
        'year': str(year),
        'month': ['01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
              ],
        'grid': '1.0/1.0',
        'time': '00:00'
    }

    req_q = {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'specific_humidity',
        'pressure_level': [ '70', '100', '125', '150', '175', '200',
                            '225', '250', '300', '350', '400', '450',
                            '500', '550', '600', '650', '700', '750',
                            '775', '800', '825', '850', '875', '900',
                            '925', '950', '975','1000',
                        ],
        'year': str(year),
        'month': ['01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
              ],
        'grid': '1.0/1.0',
        'time': '00:00',
    }

    req_t = {
        'format': 'netcdf',
        'product_type': 'monthly_averaged_reanalysis',
        'variable': 'temperature',
        'pressure_level': [ '70', '100', '125', '150', '175', '200',
                            '225', '250', '300', '350', '400', '450',
                            '500', '550', '600', '650', '700', '750',
                            '775', '800', '825', '850', '875', '900',
                            '925', '950', '975','1000',
                        ],
        'year': str(year),
        'month': ['01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
              ],
        'grid': '1.0/1.0',
        'time': '00:00',
    }

    req_u = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'u_component_of_wind',
        'pressure_level': ['250', '850'],
        'year': str(year),
        'month': ['01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
              ],
        'day': ['01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
                '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24',
                '25', '26', '27', '28', '29', '30',
                '31',
            ],
        'grid': '1.0/1.0',
        'time': ['00:00', '12:00'],
    }

    req_v = {
        'product_type': 'reanalysis',
        'format': 'netcdf',
        'variable': 'v_component_of_wind',
        'pressure_level': ['250', '850'],
        'year': str(year),
        'month': ['01', '02', '03',
                  '04', '05', '06',
                  '07', '08', '09',
                  '10', '11', '12',
              ],
        'day': ['01', '02', '03', '04', '05', '06',
                '07', '08', '09', '10', '11', '12',
                '13', '14', '15', '16', '17', '18',
                '19', '20', '21', '22', '23', '24',
                '25', '26', '27', '28', '29', '30',
                '31',
            ],
        'grid': '1.0/1.0',
        'time': ['00:00', '12:00'],
    }

    request_file(sst_fn, 'reanalysis-era5-single-levels-monthly-means', req_sst)
    request_file(sp_fn, 'reanalysis-era5-single-levels-monthly-means', req_sp)
    request_file(t_fn, 'reanalysis-era5-pressure-levels-monthly-means', req_t)
    request_file(q_fn, 'reanalysis-era5-pressure-levels-monthly-means', req_q)
    request_file(u_fn, 'reanalysis-era5-pressure-levels', req_u)
    request_file(v_fn, 'reanalysis-era5-pressure-levels', req_v)

years = list(range(year_start, year_end+1))
p = Pool(6)
output = p.map(f_request, years)
p.close()
p.join()
