# An Open-Source, Physics-Based, Tropical Cyclone Downscaling Model with Intensity-Dependent Steering
This is a publicly available, statistical-dynamical tropical cyclone downscaling model, and a derivative of the MIT tropical cyclone downscaling model. It is composed of three primary components: (1) a random seeding process that determines genesis, (2) an intensity-dependent beta-advection model that determines the track, and (3) a non-linear differential equation set that determines the intensification rate. The model is entirely forced by the large-scale environment. Downscaling ERA5 reanalysis data shows that the model is generally able to reproduce observed tropical cyclone climatology, such as the global seasonal cycle, genesis locations, track density, and lifetime maximum intensity distributions. Inter-annual variability in tropical cyclone count and power-dissipation is also well captured, on both basin-wide and global scales. The model is also able to reasonably capture the observed return period curves of landfall intensity in various sub-basins around the globe. The incorporation of an intensity-dependent steering flow is shown to lead to regionally dependent changes in power dissipation and return periods.

Citation: [Lin et al. (2023)](https://arxiv.org/abs/2302.09455) (accepted to JAMES)

## Quick Start: TLDR
If you know what you are doing and familiar with Python packages, here is a quick start command list to get the model running.

    conda env create -f environment.yml
    conda activate tc_risk
    pip install cdsapi
    [generate .cdsapirc file in home directory]
    python3 scripts/download_era5.py
    python3 run.py GL

## Environment Set Up
We strongly recommend creating a virtual environment to install the packages required to run this model. A very popular virtual environment/package manager is [Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#). We have provided the [YAML file](environment.yml) that worked in our testing of the model.

<code>conda env create -f environment.yml</code>

This requires conda to be installed, and should create a virtual environment with the name **tc_risk**. See [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file) for more details.

If you prefer to do a clean install of the required packages, the following commands will create a virtual environment:

    conda create -n tc_risk basemap cftime xarray numpy matplotlib cartopy python jupyter scipy netCDF4 dask
    pip install global-land-mask cdsapi

Note, not all Python packages are fully supported on Apple Silicon architecture yet.

## Configuring the Model
The [namelist](namelist.py) file contains various parameters used in the model. It is useful for tuning the model, as well as adding a variety model data sources. The [namelist](namelist.py) file contains self-explanatory descriptions for most of the variables, but we will add some detail here.

| Variable Name| Description |
| ------------ | ----------- |
| base_directory | master directory that all input data are located |
| output_directory | master directory to save all output data |
| exp_name | name of the experiment; output will be saved in this folder |
| dataset_type | key used in _var_keys_ for variable mapping (see Formatting Input)|
| output_interval_s | output interval of tropical cyclone tracks |
| tracks_per_year | total number of tracks to simulate each year |
| PI_reduc | reduction factor for potential intensity ($S_w$ in the manuscript) |
| p_midlevel | pressure (Pa) of which to calculate mid-level saturation entropy deficit |
| select_thermo | 1 or 2 for pseudoadiabatic and reversible thermodynamics, respectively |
| select_interp | leave it at 2 |

## Input Data
### Reanalysis Data
The manuscript uses ERA5 reanalysis data to compare model output to observations. While the user is free to use their own input data (according to the format described in the next section), we have provided a script that downloads and organizes ERA5 reanalysis data into a digestible format for the downscaling model, but it requires **cdsapi**. [Here](https://cds.climate.copernicus.eu/api-how-to) are detailed instructions on how to install cdsapi on your machine. In short, you will need to create your own account and the _cdsapirc_ key file in your home directory, then install the client via:

<code>pip install cdsapi</code>

After this, run the script [download_era5.py](scripts/download_era5.py) from the home directory. It will download data to the directory specified in <code>namelist.base_directory</code>.

Unfortunately, we have not tested the model on the reanalysis datasets of other modeling centers.

### CMIP6 Model Data
While the original manuscript does not make any climate projections with the downscaling model, we provide an example script, [download_cmip6.py](scripts/download_cmip6.py), to download and organize CMIP6 model. There is no additional account creation necessary. 

### Formatting Input
The input data needs to be formatted in relatively loose way. For a variable <code>VAR</code> In the directory <code>namelist.base_directory</code>, it finds all files that match <code>\*namelist.exp_prefix\*VAR\*.nc</code>. The files can be in organized sub-directories (such as sub-directories by year of data), or any other way that is easy for the user. The variables in the netCDF files should contain timestamps that makes them amenable to multifile dataset reading in xarray. Typically, there is nothing extra that needs to be done when downloading ERA5 reanalysis or CMIP6 climate model output.

Every modeling center, however, has different naming conventions for variables. The variable <code>namelist.var_keys</code> is a Python dictionary that allows the user to specify the standard variable name (dictionary key) and how it is represented in the dataset (dictionary value). The standard variable names used in the model are:
| Variable Name| Description |
| ------------ | ----------- |
| sst          | sea surface temperature (monthly-averaged) |
| mslp         | mean sea level pressure (monthly-averaged) |
| temp         | temperature (monthly-averaged) |
| sp_hum       | specific humidity (monthly-averaged) |
| u            | zonal wind (daily) |
| v            | meridional wind (daily) |
| lvl          | pressure vertical level |
| lon          | longitude |
| lat          | latitude |

For instance, the specific humidity is named 'q' in the ERA5 datasets, so we set the key-value pair 'sp_hum' and 'q' in the 'ERA5' (which is specified by <code>namelist.dataset_type</code>) entry in <code>namelist.var_keys</code>.

Note, when using CMIP6 model data, some GCMs use non-rectangular horizontal coordinates. The interpolation code expects rectangular grids, so you will need to regrid the NetCDF files into rectanguular coordinate. The _cdo_ library is convenient for this. In addition, some vertical grids are output in sigma coordinates. You will need the vertical grid in pressure coordinates.

## Running the Model
We have provided a simple script, [run.py](run.py), to run the model.

    conda activate tc_risk
    python3 run.py GL

where the first argument is the basin with which to downscale tracks. The first time the model is run, land masks will be generated in the source directory. This script will run the downscaling model and save pre-processed large-scale fields (potential intensity, environmental winds, thermodynamic fields) to the file directory <code>namelist.output_directory/</code>, and the downscaled tropical cyclone tracks to <code>namelist.output_directory/namelist.exp_name/</code>.

## Model Output
The tropical cyclone tracks will be output in a netCDF file. These netCDF files can be read using xarray, and are compatible with multifile dataset reads.
| Variable Name| Description |
| ------------ | ----------- |
| lon_trks     | longitude of tropical cyclone track (&deg;E) |
| lat_trks     | latitude of tropical cyclone track |
| u250_trks    | 250-hPa environmental zonal wind (m/s) |
| v250_trks    | 250-hPa environmental meridional wind (m/s) |
| u850_trks    | 850-hPa environmental zonal wind (m/s) |
| v850_trks    | 850-hPa environmental meridional wind (m/s) |
| v_trks       | maximum azimuthal wind (m/s) |
| m_trks       | non-dimensional inner core moisture |
| vmax_trks    | maximum wind, or intensity (m/s) |
| tc_basins    | basin of tropical cyclone occurrence |
| tc_month     | month of tropical cyclone occurrence |
| tc_years     | year of tropical cyclone occurrence |
| seeds_per_month | number of random seeds placed |
| time         | time since track genesis (seconds) |
    
## Potential Intensity
The model also includes a new, CAPE-based method to calculate potential intensity. The [thermo.py](thermo/thermo.py) is a file that only depends on the [constants.py](util/constants.py) file, and contains the functions to calculate potential intensity. The function, <code>CAPE_PI</code> is written in a transparent and straightforward way. There is also a vectorized method that is fast to use, but less readable than the non-vectorized code, <code>CAPE_PI_vectorized</code>.

## 24-Hour Intensity Change Distribution
Below is a comparison of the 24h-hour intensity change distribution from the downscaling model in the North Atlantic basin, as compared to observations. Only open-ocean tropical cyclones with intensities of at-least 35 knots were considered.
![image](https://github.com/linjonathan/tropical_cyclone_risk/assets/7074325/ddc7fe57-bb42-4b5f-91ed-b3f3e82d0639)


## Code Contributions
Jonathan Lin  
Raphael Rousseau-Rizzi

## Version History
Version 1.1 (2024-05-14) - see release notes  
Version 1.0 (2023-02-17) - model released

[![DOI](https://zenodo.org/badge/602773936.svg)](https://zenodo.org/badge/latestdoi/602773936)

## License
The MIT License (MIT)

Copyright (c) 2023 Jonathan Lin

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
