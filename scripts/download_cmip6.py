# Run this script from the main directory (namelist.src_directory)
import os
import glob

# In order to read namelist.py
import sys
cwd = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(cwd)
sys.path.append(parent)
import namelist

# Downloads netCDF4 files from CMIP6 web portal.
# The wget scripts are obtained from the search portal: https://esgf-node.llnl.gov/search/cmip6/.
# A sample set of scripts for obtaining GFDL-CM4 ssp585 scenario output is provided.
# Make sure you have enough space in the current directory.
# Moves all output to the base directory described in the namelist.
fns = glob.glob('%s/scripts/GFDL-CM4/*.sh' % namelist.src_directory)
for i in range(len(fns)):
    cmd = 'bash %s -s' % fns[i]
    os.system(cmd)

fn_base = namelist.base_directory
os.makedirs(fn_base, exist_ok = True)

# Move all downloaded files to namelist.base_directory.
fns = glob.glob('%s/*GFDL-CM4*.nc' % namelist.src_directory)
for i in range(len(fns)):
    cmd = 'mv %s %s/' % (fns[i], fn_base)
    os.system(cmd)
    
# Remove all wget status files.
fns = glob.glob('%s/.*.status' % namelist.src_directory)
for i in range(len(fns)):
    os.remove(fns[i])
