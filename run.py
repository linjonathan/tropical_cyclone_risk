import os
import shutil
import namelist
import sys
from scripts import generate_land_masks
from util import compute

if __name__ == '__main__':
    f_base = '%s/%s/' % (namelist.output_directory, namelist.exp_name)
    os.makedirs(f_base, exist_ok = True)
    print('Saving model output to %s' % f_base)
    shutil.copyfile('./namelist.py', '%s/namelist.py' % f_base)

    generate_land_masks.generate_land_masks()
    compute.compute_downscaling_inputs()

    print('Running tracks for basin %s...' % sys.argv[1])
    compute.run_downscaling(sys.argv[1])
