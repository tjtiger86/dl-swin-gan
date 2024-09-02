#Stupid script since i have some files to rename

import numpy as np
import os
from pathlib import Path
import sys
import glob

def main():

    data_dir = '/home/tjao/data/recon_2emaps/train-3D_5steps_2resblocks_64features_2emaps_0weight/'

    print(data_dir)
    files = glob.glob(os.path.join(data_dir, '*'))
    print(files)

    for file in files:
        print('filename is {}'.format(file))
        newname = file.replace('.h5', '')
        print('New filename is {}'.format(newname))

        os.rename(file,newname)

if __name__ == '__main__':

    main()
