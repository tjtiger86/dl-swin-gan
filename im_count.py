import numpy as np
import os
from pathlib import Path
import sys
import glob

import pickle
#import h5py

import argparse
from dl_cs.fileio import cfl


def fname_accel(file_name):

    parts = file_name.split("_")
    ndig = len(parts[2])-12 #number of digits for acceleration factor, 12 is due to extension

    return parts[0]+"_"+parts[1], parts[2][0:ndig]

def main(args):

    #Argument Parsing
    data_dir = args.data_dir
    files = glob.glob(os.path.join(data_dir, "*.im.cfl"))

    tot_im = 0
    for idx, file in enumerate(files):

        fname = os.path.basename(file)
        exam_name, accel = fname_accel(fname)

        if int(accel) == 1: 

            print("Analyzing {}".format(fname))
            image = cfl.readcfl(os.path.splitext(file)[0])             #remove .cfl for cfl.readcfl
            nx, ny, nslice, nmap, nphase, d1, d2, d3 = image.shape
            tot_im += nslice*nphase
    
    print("Total number of images is: {}".format(tot_im))

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Reconstruction Evaluation")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory of reconstructed images')
    return parser

if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)
