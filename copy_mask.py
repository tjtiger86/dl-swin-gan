# Script to copy mask from similar dataset

import pickle
import numpy as np
from pathlib import Path
#import cv2
import os

import sys
import glob
import time

import argparse

def mask_series(images):

    im = np.asarray(images[:,:,:,0,:,0,0,0])    #Taking only the first E-spirit map
    im = im.reshape(*im.shape[:3],-1)
    im = normalize(np.abs(im))

    [Nx, Ny, Nslice, Nphase] = np.shape(im)
    #print(np.shape(im))
    masks = np.empty([Nx, Ny, Nphase*Nslice])

    # Only segment once per slice, and assume it doesn't change too much
    for idx in range(Nslice):
        mask = getMask(im[:,:,idx,0])
        #mask = np.ones([Nx, Ny])
        masks[:,:,(idx*Nphase):(idx+1)*Nphase] = np.tile(mask[:,:,np.newaxis], [1, 1, Nphase])

        #plt.imshow(np.squeeze(im[idx,rois[idx][1]:(rois[idx][1]+rois[idx][3]), rois[idx][2]:(rois[idx][2]+rois[idx][4])]))
        #plt.show()
    fig, ax = plt.subplots(1,2,constrained_layout=True)
    ax[0].imshow(mask)
    ax[1].imshow(im[:,:,idx,0])
    plt.show()

    return masks

def openpickle(filename):

    infile = open(filename,'rb')
    dict = pickle.load(infile)
    infile.close()

    return dict

def writepickle(filename, data_out):
    #Writing output to pickle file
    infile = open(filename, 'wb')
    pickle.dump(data_out,infile)
    infile.close()

def main(args):

    file = args.file
    mask_file = args.mask_file

    dfile = openpickle(file)
    dmask = openpickle(mask_file)

    #Should do error checking that file and mask_file are the same size...
    N = np.size(dfile)

    #Copy mask
    for idx in range(N):

        curr_accel = dfile[idx]['accel']
        curr_name = dfile[idx]['name']

        #Could simply jsut copy mask, don't have to do ind_1accel, but lazy
        #ind_1accel = next((i for i, item in enumerate(dmask) if item['name'][:-3] == curr_name and int(item['accel']) == 1), None)
        ind_1accel = next((i for i, item in enumerate(dmask) if item['name'] == curr_name and int(item['accel']) == 1), None)
        print("curr name is {}, curr index is {}/{}, acceleration of one index is {}".format(curr_name, idx,N, ind_1accel))
        dfile[idx]['mask'] = dmask[ind_1accel]['mask']

    #Save mask
    writepickle(file, dfile)


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Reconstruction Evaluation")
    parser.add_argument('--file', type=str, required=True, help='Processed file without mask')
    parser.add_argument('--mask-file', type=str,required=True, help='Previously segmented file ')
    return parser

if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)
