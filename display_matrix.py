import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import math
import argparse
import imageio

from dl_cs.fileio import cfl
from display_data import matplot, save_gif, save_video

def normalize(im):
    return np.abs((im-np.mean(im)))/np.std(im)

def main(args):

    file_recon = args.file
    file_compare = args.dfile

    #os.path.join(args.directory, "Exam2200_Series5_12accel.im")
    animate_flag = args.animate
    #file_gt = os.path.join(args.directory, "im.dl.gt")
    plot_xt = args.xt
    phase = args.phase      #Hack code to manually pick the phase to plot
    slice = args.slice
    outfile = args.outfile
    nrows, ncols = args.matrix
    cmap = args.cmap
    scale = args.scale
    x0, x1 = args.xrange
    y0, y1 = args.yrange
    colorbar = args.cbar

    temp_image = cfl.readcfl(file_recon[0])
    nx, ny, nslice, nmap, nphase, d1, d2, d3 = temp_image.shape

    if x1==-1: 
        x1=nx
    
    if y1==-1:
        y1=ny

    lenx=x1-x0
    leny=y1-y0

    image = np.inf*np.ones((lenx*nrows, leny*ncols, nslice, nmap, nphase, d1, d2, d3))
    image_compare = np.zeros((lenx*nrows, leny*ncols, nslice, nmap, nphase, d1, d2, d3))

    for i, files in enumerate(file_recon):
        curr_row = i%nrows
        curr_col = int(np.floor(i/nrows))%ncols
        print("curr_ind is {} curr row is {} and curr_col is {}".format(i, curr_row, curr_col))
        curr_image = cfl.readcfl(files)
        curr_image = normalize(curr_image)
        #curr_image = np.abs(curr_image)
        image[curr_row*lenx:(curr_row+1)*lenx, curr_col*leny:(curr_col+1)*leny, :,:,:,:,:,:] = curr_image[x0:x1,y0:y1,:,:,:,:,:,:]
        
    if file_compare:

        for j, dfiles in enumerate(file_compare):
            curr_row = j%nrows
            curr_col = int(np.floor(j/nrows))%ncols
            print("curr_ind is {} curr row is {} and curr_col is {}".format(j, curr_row, curr_col))
            curr_image = cfl.readcfl(dfiles)
            curr_image = normalize(curr_image)
            #curr_image = np.abs(curr_image)
            image_compare[curr_row*lenx:(curr_row+1)*lenx, curr_col*leny:(curr_col+1)*leny, :,:,:,:,:,:] = curr_image[x0:x1,y0:y1,:,:,:,:,:,:]

        image_compare_tresh = image_compare
        image_compare_tresh[image_compare<1e-10]=1
        image = np.abs(image-image_compare)
        #image[image>3*scale[1]] = 0
       

    #print(np.mean(image))
    #print(np.std(image))

    if plot_xt == 1:
        image = np.transpose(image, (0,4,2,3,1,5,6,7))
    if plot_xt == 2:
        image = np.transpose(image, (4,1,2,3,0,5,6,7))

    matplot(image, animate_flag, phase, slice, outfile, scale, cmap, colorbar)


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Plotting and Analysis Code")
    parser.add_argument('--file', nargs='+', type=str, required=True, help='file name')
    parser.add_argument('--matrix', nargs=2, type=int, required=True, help='matrix size')
    parser.add_argument('--animate', type=int, default=1, help='1 for animate, 0 to click to draw frame')
    parser.add_argument('--xt', type=int, default=0, help= '0:normal plot 1:plot x-t space, 2:plot y-t space')
    parser.add_argument('--phase', type=int, default=-1, help='Pick the phase you want to plot')
    parser.add_argument('--slice', type=int, default=-1, help='Pick the image slice you want to plot')
    parser.add_argument('--outfile', type=str, default='', help='output save file for animations')
    parser.add_argument('--dfile', nargs='+', type=str, default = '', help='Comparison file, usually R=1, to calculate error')
    parser.add_argument('--scale', nargs=2, type=float, required=False, help='Image intensity scaling')
    parser.add_argument('--cmap', type=str, default='gray', help='Colormap of plot')
    parser.add_argument('--xrange', nargs=2, type=int, default=[0, -1], help='Colormap of plot')
    parser.add_argument('--yrange', nargs=2, type=int, default=[0, -1], help='Colormap of plot')
    parser.add_argument('--cbar', type=bool, default=False, help='plot colorbar')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)

