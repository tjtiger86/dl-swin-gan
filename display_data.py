import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
import math
import argparse
import imageio

from dl_cs.fileio import cfl


def matplot(image, animate_flag, phase, desired_slice, outfile, cscale, colormap, colorbar):

    if phase >= 0:
        #Just picking the phase to plot
        image = image[:,:,:,:,[phase],:,:,:]

    if desired_slice >= 0:
        #Picking slice to plot
        image = image[:,:,[desired_slice],:,:,:,:]

    nx, ny, nslice, nmap, nphase, d1, d2, d3 = image.shape

    ncol = int(math.sqrt(math.floor(nslice)))
    nrow = int(math.sqrt(math.ceil(nslice)))

    cmin = cscale[0]
    cmax = cscale[1]
    # print([ncol, nrow, nslice])

    #fig = plt.figure(figsize=(8.1,12))
    fig = plt.figure(figsize=(8,np.ceil(8*nx/ny)))
    ims = []
    for kk in range(nphase):

        slice = 1

        for ii in range(ncol):
            for jj in range(nrow):

                ax = plt.subplot(ncol,nrow,slice)
                plt.axis("off")
                if cmin==cmax:
                    im = plt.imshow(np.squeeze(image[:, :, slice-1, 0, kk]), cmap=colormap)
                else:
                    im = plt.imshow(np.squeeze(image[:, :, slice-1, 0, kk]), cmap=colormap, vmin=cmin, vmax=cmax)

                #im = plt.imshow(np.squeeze(image[:, :, slice-1, 0, kk]), cmap="gray", vmin = -1.5, vmax = 3)
                #im = plt.imshow(np.squeeze(image[:, :, slice-1, 0, kk]), cmap="gray", vmin = -2, vmax = 4)
                #im = plt.imshow(np.squeeze(image[:, :, slice-1, 0, kk]), cmap="gray")
                #im = plt.imshow(np.squeeze(image[:, :, slice-1, 0, kk]), cmap="gray", vmin = -0.5, vmax = 3)
                #im = plt.imshow(np.squeeze(image[:, :, slice-1, 0, kk]), vmin = 0, vmax = 0.5)

                # reduce white space in subplots
                ax.set_xticklabels([])
                ax.set_yticklabels([])



                fig.tight_layout()
                #ax.set_aspect('equal')
                # plt.subplots_adjust(wspace=None, hspace=None) Doesn't work
                ims.append([im])
                slice = slice + 1

                plt.subplots_adjust(wspace=0, hspace=0)

        if animate_flag != 1:
            print("Phase %d:" %(kk))
            plt.draw()
            plt.pause(1)
            input("Press enter to draw next frame:")
            #plt.close()

    if animate_flag == 1:
        ani = animation.ArtistAnimation(fig, ims, interval=5, blit=True,
                                repeat_delay=0)

        if outfile:
            print(outfile)
            save_video(outfile,ani)
            #save_gif(outfile,ims)

        if colorbar:
            plt.colorbar()
            
        plt.show()

    if animate_flag == 0:
        if outfile:
            print(outfile)
            plt.savefig(outfile)

def save_video(fname, anim):
    writervideo = animation.FFMpegWriter(fps=10)
    anim.save(fname, writer=writervideo)

def save_gif(fname,ims):
    with imageio.get_writer(fname, mode='I', duration=0.1) as writer:
        for im in ims:
            writer.append_data(im)

def main(args):

    file_recon = args.file
    #os.path.join(args.directory, "Exam2200_Series5_12accel.im")
    animate_flag = args.animate
    #file_gt = os.path.join(args.directory, "im.dl.gt")
    plot_xt = args.xt
    phase = args.phase      #Hack code to manually pick the phase to plot
    slice = args.slice
    outfile = args.outfile
    vscale = args.scale
    cmap = args.cmap
    colorbar=args.cbar

    image = cfl.readcfl(file_recon)
    image = np.abs(image)
    image = (image - np.mean(image))/np.std(image)

    if args.dfile:
        file_compare = args.dfile
        image_compare = cfl.readcfl(file_compare)
        image_compare = np.abs(image_compare)
        image_compare = (image_compare - np.mean(image_compare))/np.std(image_compare)
    else: 
        image_compare = np.zeros(np.shape(image))

    image = np.abs(image-image_compare)

    #print(np.mean(image))
    #print(np.std(image))

    nx, ny, nslice, nmap, nphase, d1, d2, d3 = image.shape

    if plot_xt == 1:
        image = np.transpose(image, (0,4,2,3,1,5,6,7))
    if plot_xt == 2:
        image = np.transpose(image, (4,1,2,3,0,5,6,7))

    matplot(image, animate_flag, phase, slice, outfile, vscale, cmap, colorbar)


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Plotting and Analysis Code")
    parser.add_argument('--file', type=str, required=True, help='file name')
    parser.add_argument('--animate', type=int, default=1, help='1 for animate, 0 to click to draw frame')
    parser.add_argument('--xt', type=int, default=0, help= '0:normal plot 1:plot x-t space, 2:plot y-t space')
    parser.add_argument('--phase', type=int, default=-1, help='Pick the phase you want to plot')
    parser.add_argument('--slice', type=int, default=-1, help='Pick the image slice you want to plot')
    parser.add_argument('--outfile', type=str, default='', help='output save file for animations')
    parser.add_argument('--dfile', type=str, default = '', help='Comparison file, usually R=1, to calculate error')
    parser.add_argument('--scale', nargs=2, type=float, required=False, help='Image intensity scaling')
    parser.add_argument('--cmap', type=str, default='gray', help='Colormap of plot')
    parser.add_argument('--cbar', type=bool, default=False, help='plot colorbar')

    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)
