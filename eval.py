import numpy as np
import os
from pathlib import Path
import sys
import glob
import time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim_calc

import pickle
#import h5py

import argparse
from dl_cs.fileio import cfl

def normalize(image):
    return (image - np.mean(image))/np.std(image)

def rmse_calc(image1, image2):
    S = np.sqrt((image1-image2)**2)
    return S.mean(), S

def ssim_mse_calc(image1, image2, ksize, win_std, overlap_flag):

    nx, ny = np.shape(image1)
    data_range = image1.max()-image1.min()
    ssim_im = np.empty([nx-ksize[0], ny-ksize[1]])
    mse_im = np.empty([nx-ksize[0], ny-ksize[1]])

    start = time.time()
    ssim, ssim_im = ssim_calc(image2, image1, data_range=data_range, full=True)
    rmse, rmse_im = rmse_calc(image1, image2)
    end = time.time()

    #print("Elapsed time Calculation of MSE and SSIM:{}".format(end-start))

    return ssim_im, ssim, rmse_im, rmse

def fname_accel(file_name):

    parts = file_name.split("_")
    ndig = len(parts[2])-12 #number of digits for acceleration factor, 12 is due to extension

    return parts[0]+"_"+parts[1], parts[2][0:ndig]


def main(args):

    #Argument Parsing
    data_dir = args.data_dir
    out_dir = args.out_dir
    k_size = [args.ksize, args.ksize]
    plot_flag = args.plot_flag

    out_name = Path(data_dir).stem
    out_file = os.path.join(out_dir, out_name + '.pkl')

    print(out_file)

    win_std = np.max(k_size)/2
    files = glob.glob(os.path.join(data_dir, "*.im.cfl"))

    overlap_flag = 1

    data_out = []

    if plot_flag == 1:

        plt.ion()
        fig, axs = plt.subplots(2,2, constrained_layout=True)

        # Initialize Plots
        im0 = axs[0,0].imshow(np.squeeze(np.zeros((100,100))), cmap='gray', vmin =-.5, vmax = 3)
        axs[0,0].axis("off")

        im1 = axs[1,0].imshow(np.squeeze(np.zeros((100,100))), cmap='gray', vmin =-.5, vmax = 3)
        axs[1,0].axis("off")
        axs[1,0].set_title('Fully Sampled')

        im2 = axs[0,1].imshow(np.squeeze(np.zeros((100,100))), cmap='inferno', vmin = 0, vmax = 1)
        axs[0,1].axis("off")
        axs[0,1].set_title('SSIM')
        cax2 = axs[0,1].inset_axes([1.04, 0, 0.05, 1], transform=axs[0,1].transAxes)
        fig.colorbar(im2, ax = axs[0,1], cax = cax2)

        im3 = axs[1,1].imshow(np.squeeze(np.zeros((100,100))), cmap='inferno', vmin = 0, vmax = 1)
        axs[1,1].axis("off")
        axs[1,1].set_title('RMSE')
        cax3 = axs[1,1].inset_axes([1.04, 0, 0.05, 1], transform=axs[1,1].transAxes)
        fig.colorbar(im3, ax = axs[1,1], cax = cax3)


    for idx, file in enumerate(files):

        fname = os.path.basename(file)
        exam_name, accel = fname_accel(fname)

        data_out.append({})
        data_out[idx]['name'] = exam_name
        data_out[idx]['accel'] = accel

        ssim_im_out = []
        rmse_im_out = []
        ssim = []
        rmse = []

        # If acceleration not 1, will perform full ssim_calculation
        if int(accel) != 1 and int(accel) != 32:

            print("Analyzing {}".format(fname))

            image = cfl.readcfl(os.path.splitext(file)[0])             #remove .cfl for cfl.readcfl
            image_fs = cfl.readcfl(os.path.join(data_dir, fname.split("_")[0]+"_"+fname.split("_")[1]+"_1accel.im"))

            image = normalize(np.abs(image))
            image_fs = normalize(np.abs(image_fs))

            nx, ny, nslice, nmap, nphase, d1, d2, d3 = image.shape

            for jj in range(nslice):
                for kk in range(nphase):

                    ssim_im_curr, ssim_curr, rmse_im_curr, rmse_curr = ssim_mse_calc(np.squeeze(image[:,:,jj,0,kk]), np.squeeze(image_fs[:,:,jj,0,kk]), k_size, win_std, overlap_flag)

                    ssim_im_out.append(ssim_im_curr)
                    rmse_im_out.append(rmse_im_curr)
                    ssim.append(ssim_curr)
                    rmse.append(rmse_curr)

                    if plot_flag == 1:

                        #Update Plot contents

                        im0.set_data(np.squeeze(image[:,:,jj,0,kk]))
                        axs[0,0].set_title('Reconstructed (accel {})'.format(accel))
                        im1.set_data(np.squeeze(image_fs[:,:,jj,0,kk]))
                        im2.set_data(np.squeeze(ssim_im_curr))
                        im3.set_data(np.squeeze(rmse_im_curr))

                        fig.suptitle("Slice {}/{} and Phase {}/{}".format(jj+1,nslice,kk+1,nphase))

                        fig.canvas.draw()
                        fig.canvas.flush_events()
                        #time.sleep(0.1)

            data_out[idx]['ssim_image'] = ssim_im_out
            data_out[idx]['rmse_image'] = rmse_im_out
            data_out[idx]['ssim'] = ssim
            data_out[idx]['rmse'] = rmse

        # If acceleration equals 1, ssim and rmse not too relevant...
        else:
            image = cfl.readcfl(os.path.splitext(file)[0])             #remove .cfl for cfl.readcfl
            image = normalize(np.abs(image))
            data_out[idx]['ssim_image'] = 1
            data_out[idx]['rmse_image'] = 0
            data_out[idx]['ssim'] = 1
            data_out[idx]['rmse'] = 0
            # data_out[idx]['image'] = image

    #Writing output to pickle file
    f = open(out_file, 'wb')
    pickle.dump(data_out,f)
    f.close()

def create_arg_parser():
    parser = argparse.ArgumentParser(description="Reconstruction Evaluation")
    parser.add_argument('--data-dir', type=str, required=True, help='Directory of reconstructed images')
    parser.add_argument('--ksize', default=5, type=int, help='Kernel Size')
    parser.add_argument('--out-dir', type=str, required=True, help='Directory of saved analysis' )
    parser.add_argument('--plot-flag', type=int, default=0, help='Plotting flag' )
    return parser

if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)
