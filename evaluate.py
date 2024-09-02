import numpy as np
import os
from pathlib import Path
import sys
import glob
import time
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim2

import h5py

import multiprocessing
from scipy.signal.windows import gaussian

import argparse
from dl_cs.fileio import cfl

def gaus_2d(ksize, std_dev):

    w1 = np.atleast_2d(gaussian(ksize[0], std_dev))
    w2 = np.atleast_2d(gaussian(ksize[1], std_dev)).T

    w = w2@w1
    return w/np.max(w)

def ssim_no_loop(im1, im2, weight, L):

    ax = 0
    N = im1.shape[ax]
    im1 = im1*weight
    im2 = im2*weight

    u1 = np.mean(im1, axis=ax)
    u2 = np.mean(im2, axis=ax)

    #u1 = np.sum(im1, axis=ax)/N
    #u2 = np.sum(im2, axis=ax)/N

    #Covariance and Variance Calculation
    var1 = np.einsum('ij,ij->j', im1-u1, im1-u1)/N
    var2 = np.einsum('ij,ij->j', im2-u1, im2-u1)/N
    cov = np.einsum('ij,ij->j', im1-u1, im2-u2)/N

    c1 = (0.3*L)**2
    c2 = (0.1*L)**2

    return ((2*u1*u2+c1)*(2*cov+c2))/((u1**2+u2**2+c1)*(var1+var2+c2))

def ssim(im1, im2, weight, L):

    #weight = np.ones(np.size(im1))
    #im1 = weight.ravel()*im1.ravel()
    #im2 = weight.ravel()*im2.ravel()

    im1 = weight*im1
    im2 = weight*im2

    u1 = np.mean(im1)
    u2 = np.mean(im2)

    cov = np.cov(im1,im2)   #Note df = 0

    c1 = (0.3*L)**2
    c2 = (0.1*L)**2

    return ((2*u1*u2+c1)*(2*cov[0,1]+c2))/((u1**2+u2**2+c1)*(cov[0,0]+cov[1,1]+c2))

def mse(im1, im2, weight):

    #im1 = weight.ravel()*im1.ravel()
    #im2 = weight.ravel()*im2.ravel()

    return np.sqrt(np.mean((im1-im2)**2))

def ssim_mse_calc(image1, image2, ksize, win_std, overlap_flag):

    nx, ny = np.shape(image1)
    w = gaus_2d(ksize, win_std)
    data_range = image1.max()-image1.min()
    ssim_im = np.empty([nx-ksize[0], ny-ksize[1]])
    mse_im = np.empty([nx-ksize[0], ny-ksize[1]])

    start2 = time.time()

    for kk in range(nx-ksize[0]):
        for jj in range(ny-ksize[1]):
            ind = (slice(kk,kk+ksize[0]), slice(jj,jj+ksize[1]))

            start = time.time()
            ssim_im[kk,jj] = ssim(image1[ind].ravel(), image2[ind].ravel(), w.ravel(), data_range)
            end = time.time()

            mse_im[kk,jj] = mse(image1[ind], image2[ind], w)

            #print("Each Loop Time {} for total of {} loops".format(end-start, (nx-ksize[0])*(ny-ksize[1])))

    end2 = time.time()

    #mse_im = abs(image1-image2)/image2

    im1_stack = np.empty((ksize[0]*ksize[1], (nx-ksize[0])*(ny-ksize[1])), dtype=float)
    im2_stack = np.empty((ksize[0]*ksize[1], (nx-ksize[0])*(ny-ksize[1])), dtype=float)
    w_stack = np.tile(np.ravel(w)[...,None], (1, (nx-ksize[0])*(ny-ksize[1])))

    loop_ind = 0

    start3 = time.time()
    for kk in range(nx-ksize[0]):
        for jj in range(ny-ksize[1]):
            ind = (slice(kk,kk+ksize[0]), slice(jj,jj+ksize[1]))
            im1_stack[:,loop_ind] = np.ravel(image1[ind])
            im2_stack[:,loop_ind] = np.ravel(image2[ind])
            loop_ind += 1

    ssim_im_v2 = ssim_no_loop(im1_stack, im2_stack, w_stack, data_range)
    ssim_im_v2 = np.reshape(ssim_im_v2, (nx-ksize[0], ny-ksize[1]))
    end3 = time.time()

    start4 = time.time()
    a, ssim_im_v3 = ssim2(image2, image1, data_range=data_range, full=True)
    end4 = time.time()

    print("im1_stack shape: {}, im2_stack shape: {} w shape: {}, ssim_im_v2 shape: {}".format(np.shape(im1_stack), np.shape(im2_stack), np.shape(w_stack), np.shape(ssim_im_v2)))
    print("Elapsed time of SSIM_calc:{}".format(end2-start2))
    print("Elapsed time of no SSIM_calc:{}".format(end3-start3))
    print("Elapsed time of no SSIM_ skimage:{}".format(end4-start4))

    return ssim_im, mse_im, ssim_im_v2, ssim_im_v3

def fname_accel(file_name):

    parts = file_name.split("_")
    ndig = len(parts[2])-12 #number of digits for acceleration factor, 12 is due to extension

    return parts[0]+"_"+parts[1], parts[2][0:ndig]

def normalize(image):
    return (image - np.mean(image))/np.std(image)

def main(args):

    #Argument Parsing
    data_dir = args.data_dir
    out_dir = args.out_dir
    k_size = [args.ksize, args.ksize]
    plot_flag = args.plot_flag

    out_name = Path(data_dir).stem
    out_file = os.path.join(out_dir, out_name, '.h5')

    print(out_name)

    win_std = np.max(k_size)/2
    files = glob.glob(os.path.join(data_dir, "*.im.cfl"))

    overlap_flag = 1

    data_out = []

    for idx, file in enumerate(files):

        fname = os.path.basename(file)
        exam_name, accel = fname_accel(fname)

        data_out.append({})
        data_out[idx]['name'] = exam_name
        data_out[idx]['accel'] = accel

        ssim_out = []
        mse_out = []

        if accel != 1:

            print("Analyzing {}".format(fname))

            image = cfl.readcfl(os.path.splitext(file)[0])             #remove .cfl for cfl.readcfl
            image_fs = cfl.readcfl(os.path.join(data_dir, fname.split("_")[0]+"_"+fname.split("_")[1]+"_1accel.im"))

            image = normalize(np.abs(image))
            image_fs = normalize(np.abs(image_fs))

            nx, ny, nslice, nmap, nphase, d1, d2, d3 = image.shape

            #if plot_flag == 1:
                #plt.ion()
                #fig, axs = plt.subplots(2,2, constrained_layout=True)

            for jj in range(nslice):
                for kk in range(nphase):

                    ssim_curr, mse_curr, ssim_v2, ssim_v3 = ssim_mse_calc(np.squeeze(image[:,:,jj,0,kk]), np.squeeze(image_fs[:,:,jj,0,kk]), k_size, win_std, overlap_flag)

                    ssim_out.append(ssim_curr)
                    mse_out.append(mse_curr)

                    if plot_flag == 1:

                        fig, axs = plt.subplots(3,2, constrained_layout=True)

                        # Plotting to visualize
                        im0 = axs[0,0].imshow(np.squeeze(image[:,:,jj,0,kk]), cmap='gray', vmin =-.5, vmax = 3)
                        axs[0,0].axis("off")
                        axs[0,0].set_title('Reconstructed (accel {})'.format(accel))

                        im1 = axs[1,0].imshow(np.squeeze(image_fs[:,:,jj,0,kk]), cmap='gray', vmin =-.5, vmax = 3)
                        axs[1,0].axis("off")
                        axs[1,0].set_title('Fully Sampled')

                        im2 = axs[0,1].imshow(np.squeeze(ssim_curr), cmap='inferno', vmin = 0, vmax = 1)
                        axs[0,1].axis("off")
                        axs[0,1].set_title('SSIM')
                        cax2 = axs[0,1].inset_axes([1.04, 0, 0.05, 1], transform=axs[0,1].transAxes)
                        fig.colorbar(im2, ax = axs[0,1], cax = cax2)

                        #im3 = axs[1,1].imshow(np.squeeze(mse_curr), cmap='inferno', vmin = 0, vmax = .5)
                        im3 = axs[1,1].imshow(np.squeeze(ssim_v2), cmap='inferno', vmin = 0, vmax = 1)
                        axs[1,1].axis("off")
                        axs[1,1].set_title('SSIM no loop')
                        cax3 = axs[1,1].inset_axes([1.04, 0, 0.05, 1], transform=axs[1,1].transAxes)
                        fig.colorbar(im3, ax = axs[1,1], cax = cax3)

                        #im3 = axs[1,1].imshow(np.squeeze(mse_curr), cmap='inferno', vmin = 0, vmax = .5)
                        im3 = axs[2,1].imshow(np.squeeze(ssim_v3), cmap='inferno', vmin = 0, vmax = 1)
                        axs[2,1].axis("off")
                        axs[2,1].set_title('SSIM skimage')
                        cax4 = axs[2,1].inset_axes([1.04, 0, 0.05, 1], transform=axs[2,1].transAxes)
                        fig.colorbar(im3, ax = axs[2,1], cax = cax4)

                        fig.suptitle("Slice {}/{} and Phase {}/{}".format(jj+1,nslice,kk+1,nphase))
                        #fig.canvas.draw()
                        #fig.canvas.flush_events()
                        plt.show()
                        #time.sleep(0.1)
                        #plt.clf()

            data_out[idx]['ssim_image'] = ssim_out
            data_out[idx]['mse_image'] = mse_out

    hf = h5py.File(out_file, 'w')
    hf.create_dataset('results', data=data_out)
    hf.close()

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
