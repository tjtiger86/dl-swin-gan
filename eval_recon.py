# Script to load pickle data and compare SE vs ResNet DL reconstructions

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
#import cv2
import os
import pdb

import sys
import glob
import time
from dl_cs.fileio import cfl

from display_data import matplot
from tabulate import tabulate

import argparse

def normalize(image):
    return (image - np.mean(image))/np.std(image)

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

def getMask(im):

    mask = np.zeros(np.shape(im))
    #cv2.imshow("image", im)
    #cv2.namedWindow("image",2)
    #cv2.resizeWindow("image", mag*x, mag*y)
    #roi = cv2.selectROI("image", im+0.5, False)
    #mask[roi[1]:(roi[1]+roi[3]),roi[0]:(roi[0]+roi[2])] = 1
    return mask
    #return np.ma.make_mask(mask)

def mean_roi(image, mask):

    image = np.array(image)
    bool_index = np.array(mask == 1, dtype = bool)

    #Note, this is a small hack code right now to get iamge and mask dimension order to
    bool_index = np.moveaxis(bool_index, 2, 0)
    #Get the mean value within the ROI
    return np.mean(image[bool_index])


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

def plot_ssim_rmse(data):

    exam = input("Exam desired:")
    series = input("Series desired:")
    accel = input("Acceleration desired:")
    name = "Exam{}_Series{}".format(exam, series)

    ind = next((i for i, item in enumerate(data) if item['name'] == name and int(item['accel']) == int(accel)), None)

    if not ind:
        retry = input("Combination of Exam{}, Series{}, and Acceleration{} not found\nTry again? y(yes), n(no) ".format(exam, series, accel))
        if retry == 'y':
            plot_ssim_rmse(data)
    else:
        slice = input("Slice desired:")
        phase = input("Phase desired:")

        image = data[ind]['rmse_image']
        image = np.transpose(image, axes=[1,2,0])
        nx, ny, nz = np.shape(image)
        print(nx, ny, nz)

        image = np.reshape(image, (nx, ny, int(nz/20), 1, 20, 1, 1, 1))
        animate_flag = 1
        matplot(image, 1, int(phase), int(slice), "test.mp4")


def main(args):

    res_file = args.res_file
    se_file = args.se_file
    im_dir = args.im_dir
    plot_error = args.plot_error
    mname1 = args.mname1
    mname2 = args.mname2

    if(im_dir is None):
        do_segment = 0
    else:
        do_segment = 1

    res_metric = openpickle(res_file)
    se_metric = openpickle(se_file)

    # Should do error checking that res and SE are the same size...
    N = np.size(res_metric)

    #Extract data into a pandas dataframe
    df = pd.DataFrame(columns = ['Model', 'Accel', 'RMSE', 'SSIM', 'RMSE_ROI', 'SSIM_ROI'])

    ########### For e-poster only ##################
    df2 = pd.DataFrame(columns = ['Model', 'Accel', 'Name', 'RMSE', 'SSIM', 'RMSE_ROI', 'SSIM_ROI'])

    #Segmentation
    if do_segment == 1:
        #Segmenting loop - only segment acceleration equaling 1
        for idx in range(N):

            curr_accel = res_metric[idx]['accel']
            curr_name = res_metric[idx]['name']

            if int(curr_accel) == 1:
            #Segmenting If required
                print("curr accel is {}".format(curr_accel))
                im_file = res_metric[idx]['name']+'_'+res_metric[idx]['accel']+'accel.im'
                im_path = os.path.join(im_dir, im_file)
                image = cfl.readcfl(im_path)
                mask = mask_series(image)
                res_metric[idx]['mask'] = mask
                se_metric[idx]['mask'] = mask


        #Segmenting loop - Copy acceleration 1 segmentations
        for idx in range(N):

            curr_accel = res_metric[idx]['accel']
            curr_name = res_metric[idx]['name']

            if int(curr_accel) != 1:

                ind_1accel = next((i for i, item in enumerate(res_metric) if item['name'] == curr_name and int(item['accel']) == 1), None)
                print("curr name is {}, curr index is {}, acceleration of one index is {}".format(curr_name, idx, ind_1accel))
                print(N)
                res_metric[idx]['mask'] = res_metric[ind_1accel]['mask']
                se_metric[idx]['mask'] = se_metric[ind_1accel]['mask']


        #Save segementation
        writepickle(res_file, res_metric)
        writepickle(se_file, se_metric)

    #Analysis with Box-Whisker
    for idx in range(N):
        curr_accel = res_metric[idx]['accel']

        if int(curr_accel) != 1:
            r_mean_ssim = np.mean(res_metric[idx]['ssim'])
            r_mean_rmse = np.mean(res_metric[idx]['rmse'])

            s_mean_ssim = np.mean(se_metric[idx]['ssim'])
            s_mean_rmse = np.mean(se_metric[idx]['rmse'])

            r_mean_ssim_roi = mean_roi(res_metric[idx]['ssim_image'], res_metric[idx]['mask'])
            r_mean_rmse_roi = mean_roi(res_metric[idx]['rmse_image'], res_metric[idx]['mask'])

            s_mean_ssim_roi = mean_roi(se_metric[idx]['ssim_image'], se_metric[idx]['mask'])
            s_mean_rmse_roi = mean_roi(se_metric[idx]['rmse_image'], se_metric[idx]['mask'])

            # Each loop goes through both resnet and se
            # df.loc[str(2*idx)] = ['RES', curr_accel, r_mean_rmse, r_mean_ssim, r_mean_rmse_roi, r_mean_ssim_roi]
            # df.loc[str(2*idx + 1)] = ['SE', curr_accel, s_mean_rmse, s_mean_ssim, s_mean_rmse_roi, s_mean_ssim_roi]

            df.loc[str(2*idx)] = [mname1, curr_accel, r_mean_rmse, r_mean_ssim, r_mean_rmse_roi, r_mean_ssim_roi]
            df.loc[str(2*idx + 1)] = [mname2, curr_accel, s_mean_rmse, s_mean_ssim, s_mean_rmse_roi, s_mean_ssim_roi]

            df = df.sort_values(by=['Accel'])

    ########## - e-poster only - Listing values only for certain acceleration ##########

    for idx in range(N):
        curr_accel = res_metric[idx]['accel']
        curr_name = res_metric[idx]['name']

        if int(curr_accel) == 12:
            r_mean_ssim = np.mean(res_metric[idx]['ssim'])
            r_mean_rmse = np.mean(res_metric[idx]['rmse'])

            s_mean_ssim = np.mean(se_metric[idx]['ssim'])
            s_mean_rmse = np.mean(se_metric[idx]['rmse'])

            r_mean_ssim_roi = mean_roi(res_metric[idx]['ssim_image'], res_metric[idx]['mask'])
            r_mean_rmse_roi = mean_roi(res_metric[idx]['rmse_image'], res_metric[idx]['mask'])

            s_mean_ssim_roi = mean_roi(se_metric[idx]['ssim_image'], se_metric[idx]['mask'])
            s_mean_rmse_roi = mean_roi(se_metric[idx]['rmse_image'], se_metric[idx]['mask'])

            # Each loop goes through both resnet and se
            df2.loc[str(2*idx)] = [mname1, curr_accel, curr_name, r_mean_rmse, r_mean_ssim, r_mean_rmse_roi, r_mean_ssim_roi]
            df2.loc[str(2*idx + 1)] = [mname2, curr_accel, curr_name, s_mean_rmse, s_mean_ssim, s_mean_rmse_roi, s_mean_ssim_roi]

            df2 = df2.sort_values(by=['Name'])

    ########## - e-poster only - Listing values only for certain acceleration ##########


    #pdb. set_trace()

    #########################Just for abstract!!! ###############################

    LAname = 'Exam2200_Series8'
    SAname = 'Exam5050_Series12'
    LAaccel = [10, 12, 14]
    SAaccel = [12, 14, 16]

    for idx in range(N):

        curr_accel = res_metric[idx]['accel']
        curr_name = res_metric[idx]['name']

        if int(curr_accel) in LAaccel and curr_name == LAname:
            print("Long Axis: curr name is {},  acceleration is {}\n".format(curr_name, curr_accel))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname1, res_metric[idx]['ssim'][19], res_metric[idx]['rmse'][19]))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname2, se_metric[idx]['ssim'][19], se_metric[idx]['rmse'][19]))

        if int(curr_accel) in SAaccel and curr_name == SAname:
            print("Short Axis: curr name is {},  acceleration is {}\n".format(curr_name, curr_accel))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname1, res_metric[idx]['ssim'][54], res_metric[idx]['rmse'][54]))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname2, se_metric[idx]['ssim'][54], se_metric[idx]['rmse'][54]))

    #########################Just for abstract!!! ###############################


    #########################Just for e-poster!!! ###############################

    #for idx in range(N):
    #    print("Acceleration: {}", df2.loc[str(2*idx)]['Accel'])
    #    print("Name: {}", df2.loc[str(2*idx)]['Name'])

    print(tabulate(df2, headers = 'keys', tablefmt = 'psql'))

    #########################Just for e-poster!!! ###############################

    for idx in range(N):

        curr_accel = res_metric[idx]['accel']
        curr_name = res_metric[idx]['name']

        if int(curr_accel) in LAaccel and curr_name == LAname:
            print("Long Axis: curr name is {},  acceleration is {}\n".format(curr_name, curr_accel))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname1, res_metric[idx]['ssim'][19], res_metric[idx]['rmse'][19]))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname2, se_metric[idx]['ssim'][19], se_metric[idx]['rmse'][19]))

        if int(curr_accel) in SAaccel and curr_name == SAname:
            print("Short Axis: curr name is {},  acceleration is {}\n".format(curr_name, curr_accel))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname1, res_metric[idx]['ssim'][54], res_metric[idx]['rmse'][54]))
            print("{} --- SSIM: {}, RMSE: {}\n".format(mname2, se_metric[idx]['ssim'][54], se_metric[idx]['rmse'][54]))


    #Plotting the ssim and mse
    if plot_error:
        plot_ssim_rmse(res_metric)
        plot_ssim_rmse(se_metric)

    #plot ssim and rmse by accleration factor
    sns.set_style("darkgrid")
    
    fig, axs = plt.subplots(1,2,constrained_layout=True, figsize=(14,8))
    ax1 = sns.boxplot(x = df['Accel'], y = df['SSIM'], hue = df['Model'], ax=axs[0], showfliers = False)
    axs[0].set_title('SSIM', size=22)
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel('',size=1)
    ax1.set_xlabel('Acceleration',size=16)
    ax1.legend(title="")
    
    ax2 = sns.boxplot(x = df['Accel'], y = df['RMSE'], hue = df['Model'], ax=axs[1], showfliers = False)
    axs[1].set_title('RMSE', size=22)
    ax2.tick_params(labelsize=16)
    ax2.set_ylabel('',size=1)
    ax2.set_xlabel('Acceleration',size=16)
    ax2.legend(title="")
    
    plt.setp(ax1.get_legend().get_texts(), fontsize='14') 
    plt.setp(ax2.get_legend().get_texts(), fontsize='14')

    #plt.plot(df["Accel"], df["SSIM"])
    plt.show()

    fig, axs = plt.subplots(1,2,constrained_layout=True, figsize=(14,8))
    ax1 = sns.boxplot(x = df['Accel'], y = df['SSIM_ROI'], hue = df['Model'], ax=axs[0], showfliers = False)
    axs[0].set_title('SSIM Heart ROI ', fontsize=22)
    ax1.tick_params(labelsize=16)
    ax1.set_ylabel('',size=1)
    ax1.set_xlabel('Acceleration',size=16)
    ax1.legend(title="")

    
    ax2 = sns.boxplot(x = df['Accel'], y = df['RMSE_ROI'], hue = df['Model'], ax=axs[1], showfliers = False)
    axs[1].set_title('RMSE Heart ROI', fontsize=18)
    ax2.tick_params(labelsize=16)
    ax2.set_ylabel('',size=1)
    ax2.set_xlabel('Acceleration',size=16)
    ax2.legend(title="")

    plt.setp(ax1.get_legend().get_texts(), fontsize='12') 
    plt.setp(ax2.get_legend().get_texts(), fontsize='12')

    #plt.plot(df["Accel"], df["SSIM"])
    plt.show()

    #save results in output .csv file?

    #display data for side by side comparison of SE and Resnet


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Reconstruction Evaluation")
    parser.add_argument('--res-file', type=str, required=True, help='Calculated ResNet ssim/rmse pickle file')
    parser.add_argument('--se-file', type=str,required=True, help='Calculated SE ssim/rmse pickle file')
    #parser.add_argument('--res-dir', type=str, required=True, help='Directory of reconstructed ResNet image data' )
    #parser.add_argument('--se-dir', type=str, required=True, help='Directory of reconstructed SE image data' )
    parser.add_argument('--plot-flag', type=int, default=0, help='Plotting flag' )
    parser.add_argument('--im-dir', type=str, help='if segmenting, requires this folder')
    parser.add_argument('--plot-error', type=int, default=0, help='1 to plot ssim/mse')
    parser.add_argument('--mname1', type=str, default='RES', help='name of model 1')
    parser.add_argument('--mname2', type=str, default='SE', help='name of model 2')
    return parser

if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])

    main(args)
