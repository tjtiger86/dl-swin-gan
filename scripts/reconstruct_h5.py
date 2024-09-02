"""
Inference script for MRI unrolled recon.
"""
#Import Plotting
#import matplotlib
#matplotlib.use('TKagg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import sys
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm
import glob
import h5py
import random

import torch
from torch import nn
from torch.utils.data import (Dataset, DataLoader)

# PyTorch lightning modules
import pytorch_lightning as pl

# Custom dl_cs modules
from dl_cs.config import load_cfg
from dl_cs.fileio import cfl
from dl_cs.mri import utils
from dl_cs.mri import transforms as T
from dl_cs.models import unrolledSE
from dl_cs.models import unrolledCBAM
from dl_cs.models import unrolled
from dl_cs.models import unrolledswin
from dl_cs.mri import subsample as ss
# from dl_cs.data.dataset import Hdf5Dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.enabled = True


class LitUnrolledResNet(pl.LightningModule):
    """
    Unrolled model inside a PyTorch Lightning module.
    """
    def __init__(self, config, data_parallel=False):
        super().__init__()

        self.config = config
        self.data_parallel = data_parallel

        if config.MODEL.META_ARCHITECTURE == 'dlespirit':
                self.model = unrolled.ProximalGradientDescent(config)
        elif config.MODEL.META_ARCHITECTURE == 'modl':
                self.model = unrolled.HalfQuadraticSplitting(config)
        else:
            raise ValueError('Meta architecture in config file not recognized!')

    def forward(self, kspace, mask, maps, initial_guess):
        if self.data_parallel:
            # Necessary because nn.DataParallel converts tensors to float32 (why..?)
            kspace = torch.view_as_complex(kspace)
            maps = torch.view_as_complex(maps)
            initial_guess = torch.view_as_complex(initial_guess)

        return self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)

class LitUnrolledSE(pl.LightningModule):
    """
    Unrolled model inside a PyTorch Lightning module.
    """
    def __init__(self, config, data_parallel=False):
        super().__init__()

        self.config = config
        self.data_parallel = data_parallel

        if config.MODEL.META_ARCHITECTURE == 'dlespirit':
                self.model = unrolledSE.ProximalGradientDescent(config)
        elif config.MODEL.META_ARCHITECTURE == 'modl':
                self.model = unrolledSE.HalfQuadraticSplitting(config)
        else:
            raise ValueError('Meta architecture in config file not recognized!')

    def forward(self, kspace, mask, maps, initial_guess):
        if self.data_parallel:
            # Necessary because nn.DataParallel converts tensors to float32 (why..?)
            kspace = torch.view_as_complex(kspace)
            maps = torch.view_as_complex(maps)
            initial_guess = torch.view_as_complex(initial_guess)

        return self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)

class LitUnrolledSWIN(pl.LightningModule):
    """
    Unrolled model inside a PyTorch Lightning module.
    """
    def __init__(self, config, data_parallel=False):
        super().__init__()

        self.config = config
        self.data_parallel = data_parallel

        if config.MODEL.META_ARCHITECTURE == 'dlespirit':
                self.model = unrolledswin.ProximalGradientDescent(config)
        elif config.MODEL.META_ARCHITECTURE == 'modl':
                self.model = unrolledswin.HalfQuadraticSplitting(config)
        else:
            raise ValueError('Meta architecture in config file not recognized!')

    def forward(self, kspace, mask, maps, initial_guess):
        if self.data_parallel:
            # Necessary because nn.DataParallel converts tensors to float32 (why..?)
            kspace = torch.view_as_complex(kspace)
            maps = torch.view_as_complex(maps)
            initial_guess = torch.view_as_complex(initial_guess)

        return self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)


class LitUnrolledCBAM(pl.LightningModule):
    """
    Unrolled model inside a PyTorch Lightning module.
    """
    def __init__(self, config, data_parallel=False):
        super().__init__()

        self.config = config
        self.data_parallel = data_parallel

        if config.MODEL.META_ARCHITECTURE == 'dlespirit':
                self.model = unrolledCBAM.ProximalGradientDescent(config)
        elif config.MODEL.META_ARCHITECTURE == 'modl':
                self.model = unrolledCBAM.HalfQuadraticSplitting(config)
        else:
            raise ValueError('Meta architecture in config file not recognized!')

    def forward(self, kspace, mask, maps, initial_guess):
        if self.data_parallel:
            # Necessary because nn.DataParallel converts tensors to float32 (why..?)
            kspace = torch.view_as_complex(kspace)
            maps = torch.view_as_complex(maps)
            initial_guess = torch.view_as_complex(initial_guess)

        return self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)

'''
kspace, mask, maps, initial_guess, scale, target = batch
pred = self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)
'''


class Hdf5Dataset(Dataset):
    """
    A generic PyTorch Dataset class that provides access to 2D MRI data
    contained in an HDF5 file.

    Each HDF5 file should correspond to a single patient. It contains three keys:
    - 'kspace': np.array with dimensions [slices, kx, ky, ..., coils]
    - 'maps': np.array with dimensions [slices, x, y, ..., coils]
    - 'target': np.array with dimensions [slices, x, y, ...]
    """
    def __init__(self, files, transform, sample_rate=1.0):
        self.transform = transform
        self.examples = []
        #files = glob.glob(os.path.join(root_directory, '*.h5'))

        if sample_rate < 1.0:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        #for filename in sorted(files):
        filename = files
        kspace = h5py.File(filename, 'r')['kspace']
        num_slices = kspace.shape[0]
        self.examples += [(filename, slice) for slice in range(num_slices)]

        # Below is wrong
        maps = h5py.File(filename, 'r')['maps']
        # Get data dimensions
        """
        shape_x = kspace.shape[0]  # kx
        shape_y = kspace.shape[1]  # ky
        num_slices = kspace.shape[2]  # slice
        num_coils = kspace.shape[3]  # coils
        #num_echoes = kspace.shape[5]  # echoes / flow encodes
        #num_phases = kspace.shape[7]  # cardiac phases
        num_emaps = maps.shape[4]
        """

        shape_x = kspace.shape[4]  # kx
        shape_y = kspace.shape[3]  # ky
        num_slices = kspace.shape[0]  # slice
        num_coils = kspace.shape[1]  # coils
        #num_echoes = kspace.shape[5]  # echoes / flow encodes
        num_phases = kspace.shape[2]  # cardiac phases
        num_emaps = maps.shape[1]

        # print("shape_x is %d shape_y is %d, num_slices is %d, num_coils is %d, num_phases is %d num_emaps is %d " %(shape_x,shape_y, num_slices, num_coils, num_phases, num_emaps))

        #self.image_dims = (num_slices, num_echoes, num_emaps, num_phases, shape_y, shape_x)
        self.image_dims = (num_slices, num_emaps, num_phases, shape_y, shape_x)

    def write(self, file_im, images):
        """
        Write out CFL file with reconstructed images.
        """

        def write_im(name, array, order='C'):
           h = open(name + ".hdr", "w")
           h.write('# Dimensions\n')
           if order=='C':
               for i in array.shape[::-1]:
                   h.write("%d " % i)
           else:
               for i in (array.shape):
                   h.write("%d " % i)
           h.write('\n')
           h.close()

           d = open(name + ".cfl", "w")
           if order=='C':
               array.astype(np.complex64).tofile(d)
           else:
               # tranpose for column-major order
               array.T.astype(np.complex64).tofile(d)
           d.close()

        images = torch.cat(images, dim=0).numpy()
        images = np.reshape(images, self.image_dims)
        images = np.transpose(images, (4, 3, 0, 1, 2))  # [x, y, sl, emap, ec, ph]
        images = images[:, :, :, :, :, None, None, None]

        write_im(file_im, images, order='F')


    def __len__(self):
        """
        Returns number of examples.
        """
        return len(self.examples)


    def __getitem__(self, index):
        """
        Generates one sample of data.
        """
        filename, slice = self.examples[index]
        with h5py.File(filename, 'r') as data:
            kspace = data['kspace'][slice]
            maps = data['maps'][slice]
            target = data['target'][slice]

        # return self.transform(kspace, maps, target, filename)
        return self.transform(kspace, maps)


class DataTransform:
    """
    Module for pre-processing cine data for inference.
    """
    def __init__(self, config):
        # Method for initializing network (sliding window)
        self.slwin_init = config.MODEL.PARAMETERS.SLWIN_INIT


    def __call__(self, kspace, maps):
        # Convert everything from NumPy arrays to PyTorch tensors
        kspace = torch.from_numpy(kspace).unsqueeze(0)
        maps = torch.from_numpy(maps).unsqueeze(0)


        # Get the undersampling mask from input k-space
        mask = utils.get_mask(kspace)
        mask = mask[:, 0, None]  # no need to store all coils...

        # Perform fftmod to avoid fftshift before/after each fft
        # kspace = utils.fftmod(kspace)
        # maps = utils.fftmod(maps)

        # Initialize ESPIRiT model
        A = T.SenseModel(maps, weights=None)

        # Compute normalization factor (based on 95% max signal level in view-shared dataa)
        averaged_kspace = utils.time_average(kspace, dim=2)
        image = A(averaged_kspace, adjoint=True)
        magnitude = torch.abs(image).reshape(-1)
        k = int(round(0.05 * magnitude.numel()))
        scale = torch.min(torch.topk(magnitude, k).values)

        # Normalize k-space
        kspace /= scale

        # Compute network initialization
        if self.slwin_init:
            init_kspace = utils.sliding_window(kspace, dim=2, window_size=5)
        else:
            init_kspace = kspace
        init_image = A(init_kspace, adjoint=True)

        # Get rid of batch dimension...
        kspace = kspace.squeeze(0)
        mask = mask.squeeze(0)
        maps = maps.squeeze(0)
        init_image = init_image.squeeze(0)

        return kspace, mask, maps, init_image, scale

class DataTransformSS:
    """
    Module for pre-processing cine data for inference.
    """
    def __init__(self, acceleration, config):
        # Method for initializing network (sliding window)
        self.slwin_init = config.MODEL.PARAMETERS.SLWIN_INIT

        #Taken in to subsample the data
        acceleration_range = (acceleration, acceleration)
        self.mask_func = ss.VDktMaskFunc(acceleration_range,
                                            sim_partial_kx=config.AUG_TRAIN.UNDERSAMPLE.PARTIAL_KX,
                                            sim_partial_ky=config.AUG_TRAIN.UNDERSAMPLE.PARTIAL_KY)

    def __call__(self, kspace, maps):
        # Convert everything from NumPy arrays to PyTorch tensors
        kspace = torch.from_numpy(kspace).unsqueeze(0)
        maps = torch.from_numpy(maps).unsqueeze(0)


        #Taken in to subsample the data
        kspace, mask = ss.subsample(kspace, self.mask_func, seed=1000, mode='3D')


        # Perform fftmod to avoid fftshift before/after each fft
        # kspace = utils.fftmod(kspace)
        # maps = utils.fftmod(maps)

        # Initialize ESPIRiT model
        A = T.SenseModel(maps, weights=None)

        # Compute normalization factor (based on 95% max signal level in view-shared dataa)
        averaged_kspace = utils.time_average(kspace, dim=2)
        image = A(averaged_kspace, adjoint=True)
        magnitude = torch.abs(image).reshape(-1)
        k = int(round(0.05 * magnitude.numel()))
        scale = torch.min(torch.topk(magnitude, k).values)

        # Normalize k-space
        kspace /= scale

        # Compute network initialization
        if self.slwin_init:
            init_kspace = utils.sliding_window(kspace, dim=2, window_size=5)
        else:
            init_kspace = kspace
        init_image = A(init_kspace, adjoint=True)

        # Get rid of batch dimension...
        kspace = kspace.squeeze(0)
        mask = mask.squeeze(0)
        maps = maps.squeeze(0)
        init_image = init_image.squeeze(0)

        return kspace, mask, maps, init_image, scale

def main(args):
    # Get filenames
    data_file = args.file
    directory = args.out_directory
    accel = args.acceleration

    #file_images = os.path.join(directory, str(Path.basename(data_file)) + "_" + str(accel) + "accel.im")   #Reconstructed image file name
    file_images = os.path.join(directory, str(os.path.splitext(os.path.basename(data_file))[0])+"_{}accel.im".format(accel))

    logger.info("Current image: {}".format(data_file))
    # Get device name

    if args.multi_gpu:
        logger.info('Running on multiple GPU devices...')
        num_devices = torch.cuda.device_count()
        logger.info(f'Detected {num_devices} devices!')
        device = torch.device('cuda')
    else:
        num_devices = 1
        if args.device > -1:
            logger.info(f'Running on GPU device #{args.device}...')
            device = torch.device(f'cuda:{args.device}')
        else:
            logger.info(f'Running on CPU...')
            device = torch.device('cpu')

    logger.info(f'Loading model {args.ckpt}...')
    config = load_cfg(args.config_file)

    if args.model == 'RES':
        unrolled = LitUnrolledResNet.load_from_checkpoint(args.ckpt, config=config, data_parallel=args.multi_gpu)
    elif args.model == 'SE':
        unrolled = LitUnrolledSE.load_from_checkpoint(args.ckpt, config=config, data_parallel=args.multi_gpu)
    elif args.model == 'CBAM':
        unrolled = LitUnrolledCBAM.load_from_checkpoint(args.ckpt, config=config, data_parallel=args.multi_gpu)
    elif args.model == 'SWIN':
        unrolled = LitUnrolledSWIN.load_from_checkpoint(args.ckpt, config=config, data_parallel=args.multi_gpu)

    unrolled.freeze()  # similar to running with torch.no_grad()
    if args.multi_gpu:
        unrolled = nn.DataParallel(unrolled).to(device)
    else:
        unrolled = unrolled.to(device)


    logger.info('Loading H5 data...')

    if accel > 1:
    #   Undersampled Data
        eval_data = Hdf5Dataset(files=data_file, transform=DataTransformSS(accel, config))
        data_loader = DataLoader(dataset=eval_data, batch_size=(args.batch_size * num_devices), pin_memory=True)

    else:
        #Fully-sampled Data
        eval_data = Hdf5Dataset(files=data_file, transform=DataTransform(config))
        data_loader = DataLoader(dataset=eval_data, batch_size=(args.batch_size * num_devices), pin_memory=True)

    logger.info('Running inference...')
    pbar = tqdm(total=len(data_loader), desc='DL Recon', unit='slice')

    # Start timer for reconstruction
    start = time.time()

    out = []
    for data in data_loader:
        # Load batch of data
        kspace, mask, maps, init_image, scale = data

        # Move to GPU
        kspace = kspace.to(device)
        mask = mask.to(device)
        maps = maps.to(device)
        init_image = init_image.to(device)

        if accel > 1:

            # Perform inference
            images = unrolled(kspace, mask, maps, init_image)

            if args.multi_gpu:
                # Necessary because nn.DataParallel converts tensors to float32 (why..?)
                images = torch.view_as_complex(images)

            # Re-scale
            out.append(scale[:, None, None, None, None] * images.cpu())

        #No acceleration, just use fully sampled init_image reconstruction
        else:
            out.append(scale[:, None, None, None, None] * init_image.cpu())
        # Update progress bar

        pbar.update(n=args.batch_size)


    # End timer for reconstruction
    end = time.time()
    logger.info(f'Elapsed time (reconstruction): {end - start} s')
    pbar.close()

    logger.info('Writing images to {}'.format(file_images))
    eval_data.write(file_images, out)

    #Small Preview

    '''
    temp = images.cpu()
    temp2 = init_image.cpu()

    plt.subplot(121)
    plt.imshow(np.abs(temp[0,0,0,:,:]), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.abs(temp2[0,0,0,:,:]), cmap="gray")
    plt.axis("off")
    plt.show()
    '''
    return


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    parser.add_argument('--file', type=str, required=True, help='Name of the H5 file with kspace, mask, and map')
    parser.add_argument('--model', type=str, required=True, help='RES (resnet) or SE (squeeze excitation) ')
    parser.add_argument('--acceleration', type=int, default=1, help='Undersampling Accelration Factor')
    parser.add_argument('--out-directory', type=str, required=True, help='Output Directory')
    parser.add_argument('--ckpt', type=str, required=True, help='Model checkpoint file')
    parser.add_argument('--batch-size', type=int, default=1, help='Model checkpoint file')
    parser.add_argument('--config-file', type=str, required=True, help='Training config file (yaml)')
    parser.add_argument('--device', type=int, default=-1, help='GPU device')
    parser.add_argument('--multi-gpu', action='store_true', help='Uses multiple GPUs for inference (overrides device flag)')
    parser.add_argument('--verbose', action='store_true', help='Turn on debug statements')
    return parser


if __name__ == '__main__':
    args = create_arg_parser().parse_args(sys.argv[1:])

    start = time.time()
    main(args)
    end = time.time()

    logger.info('Script complete.')
    logger.info(f'Elapsed time (total): {end-start} s')
