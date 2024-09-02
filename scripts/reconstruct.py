"""
Inference script for MRI unrolled recon.
"""
import os
import sys
import time
import logging
import argparse
import numpy as np
from tqdm import tqdm

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
from dl_cs.models import unrolled

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.backends.cudnn.enabled = True


class LitUnrolled(pl.LightningModule):
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

    def forward(self, kspace, maps, mask, initial_guess):
        if self.data_parallel:
            # Necessary because nn.DataParallel converts tensors to float32 (why..?)
            kspace = torch.view_as_complex(kspace)
            maps = torch.view_as_complex(maps)
            initial_guess = torch.view_as_complex(initial_guess)

        return self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)


class CflDataset(Dataset):
    """
    """
    def __init__(self, file_ks, file_maps, transform):
        self.transform = transform
        self.examples = []

        # Load CFL files into numpy arrays
        kspace = cfl.read(file_ks, order='F')
        maps = cfl.read(file_maps, order='F')

        # Get data dimensions
        shape_x = kspace.shape[0]  # kx
        shape_y = kspace.shape[1]  # ky
        num_slices = kspace.shape[2]  # slice
        num_coils = kspace.shape[3]  # coils
        num_echoes = kspace.shape[5]  # echoes / flow encodes
        num_phases = kspace.shape[7]  # cardiac phases
        num_emaps = maps.shape[4]

        # Dimensions of arrays
        kspace_dims = (shape_x, shape_y, num_slices, num_coils, num_echoes, num_phases)
        map_dims = (shape_x, shape_y, num_slices, 1, num_coils, num_emaps)
        self.image_dims = (num_slices, num_echoes, num_emaps, num_phases, shape_y, shape_x)

        kspace = np.reshape(kspace, kspace_dims)
        maps = np.reshape(maps, map_dims)

        # Permute arrays to be "channels_first"
        kspace = np.transpose(kspace, (2, 4, 3, 5, 1, 0))  # [sl, ec, coil, ph, y, x]
        maps = np.transpose(maps, (2, 5, 4, 3, 1, 0))      # [sl, em, coil, 1, y, x]

        # Create list of examples
        for ec in range(num_echoes):
            for sl in range(num_slices):
                kspace_ex = kspace[sl, ec]
                maps_ex = maps[sl]  # same for all echoes
                self.examples.append([kspace_ex, maps_ex])

    def write(self, file_im, images):
        """
        Write out CFL file with reconstructed images.
        """
        images = torch.cat(images, dim=0).numpy()
        images = np.reshape(images, self.image_dims)
        images = np.transpose(images, (5, 4, 0, 2, 1, 3))  # [x, y, sl, emap, ec, ph]
        images = images[:, :, :, None, :, :, None, :]

        cfl.write(file_im, images, order='F')

    def __len__(self):
        """
        Returns number of examples.
        """
        return len(self.examples)

    def __getitem__(self, index):
        """
        Generates one sample of data.
        """
        kspace, maps = self.examples[index]
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
        kspace = utils.fftmod(kspace)
        maps = utils.fftmod(maps)

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
        maps = maps.squeeze(0)
        mask = mask.squeeze(0)
        init_image = init_image.squeeze(0)

        return kspace, maps, mask, init_image, scale


def main(args):
    # Get filenames
    file_kspace = os.path.join(args.directory, args.kspace)
    file_maps = os.path.join(args.directory, args.maps)
    file_images = os.path.join(args.directory, args.out)

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
    unrolled = LitUnrolled.load_from_checkpoint(args.ckpt, config=config, data_parallel=args.multi_gpu)
    unrolled.freeze()  # similar to running with torch.no_grad()
    if args.multi_gpu:
        unrolled = nn.DataParallel(unrolled).to(device)
    else:
        unrolled = unrolled.to(device)

    logger.info('Loading CFL data...')
    eval_data = CflDataset(file_kspace, file_maps, transform=DataTransform(config))
    data_loader = DataLoader(dataset=eval_data, batch_size=(args.batch_size * num_devices), pin_memory=True)

    logger.info('Running inference...')
    pbar = tqdm(total=len(data_loader), desc='DL Recon', unit='slice')

    # Start timer for reconstruction
    start = time.time()

    out = []
    for data in data_loader:
        # Load batch of data
        kspace, maps, mask, init_image, scale = data

        # Move to GPU
        kspace = kspace.to(device)
        maps = maps.to(device)
        mask = mask.to(device)
        init_image = init_image.to(device)

        # Perform inference
        images = unrolled(kspace, maps, mask, init_image)

        if args.multi_gpu:
            # Necessary because nn.DataParallel converts tensors to float32 (why..?)
            images = torch.view_as_complex(images)

        # Re-scale
        out.append(scale[:, None, None, None, None] * images.cpu())

        # Update progress bar
        pbar.update(n=args.batch_size)

    # End timer for reconstruction
    end = time.time()
    logger.info(f'Elapsed time (reconstruction): {end - start} s')

    pbar.close()

    logger.info('Writing images...')
    eval_data.write(file_images, out)

    return


def create_arg_parser():
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    parser.add_argument('--directory', type=str, required=True, help='Directory with raw data files')
    parser.add_argument('--kspace', type=str, default='ks', help='k-Space file (CFL)')
    parser.add_argument('--maps', type=str, default='maps', help='Sensitivity maps file (CFL)')
    parser.add_argument('--out', type=str, default='im.dl', help='Output - images file (CFL)')
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
