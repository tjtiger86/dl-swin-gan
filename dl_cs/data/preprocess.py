"""
Classes for data pre-processing.
"""
import torch
import numpy as np

from dl_cs.mri import transforms as T
from dl_cs.mri import subsample as ss
from dl_cs.mri import lowrank as lr
from dl_cs.mri import utils

class Preprocess:
    """
    Abstract module for pre-processing k-space data during training.

    Specifically, this module should apply pre-processing steps such as
    data augmentation, undersampling, normalization, etc.
    """
    def __init__(self, config, use_seed=False):
        self.config = config
        self.use_seed = use_seed
        self.rng = np.random.RandomState()

    def _augment(self, kspace, maps, target, seed):
        raise ValueError('Not implemented for abstract data class!')

    def __call__(self, kspace, maps, target, fname):
        raise ValueError('Not implemented for abstract data class!')


class CinePreprocess(Preprocess):
    """
    Module for pre-processing cine data during training.

    Specifically, this module takes in fully-sampled k-space, sensitivity maps,
    and the target image. From these it forms the undersampled network input,
    applies data augmentation, scaling, etc.
    """
    def __init__(self, config, lr_decom=False, use_seed=False):
        super().__init__(config, use_seed)

        # Initialize undersampling function
        acceleration_range = config.AUG_TRAIN.UNDERSAMPLE.ACCELERATIONS
        self.mask_func = ss.VDktMaskFunc(acceleration_range,
                                         sim_partial_kx=config.AUG_TRAIN.UNDERSAMPLE.PARTIAL_KX,
                                         sim_partial_ky=config.AUG_TRAIN.UNDERSAMPLE.PARTIAL_KY)

        # DSLR-specific parameters (not used if lr_decom=False)
        self.lr_decom = lr_decom
        self.block_size = config.MODEL.PARAMETERS.DSLR.BLOCK_SIZE
        self.num_basis = config.MODEL.PARAMETERS.DSLR.NUM_BASIS
        self.overlapping = config.MODEL.PARAMETERS.DSLR.OVERLAPPING

    def _augment(self, kspace, maps, target, seed):
        """
        Data augmentation processes (For now, this just includes cropping and flipping across x, y, t dimensions)
        """
        self.rng.seed(seed)

        # Convert undersampled k-space into multicoil images
        F = T.FFT(ndims=2)
        multicoil_images = F(kspace, adjoint=True)

        # Crop along the readout dimension
        crop_size = self.config.AUG_TRAIN.CROP_READOUT
        if crop_size > 0:
            # First pick a random center about which to crop
            shape_x = multicoil_images.shape[-1]
            mean_x = shape_x // 2 + 1
            std_x = crop_size // 2
            crop_center_x = int(self.rng.normal(loc=mean_x, scale=std_x))

            # Clip center so that crop indices don't over-range
            min_x = crop_size // 2
            max_x = shape_x - crop_size // 2 - 1
            crop_center_x = np.clip(crop_center_x, a_min=min_x, a_max=max_x)

            # Crop data
            start_x = crop_center_x - crop_size//2 + 1
            end_x = start_x + crop_size
            multicoil_images = multicoil_images[..., start_x:end_x]
            maps = maps[..., start_x:end_x]
            target = target[..., start_x:end_x]

        # Crop along the phase encode dimension
        crop_size_y = self.config.AUG_TRAIN.ZPAD_PE
        if crop_size_y > 0:
            # First pick a random center about which to crop
            shape_y = multicoil_images.shape[-2]
            mean_y = shape_y // 2 + 1
            std_y = crop_size_y // 2
            crop_center_y = int(self.rng.normal(loc=mean_y, scale=std_y))

            # Clip center so that crop indices don't over-range
            min_y = crop_size_y // 2
            max_y = shape_y - crop_size_y // 2 - 1
            crop_center_y = np.clip(crop_center_y, a_min=min_y, a_max=max_y)

            # Crop data
            start_y = crop_center_y - crop_size_y//2 + 1
            end_y = start_y + crop_size_y
            multicoil_images = multicoil_images[..., start_y:end_y,:]
            maps = maps[..., start_y:end_y,:]
            target = target[..., start_y:end_y,:]

        # Random flips across readout
        if self.rng.rand() > 0.5:
            multicoil_images = torch.flip(multicoil_images, dims=(-1,))
            maps = torch.flip(maps, dims=(-1,))
            target = torch.flip(target, dims=(-1,))

        # Random flips across phase encode direction
        if self.rng.rand() > 0.5:
            multicoil_images = torch.flip(multicoil_images, dims=(-2,))
            maps = torch.flip(maps, dims=(-2,))
            target = torch.flip(target, dims=(-2,))

        # Random flips through time
        if self.rng.rand() > 0.5:
            multicoil_images = torch.flip(multicoil_images, dims=(-3,))
            target = torch.flip(target, dims=(-3,))

        # Convert images back to k-space
        kspace = F(multicoil_images)

        return kspace, maps, target

    def __call__(self, kspace, maps, target, fname):
        seed = None if not self.use_seed else tuple(map(ord, fname))

        # Convert everything from numpy arrays to tensors
        kspace = torch.from_numpy(kspace).unsqueeze(0)
        maps = torch.from_numpy(maps).unsqueeze(0)
        target = torch.from_numpy(target).unsqueeze(0)

        # Apply random data augmentation
        kspace, maps, target = self._augment(kspace, maps, target, seed)

        # Initialize ESPIRiT model
        A = T.SenseModel(maps, weights=None)

        # Create ground truth
        target = A(kspace, adjoint=True)

        # Undersample k-space data
        masked_kspace, mask = ss.subsample(kspace, self.mask_func, seed, mode='3D')

        # Compute normalization factor (based on 95% max signal level in view-shared dataa)
        averaged_kspace = utils.time_average(masked_kspace, dim=2)
        image = A(averaged_kspace, adjoint=True)
        magnitude = torch.abs(image).reshape(-1)
        k = int(round(0.05 * magnitude.numel()))
        scale = torch.min(torch.topk(magnitude, k).values)

        # Normalize k-space and target images
        masked_kspace /= scale
        target /= scale

        # Compute network initialization
        if self.config.MODEL.PARAMETERS.SLWIN_INIT:
            init_kspace = utils.sliding_window(masked_kspace, dim=2, window_size=5)
        else:
            init_kspace = masked_kspace
        init_image = A(init_kspace, adjoint=True)

        if self.lr_decom:
            decompose = lr.Decompose(self.block_size, self.num_basis, target.shape, overlapping=self.overlapping)
            L_init, R_init = decompose(init_image)

        # Get rid of batch dimension...
        masked_kspace = masked_kspace.squeeze(0)
        mask = mask.squeeze(0)
        maps = maps.squeeze(0)
        init_image = init_image.squeeze(0)
        target = target.squeeze(0)

        if self.lr_decom:
            return masked_kspace, mask, maps, L_init, R_init, init_image, scale, target
        else:
            return masked_kspace, mask, maps, init_image, scale, target
