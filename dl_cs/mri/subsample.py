"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
from math import floor, ceil


class MaskFunc:
    """
    Abstract MaskFunc class for creating undersampling masks of a specified shape.
    """

    def __init__(self, accelerations):
        """
        Args:
            accelerations (List[int]): Range of acceleration rates to simulate.
        """
        self.accelerations = accelerations
        self.rng = np.random.RandomState()

    def choose_acceleration(self):
        """
        Chooses a random acceleration rate given a range.
        """
        accel_range = self.accelerations[1] - self.accelerations[0]
        acceleration = self.accelerations[0] + accel_range * self.rng.rand()
        return acceleration


class RandomMaskFunc(MaskFunc):
    """
    RandomMaskFunc creates a 2D uniformly random undersampling mask.
    """

    def __init__(self, accelerations, calib_size):
        super().__init__(accelerations)
        self.calib_size = calib_size

    def __call__(self, out_shape, seed=None):
        # self.rng.seed(seed)

        # Design parameters for mask
        nky = out_shape[1]
        nkz = out_shape[2]
        acceleration = self.choose_acceleration()
        prob = 1.0 / acceleration

        # Generate undersampling mask
        mask = torch.rand([nky, nkz], dtype=torch.float32)
        mask = torch.where(mask < prob, torch.Tensor([1]), torch.Tensor([0]))

        # Add calibration region
        calib = [self.calib_size, self.calib_size]
        mask[int(nky / 2 - calib[-2] / 2):int(nky / 2 + calib[-2] / 2),
        int(nkz / 2 - calib[-1] / 2):int(nkz / 2 + calib[-1] / 2)] = torch.Tensor([1])

        return mask.reshape(out_shape)


class VDktMaskFunc(MaskFunc):
    """
    VDktMaskFunc creates a variable-density undersampling mask in k-t space.
    """

    def __init__(self, accelerations, sim_partial_kx=0.25, sim_partial_ky=0.0):
        """
        Args:
            accelerations (List[int]): Range of acceleration rates to simulate.
            sim_partial_kx (bool): Simulates partial readout
            sim_partial_ky (bool): Simulates partial phase encoding
        """
        super().__init__(accelerations)
        self.sim_partial_kx = sim_partial_kx
        self.sim_partial_ky = sim_partial_ky
        self.golden_ratio = 0.618034

    def __call__(self, out_shape, seed=None):
        """
        Args:
            shape (iterable[int]): The shape of the mask to be created format [H, W, D]
            seed (int, optional): Seed for the random number generator. Setting the seed
                ensures the same mask is generated each time for the same shape.
        Returns:
            torch.Tensor: A mask of the specified shape.
        """

        self.rng.seed(seed)

        # Design parameters for mask
        nkx = out_shape[4]
        nky = out_shape[3]
        nphases = out_shape[2]
        acceleration_rate = self.choose_acceleration()

        # Generate ky-t mask
        if self.sim_partial_ky > 0.0:
            mask = self.vdkt_partial_ky(nky, nphases, acceleration_rate, partialFourierFactor=self.sim_partial_ky)
        else:
            mask = self.vdkt(nky, nphases, acceleration_rate)

        # Simulate partial echo
        if self.sim_partial_kx > 0.0:
            mask = np.stack(nkx * [mask], axis=0)
            mask[:int(self.sim_partial_kx * nkx)] = 0

        # Reverse dimensions
        mask = mask.transpose(2, 1, 0)

        # Reshape the mask
        mask = torch.from_numpy(mask.reshape(out_shape).astype(np.float32))

        return mask

    def goldenratio_shift(self, accel, nt):
        return np.round(np.arange(0, nt) * self.golden_ratio * accel) % accel

    def vdkt(self, ny, nt, accel, nCal=1, vdDegree=1.5, vdFactor=None,
             perturbFactor=0.4, adhereFactor=0.33):
        """
        Generates variable-density k-t undersampling mask for dynamic 2D imaging data.

        Written by Peng Lai, GE Healthcare, 2018.
        """
        vdDegree = max(vdDegree, 0.0)
        perturbFactor = min(max(perturbFactor, 0.0), 1.0)
        adhereFactor = min(max(adhereFactor, 0.0), 1.0)
        nCal = max(nCal, 0)

        if vdFactor == None or vdFactor > accel:
            vdFactor = accel

        yCent = floor(ny / 2.0)
        yRadius = (ny - 1) / 2.0

        if vdDegree > 0:
            vdFactor = vdFactor ** (1.0 / vdDegree)
        accel_aCoef = (vdFactor - 1.0) / vdFactor
        accel_bCoef = 1.0 / vdFactor

        ktMask = np.zeros([ny, nt], np.float32)
        ktShift = self.goldenratio_shift(accel, nt)

        for t in range(0, nt):
            # inital sampling with uiform density kt
            ySamp = np.arange(ktShift[t], ny, accel)

            # add random perturbation with certain adherence
            if perturbFactor > 0:
                for n in range(0, ySamp.size):
                    if ySamp[n] < perturbFactor * accel or ySamp[n] >= ny - perturbFactor * accel:
                        continue

                    yPerturb = perturbFactor * accel * (self.rng.rand() - 0.5)
                    ySamp[n] += yPerturb

                    if n > 0:
                        ySamp[n - 1] += adhereFactor * yPerturb
                    if n < ySamp.size - 1:
                        ySamp[n + 1] += adhereFactor * yPerturb

            ySamp = np.clip(ySamp, 0, ny - 1)
            ySamp = (ySamp - yRadius) / yRadius
            ySamp = ySamp * (accel_aCoef * np.abs(ySamp) + accel_bCoef) ** vdDegree

            ind = np.argsort(np.abs(ySamp))
            ySamp = ySamp[ind]
            yUppHalf = np.where(ySamp >= 0)[0]
            yLowHalf = np.where(ySamp < 0)[0]

            # fit upper half k-space to Cartesian grid
            yAdjFactor = 1.0
            yEdge = floor(ySamp[yUppHalf[0]] * yRadius + yRadius + 0.0001)
            yOffset = 0.0

            for n in range(0, yUppHalf.size):
                # add a very small float 0.0001 to be tolerant to numerical error with floor()
                yLoc = min(floor((yOffset + (ySamp[yUppHalf[n]] - yOffset) * yAdjFactor) * yRadius + yRadius + 0.0001),
                           ny - 1)

                if ktMask[yLoc, t] == 0:
                    ktMask[yLoc, t] = 1
                    yEdge = yLoc + 1
                else:
                    ktMask[yEdge, t] = 1
                    yOffset = ySamp[yUppHalf[n]]
                    yAdjFactor = (yRadius - float(yEdge - yRadius)) / (yRadius * (1 - abs(yOffset)))
                    yEdge += 1

            # fit lower half k-space to Cartesian grid
            yAdjFactor = 1.0
            yEdge = floor(ySamp[yLowHalf[0]] * yRadius + yRadius + 0.0001)
            yOffset = 0.0

            if ktMask[yEdge, t] == 1:
                yEdge -= 1
                yOffset = ySamp[yLowHalf[0]]
                yAdjFactor = (yRadius + float(yEdge - yRadius)) / (yRadius * (1.0 - abs(yOffset)))

            for n in range(0, yLowHalf.size):
                yLoc = max(floor((yOffset + (ySamp[yLowHalf[n]] - yOffset) * yAdjFactor) * yRadius + yRadius + 0.0001),
                           0)

                if ktMask[yLoc, t] == 0:
                    ktMask[yLoc, t] = 1
                    yEdge = yLoc + 1

                else:
                    ktMask[yEdge, t] = 1
                    yOffset = ySamp[yLowHalf[n]]
                    yAdjFactor = (yRadius - float(yEdge - yRadius)) / (yRadius * (1 - abs(yOffset)))
                    yEdge -= 1

        # at last, add calibration data
        ktMask[(yCent - ceil(nCal / 2)):(yCent + nCal - 1 - ceil(nCal / 2)), :] = 1

        return ktMask

    def vdkt_partial_ky(self, nky, nphases, tgt_accel, partialFourierFactor=0.25, tol=0.1, max_iter=10):
        """Performs binary search for k-t mask with correct target acceleration."""

        # Define binary search range
        search_range = [1.0, tgt_accel]
        act_accel = 1.0  # initialize actual acceleration variable
        iter = 0  # number of binary search iterations

        while (abs(act_accel - tgt_accel) > tol) and (iter < max_iter):
            # Guess the next acceleration rate
            curr_accel = 0.5 * (search_range[0] + search_range[1])

            # Generate a new mask
            mask = self.vdkt(nky, nphases, curr_accel)

            # CMS: simulate partial Fourier scheme with alternating ky lines
            nyMask = int(nky * partialFourierFactor)
            mask[(nky - nyMask):nky, 0::2] = 0
            mask[0:nyMask, 1::2] = 0

            # Get new actual acceleration rate (will be higher than curr_accel)
            act_accel = (nky * nphases) / np.sum(mask)

            # Adjust the search range for next iteration
            if act_accel < tgt_accel:
                search_range[0] = curr_accel  # lower bound
            else:
                search_range[1] = curr_accel  # upper bound

            iter += 1

        return mask


def subsample(data, mask_func, seed=None, mode='2D'):
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data (torch.Tensor): The input k-space data. This should have at least 3 dimensions, where
            dimensions -3 and -2 are the spatial dimensions, and the final dimension has size
            2 (for complex values).
        mask_func (callable): A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed (int or 1-d array_like, optional): Seed for the random number generator.

    Returns:
        (tuple): tuple containing:
            masked data (torch.Tensor): Subsampled k-space data
            mask (torch.Tensor): The generated mask
    """
    data_shape = tuple(data.shape)  # [N, coils, ..., y, x]
    if mode is '2D':
        mask_shape = (1, 1,) + data_shape[2:4]
    elif mode is '3D':
        mask_shape = (1, 1,) + data_shape[2:5]
    else:
        raise ValueError('Only 2D and 3D undersampling masks are supported.')
    mask = mask_func(mask_shape, seed)

    return mask * data, mask



