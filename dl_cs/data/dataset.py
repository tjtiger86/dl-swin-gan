"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import os
import glob
import h5py
import random
from torch.utils.data import Dataset


class Hdf5Dataset(Dataset):
    """
    A generic PyTorch Dataset class that provides access to 2D MRI data
    contained in an HDF5 file.

    Each HDF5 file should correspond to a single patient. It contains three keys:
    - 'kspace': np.array with dimensions [slices, kx, ky, ..., coils]
    - 'maps': np.array with dimensions [slices, x, y, ..., coils]
    - 'target': np.array with dimensions [slices, x, y, ...]
    """
    def __init__(self, root_directory, transform, sample_rate=1.0):
        self.transform = transform
        self.examples = []
        files = glob.glob(os.path.join(root_directory, '*.h5'))

        if sample_rate < 1.0:
            random.shuffle(files)
            num_files = round(len(files) * sample_rate)
            files = files[:num_files]

        for filename in sorted(files):
            kspace = h5py.File(filename, 'r')['kspace']
            num_slices = kspace.shape[0]
            self.examples += [(filename, slice) for slice in range(num_slices)]

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

        return self.transform(kspace, maps, target, filename)
