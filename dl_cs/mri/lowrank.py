"""
Written by Christopher Sandino, 2020.
"""

import torch
from torch import nn

from dl_cs.mri import utils

from typing import List, Tuple, Optional


class ArrayToBlocks(nn.Module):
    def __init__(self,
                 block_size: int,
                 image_shape: List[int],
                 overlapping: Optional[bool] = False) -> None:
        """
        An array to blocks linear operator.
          - extract() converts 5D array into patches stacked along batch dim  [N, c*bx*by, t]
          - combine() combines spatial patches into a 5D array [1, c, t, y, x]
        """
        super(ArrayToBlocks, self).__init__()

        # TODO: Add support for non-overlapping blocks
        assert overlapping is True

        # Get image / block dimensions
        self.block_size = block_size
        self.image_shape = image_shape
        _, self.ne, self.nt, self.ny, self.nx = image_shape

        # Overlapping vs. non-overlapping block settings
        if overlapping:
            self.block_stride = self.block_size // 2
            # Use Hanning window to reduce blocking artifacts
            win1d = torch.hann_window(block_size, dtype=torch.float32) ** 0.5
            win = win1d[None, :, None] * win1d[None, None, :]
            win = win.repeat(self.ne, 1, 1).reshape((1, self.ne * (block_size ** 2), 1))

        else:
            self.block_stride = self.block_size
            win = torch.tensor([1.0], dtype=torch.float32)
        # Note: this creates a new member variable called self.win
        self.register_buffer('win', win)

        # Get pad shape
        self.pad_x, self.pad_y = self._get_pad_sizes()
        self.nx_pad = self.pad_x[0] + self.nx + self.pad_x[1]
        self.ny_pad = self.pad_y[0] + self.ny + self.pad_y[1]

        # Compute total number of blocks for padded array
        self.num_blocks_x = int((self.nx_pad - self.block_size) / self.block_stride + 1)
        self.num_blocks_y = int((self.ny_pad - self.block_size) / self.block_stride + 1)
        self.num_blocks = self.num_blocks_x * self.num_blocks_y

        # Compute re-weighting function
        self.register_buffer('weights', None)
        all_ones = torch.ones(image_shape, dtype=torch.complex64)
        self.weights = self.combine(self.extract(all_ones))

    def _get_pad_sizes(self) -> Tuple[Tuple[int]]:
        # Pad the image so that an integer number of blocks fits across each dimension
        num_blocks_x = (self.nx // self.block_size) + 1
        num_blocks_y = (self.ny // self.block_size) + 1

        # pad size along x
        pad_x_left = (self.block_size * num_blocks_x - self.nx) // 2
        pad_x_right = pad_x_left if self.nx % 2 == 0 else pad_x_left + 1
        pad_x = (pad_x_left, pad_x_right)

        # pad size along y
        pad_y_left = (self.block_size * num_blocks_y - self.ny) // 2
        pad_y_right = pad_y_left if self.ny % 2 == 0 else pad_y_left + 1
        pad_y = (pad_y_left, pad_y_right)

        return pad_x, pad_y

    def _unfold(self, images: torch.Tensor) -> torch.Tensor:
        """
        This function converts spatiotemporal image data with size (1, ne, nt, ny, nx)
        into N blocks with shape (N, ne*b^2, nt) where N is the number of blocks, and
        b is the block size. It uses torch.Tensor.unfold() to do this.
        """

        # Get rid of batch dimension (TODO: Add support for minibatches)
        images = images.squeeze(0)

        # Unfold along Y, then unfold along X
        images = images.unfold(2, self.block_size, self.block_stride)
        images = images.unfold(3, self.block_size, self.block_stride)

        # Reshape into (N, ne*b^2, nt)
        images = images.permute(2, 3, 0, 4, 5, 1).reshape((self.num_blocks, self.ne * (self.block_size ** 2), self.nt))

        return images

    def _fold(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        This function converts a set of N blocks with shape (N, ne*b^2, nt)
        back into the image representation (1, ne, nt, ny, nx). This function
        technically already exists in PyTorch (nn.Fold), however, it does not
        currently support auto-differentiation with tensors that have type complex64.
        So I have re-written it myself with some assumptions:
        1) The 3-D blocks of size (b, b, nt) are extracted from spatiotemporal data
           Note that these functions do not yet support decimation through time.
        2) The 3-D blocks are overlapping - i.e. the centers of each block are spaced by b/2.
        3) The original image is zero-padded such that an odd integer number of blocks fit across X, Y

        Describing whats going on in this function is kind of hard without a picture...
              ___ ___ ___      ___ ___      ___
             | 0 | 1 | 0 |    | 0 | 0 |    | 1 |    ___ ___     ___
              --- --- ---      --- ---   +  ---  + | 2 | 2 | + | 3 |
             | 2 | 3 | 2 | -> | 0 | 0 |    | 1 |    --- ---     ---
              --- --- ---      --- ---      ---
             | 0 | 1 | 0 |
              --- --- ---

        But basically, I'm taking the blocks in each group (0-3) and adding padded versions
        together to form the images tensor. Groups 1, 2, 3 have to be padded to have the same
        dimensions of group 0, which happens to be the dimensions of our original images array.
        """

        # Get 2-D grid of blocks
        blocks_dims = (1, self.num_blocks_y, self.num_blocks_x, self.ne, self.block_size, self.block_size, self.nt)
        blocks = blocks.view(blocks_dims).permute(0, 3, 6, 1, 4, 2, 5)  # [1, ne, nt, nblksy, blky, nblksx, blkx]

        # Get blocks indexed by even numbers in both Y and X dims (group 0)
        even_even_dims = (1, self.ne, self.nt, self.ny_pad, self.nx_pad)
        images = blocks[..., 0::2, :, 0::2, :].reshape(even_even_dims)

        # Get blocks indexed by odd numbers in Y dim, and even numbers in X dim (group 1)
        odd_even_dims = (1, self.ne, self.nt, self.ny_pad - self.block_size, self.nx_pad)
        images_odd_even = blocks[..., 1::2, :, 0::2, :].reshape(odd_even_dims)
        images += nn.functional.pad(images_odd_even, 2 * (0,) + 2 * (self.block_stride,), mode='constant')

        # Get blocks indexed by even numbers in Y dim, and odd numbers in X dim (group 2)
        even_odd_dims = (1, self.ne, self.nt, self.ny_pad, self.nx_pad - self.block_size)
        images_even_odd = blocks[..., 0::2, :, 1::2, :].reshape(even_odd_dims)
        images += nn.functional.pad(images_even_odd, 2 * (self.block_stride,), mode='constant')

        # Get blocks indexed by odd numbers in both Y and X dimensions (group 3)
        odd_odd_dims = (1, self.ne, self.nt, self.ny_pad - self.block_size, self.nx_pad - self.block_size)
        images_odd_odd = blocks[..., 1::2, :, 1::2, :].reshape(odd_odd_dims)
        images += nn.functional.pad(images_odd_odd, 4 * (self.block_stride,), mode='constant')

        return images

    def extract(self, data: torch.Tensor) -> torch.Tensor:
        # Input data has shape: (1, ne, nt, ny, nx)

        # Pad array with zeros
        data = nn.functional.pad(data, self.pad_x + self.pad_y, mode='constant')

        # Unfold images into blocks
        data = self._unfold(data)

        # Apply sqrt(window) to reduce blocking artifacts
        data *= self.win

        return data

    def combine(self, data: torch.Tensor) -> torch.Tensor:
        # Input data has shape: (N, ne*b^2, nt)

        # Apply sqrt(window) to reduce blocking artifacts
        data *= self.win

        # Fold along X/Y simultaneously
        data = self._fold(data)

        # Crop zero-padded images
        data = utils.center_crop(data, shapes=[self.ny, self.nx], dims=[-2, -1])

        if self.weights is not None:
            data /= (self.weights + 1e-8)

        return data

    def forward(self,
                input: torch.Tensor,
                adjoint: Optional[bool] = False) -> torch.Tensor:

        if adjoint:
            return self.combine(input)
        else:
            return self.extract(input)


class Decompose(nn.Module):
    def __init__(self,
                 block_size: int,
                 rank: int,
                 image_shape: List[int],
                 overlapping: Optional[bool] = False,
                 device: str = 'cpu') -> None:
        """
        A linear operator which converts dynamic 2-D data: [1, x, y, t, c] into
        N blocks of size b x b: [N, b, b, t, c]. Then each block is decomposed
        into simpler basis functions L: [N, b, b, 1, r], and R: [N, 1, 1, t, r].
          - decompose():
          - compose()
        """
        super(Decompose, self).__init__()

        # SVD for torch.complex64 is not yet supported on GPU :(
        assert device == 'cpu'

        self.block_size = block_size
        self.rank = rank
        self.block_op = ArrayToBlocks(block_size, image_shape, overlapping).to(device)

    def decompose(self,
                  images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # get data dimensions
        nt = images.shape[2]
        ne = images.shape[1]
        bx = by = self.block_size
        rk = self.rank

        # convert image data into blocks
        blocks = self.block_op(images)  # [N, b^2*e, t]
        nblocks = blocks.shape[0]

        # perform SVD
        U, S, V = torch.svd(blocks, compute_uv=True)

        # Truncate singular values and corresponding singular vectors
        U = U[:, :, :rk]  # [nblocks, nx*ny*ne, rank]
        S = S[:, :rk]  # [nblocks, rank]
        V = V[:, :, :rk]  # [nblocks, nt, rank]

        # Combine and reshape matrices
        S_sqrt = S.reshape((nblocks, 1, rk)).sqrt()
        L = U.reshape((nblocks, bx*by*ne, rk)) * S_sqrt
        R = V.reshape((nblocks, nt, rk)) * S_sqrt

        return L, R

    def btranspose(self,
                   matrix_batch: torch.Tensor) -> torch.Tensor:
        """
        Takes Hermitian transpose of a batch of matrices.
        """
        return matrix_batch.conj().permute(0, 2, 1)

    def compose(self,
                L: torch.Tensor,
                R: torch.Tensor) -> torch.Tensor:

        blocks = torch.bmm(L, self.btranspose(R))  # torch.sum(L * R.conj(), dim=-1)
        images = self.block_op(blocks, adjoint=True)
        return images

    def forward(self, data, adjoint=False):
        if adjoint:
            L, R = data
            images = self.compose(L, R)
            return images
        else:
            L, R = self.decompose(data)
            return L, R