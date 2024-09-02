"""
Written by Christopher Sandino, 2020.
"""

import torch
import torch.fft
from torch import nn

from typing import List, Tuple, Optional


class FFT(nn.Module):
    """
    A module that computes centered, N-dimensional FFT of an input array.
    """
    def __init__(self,
                 ndims: int,
                 norm: Optional[str] = 'ortho') -> None:
        super(FFT, self).__init__()

        # Save variables
        self.ndims = ndims
        self.norm = norm

        # Dimensions across which to take FFT/IFFT
        self.fft_dims = [i for i in range(-1, -1-ndims, -1)]

    def forward(self,
                data: torch.Tensor,
                adjoint: Optional[torch.Tensor] = False,
                centered: Optional[bool] = False):

        assert torch.is_complex(data)  # force complex

        if centered:
            data = torch.fft.ifftshift(data, dim=self.fft_dims)

        if adjoint:
            data = torch.fft.ifftn(data, dim=self.fft_dims, norm=self.norm)
        else:
            data = torch.fft.fftn(data, dim=self.fft_dims, norm=self.norm)

        if centered:
            data = torch.fft.fftshift(data, dim=self.fft_dims)

        return data


class SenseModel(nn.Module):
    """
    A module that computes forward and adjoint SENSE operations.

    The forward operation computes:
        y = (W * F * S) x

    The adjoint operation computes:
        x = (S^H * F^H * W) y

    """
    def __init__(self,
                 maps: torch.Tensor,
                 weights: Optional[torch.Tensor] = None) -> None:
        """
        Args:
            - maps (torch.Tensor[complex64]): dimensions [batch_size, emaps, coils, 1, y, x]
            - weights (torch.Tensor): dimensions [batch_size, coils, t, y, x]
        """
        super(SenseModel, self).__init__()

        assert torch.is_complex(maps)  # force complex

        # Save variables
        self.maps = maps

        # Initialize FFT object
        ndims = len(list(maps[0, 0, 0, ...].squeeze().shape))
        self.fft = FFT(ndims)

        if weights is None:
            self.weights = 1.0
        else:
            self.weights = weights

    def _adjoint_op(self,
                    data: torch.Tensor) -> torch.Tensor:

        # k-space -> image
        data = self.fft(self.weights * data, adjoint=True)
        data = data.unsqueeze(1) * torch.conj(self.maps)
        return data.sum(2)

    def _forward_op(self,
                    data: torch.Tensor) -> torch.Tensor:

        # image -> k-space
        data = data.unsqueeze(2) * self.maps
        data = self.weights * self.fft(data.sum(1))
        return data

    def forward(self,
                data: torch.Tensor,
                adjoint: Optional[bool] = False) -> torch.Tensor:

        assert torch.is_complex(data)

        if adjoint:
            data = self._adjoint_op(data)
        else:
            data = self._forward_op(data)

        return data