"""
Written by Christopher Sandino, 2020.
"""
import torch


def fftmod(out):
    """Performs a modulated FFT on the input, that is multiplying every other line by exp(j*pi) which is a shift of N/2, hence modulating the output by +/- pi.
    Args:
        out (array_like): Input to the FFT.
    Returns:
        The modulated FFT of the input.

    Written by Jon Tamir.
    """
    out[..., ::2] *= -1
    out[..., ::2, :] *= -1
    out *= -1
    return out


def root_sum_of_squares(x, dim=0):
    """
    Compute the root sum-of-squares (RSS) transform along a given dimension of a complex-valued tensor.
    """
    return torch.sqrt(torch.sum(torch.abs(x)**2, dim=dim))


def time_average(data, dim, eps=1e-6, keepdim=True):
    """
    Computes time average across a specified axis.
    """
    mask = get_mask(data)
    return data.sum(dim, keepdim=keepdim) / (mask.sum(dim, keepdim=keepdim) + eps)


def sliding_window(data, dim, window_size):
    """
    Computes sliding window with circular boundary conditions across a specified axis.
    """
    assert 0 < window_size <= data.shape[dim]

    windows = [None] * data.shape[dim]
    for i in range(data.shape[dim]):
        data_slide = torch.roll(data, int(window_size/2)-i, dim)
        window = data_slide.narrow(dim, 0, window_size)
        windows[i] = time_average(window, dim)

    return torch.cat(windows, dim=dim)


def center_crop(data, shapes, dims):
    """
    Apply a center crop to a multi-dimensional tensor.

    Args:
        data (torch.Tensor): The input tensor to be center cropped.
        dims (list of ints): List of dimensions across which to perform crop.
        shapes (list of ints): Size of center crop in each dimension listed in dims.
    """
    for i, dim in enumerate(dims):
        assert 0 < shapes[i] <= data.shape[dim]
        idx_start = (data.shape[dim] - shapes[i]) // 2
        data = data.narrow(dim, idx_start, shapes[i])

    return data


def get_mask(data, eps=1e-12):
    """
    Finds k-space sampling mask given k-space data.
    """
    assert torch.is_complex(data) # force complex

    magnitude = torch.abs(data)
    mask = torch.where(magnitude > eps,
                       torch.ones_like(magnitude),
                       torch.zeros_like(magnitude))
    return mask
