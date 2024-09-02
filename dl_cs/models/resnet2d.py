"""
Implementations of various layers and networks for 4-D data [N, H, W, C].

by Christopher M. Sandino (sandino@stanford.edu), 2020.

"""

import torch
from torch import nn
from dl_cs.mri.utils import center_crop

class Normalization(nn.Module):
    """
    A generic class for normalization layers.
    """
    def __init__(self, in_chans, type):
        super(Normalization, self).__init__()

        if type == 'none':
            self.norm = nn.Identity()
        elif type == 'instance':
            self.norm = nn.InstanceNorm2d(in_chans, affine=False)
        elif type == 'batch':
            self.norm = nn.BatchNorm2d(in_chans, affine=False)
        else:
            raise ValueError('Invalid normalization type: %s' % type)

    def forward(self, input):
        if input.is_complex():
            return torch.complex(self.norm(input.real), self.norm(input.imag))
        else:
            return self.norm(input)


class Activation(nn.Module):
    """
    A generic class for activation layers.
    """
    def __init__(self, type):
        super(Activation, self).__init__()

        if type == 'none':
            self.activ = nn.Identity()
        elif type == 'relu':
            self.activ = nn.ReLU(inplace=True)
        elif type == 'leaky_relu':
            self.activ = nn.LeakyReLU(inplace=True)
        else:
            raise ValueError('Invalid activation type: %s' % type)

    def forward(self, input):
        if input.is_complex():
            return torch.complex(self.activ(input.real), self.activ(input.imag))
        else:
            return self.activ(input)


class Conv2d(nn.Module):
    """
    A simple 2D convolutional operator.
    """
    def __init__(self, in_chans, out_chans, kernel_size):
        super(Conv2d, self).__init__()

        # Force padding such that the shapes of input and output match
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, padding=padding)

    def forward(self, input):
        return self.conv(input)


class ComplexConv2d(nn.Module):
    """
    A 2D convolutional operator that supports complex values.

    Based on implementation described by:
        EK Cole, et al. "Analysis of deep complex-valued CNNs for MRI reconstruction," arXiv:2004.01738.
    """
    def __init__(self, in_chans, out_chans, kernel_size):
        super(ComplexConv2d, self).__init__()

        # Force padding such that the shapes of input and output match
        padding = (kernel_size - 1) // 2

        # The complex convolution operator has two sets of weights: X, Y
        self.conv_r = nn.Conv2d(in_chans, out_chans, kernel_size, padding=padding)
        self.conv_i = nn.Conv2d(in_chans, out_chans, kernel_size, padding=padding)

    def forward(self, input):
        # The input has real and imaginary parts: a, b
        # The output of the convolution (Z) can be written as:
        #   Z = (X + iY) * (a + ib)
        #     = (X*a - Y*b) + i(X*b + Y*a)

        # Compute real part of output
        output_real = self.conv_r(input.real)
        output_real -= self.conv_i(input.imag)

        # Compute imaginary part of output
        output_imag = self.conv_r(input.imag)
        output_imag += self.conv_i(input.real)

        return torch.complex(output_real, output_imag)


class ConvBlock(nn.Module):
    """
    A 2D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv

    Based on implementation described by:
        K He, et al. "Identity Mappings in Deep Residual Networks" arXiv:1603.05027
    """
    def __init__(self, in_chans, out_chans, kernel_size,
                 act_type='relu', norm_type='none', is_complex=False):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
        """
        super(ConvBlock, self).__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.is_complex = is_complex
        self.name = 'ComplexConv2D' if is_complex else 'Conv2D'

        # Define normalization and activation layers
        normalization = Normalization(in_chans, norm_type)
        activation = Activation(act_type)

        if is_complex:
            convolution = ComplexConv2d(in_chans, out_chans, kernel_size=kernel_size)
        else:
            convolution = Conv2d(in_chans, out_chans, kernel_size=kernel_size)

        # Define forward pass (pre-activation)
        self.layers = nn.Sequential(
            normalization,
            activation,
            convolution
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        return self.layers(input)

    def __repr__(self):
        return f'{self.name}(in_chans={self.in_chans}, out_chans={self.out_chans})'


class ResBlock(nn.Module):
    """
    A ResNet block that consists of two convolutional layers followed by a residual connection.
    """

    def __init__(self, in_chans, out_chans, kernel_size, is_complex=False):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            drop_prob (float): Dropout probability.
        """
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(in_chans, out_chans, kernel_size, act_type='relu', is_complex=is_complex),
            ConvBlock(out_chans, out_chans, kernel_size, act_type='relu', is_complex=is_complex)
        )

        if in_chans != out_chans:
            self.resample = ConvBlock(in_chans, out_chans, 1, act_type='none', is_complex=is_complex)
        else:
            self.resample = nn.Identity()

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """

        # To have a residual connection, number of inputs must be equal to outputs
        shortcut = self.resample(input)

        return self.layers(input) + shortcut


class ResNet(nn.Module):
    """
    Prototype for 2D ResNet architecture.
    """

    def __init__(self, num_resblocks, in_chans, chans, kernel_size, use_complex_layers=False, circular_pad=False):
        """

        """
        super(ResNet, self).__init__()

        self.use_complex_layers = use_complex_layers
        self.circular_pad = circular_pad
        self.pad_size = (2*num_resblocks + 2) * (kernel_size - 1) // 2
        chans = int(chans/1.4142)+1 if use_complex_layers else chans

        # Declare initial conv layer
        self.init_layer = ConvBlock(in_chans, chans, kernel_size, act_type='none', is_complex=use_complex_layers)

        # Declare ResBlock layers
        self.res_blocks = nn.ModuleList([])
        for _ in range(num_resblocks):
            self.res_blocks += [ResBlock(chans, chans, kernel_size, use_complex_layers)]

        # Declare final conv layer (down-sample to original in_chans)
        self.final_layer = ConvBlock(chans, in_chans, kernel_size, act_type='relu', is_complex=use_complex_layers)

    def _preprocess(self, input):
        if not self.use_complex_layers:
            # Convert complex tensor into real representation
            # [N, C, Y, X] complex64 -> [N, 2*C, Y, X] float32
            input = torch.cat((input.real, input.imag), dim=1)

        return input

    def _postprocess(self, output):
        if not self.use_complex_layers:
            # Convert real tensor back into complex representation
            # [N, 2*C, Y, X] float32 -> [N, C, Y, X] complex64
            chans = output.shape[1]
            output = torch.complex(output[:, :(chans // 2)], output[:, (chans // 2):])

        return output

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [N, H, W, C] and type torch.complex64

        Returns:
            (torch.Tensor): Output tensor of shape [N, H, W, C] and type torch.complex64
        """

        # Pre-process input data...
        input = self._preprocess(input)

        # Perform forward pass through the network
        output = self.init_layer(input)
        for res_block in self.res_blocks:
            output = res_block(output)
        output = self.final_layer(output) + input

        # Post-process output data...
        output = self._postprocess(output)

        return output

