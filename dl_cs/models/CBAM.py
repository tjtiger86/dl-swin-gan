"""
Implementations of various layers and networks for 5-D data [N, H, W, D, C].

by Terrence Jao (tjao@stanford.edu), 2021.
Using a

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
            self.norm = nn.InstanceNorm3d(in_chans, affine=False)
        elif type == 'batch':
            self.norm = nn.BatchNorm3d(in_chans, affine=False)
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
        elif type == 'sigmoid':
            self.activ = nn.Sigmoid()
        else:
            raise ValueError('Invalid activation type: %s' % type)

    def forward(self, input):
        if input.is_complex():
            return torch.complex(self.activ(input.real), self.activ(input.imag))
        else:
            return self.activ(input)

class GlobalMaxPool(nn.Module):
    """
    A generic class for Max pooling of layers, which is essentially
    a global mean of each channel.
    """
    def __init__(self, type):
        super(GlobalMaxPool, self).__init__()

        self.mpool = nn.Identity()  #Do I even need this?

        if type == 'channel':
            self.type = 'channel'
        elif type == 'spatial':
            self.type = 'spatial'
        else:
            raise ValueError('Invalid type')

    def forward(self, input):

        if self.type == 'channel':
            mp = torch.amax(input, dim=(2,3,4))
        elif self.type == 'spatial':
            mp = torch.amax(input, dim=1)

        if input.is_complex():
            return torch.complex(mp.real, mp.imag)
        else:
            return mp


class GlobalAvgPool(nn.Module):
    """
    A generic class for global average pooling of layers, which is essentially
    a global mean of each channel.
    """
    def __init__(self, type):
        super(GlobalAvgPool, self).__init__()

        self.apool = nn.Identity()  #Do I even need this?

        if type == 'channel':
            self.type = 'channel'
        elif type == 'spatial':
            self.type = 'spatial'
        else:
            raise ValueError('Invalid type')

    def forward(self, input):

        if self.type == 'channel':
            ap = torch.mean(input, (2,3,4))
        elif self.type == 'spatial':
            ap = torch.mean(input, 1)

        if input.is_complex():
            return torch.complex(ap.real, ap.imag)
        else:
            return ap


class FC(nn.Module):
    """
    A generic class for fully connected layers.
    """
    def __init__(self, in_chans, out_chans):
        super(FC, self).__init__()

        self.fc = nn.Linear(in_chans, out_chans)


    def forward(self, input):
        if input.is_complex():
            return torch.complex(self.fc(input.real), self.fc(input.imag))
        else:
            return self.fc(input)


class Conv3d(nn.Module):
    """
    A simple 3D convolutional operator.
    """
    def __init__(self, in_chans, out_chans, kernel_size):
        super(Conv3d, self).__init__()

        # Force padding such that the shapes of input and output match
        padding = (kernel_size - 1) // 2

        self.conv = nn.Conv3d(in_chans, out_chans, kernel_size, padding=padding)

    def forward(self, input):

        return self.conv(input)


class ComplexConv3d(nn.Module):
    """
    A 3D convolutional operator that supports complex values.

    Based on implementation described by:
        EK Cole, et al. "Analysis of deep complex-valued CNNs for MRI reconstruction," arXiv:2004.01738.
    """
    def __init__(self, in_chans, out_chans, kernel_size):
        super(ComplexConv3d, self).__init__()

        # Force padding such that the shapes of input and output match
        padding = (kernel_size - 1) // 2

        # The complex convolution operator has two sets of weights: X, Y
        self.conv_r = nn.Conv3d(in_chans, out_chans, kernel_size, padding=padding)
        self.conv_i = nn.Conv3d(in_chans, out_chans, kernel_size, padding=padding)

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


class SeparableConv3d(nn.Module):
    """
    A separable 3D convolutional operator.
    """
    def __init__(self, in_chans, out_chans, kernel_size, spatial_chans=None,
                 act_type='relu', is_complex=False):
        """
        Args:
            in_chans (int): Number of channels in the input.
            out_chans (int): Number of channels in the output.
            kernel_size (int): Size of kernel (repeated for all three dimensions).
        """
        super(SeparableConv3d, self).__init__()

        # Force padding such that the shapes of input and output match
        padding = (kernel_size - 1) // 2

        sp_kernel_size = (1, kernel_size, kernel_size)
        sp_pad_size = (0, padding, padding)
        t_kernel_size = (kernel_size, 1, 1)
        t_pad_size = (padding, 0, 0)

        if spatial_chans is None:
            # Force number of spatial features, such that the total number of
            # parameters is the same as a nn.Conv3D(in_chans, out_chans)
            spatial_chans = (kernel_size**3)*in_chans*out_chans
            spatial_chans /= (kernel_size**2)*in_chans + kernel_size*out_chans
            spatial_chans = int(spatial_chans)

        # Define each layer in SeparableConv3d block
        if is_complex:
            conv_2s = ComplexConv3d(in_chans, spatial_chans, kernel_size=sp_kernel_size, padding=sp_pad_size)
            conv_1t = ComplexConv3d(spatial_chans, out_chans, kernel_size=t_kernel_size, padding=t_pad_size)
        else:
            conv_2s = Conv3d(in_chans, spatial_chans, kernel_size=sp_kernel_size, padding=sp_pad_size)
            conv_1t = Conv3d(spatial_chans, out_chans, kernel_size=t_kernel_size, padding=t_pad_size)

        # Define choices for intermediate activation layer
        activation = Activation(act_type)

        # Define the forward pass
        self.layers = nn.Sequential(conv_2s, activation, conv_1t)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        return self.layers(input)


class ConvBlock(nn.Module):
    """
    A 3D Convolutional Block that consists of Norm -> ReLU -> Dropout -> Conv

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
        self.name = 'ComplexConv3D' if is_complex else 'Conv3D'

        # Define normalization and activation layers
        normalization = Normalization(in_chans, norm_type)
        activation = Activation(act_type)

        if is_complex:
            convolution = ComplexConv3d(in_chans, out_chans, kernel_size=kernel_size)
        else:
            convolution = Conv3d(in_chans, out_chans, kernel_size=kernel_size)

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

    def __init__(self, chans, kernel_size, act_type='relu', is_complex=False):
        """
        Args:
            chans (int): Number of channels.
            drop_prob (float): Dropout probability.
        """
        super(ResBlock, self).__init__()

        self.layers = nn.Sequential(
            ConvBlock(chans, chans, kernel_size, act_type=act_type, is_complex=is_complex),
            ConvBlock(chans, chans, kernel_size, act_type=act_type, is_complex=is_complex)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        return self.layers(input) + input


class SeBlock(nn.Module):
    """
    A Squeeze Excitation Block that consists of Global Average Pooling -> FC
    -> ReLU -> FC -> Sigmoid. This usually runs in parallel to a inception block
    or feed into the

    Based on implementation described by:
        J Hu, et al. "Squeeze-and-Excitation Networks" arXiv:1709.01507
    """
    def __init__(self, out_chans, rr, norm_type='none', is_complex=False):
        """
        Args:
            out_chans (int): Number of channels in the output of the Residual block
            rr (int): Reduction ratio the number of FC channels before 1st ReLU
        """
        super(SeBlock, self).__init__()

        self.is_complex = is_complex
        # self.name = 'ComplexConv3D' if is_complex else 'Conv3D'

        # Define normalization and activation layers

        globalpooling = GlobalAvgPool("channel")
        FC1 = FC(out_chans, rr)
        activation1 = Activation("relu")
        FC2 = FC(rr, out_chans)
        activation2 = Activation("sigmoid")

        # Define forward pass (pre-activation)
        self.layers = nn.Sequential(
            globalpooling,
            FC1,
            activation1,
            FC2,
            activation2
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans]
        """

        return self.layers(input)


class CABlock(nn.Module):
    """
    A Channel attention block that consists of Global Average Pooling -> FC
    -> ReLU -> FC -> Sigmoid summed with Global Max Pooling -> FC -> RELU -> FC.
    This is esentially the Squeeze excitation block proposed by Hu. et. al. with
    addition of both a max and average pooling.

    Based on implmentation described by:
    S Woo et. al. "CBAM Convolutional Block Attention Module"  	arXiv:1807.06521

    """
    def __init__(self, out_chans, rr, norm_type='none', is_complex=False):
        """
        Args:
            out_chans (int): Number of channels in the output of the Residual block
            rr (int): Reduction ratio the number of FC channels before 1st ReLU
        """
        super(CABlock, self).__init__()

        self.is_complex = is_complex
        # self.name = 'ComplexConv3D' if is_complex else 'Conv3D'

        # Define normalization and activation layers
        avgpool = GlobalAvgPool("channel")
        maxpool = GlobalMaxPool("channel")

        FC1 = FC(out_chans, rr)
        activation1 = Activation("relu")
        FC2 = FC(rr, out_chans)
        activation2 = Activation("sigmoid")

        # Define forward pass (pre-activation)
        self.layers = nn.Sequential(
            FC1,
            activation1,
            FC2,
            activation2
        )

        self.avgpool = nn.Sequential(avgpool)
        #self.maxpool = nn.Sequential(maxpool)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans]
        """
        #return self.layers(self.maxpool(input))+self.layers(self.avgpool(input))
        return self.layers(self.avgpool(input))


class SABlock(nn.Module):
    """
    A Spatial attention block that consists of concatenated Average Pooling and
    max pooling across channels with a 7x7 convlution.

    Based on implmentation described by:
    S Woo et. al. "CBAM Convolutional Block Attention Module"  	arXiv:1807.06521

    """
    def __init__(self, act_type='relu', norm_type='none', is_complex=False):
        """
        Args:
            out_chans (int): Number of channels in the output of the Residual block
        """
        super(SABlock, self).__init__()

        self.is_complex = is_complex
        # self.name = 'ComplexConv3D' if is_complex else 'Conv3D'

        if is_complex:
            #convolution = ComplexConv3d(out_chans=1, in_chans=2, kernel_size=5)
            convolution = ComplexConv3d(out_chans=1, in_chans=1, kernel_size=5)
        else:
            #convolution = Conv3d(out_chans=1, in_chans=2, kernel_size=5) #PARAMETER SET TO 5
            convolution = Conv3d(out_chans=1, in_chans=1, kernel_size=5)

        # Define normalization and activation layers

        #maxpool = GlobalMaxPool("spatial")
        avgpool = GlobalAvgPool("spatial")

        self.avgpool = nn.Sequential(avgpool)
        #self.maxpool = nn.Sequential(maxpool)

        # Define forward pass (pre-activation)
        self.layers = nn.Sequential(
            convolution
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, depth, width, height]
        """

        #catinput = torch.cat((self.avgpool(input)[:,None,:,:,:], self.maxpool(input)[:,None,:,:,:]), dim=1)
        return self.layers(self.avgpool(input)[:,None,:,:,:])
        #return self.layers(catinput)

class CBAMResBlock(nn.Module):
    """
    A CBAMResNet block that consists of two convolutional layers scaled by
    output of a SeBlock followed by a residual connection.
    """

    def __init__(self, chans, kernel_size, rr, act_type='relu', is_complex=False):
        """
        Args:
            chans (int): Number of channels.
            drop_prob (float): Dropout probability.
        """
        super(CBAMResBlock, self).__init__()

        self.layers1 = nn.Sequential(
            ConvBlock(chans, chans, kernel_size, act_type=act_type, is_complex=is_complex),
            ConvBlock(chans, chans, kernel_size, act_type=act_type, is_complex=is_complex)
        )

        self.CAmodule = nn.Sequential(
            CABlock(chans, rr, is_complex=is_complex)
        )

        self.SAmodule = nn.Sequential(
            SABlock(is_complex=is_complex)
        )

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        #Output of CAmodule is [batch_size, self.out_chans], so it needs to be repeated
        #to match dimension of Input

        [batch_size, in_chans, depth, width, height] = input.size()

        residual = self.layers1(input)
        residual *= self.CAmodule(residual)[:,:,None,None,None].repeat(1, 1, depth, width, height)
        residual *= self.SAmodule(residual).repeat(1, in_chans, 1, 1, 1)

        return residual + input



class SeResBlock(nn.Module):
    """
    A SeResNet block that consists of two convolutional layers scaled by
    output of a SeBlock followed by a residual connection.
    """

    def __init__(self, chans, kernel_size, rr, act_type='relu', is_complex=False):
        """
        Args:
            chans (int): Number of channels.
            drop_prob (float): Dropout probability.
        """
        super(SeResBlock, self).__init__()

        self.layers1 = nn.Sequential(
            ConvBlock(chans, chans, kernel_size, act_type=act_type, is_complex=is_complex),
            ConvBlock(chans, chans, kernel_size, act_type=act_type, is_complex=is_complex)
        )

        self.layers2 = SeBlock(chans, rr, is_complex=is_complex)

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        #Output of layers2 is [batch_size, self.out_chans], so it needs to be repeated
        #to match dimension of Input

        [batch_size, in_chans, depth, width, height] = input.size()

        residual = self.layers1(input)
        residual *= self.layers2(residual)[:,:,None,None,None].repeat(1, 1, depth, width, height)

        return residual + input


class CBAMResNet(nn.Module):
    """
    Prototype for 3D ResNet architecture.
    """

    def __init__(self, num_resblocks, in_chans, chans, kernel_size, rr, act_type='relu', use_complex_layers=False, circular_pad=True):
        """

        """
        super(CBAMResNet, self).__init__()

        self.use_complex_layers = use_complex_layers
        self.circular_pad = circular_pad
        self.pad_size = (2*num_resblocks + 2) * (kernel_size - 1) // 2
        chans = int(chans/1.4142)+1 if use_complex_layers else chans

        # Declare initial conv layer
        self.init_layer = ConvBlock(in_chans, chans, kernel_size, act_type='none', is_complex=use_complex_layers)

        # Declare ResBlock layers
        self.se_res_blocks = nn.ModuleList([])
        for _ in range(num_resblocks):
            self.se_res_blocks += [CBAMResBlock(chans, kernel_size, rr, act_type=act_type, is_complex=use_complex_layers)]

        # Declare final conv layer (down-sample to original in_chans)
        self.final_layer = ConvBlock(chans, in_chans, kernel_size, act_type=act_type, is_complex=use_complex_layers)

    def _preprocess(self, input):

        if not self.use_complex_layers:
            # Convert complex tensor into real representation
            # [N, C, T, Y, X] complex64 -> [N, 2*C, T, Y, X] float32
            input = torch.cat((input.real, input.imag), dim=1)

        # Circularly pad array through time
        if self.circular_pad:
            pad_dims = (0, 0, 0, 0) + 2 * (self.pad_size,)
            input = nn.functional.pad(input, pad_dims, mode='circular')

        return input

    def _postprocess(self, output):
        # Crop back to the original shape
        output = center_crop(output, dims=[2], shapes=[output.shape[2]-2*self.pad_size])

        if not self.use_complex_layers:
            # Convert real tensor back into complex representation
            # [N, 2*C, T, Y, X] float32 -> [N, C, T, Y, X] complex64
            chans = output.shape[1]
            output = torch.complex(output[:, :(chans//2)], output[:, (chans//2):])

        return output

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [N, C, T, Y, X]

        Returns:
            (torch.Tensor): Output tensor of shape [N, C, T, Y, X]
        """

        # Pre-process input data...
        input = self._preprocess(input)

        # Perform forward pass through the network
        output = self.init_layer(input)
        for se_res_block in self.se_res_blocks:
            output = se_res_block(output)
        output = self.final_layer(output) + input

        # Post-process output data...
        output = self._postprocess(output)

        return output
