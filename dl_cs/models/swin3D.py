"""
Implementations of various layers and networks for 5-D data [N, H, W, D, C].

by Terrence Jao (tjao@stanford.edu), 2021.


"""
import torch
from torch import nn
from dl_cs.mri.utils import center_crop
#from torchvision.models.video.swin_transformer import SwinTransformer3d
#from torchvision.models.video.swin_transformer import swin3d_t
#from dl_cs.models.swin_transformer_mri import swin3d_t_mr as swin3d_t
from dl_cs.models.video_swin_transformer_mri_downsample import SwinTransformer3D

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
    def __init__(self):
        super(GlobalMaxPool, self).__init__()

        self.gpool = nn.Identity()  #Do I even need this?

    def forward(self, input):

        gp = torch.max(input, [2,3,4])

        if input.is_complex():
            return torch.complex(gp.real, gp.imag)
        else:
            return gp


class GlobalAvgPool(nn.Module):
    """
    A generic class for global average pooling of layers, which is essentially
    a global mean of each channel.
    """
    def __init__(self):
        super(GlobalAvgPool, self).__init__()

        self.gpool = nn.Identity()  #Do I even need this?

    def forward(self, input):

        gp = torch.mean(input, [2,3,4])

        if input.is_complex():
            return torch.complex(gp.real, gp.imag)
        else:
            return gp


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

class SwinTransformer3DBlock(nn.Module):
    """
    A Swin Transformer 3D block as a replacement for ResNet 3D block.
    """
    def __init__(self, in_chans, chans, window_size, num_heads, num_layers, is_complex=False):

        self.is_complex = is_complex

        super(SwinTransformer3DBlock, self).__init__()

        #self.transformer = swin3d_t(in_channel = in_chans, embed_dim = chans)
        self.transformer = SwinTransformer3D(in_chans=in_chans,embed_dim=chans, depths=[6], num_heads=[8], window_size=(7,8,8))

    def forward(self, input):
        """
        Args:
            input (torch.Tensor): Input tensor of shape [batch_size, self.in_chans, depth, width, height]

        Returns:
            (torch.Tensor): Output tensor of shape [batch_size, self.out_chans, depth, width, height]
        """
        return self.transformer(input)

class ResSwinTransformer3DBlock(nn.Module):
    """
    A ResSwinTransformer3DBlock with Swin Transformer 3D as a replacement for ResNet 3D block.
    """
    def __init__(self, in_chans, chans, window_size, num_heads, num_layers, act_type='relu', is_complex=False):
        super(ResSwinTransformer3DBlock, self).__init__()

        self.layers = nn.Sequential(
            SwinTransformer3DBlock(in_chans, chans, window_size, num_heads, num_layers, is_complex=is_complex),
            ConvBlock(chans, chans, kernel_size=3, act_type=act_type, is_complex=is_complex)
        )

    def forward(self, input):
        return self.layers(input) + input
    
class DeepFeatureExtraction(nn.Module):
    """
    Deep feature extraction block 
    """

    def __init__(self, in_chans, chans, window_size, num_heads, num_layers, num_swinblocks, act_type='relu', is_complex=False):
        super(DeepFeatureExtraction, self).__init__()

        self.resswin_blocks = nn.ModuleList([])
        for _ in range(num_swinblocks):
            self.resswin_blocks += [ResSwinTransformer3DBlock(in_chans, chans, window_size, num_heads, num_layers, act_type=act_type, is_complex=is_complex)]

        self.layers = nn.Sequential(
            *self.resswin_blocks,
            ConvBlock(chans, chans, kernel_size=3, act_type=act_type, is_complex=is_complex)
            )

    def forward(self, input):
        
        """
        output = input
        for resswin_blocks in self.resswin_blocks:
            output = resswin_blocks(output)

        return output + input
        """      
        return self.layers(input) + input 
        #return self.layers(input)
       
class SwinTransformer3DNet(nn.Module):
    """
    SwinMR architecture with Swin Transformer 3D blocks.
    """
    def __init__(self, num_swinblocks, in_chans, chans, kernel_size,  window_size, act_type='relu', num_heads=[4], num_layers=[4], use_complex_layers=False, circular_pad=True):
        super(SwinTransformer3DNet, self).__init__()

        self.use_complex_layers = use_complex_layers
        self.circular_pad = circular_pad 
        self.pad_size = (2*num_swinblocks + 2) * (kernel_size - 1) // 2
        chans = int(chans/1.4142)+1 if use_complex_layers else chans

        # Declare initial conv layer - Shallow feature Extraction
        self.SFE = ConvBlock(in_chans, chans, kernel_size=3, act_type='none', is_complex=use_complex_layers)
        #print("inchans = {}, chans = {}, is_complex = {}".format(in_chans, chans, use_complex_layers))

        #self.DFE = DeepFeatureExtraction(in_chans, chans, window_size, num_heads, num_layers, num_swinblocks, act_type=act_type, is_complex=use_complex_layers)
        self.DFE = DeepFeatureExtraction(chans, chans, window_size, num_heads, num_layers, num_swinblocks, act_type=act_type, is_complex=use_complex_layers)
        
        # Declare final conv layer (down-sample to original in_chans) - HQ image reconstruction
        self.final_layer = ConvBlock(chans, in_chans, kernel_size=3, act_type=act_type, is_complex=use_complex_layers)
        #self.fl = ConvBlock(chans, in_chans, kernel_size=1, act_type=act_type, is_complex=use_complex_layers)

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

    def forward(self, x):
        # Pre-process input data...
        x = self._preprocess(x)

        x = self.SFE(x)
        #print("size of output after SFE is {}".format(output.size()))
        x = self.DFE(x)
        #print("size of output after DFE is {}".format(output.size()))
        x = self.final_layer(x)
        #x = self.fl(x)
        
        x = self._postprocess(x)
        #print("size of output after post_process Layer is {}".format(output.size()))

        return x
