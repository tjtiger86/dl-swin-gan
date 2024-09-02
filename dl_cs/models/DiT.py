"""
Implementations of various layers and networks for 5-D data [N, H, W, D, C].

by Terrence Jao (tjao@stanford.edu), 2021.


"""
import torch
import collections.abc
import math
import itertools
from math import prod
import numpy as np
from torch import nn, Tensor
import torch.nn.functional as F
from dl_cs.mri.utils import center_crop
from dl_cs.models.video_swin_transformer_mri_downsample import SwinTransformer3D
from timm.models.vision_transformer import PatchEmbed, Attention, Mlp
from typing import Any, Callable, List, Optional, Tuple
from math import ceil, floor

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def to_3tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x, x)

def calc_num_patch(x, patch_size):

    patch_size = to_3tuple(patch_size)
    _, _, D, H, W = x.size()
    
    if D % patch_size[0] != 0:
        padD = patch_size[0] - D % patch_size[0] 
    else: 
        padD = 0
    if H % patch_size[1] != 0:
        padH = patch_size[1] - H % patch_size[1]
    else:
        padH = 0
    if W % patch_size[2] != 0:
        padW = patch_size[2] - W % patch_size[2]
    else:
        padW = 0

    pad = np.array([padD, padH, padW])
    grid_size = np.array( [(D+padD) // patch_size[0], (H+padH) // patch_size[1], (W+padW) // patch_size[2] ])
    num_patch = grid_size[0] * grid_size[1] * grid_size[2]

    #print(f"number of patches in calc_num_patch is: {grid_size}")
    return num_patch, grid_size, pad

def factorize(x, patchify_size, flag):
    b, d, f, h, w = patchify_size

    if flag == 0:
        #Temporal Factorize
        x = x.reshape(b*f, h*w, d)
    else:
        #Spatial Factorize
        x = x.reshape(b, f, h, w, d).permute(0, 2, 3, 1, 4).reshape(b*h*w, f, d)
    
    return x
    
def unfactorize(x, pathcify_size, flag):
    b, d, f, h, w = pathcify_size
    if flag == 0:
        #Temporal Factorize
        x = x.reshape(b, f*h*w, d)
    else:
        #Spatial Factorize
        x = x.reshape(b, h, w, f, d).permute(0, 3, 1, 2, 4).reshape(b, f*h*w, d)

    return x

class PatchEmbed3D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, image_size = (224, 224, 20), patch_size=(2,4,4), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        patch_size = to_3tuple(patch_size)
        image_size = to_3tuple(image_size)

        self.patch_size = patch_size
        self.image_size = image_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        #print("image_size = {} and patch_size = {}".format(image_size, patch_size))
        
        #self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1], image_size[2] // patch_size[2] )
        #self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()

        image_size = (D, H, W)

        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        patchify_size = x.size()
        #print(f"patchify size is {patchify_size}")
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, D, Wh, Ww)

        #print(f"number of patches in patchEmbed3D is: {x.shape[2]*x.shape[3]*x.shape[4]}")
        
        x = x.reshape(shape=(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]*x.shape[4]))
        x = x.permute(0,2,1)

        return x, patchify_size

class PatchUnembed3D(nn.Module):
    #Patch UnEmbedding to Video.
    """
    Args:
        patch_size (List[int]): Patch token size.
        pre_size (List[int]): Original Size before patch_embed. 
        in_channels (int): Number of input channels. Default: 3
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size = (2,4,4), in_channels = 3, embed_dim = 96, norm_layer = None):
        super().__init__()
        
        patch_size = to_3tuple(patch_size)
        self.patch_size = patch_size

        self.proj = nn.ConvTranspose3d(embed_dim, in_channels, kernel_size=patch_size, stride=patch_size,)

        if norm_layer is not None:
            self.norm = norm_layer(in_channels)
        else:
            self.norm = nn.Identity()

    def forward(self, x: Tensor, pre_size: List[int]) -> Tensor:
        # Forward function.
        
        x = self.proj(x)  # B C F H W

        #Cropping
        curr_size = x.size()
        print(f"curr_size is {curr_size} and pre_size is {pre_size}")
        diff = [curr_size[j] - pre_size[j] for j in range(5)]
        x = x[:,:,ceil(diff[2]/2):curr_size[2]-floor((diff[2]/2)), 
            ceil(diff[3]/2):curr_size[3]-floor((diff[3]/2)), 
            ceil(diff[4]/2):curr_size[4]-floor((diff[4]/2))]
    
        x = x.permute(0, 2, 3, 4, 1)  # B F H W C
        if self.norm is not None:
            x = self.norm(x)

        x = x.permute(0, 4, 1, 2, 3)  # B C F H W
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class PosEmbed(nn.Module):
    """
    Will use fixed sin-cos embedding, however, image sizes during training may change, so cannot have it fixed like original implementation
    """
    def __init__(self, patch_size, hidden_size, max_grid_size=(128,128,15)):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.max_grid_size = max_grid_size

        self.pos_embed_table = nn.Parameter(torch.zeros(1, prod(self.max_grid_size), self.hidden_size), requires_grad = False)
        pos = get_3d_sincos_pos_embed(self.hidden_size, self.max_grid_size)
        self.pos_embed_table.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        #self.pos_embed = torch.empty(0, requires_grad = False)
    def forward(self, grid_size):
        """
        The max embedding is concatenated as ([emb_h, emb_w, emb_t], axis=1), just need to extract
        """
        F, H, W = grid_size
        max_F, max_H, max_W = self.max_grid_size
        index = np.array([f + h*max_F + w*max_F*max_H for w, h, f in itertools.product(range(F), range(H), range(W))])

        #index = np.array([w + h*max_W + f*max_H*max_W for f, h, w in itertools.product(range(F), range(H), range(W))])

        #print("grid_size is {}\nMax grid_size is {}".format(grid_size, self.max_grid_size))
        #print("Self.pos_embed size is {}".format(self.pos_embed_table.size()))
        
        """
        index2 = np.meshgrid(np.arange(F), np.arange(H), np.arange(W))  # here w goes first
        index2 = np.stack(index2, axis=0)
        index2 = np.reshape(index2, (3, int(prod(np.shape(index2))/3)))
        index2 = index2[0]
        
        print("index shape is {}".format(index.size))
        print("index2 shape is {}".format(index2.size))

        print(f"Index range: {index.min()} to {index.max()}")
        print(f"Index2 range: {index2.min()} to {index2.max()}")
        """

        """ Below Works """
        """
        index = np.meshgrid(np.arange(F), np.arange(H), np.arange(W))  # here w goes first
        index = np.stack(index, axis=0)
        index = np.reshape(index, (3, int(prod(np.shape(index))/3)))
        index = index[0]
        #index = np.ndarray.flatten(index, 'F')
        """

        #print("this is index size{}".format(np.shape(index)))

        return self.pos_embed_table[:, index, :]
    
#################################################################################
#                                 Core DiT Model                                #
#################################################################################

class DiTBlockFactor(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm3 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        )

    def forward(self, x, c, patchify_size):

        shift_msa_spatial, scale_msa_spatial, gate_msa_spatial, shift_msa_temporal, scale_msa_temporal, gate_msa_temporal, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        
        #Spatial Self Attention
        residual = x
        x = modulate(self.norm1(x), shift_msa_spatial, scale_msa_spatial)
        x = self.attn(factorize(x, patchify_size,1))
        x = unfactorize(x, patchify_size, 1)
        x = gate_msa_spatial.unsqueeze(1)*x + residual

        #Temporal Self Attention
        residual = x 
        x = modulate(self.norm2(x), shift_msa_spatial, scale_msa_spatial)
        x = self.attn(factorize(x, patchify_size, 0))
        x = unfactorize(x, patchify_size, 0)
        x = gate_msa_temporal.unsqueeze(1)*x + residual

        #MLP
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm3(x), shift_mlp, scale_mlp))
        
        return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

        #self.adaLN_modulation = nn.Sequential(
        #    nn.SiLU(),
        #    nn.Linear(hidden_size, 9 * hidden_size, bias=True)
        #)

    def forward(self, x, c):
        #shift_msa_spatial, scale_msa_spatial, gate_msa_spatial, shift_msa_temporal, scale_msa_temporal, gate_msa_temporal, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(9, dim=1)
        #x = x + gate_msa_spatial.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa_spatial, scale_msa_spatial))
        #x = x + gate_msa_temporal.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa_temporal, scale_msa_temporal))
        
        shift_msa_spatiotemporal, scale_msa_spatiotemporal, gate_msa_spatiotemporal, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa_spatiotemporal.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa_spatiotemporal, scale_msa_spatiotemporal))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()

        patch_size = to_3tuple(patch_size)
        
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * patch_size[2] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=(28, 180, 64),
        patch_size= (2, 4, 4),
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1,
        learn_sigma=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads

        self.x_embedder = PatchEmbed3D(input_size, patch_size, in_channels, hidden_size, norm_layer= None)
        #self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        
        self.x_unembedder = PatchUnembed3D(patch_size = patch_size, in_channels=in_channels, embed_dim = hidden_size , norm_layer= None)

        #num_patches = self.x_embedder.num_patches
       
        # Will use fixed sin-cos embedding:
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.pos_embedder = PosEmbed(patch_size, hidden_size)
        
        self.blocks = nn.ModuleList([
            #DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
            DiTBlockFactor(hidden_size, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        
        #print("size of self.pos_embed before get_3d_sincos is {}".format(self.pos_embed.size()))
        #print("number of patches is {}".format(self.x_embedder.num_patches))

        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        #pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** (1/3)))
        
        #pos_embed = get_3d_sincos_pos_embed(self.pos_embed.shape[-1], self.x_embedder.grid_size)
        #print("size of pos_embed after get_3d_sincos is {}".format(np.size(pos_embed,1)))
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs
    
    def unpatchify2(self, x, x_orig_size, pad):
        """
        x: (N, T, patch_size[0]*patch_size[1]*patch_size[2] * C)
        imgs: (N, F, H, W, C)
        """
        c = self.out_channels
        p0 = self.x_embedder.patch_size[0]
        p1 = self.x_embedder.patch_size[1]
        p2 = self.x_embedder.patch_size[2]
        #h = w = int(x.shape[1] ** 0.5)
    
        f, h, w = x_orig_size + pad

        #print(f"f/p is {f/p0}, h/p is {h/p1}, and w/p is {w/p2}")
        assert f/p0 * h/p1 * w/p2 == x.shape[1]

        x = x.reshape(shape=(x.shape[0], int(f/p0), int(h/p1), int(w/p2), p0, p1, p2, c))
        
        x = torch.einsum('nfhwpqrc->ncfphqwr', x)
        
        #imgs = x.reshape(shape=(x.shape[0], c, f, h, w))
        x = x.reshape(shape=(x.shape[0], c, f, h, w))

        #Cropping away padding
        x = x[:,:,ceil(pad[0]/2):f-floor((pad[0]/2)), 
            ceil(pad[1]/2):h-floor((pad[1]/2)), 
            ceil(pad[2]/2):w-floor((pad[2]/2))]
        
        return x


    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, F, H, W) tensor of spatiotemporal inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        #print("size of input x is {}, size of input t is {}".format(x.size(), t.size()))
        #print("size of self.x_embedder(x) is {}, size of self.pos_embed is {}".format(self.x_embedder(x).size(), self.pos_embed.size()))
        
        #x = self.x_embedder(x) + self.pos_embedder(x)  # (N, T, D), where T = F* H * W / patch_size ** 3
      
        
        #x = self.x_embedder(x) + self.pos_embed
        #pos_embed = self.pos_embedder(grid_size)
        #x = self.x_embedder(x) + pos_embed

        #Forward Step with DitBlockFactor
        
        _,_,F,H,W = x.size()
        x_orig_size = np.array([F,H,W])

        _, grid_size, pad = calc_num_patch(x, self.patch_size)
        x, unpatchify_size = self.x_embedder(x) 
        x = x + self.pos_embedder(grid_size)
        t = self.t_embedder(t)                      # (N, D)
        y = self.y_embedder(y, self.training)       # (N, D)
        c = t + y                                   # (N, D)
        for block in self.blocks:
            x = block(x, c, unpatchify_size)        # (N, T, D)

        x = self.final_layer(x, c)                  # (N, T, patch_size[0]*patch_size[1]*patch_size[2] * out_channels)
        x = self.unpatchify2(x, x_orig_size, pad)   # (N, out_channels, F, H, W)

        """
        
        #Forward Step with DitBlock
        _,_,F,H,W = x.size()
        x_orig_size = np.array([F,H,W])

        _, grid_size, pad = calc_num_patch(x, self.patch_size)
        x, _ = self.x_embedder(x) 
        x = x + self.pos_embedder(grid_size)
        t = self.t_embedder(t)                      # (N, D)
        y = self.y_embedder(y, self.training)       # (N, D)
        c = t + y                                   # (N, D)
        for block in self.blocks:
            x = block(x, c)                         # (N, T, D)

        #print("size of x is {} after block".format(x.size()))
        x = self.final_layer(x, c)                  # (N, T, patch_size[0]*patch_size[1]*patch_size[2] * out_channels)
        #print("size of x is {} after final layer".format(x.size()))

        
        #x = self.x_unembedder(x, x_orig_size)            # (N, out_channels, F, H, W) 
        x = self.unpatchify2(x, x_orig_size, pad)         # (N, out_channels, F, H, W)
    
        """
        
        """"
        #Debug forward step to seew if pathembed and patch_unembed are inverses of each other
        x_orig_size = np.array(x.size())

        x_orig = x
        print(f"x_orig size: {x_orig.size()}")
        print(f"x_orig samples: {x_orig[0,0,:,0,0]}")
        x, patchify_size = self.x_embedder(x)
        
        #x = x.permute(0,2,1)
        #x = x.reshape(shape=patchify_size)

        print(f"x reshape size is {x.size()}")
        x = self.x_unembedder(x, x_orig_size)
        #x = self.unpatchify2(x, x_orig_size, pad)

        
        print(f"x samples: {x[0,0,:,0,0]}")
        print(f"they are equal == {torch.eq(x_orig, x)}")
        """

        #Release Memory? 
        #pos_embed.detach()
        #del(pos_embed)
        #print("size of x is {} after unpatchify2".format(x.size()))

        return x
    

    def forward_2D(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W, D) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)                   # (N, D)
        y = self.y_embedder(y, self.training)    # (N, D)
        c = t + y                                # (N, D)
        for block in self.blocks:
            x = block(x, c)                      # (N, T, D)
        x = self.final_layer(x, c)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
    
#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_3d_sincos_pos_embed_new(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: tuple of three integers (depth, height, width)
    return:
    pos_embed: [grid_size_d * grid_size_h * grid_size_w, embed_dim] or 
               [extra_tokens + grid_size_d * grid_size_h * grid_size_w, embed_dim] (w/ or w/o cls_token)
    """
    grid0, grid1, grid2 = grid_size
    
    # Generate 1D grids
    grid_t = np.arange(grid0, dtype=np.float32)
    grid_w = np.arange(grid1, dtype=np.float32)
    grid_h = np.arange(grid2, dtype=np.float32)
    
    # Use broadcasting to create 3D grids without meshgrid
    grid_t = grid_t[:, np.newaxis, np.newaxis]  # (depth, 1, 1)
    grid_w = grid_w[np.newaxis, :, np.newaxis]  # (1, height, 1)
    grid_h = grid_h[np.newaxis, np.newaxis, :]  # (1, 1, width)
    
    # Flatten the grid positions
    grid_t = grid_t.flatten()
    grid_w = grid_w.flatten()
    grid_h = grid_h.flatten()
    
    # Combine grid positions into a single array
    grid = np.stack([grid_t, grid_w, grid_h], axis=-1)  # (grid_size_d * grid_size_h * grid_size_w, 3)
    
    # Compute the positional embeddings directly
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_3d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height, width, and frames
    return:
    pos_embed: [grid_size_h*grid_size_w*grid_size_f, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_t = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid_h = np.arange(grid_size[2], dtype=np.float32)
    grid = np.meshgrid(grid_t, grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape(3, 1, grid_size[0], grid_size[1], grid_size[2])
    #print("shape of meshgrid is {}".format(np.shape(grid)))
    pos_embed = get_3d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_3d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0  #embed_dim % 3 == 0? 

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[0])  # (H, D/3)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[1])  # (W, D/3)
    emb_t = get_1d_sincos_pos_embed_from_grid(embed_dim // 3, grid[2])  # (F, D/3)

    #print("emb_h size is {}, emb_w size is {}, emb_t size is {}".format(np.shape(emb_h), np.shape(emb_w), np.shape(emb_t)))

    emb = np.concatenate([emb_h, emb_w, emb_t], axis=1) # (H*W*F, D)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################

def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

def DiT_Cust(**kwargs):
    return DiT(depth=6, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


DiT_models = {
    'DiT-XL/2': DiT_XL_2,  'DiT-XL/4': DiT_XL_4,  'DiT-XL/8': DiT_XL_8,
    'DiT-L/2':  DiT_L_2,   'DiT-L/4':  DiT_L_4,   'DiT-L/8':  DiT_L_8,
    'DiT-B/2':  DiT_B_2,   'DiT-B/4':  DiT_B_4,   'DiT-B/8':  DiT_B_8,
    'DiT-S/2':  DiT_S_2,   'DiT-S/4':  DiT_S_4,   'DiT-S/8':  DiT_S_8,
    'DiT-Cust': DiT_Cust
}

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

class DiTNet(nn.Module):
    """
    DiTMR architecture with DiT 3D blocks
    """   
    def __init__(self, num_blocks, in_chans, chans, kernel_size, act_type='relu', num_heads=6, num_layers=12, use_complex_layers=False, circular_pad=True, learn_sigma=False):
        super(DiTNet, self).__init__()

        self.use_complex_layers = use_complex_layers
        self.circular_pad = circular_pad 
        self.pad_size = (2*num_blocks + 2) * (kernel_size - 1) // 2
        #chans = int(chans/1.4142)+1 if use_complex_layers else chans

        # Declare initial conv layer - Shallow feature Extraction - this was part of SWINIR
        self.SFE = ConvBlock(in_chans, chans, kernel_size=3, act_type='none', is_complex=use_complex_layers)
        #print("depth = {}, inchans = {}, chans = {}, is_complex = {}".format(num_layers, in_chans, chans, use_complex_layers))

        #self.DiT = DiT(depth=num_layers, hidden_size=chans, patch_size=(2, 4, 4), num_heads=num_heads, in_channels = chans, learn_sigma=learn_sigma)
        
        self.DiT = DiT(depth=num_layers, hidden_size=chans, patch_size=(2, 4, 4), num_heads=num_heads, in_channels = in_chans, learn_sigma=learn_sigma)
        
        # Declare final conv layer (down-sample to original in_chans) - HQ image reconstruction
        self.final_layer = ConvBlock(chans, in_chans, kernel_size=3, act_type=act_type, is_complex=use_complex_layers)
        
        #self.final_layer = ConvBlock(in_chans, in_chans, kernel_size=3, act_type=act_type, is_complex=use_complex_layers)
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
    
    """
    def forward(self, x, t, c):
        # Pre-process input data...
        x = self._preprocess(x)

        res = self.SFE(x)
        #print("size of x before DiT is {}".format(x.size()))
        x = self.DiT(res, t, c)
        
        #Get rid of some of the blocky artifacts from patch unembedding
        x = self.final_layer(x+res) 
        #x = self.fl(x)
        
        x = self._postprocess(x)
        #print("size of output after post_process Layer is {}".format(output.size()))

        return x
    """

    def forward(self, x, t, c):
        # Pre-process input data...
        x = self._preprocess(x)

        #print("size of x before DiT is {}".format(x.size()))
        x = self.DiT(x, t, c)
        
        #Get rid of some of the blocky artifacts from patch unembedding
        #x = self.fl(x)
        
        x = self._postprocess(x)
        #print("size of output after post_process Layer is {}".format(output.size()))
        return x
    
class DiTResNet(nn.Module):
    """
    DiTMR architecture with DiT 3D blocks
    """   
    def __init__(self, num_blocks, in_chans, chans, kernel_size, act_type='relu', num_heads=6, num_layers=12, use_complex_layers=False, circular_pad=True, learn_sigma=False):
        super(DiTResNet, self).__init__()

        self.use_complex_layers = use_complex_layers
        self.circular_pad = circular_pad 
        self.pad_size = (2*num_blocks + 2) * (kernel_size - 1) // 2
        #chans = int(chans/1.4142)+1 if use_complex_layers else chans

        # Declare initial conv layer - Shallow feature Extraction - this was part of SWINIR
        self.SFE = ConvBlock(in_chans, chans, kernel_size=3, act_type='none', is_complex=use_complex_layers)
        #print("depth = {}, inchans = {}, chans = {}, is_complex = {}".format(num_layers, in_chans, chans, use_complex_layers))
        self.DiT = DiT(depth=num_layers, hidden_size=chans, patch_size=(2, 4, 4), num_heads=num_heads, in_channels = chans, learn_sigma=learn_sigma)
        
        # Declare final conv layer (down-sample to original in_chans) - HQ image reconstruction
        self.final_layer = ConvBlock(chans, in_chans, kernel_size=3, act_type=act_type, is_complex=use_complex_layers)
        
        #self.final_layer = ConvBlock(in_chans, in_chans, kernel_size=3, act_type=act_type, is_complex=use_complex_layers)
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
    
    
    def forward(self, x, t, c):
        # Pre-process input data...
        x = self._preprocess(x)

        print(f"x size is {x.size()}")
        res = self.SFE(x)
        #print("size of x before DiT is {}".format(x.size()))
        x = self.DiT(res, t, c)
        
        #Get rid of some of the blocky artifacts from patch unembedding
        x = self.final_layer(x+res) 
        #x = self.fl(x)
        
        x = self._postprocess(x)
        #print("size of output after post_process Layer is {}".format(output.size()))

        return x
    
