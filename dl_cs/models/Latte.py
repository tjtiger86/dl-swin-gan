# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import itertools
import collections.abc
import torch
import torch.nn as nn
import numpy as np
from math import ceil, floor
import torch.nn.functional as F
from dl_cs.mri.utils import center_crop

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def to_2tuple(x):
    if isinstance(x, collections.abc.Iterable):
        return x
    return (x, x)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1,2).contiguous()
            k_xf = k.transpose(1,2).contiguous()
            v_xf = v.transpose(1,2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PatchEmbed2D(nn.Module):
    """ Video to Patch Embedding.
    Args:
        patch_size (int): Patch token size. Default: (2,4,4).
        in_chans (int): Number of input video channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, image_size = (224, 224), patch_size=(2,2), in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

        patch_size = to_2tuple(patch_size)
        image_size = to_2tuple(image_size)

        self.patch_size = patch_size
        self.image_size = image_size

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        #print("image_size = {} and patch_size = {}".format(image_size, patch_size))
        
        #self.grid_size = (image_size[0] // patch_size[0], image_size[1] // patch_size[1], image_size[2] // patch_size[2] )
        #self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, H, W = x.size()

        image_size = (H, W)

        if W % self.patch_size[1] != 0:
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1]))
        if H % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[0] - H % self.patch_size[0]))

        x = self.proj(x)  # B C Wh Ww
        patchify_size = x.size()
        #print(f"patchify size is {patchify_size}")
        if self.norm is not None:
            Wh, Ww = x.size(2), x.size(3)
            x = x.flatten(2).transpose(1, 2)
            x = self.norm(x)
            x = x.transpose(1, 2).view(-1, self.embed_dim, Wh, Ww)

        #print(f"number of patches in patchEmbed3D is: {x.shape[2]*x.shape[3]*x.shape[4]}")
        
        x = x.reshape(shape=(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
        x = x.permute(0,2,1)

        return x

class TempEmbed(nn.Module):

    def __init__(self, hidden_size, max_frames = 100):
        super().__init__()
        self.temp_embed_table = nn.Parameter(torch.zeros(1, max_frames, hidden_size), requires_grad=False)
        temp_embed_table = get_1d_sincos_temp_embed(hidden_size, max_frames)
        self.temp_embed_table.data.copy_(torch.from_numpy(temp_embed_table).float().unsqueeze(0))

    def forward(self, frames):
        #print(f"size of self temp embed table is {self.temp_embed_table.size()}")
        return self.temp_embed_table[:, :frames,:]

class PosEmbed(nn.Module):
    """
    Will use fixed sin-cos embedding, however, image sizes during training may change, so cannot have it fixed like original implementation
    """
    def __init__(self, patch_size, hidden_size, max_grid_size=(128,128)):
        super().__init__()
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.max_grid_size = max_grid_size

        self.pos_embed_table = nn.Parameter(torch.zeros(1, math.prod(self.max_grid_size), self.hidden_size), requires_grad = False)
        pos = get_2d_sincos_pos_embed(self.hidden_size, self.max_grid_size)
        self.pos_embed_table.data.copy_(torch.from_numpy(pos).float().unsqueeze(0))

        #pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        #self.pos_embed = torch.empty(0, requires_grad = False)
    def forward(self, grid_size):
        """
        The max embedding is concatenated as ([emb_h, emb_w, emb_t], axis=1), just need to extract
        """
        H, W = grid_size
        max_H, max_W = self.max_grid_size
        index = np.array([h + w*max_H for w, h in itertools.product(range(H), range(W))])

        #print("grid_size is {}\nMax grid_size is {}".format(grid_size, self.max_grid_size))
        #print("Self.pos_embed size is {}".format(self.pos_embed_table.size()))

        #print(f"size of self pos_embed_table is {self.pos_embed_table.size()}")
        return self.pos_embed_table[:, index]

def calc_num_patch(x, patch_size):

    patch_size = to_2tuple(patch_size)
    _, _, _, H, W = x.size()
    
    if H % patch_size[0] != 0:
        padH = patch_size[0] - H % patch_size[0]
    else:
        padH = 0
    if W % patch_size[1] != 0:
        padW = patch_size[1] - W % patch_size[1]
    else:
        padW = 0

    pad = np.array([padH, padW])
    grid_size = np.array( [ (H+padH) // patch_size[0], (W+padW) // patch_size[1] ])
    num_patch = grid_size[0] * grid_size[1] 

    #print(f"number of patches in calc_num_patch is: {grid_size}")
    return num_patch, grid_size, pad

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

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

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
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


#################################################################################
#                                 Core Latte Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A Latte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
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

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of Latte.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size[0] * patch_size[1] * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class Latte(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=1,
        attention_mode='math',
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.extras = extras
        self.num_frames = num_frames

        #self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.x_embedder = PatchEmbed2D(patch_size=patch_size, in_chans=in_channels, embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(77 * 768, hidden_size, bias=True)
        )

        #num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        #self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        #self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)

        self.pos_embedder = PosEmbed(patch_size=patch_size, hidden_size=hidden_size)
        self.temp_embedder = TempEmbed(hidden_size=hidden_size)
        self.hidden_size =  hidden_size

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode) for _ in range(depth)
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
        #self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        #temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        #self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Latte blocks:
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
        x: (N, T, patch_size[0]*patch_size[1] * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p0 = self.x_embedder.patch_size[0]
        p1 = self.x_embedder.patch_size[1]
        #h = w = int(x.shape[1] ** 0.5)
    
        h, w = x_orig_size + pad

        #print(f"f/p is {f/p0}, h/p is {h/p1}, and w/p is {w/p2}")
        assert h/p0 * w/p1 == x.shape[1]

        x = x.reshape(shape=(x.shape[0], int(h/p0), int(w/p1), p0, p1, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        x = x.reshape(shape=(x.shape[0], c, h, w))

        #Cropping away padding
        x = x[:,:,ceil(pad[0]/2):h-floor((pad[0]/2)), 
            ceil(pad[1]/2):w-floor((pad[1]/2))]
        
        return x

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, 
                x, 
                t, 
                y=None, 
                text_embedding=None, 
                use_fp16=False):
        """
        Forward pass of Latte.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)

        # Keep track of original x size for unpatchify later
        batches, channels, frames, height, width = x.shape 
        x_orig_size = np.array([height, width])
        _, grid_size, pad = calc_num_patch(x, self.patch_size)
        
        x = torch.permute(x, (0,2,1,3,4))
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        
        x = self.x_embedder(x) 
        pos_embed = self.pos_embedder(grid_size)
        #print(f"pos embed size is {pos_embed.size()}")
        x = x + pos_embed

        temp_embed = self.temp_embedder(frames)
        #print(f"temp embed size is {temp_embed.size()}")

        t = self.t_embedder(t, use_fp16=use_fp16)                  
        timestep_spatial = repeat(t, 'n d -> (n c) d', c=temp_embed.shape[1]) 
        timestep_temp = repeat(t, 'n d -> (n c) d', c=pos_embed.shape[1])

        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            y_spatial = repeat(y, 'n d -> (n c) d', c=temp_embed.shape[1]) 
            y_temp = repeat(y, 'n d -> (n c) d', c=self.pos_embed.shape[1])
        elif self.extras == 78:
            text_embedding = self.text_embedding_projection(text_embedding.reshape(batches, -1))
            text_embedding_spatial = repeat(text_embedding, 'n d -> (n c) d', c=temp_embed.shape[1])
            text_embedding_temp = repeat(text_embedding, 'n d -> (n c) d', c=pos_embed.shape[1])

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            else:
                c = timestep_spatial
            x  = spatial_block(x, c)

            x = rearrange(x, '(b f) t d -> (b t) f d', b=batches)
            # Add Time Embedding
            if i == 0:
                #print(f"size of x is {x.size()}")
                #print(f"size of temp_embed is {temp_embed.size()}")
                x = x + temp_embed

            if self.extras == 2:
                c = timestep_temp + y_temp
            elif self.extras == 78:
                c = timestep_temp + text_embedding_temp
            else:
                c = timestep_temp

            x = temp_block(x, c)
            x = rearrange(x, '(b t) f d -> (b f) t d', b=batches)

        if self.extras == 2:
            c = timestep_spatial + y_spatial
        else:
            c = timestep_spatial
        x = self.final_layer(x, c)               
        #x = self.unpatchify(x)
        x = self.unpatchify2(x, x_orig_size, pad)                   
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        x = torch.permute(x, (0,2,1,3,4))

        del(pos_embed)
        del(temp_embed)
        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, use_fp16=False, text_embedding=None):
        """
        Forward pass of Latte, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, y=y, use_fp16=use_fp16, text_embedding=text_embedding)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
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
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#################################################################################
#                                   Latte Configs                                  #
#################################################################################

def Latte_XL_2(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def Latte_XL_4(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def Latte_XL_8(**kwargs):
    return Latte(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def Latte_L_2(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def Latte_L_4(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def Latte_L_8(**kwargs):
    return Latte(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def Latte_B_2(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def Latte_B_4(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def Latte_B_8(**kwargs):
    return Latte(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def Latte_S_2(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def Latte_S_4(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def Latte_S_8(**kwargs):
    return Latte(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


Latte_models = {
    'Latte-XL/2': Latte_XL_2,  'Latte-XL/4': Latte_XL_4,  'Latte-XL/8': Latte_XL_8,
    'Latte-L/2':  Latte_L_2,   'Latte-L/4':  Latte_L_4,   'Latte-L/8':  Latte_L_8,
    'Latte-B/2':  Latte_B_2,   'Latte-B/4':  Latte_B_4,   'Latte-B/8':  Latte_B_8,
    'Latte-S/2':  Latte_S_2,   'Latte-S/4':  Latte_S_4,   'Latte-S/8':  Latte_S_8,
}
"""
if __name__ == '__main__':

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    img = torch.randn(3, 16, 4, 32, 32).to(device)
    t = torch.tensor([1, 2, 3]).to(device)
    y = torch.tensor([1, 2, 3]).to(device)
    network = Latte_XL_2().to(device)
    from thop import profile 
    flops, params = profile(network, inputs=(img, t))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')
    # y_embeder = LabelEmbedder(num_classes=101, hidden_size=768, dropout_prob=0.5).to(device)
    # lora.mark_only_lora_as_trainable(network)
    # out = y_embeder(y, True)
    # out = network(img, t, y)
    # print(out.shape)
"""

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


class LatteNet(nn.Module):
    """
    DiTMR architecture with DiT 3D blocks
    """   
    def __init__(self, num_blocks, in_chans, chans, kernel_size, act_type='relu', num_heads=6, num_layers=12, use_complex_layers=False, circular_pad=True, learn_sigma=False):
        super(LatteNet, self).__init__()

        self.use_complex_layers = use_complex_layers
        self.circular_pad = circular_pad 
        self.pad_size = (2*num_blocks + 2) * (kernel_size - 1) // 2
        #chans = int(chans/1.4142)+1 if use_complex_layers else chans

        # Declare initial conv layer - Shallow feature Extraction - this was part of SWINIR
        self.SFE = ConvBlock(in_chans, chans, kernel_size=3, act_type='none', is_complex=use_complex_layers)
        #print("depth = {}, inchans = {}, chans = {}, is_complex = {}".format(num_layers, in_chans, chans, use_complex_layers))

        self.Latte = Latte(depth=num_layers, hidden_size=chans, patch_size=(4, 4), num_heads=num_heads, in_channels = in_chans, learn_sigma=learn_sigma)
        
        # Declare final conv layer (down-sample to original in_chans) - HQ image reconstruction
        self.final_layer = ConvBlock(chans, in_chans, kernel_size=3, act_type=act_type, is_complex=use_complex_layers)
        
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
        
        x = self._postprocess(x)
        #print("size of output after post_process Layer is {}".format(output.size()))

        return x
    """

    def forward(self, x, t, c):
        # Pre-process input data...
        x = self._preprocess(x)

        #print("size of x before DiT is {}".format(x.size()))
        x = self.Latte(x, t, c)
        
        #Get rid of some of the blocky artifacts from patch unembedding
        #x = self.fl(x)
        
        x = self._postprocess(x)
        #print("size of output after post_process Layer is {}".format(output.size()))
        return x