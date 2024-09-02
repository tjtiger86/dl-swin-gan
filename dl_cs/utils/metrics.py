"""Metric utilities

All metrics take a reference scan (ref) and an output/reconstructed scan (x).
"""

import torch
import torch.nn.functional as F
from dl_cs.utils import vgg_preceptual_loss as vggpl
from dl_cs.utils import VGGloss as vgg_loss

def calc_weight(ref: torch.Tensor):
    """Calculate weight - simple multiply by through time standard deviation
    """
    nbatch, nchannel, nt, ny, nx = ref.shape
    W = torch.reshape(torch.repeat_interleave(torch.abs(torch.std(ref, dim=2)), nt, dim=2), ref.shape)

    return W


def l2(ref: torch.Tensor, pred: torch.Tensor, weight: bool):
    """L2 loss.
    """
    if weight is True:
        W = calc_weight(ref)
    else:
        W = torch.ones(ref.shape, device=ref.device)

    return torch.sqrt(torch.mean(torch.abs(W*(ref - pred))**2))


def l1(ref: torch.Tensor, pred: torch.Tensor, weight: bool):
    """L1 loss.
    """
    if weight is True:
        W = calc_weight(ref)
    else:
        W = torch.ones(ref.shape, device=ref.device)

    return torch.mean(torch.abs(W*(ref - pred)))

def vggloss(ref: torch.Tensor, pred: torch.Tensor):
    """VGG Perceptual Loss with Weights 0.65, 0.3, and 0.05 for layers, 0,3,6
    """
    #vggmodel = vggpl.VGGPerceptualLoss()
    #vggmodel = vggmodel.to(device = ref.device)

    loss = 0

    vggmodel = vgg_loss.VGG_Loss()
    vggmodel = vggmodel.to(device = ref.device)

    nbatch, nchannel, nt, ny, nx = ref.shape

    """ Implementation 1:"""

    """ VGG loss for both E-spirit maps, then 0 padding to have 3 channels for VGG,
    note, this implementation suboptimal, since VGG trained on RGB image,
    with each channel having different color information, meanwhile, espirit
    maps not correlated like that"""

    """
    for kk in range(nt):

        curr_ref = ref[:,:,kk,:,:]
        curr_pred = pred[:,:,kk,:,:]


        curr_ref = F.pad(input=curr_ref, pad=(0, 0, 0, 0, 0, 1), mode='constant', value=0)
        curr_pred = F.pad(input=curr_pred, pad=(0, 0, 0, 0, 0, 1), mode='constant', value=0)

        loss += vggmodel(curr_ref, curr_pred)



    """

    """
    Implementation 2:
    Separate each emap and have it contribute to the vgg loss. Set real
    and imagingary as the different channels and zero-pad the last channel.
    Have a version that only looks at the first e-spirit channel"""

    # Only take the first Emap channel
    ref = ref[:,1,:,:,:]
    pred = pred[:,1,:,:,:]

    # Concatenate complex values
    if torch.is_complex(ref): 
        ref = torch.cat((torch.real(ref), torch.imag(ref)), 0)
        pred = torch.cat((torch.real(pred), torch.imag(pred)), 0)
        #zero pad third channel
        ref = F.pad(input=ref, pad=(0, 0, 0, 0, 0, 0, 0, 1), mode='constant', value=0)
        pred = F.pad(input=pred, pad=(0, 0, 0, 0, 0, 0, 0, 1), mode='constant', value=0)
    else: 
        ref = F.pad(input=ref, pad=(0, 0, 0, 0, 0, 0, 0, 2), mode='constant', value=0)
        pred = F.pad(input=pred, pad=(0, 0, 0, 0, 0, 0, 0, 2), mode='constant', value=0)
    
    for kk in range(nt):  #loop between slices
        loss += vggmodel(torch.unsqueeze(ref[:,kk,:,:], dim=0), torch.unsqueeze(pred[:,kk,:,:], dim=0))

    #print(loss)
    """ VGG loss for both espirti map, but not splitting complex numbers """
    """
    for kk in range(nt):
        for jj in range(nchannel):

            curr_ref = torch.unsqueeze(ref[:,jj,kk,:,:], dim=0)
            curr_pred = torch.unsqueeze(pred[:,jj,kk,:,:], dim=0)


            curr_ref = F.pad(input=curr_ref, pad=(0, 0, 0, 0, 0, 2), mode='constant', value=0)
            curr_pred = F.pad(input=curr_pred, pad=(0, 0, 0, 0, 0, 2), mode='constant', value=0)

            loss += vggmodel(curr_ref, curr_pred)
    """

    return loss



def psnr(ref: torch.Tensor, pred: torch.Tensor, weight: bool):
    """Peak signal-to-noise ratio (PSNR)
    """
    scale = torch.abs(ref).max()
    return 20 * torch.log10(scale / l2(ref, pred, weight))


def perp_loss(ref: torch.Tensor, pred: torch.Tensor, weight: bool):
    """Perpendicular loss.

    Based on M. Terpstra, et al. Rethinking complex image
    reconstruction: Perp-loss for improved complex image
    reconstruction with deep learning. ISMRM 2021.
    """
    if weight is True:
        W = calc_weight(ref)
    else:
        W = torch.ones(ref.shape, device=ref.device)


    # Only valid for complex-valued tensors
    assert ref.is_complex()
    assert pred.is_complex()

    # Perp-loss = P(pred, ref) + l1(|pred|, |ref|)

    # Compute P - normalized absolute cross product between pred, ref
    P = torch.abs(W*pred.real*ref.imag-W*pred.imag*ref.real) / torch.abs(W*ref)

    # Compute M - magnitude loss
    M = torch.abs(torch.abs(W*ref) - torch.abs(W*pred))

    return torch.mean(P + M)
