"""
Unrolled Compressed Sensing (3D)
by Christopher M. Sandino (sandino@stanford.edu), 2020.

"""

import torch
from torch import nn
import torch.utils.checkpoint as cp

from dl_cs.models.se3d import SeResNet
from dl_cs.mri.algorithms import ConjugateGradient


class UnrolledSeResNet(nn.Module):
    """
    Abstract class for an unrolled neural network.
    """
    def __init__(self, config):
        super(UnrolledSeResNet, self).__init__()

        # Load network config parameters
        self.num_unrolls = config.MODEL.PARAMETERS.NUM_UNROLLS
        self.num_resblocks = config.MODEL.PARAMETERS.NUM_RESBLOCKS
        self.rr = config.MODEL.PARAMETERS.RR
        self.num_features = config.MODEL.PARAMETERS.NUM_FEATURES
        self.kernel_size = config.MODEL.PARAMETERS.CONV_BLOCK.KERNEL_SIZE[0]
        self.num_emaps = config.MODEL.PARAMETERS.NUM_EMAPS
        self.share_weights = config.MODEL.PARAMETERS.SHARE_WEIGHTS
        self.fix_step_size = config.MODEL.PARAMETERS.FIX_STEP_SIZE
        self.use_complex_layers = config.MODEL.PARAMETERS.CONV_BLOCK.COMPLEX
        self.circular_pad = config.MODEL.PARAMETERS.CONV_BLOCK.CIRCULAR_PAD
        self.do_checkpoint = config.MODEL.PARAMETERS.GRAD_CHECKPOINT

        # Initialize network module
        self.cnn_update = self.init_nets()

    def init_nets(self):
        """
        Initializes convolutional neural networks (CNN) to be interleaved with
        data consistency updates.
        """
        # Figure out how many channels does this data have
        in_chans = self.num_emaps if self.use_complex_layers else 2 * self.num_emaps

        # Read in network parameters
        seresnet_params = dict(in_chans=in_chans,
                             chans=self.num_features,
                             num_resblocks=self.num_resblocks,
                             use_complex_layers=self.use_complex_layers,
                             kernel_size=self.kernel_size,
                             circular_pad=self.circular_pad,
                             rr = self.rr
                             )

        # Declare CNNs for each unrolled iteration
        if self.share_weights:
            # Initializes copies of the same network
            nets = nn.ModuleList([SeResNet(**seresnet_params)] * self.num_unrolls)
        else:
            # Initializes unique networks
            nets = nn.ModuleList([SeResNet(**seresnet_params) for _ in range(self.num_unrolls)])

        return nets

    def forward(self, y, A, x0=None):
        """
        Performs the forward pass through the unrolled network.
        """
        raise NotImplementedError


class ProximalGradientDescent(UnrolledSeResNet):
    """
    Implementation of proximal gradient descent (PGD) solver for the
    regularized least squares problem:
        \\underset{x}{minimize} || y - Ax ||_2^2 + R(x)
    """

    def __init__(self, config):
        super().__init__(config)

        # PGD step size
        self.step_size = nn.Parameter(torch.tensor([-2.0], dtype=torch.float32),
                                      requires_grad=(not self.fix_step_size))

    def forward(self, y, A, x0=None):
        """
        Performs the forward pass through the unrolled network.
        """
        # Pre-compute A^T(y)
        ATy = A(y, adjoint=True)

        # Get initial guess
        xi = ATy if x0 is None else x0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            xi.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(x):
                # Data consistency update
                x = x + self.step_size * (A(A(x), adjoint=True) - ATy)
                # CNN update
                x = self.cnn_update[i](x)
                return x
            return update_fn

        # Begin unrolled network.
        for i in range(self.num_unrolls):
            if self.do_checkpoint:
                xi = cp.checkpoint(update(i), xi)
            else:
                xi = update(i)(xi)

        return xi


class HalfQuadraticSplitting(UnrolledSeResNet):
    """
    Implementation of half quadratic splitting (HQS) solver for the
    regularized least squares problem:
        \\underset{x, z}{minimize} || y - Ax ||_2^2 + mu*||x - z||_2^2 + R(z)
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_cg_iter = config.MODEL.PARAMETERS.MODL.NUM_CG_STEPS
        self.lamda = nn.Parameter(torch.tensor([0.1], dtype=torch.float32),
                                      requires_grad=(not self.fix_step_size))

    def forward(self, y, A, x0=None):
        """
        Performs the forward pass through the unrolled network.
        """
        # Pre-compute A^T(y)
        ATy = A(y, adjoint=True)

        # Get initial guess
        xi = ATy if x0 is None else x0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            xi.requires_grad_()

        # Define normal equations and CG solver
        model_normal = lambda m: A(A(m), adjoint=True) + self.lamda * m
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)

        # Define update rule
        def update(i):
            def update_fn(x):
                z = self.cnn_update[i](x)
                x = cg_solve(x, ATy + self.lamda * z)
                return x
            return update_fn

        # Begin unrolled network.
        for i in range(self.num_unrolls):
            if self.do_checkpoint:
                xi = cp.checkpoint(update(i), xi)
            else:
                xi = update(i)(xi)

        return xi
