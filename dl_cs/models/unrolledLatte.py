"""
Unrolled Compressed Sensing (3D)
by Christopher M. Sandino (sandino@stanford.edu), 2020.

"""

import torch
from torch import nn
import torch.utils.checkpoint as cp

from dl_cs.models.Latte import LatteNet
from dl_cs.mri.algorithms import ConjugateGradient


class UnrolledLatteNet(nn.Module):
    """
    Abstract class for an unrolled neural network.
    """
    def __init__(self, config):
        super(UnrolledLatteNet, self).__init__()

        # Load network config parameters
        self.num_unrolls = config.MODEL.PARAMETERS.NUM_UNROLLS
        self.num_blocks = config.MODEL.PARAMETERS.NUM_RESBLOCKS
        self.num_features = config.MODEL.PARAMETERS.NUM_FEATURES
        self.num_layers = config.MODEL.PARAMETERS.NUM_LAYERS
        self.num_heads = config.MODEL.PARAMETERS.NUM_HEADS
        self.kernel_size = config.MODEL.PARAMETERS.CONV_BLOCK.KERNEL_SIZE[0]
        self.num_emaps = config.MODEL.PARAMETERS.NUM_EMAPS
        self.share_weights = config.MODEL.PARAMETERS.SHARE_WEIGHTS
        self.fix_step_size = config.MODEL.PARAMETERS.FIX_STEP_SIZE
        self.use_complex_layers = config.MODEL.PARAMETERS.CONV_BLOCK.COMPLEX
        self.circular_pad = config.MODEL.PARAMETERS.CONV_BLOCK.CIRCULAR_PAD
        self.do_checkpoint = config.MODEL.PARAMETERS.GRAD_CHECKPOINT
        self.learn_sigma = config.MODEL.PARAMETERS.LEARN_SIGMA
        
        # Initialize network module
        self.nn_update = self.init_nets()

    def init_nets(self):
        """
        Initializes  neural networks (NN) to be interleaved with
        data consistency updates.
        """
        # Figure out how many channels does this data have
        in_chans = self.num_emaps if self.use_complex_layers else 2 * self.num_emaps

        # Read in network parameters
        dit_params = dict(in_chans=in_chans,
                             chans=self.num_features,
                             num_blocks=self.num_blocks,
                             use_complex_layers=self.use_complex_layers,
                             kernel_size=self.kernel_size,
                             circular_pad=self.circular_pad,
                             num_heads = self.num_heads,
                             num_layers = self.num_layers,
                             learn_sigma = False,
                             )
        
        if self.learn_sigma == True:

            #Final DiT unroll needs to learn sigma for the loss function
            dit_final_params = dict(in_chans=in_chans,
                             chans=self.num_features,
                             num_blocks=self.num_blocks,
                             use_complex_layers=self.use_complex_layers,
                             kernel_size=self.kernel_size,
                             circular_pad=self.circular_pad,
                             num_heads = self.num_heads,
                             num_layers = self.num_layers,
                             learn_sigma = self.learn_sigma,
                             )
        
            # Declare NNs for each unrolled iteration
            if self.share_weights:
            # Initializes copies of the same network
                nets = nn.ModuleList([LatteNet(**dit_params)] * self.num_unrolls-1)
            else:
            # Initializes unique networks
                nets = nn.ModuleList([LatteNet(**dit_params) for _ in range(self.num_unrolls-1)])
        
            nets.append(LatteNet(**dit_final_params))
        else:  
            if self.share_weights:
                # Initializes copies of the same network
                nets = nn.ModuleList([LatteNet(**dit_params)] * self.num_unrolls)
            else:
                # Initializes unique networks
                nets = nn.ModuleList([LatteNet(**dit_params) for _ in range(self.num_unrolls)])
        
        return nets

    def forward(self, y, A, t, c, x0=None):
        """
        Performs the forward pass through the unrolled network.
        """
        raise NotImplementedError
    


class DDPM(UnrolledLatteNet):

    """
    In DDPM we are only predicting the noise, we do not need data consistency during training? 
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, x0, t, A, A_1, A_F, fs, c):
        # We still have A, A_1, A_F, model inputs for p-sampling during reconstruction
        # Necessary when gradient checkpointing is on
        
        # Get initial guess
        xi = x0

        if self.training and self.do_checkpoint:
            xi.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(x):
                # NN update
                x = self.nn_update[i](x, t, c)
                return x
            return update_fn

        # Begin unrolled network.
        for i in range(self.num_unrolls):
            if self.do_checkpoint:
                xi = cp.checkpoint(update(i), xi)
            else:
                xi = update(i)(xi)
        return xi

class DataConsistency(UnrolledLatteNet):
    
    """
    In the diffusion case, we are not using PGD to solve the least squares problem. The supposed "unrolls" is just to inject
    Data consistency periodically into the network
    """

    def __init__(self, config):
        super().__init__(config)

    def forward(self, x0, t, A, A_1, A_F, A_S, fs, c):
        """
        Data Consistency
        """
        # Pre-compute A^T(y)
        ATy = x0

        # Get initial guess
        xi = x0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            xi.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(x):

                # NN update
                x = self.nn_update[i](x, t, c)

                # Data consistency after neural network
                #print(f"x.size is {x.size()}, A_1(x) size is {A_1(x).size()} and A(x0) size is {A(x0).size()}")
                x =  A_F(A_1(x) + A(x0), adjoint=True)

                return x
            return update_fn

        # Begin unrolled network.
        for i in range(self.num_unrolls):
            if self.do_checkpoint:
                xi = cp.checkpoint(update(i), xi)
            else:
                xi = update(i)(xi)
        return xi

class ProximalGradientDescent(UnrolledLatteNet):
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


    def forward(self, x0, t, A, c):
        #Reimplement forward pass without feeding k-space, pass in x0 = ATy directly so diffusion forward model 
        #can more easily add noise into the image domain
        #Performs the forward pass through the unrolled network.
        
        # Pre-compute A^T(y)
        ATy = x0

        # Get initial guess
        xi = x0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            xi.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(x):
                # Data consistency update -> replace the data of 
                x = x + self.step_size * (A(A(x), adjoint=True) - ATy)
                # NN update
                x = self.nn_update[i](x, t, c)

                return x
            return update_fn

        # Begin unrolled network.
        for i in range(self.num_unrolls):
            if self.do_checkpoint:
                xi = cp.checkpoint(update(i), xi)
            else:
                xi = update(i)(xi)

        return xi

    """
    def forward(self, y, t, A, c, x0=None):
        
        #Performs the forward pass through the unrolled network.
        
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
                # NN update
                x = self.nn_update[i](x, t, c)
                return x
            return update_fn

        # Begin unrolled network.
        for i in range(self.num_unrolls):
            if self.do_checkpoint:
                xi = cp.checkpoint(update(i), xi)
            else:
                xi = update(i)(xi)

        return xi
    """

class HalfQuadraticSplitting(UnrolledLatteNet):
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

    def forward(self, y, t, A, c, x0=None):
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
                z = self.nn_update[i](x, t, c)
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
