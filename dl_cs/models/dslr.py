"""
Unrolled Compressed Sensing (3D)
by Christopher M. Sandino (sandino@stanford.edu), 2020.

"""

import torch
from torch import nn
import torch.utils.checkpoint as cp

from dl_cs.models.rnn import RNN
from dl_cs.models.resnet1d import ResNet as ResNet1D
from dl_cs.models.resnet2d import ResNet as ResNet2D
from dl_cs.mri.algorithms import ConjugateGradient
from dl_cs.mri.algorithms import PowerMethod


class UnrolledLRNet(nn.Module):
    """
    Abstract class for an unrolled neural network.
    """
    def __init__(self, config):
        super(UnrolledLRNet, self).__init__()

        # Load config parameters
        self.num_unrolls = config.MODEL.PARAMETERS.NUM_UNROLLS
        self.num_resblocks = config.MODEL.PARAMETERS.NUM_RESBLOCKS
        self.num_features = config.MODEL.PARAMETERS.NUM_FEATURES
        self.kernel_size = config.MODEL.PARAMETERS.CONV_BLOCK.KERNEL_SIZE[0]
        self.num_emaps = config.MODEL.PARAMETERS.NUM_EMAPS
        self.share_weights = config.MODEL.PARAMETERS.SHARE_WEIGHTS
        self.fix_step_size = config.MODEL.PARAMETERS.FIX_STEP_SIZE
        self.use_complex_layers = config.MODEL.PARAMETERS.CONV_BLOCK.COMPLEX
        self.circular_pad = config.MODEL.PARAMETERS.CONV_BLOCK.CIRCULAR_PAD
        self.do_checkpoint = config.MODEL.PARAMETERS.GRAD_CHECKPOINT
        self.block_size = config.MODEL.PARAMETERS.DSLR.BLOCK_SIZE
        self.num_basis = config.MODEL.PARAMETERS.DSLR.NUM_BASIS

        # Initialize 2D and 1D networks
        self.spatial_cnn_update = self.init_spatial_nets()
        self.temporal_cnn_update = self.init_temporal_nets()

    def init_spatial_nets(self):
        """
        """
        # Figure out how many input channels does this data have
        if self.use_complex_layers:
            in_chans = self.num_emaps * self.num_basis
        else:
            in_chans = 2 * self.num_emaps * self.num_basis

        # Read in network parameters
        resnet_params = dict(in_chans=in_chans,
                             chans=self.num_features,
                             num_resblocks=self.num_resblocks,
                             use_complex_layers=self.use_complex_layers,
                             kernel_size=self.kernel_size,
                             circular_pad=False,  # don't need to circularly pad spatial basis
                             )

        # Declare CNNs for each unrolled iteration
        if self.share_weights:
            # Initializes copies of the same network
            nets = nn.ModuleList([ResNet2D(**resnet_params)] * self.num_unrolls)
        else:
            # Initializes unique networks
            nets = nn.ModuleList([ResNet2D(**resnet_params) for _ in range(self.num_unrolls)])

        return nets

    def init_temporal_nets(self):
        """
        """
        # Figure out how many input channels does this data have
        if self.use_complex_layers:
            in_chans = self.num_basis
        else:
            in_chans = 2 * self.num_basis

        # Read in network parameters
        resnet_params = dict(in_chans=in_chans,
                             chans=self.num_features,
                             num_resblocks=self.num_resblocks,
                             use_complex_layers=self.use_complex_layers,
                             kernel_size=self.kernel_size,
                             circular_pad=self.circular_pad,
                             )

        # Declare CNNs for each unrolled iteration
        if self.share_weights:
            # Initializes copies of the same network
            nets = nn.ModuleList([ResNet1D(**resnet_params)] * self.num_unrolls)
        else:
            # Initializes unique networks
            nets = nn.ModuleList([ResNet1D(**resnet_params) for _ in range(self.num_unrolls)])

        return nets

    def init_recurrent_nets(self):
        # Figure out how many input channels does this data have
        if self.use_complex_layers:
            in_chans = self.num_basis
        else:
            in_chans = 2 * self.num_basis

        # Read in network parameters
        rnn_params = dict(in_chans=in_chans,
                          hidden_size=self.num_features,
                          num_layers=3,
                          bidirectional=True)

        # Declare RNNs for each unrolled iteration
        if self.share_weights:
            # Initializes copies of the same network
            nets = nn.ModuleList([RNN(**rnn_params)] * self.num_unrolls)
        else:
            # Initializes unique networks
            nets = nn.ModuleList([RNN(**rnn_params) for _ in range(self.num_unrolls)])

        return nets

    def btranspose(self, matrix: torch.Tensor) -> torch.Tensor:
        """
        Takes Hermitian transpose of batch of matrices
        """
        return matrix.conj().permute(0, 2, 1)

    def compose(self,
                L: torch.Tensor,
                R: torch.Tensor,
                BlockOp: nn.Module) -> torch.Tensor:
        """
        Converts LR representation into images.
        """
        blocks = torch.bmm(L, self.btranspose(R))
        images = BlockOp(blocks, adjoint=True)
        return images

    def cnn_update_L(self, L, iter):
        """
        Perform spatial CNN update on L
        """
        # Get shapes for reshaping
        num_blocks = L.shape[0]
        shape_before = (num_blocks, self.num_basis*self.num_emaps, self.block_size, self.block_size)
        shape_after = (num_blocks, self.num_basis, self.num_emaps*(self.block_size**2))

        # Perform L-CNN update
        L = L.permute(0, 2, 1).reshape(shape_before)
        L = self.spatial_cnn_update[iter](L)
        L = L.reshape(shape_after).permute(0, 2, 1)

        return L

    def cnn_update_R(self, R, iter):
        """
        Perform spatial CNN update on L
        """
        # Perform R-CNN update
        R = R.permute(0, 2, 1)
        R = self.temporal_cnn_update[iter](R)
        R = R.permute(0, 2, 1)

        return R

    def forward(self, y, A, BlockOp, L0, R0):
        """
        Performs the forward pass through the unrolled network.
        """
        raise NotImplementedError


class AltMinPGD(UnrolledLRNet):
    """
    Implementation of alternating minimization solver to solve the following
    low-rank optimization problem:

        argmin_{L,R} || Y - A(LR^H) ||_F^2 + Psi(L) + Phi(R)
    """

    def __init__(self, config):
        super().__init__(config)

        # Initialize power method algorithm
        self.power_method = PowerMethod(num_iter=10)

    def get_step_sizes(self, L, R, alpha=0.9):
        """
        Computes step size based on eigenvalues of L'*L and R'*R.
        """
        # Compute L step size (based on R)
        E = self.power_method(R)
        step_size_L = -1.0 * alpha / E.max()

        # Compute R step size (based on L)
        E = self.power_method(L)
        step_size_R = -1.0 * alpha / E.max()

        return step_size_L, step_size_R

    def dc_update(self, L_prev, R_prev, A, BlockOp, ATy):
        """
        Use CG to perform direct model inversion and solve for new L, R
        """

        # Compute gradients of ||Y - ALR'||_2 w.r.t. L, R
        grad_x = BlockOp(A(A(self.compose(L_prev, R_prev, BlockOp)), adjoint=True) - ATy)
        grad_L = torch.bmm(grad_x, R_prev)
        grad_R = torch.bmm(self.btranspose(grad_x), L_prev)

        # L, R model updates
        step_size_L, step_size_R = self.get_step_sizes(L_prev, R_prev)
        L_next = L_prev + step_size_L * grad_L
        R_next = R_prev + step_size_R * grad_R

        return L_next, R_next

    def forward(self, y, A, BlockOp, L0, R0):
        # Pre-compute ATy for convenience
        ATy = A(y, adjoint=True)

        # Get initial guesses
        Li = L0
        Ri = R0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            Li.requires_grad_()
            Ri.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(L, R):
                # dc updates
                L, R = self.dc_update(L, R, A, BlockOp, ATy)

                # prox updates
                L = self.cnn_update_L(L, i)
                R = self.cnn_update_R(R, i)

                return L, R

            return update_fn

        # Iteratively update L, R basis functions
        for i in range(self.num_unrolls):
            if self.training and self.do_checkpoint:
                Li, Ri = cp.checkpoint(update(i), Li, Ri)
            else:
                Li, Ri = update(i)(Li, Ri)

        # Compose L, R into image
        xi = self.compose(Li, Ri, BlockOp)

        return xi


class AltMinCGv1(UnrolledLRNet):
    """
    Implementation of alternating minimization solver to solve the following
    low-rank optimization problem:

        argmin_{L,R} || Y - A(LR^H) ||_F^2 + Psi(L) + Phi(R)
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_cg_iter = config.MODEL.PARAMETERS.DSLR.NUM_CG_STEPS

    def dc_update_L(self, L_update, R_fixed, A, ATy, BlockOp):
        """
        Apply data consistency to L using conjugate gradient while keeping R fixed.
        """

        # Define normal equations for L
        def model_normal(L):
            x = BlockOp(A(A(self.compose(L, R_fixed, BlockOp)), adjoint=True))
            return torch.bmm(x, R_fixed)  # [N, b^2*e, nt] x [N, nt, nr] = [N, b^2*e, nr]

        # Run DC update (using CG solver)
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        L_update = cg_solve(L_update, torch.bmm(ATy, R_fixed))

        return L_update

    def dc_update_R(self, R_update, L_fixed, A, ATy, BlockOp):
        """
        Apply data consistency to R using conjugate gradient while keeping L fixed.
        """

        # Define normal equations for R
        def model_normal(R):
            x = BlockOp(A(A(self.compose(L_fixed, R, BlockOp)), adjoint=True))
            return torch.bmm(self.btranspose(x), L_fixed)  # [N, nt, b^2*e] x [N, b^2*e, nr] = [N, nt, nr]

        # Run DC update (using CG solver)
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        R_update = cg_solve(R_update, torch.bmm(self.btranspose(ATy), L_fixed))

        return R_update

    def forward(self, y, A, BlockOp, L0, R0):
        # Pre-compute ATy for convenience
        ATy = BlockOp(A(y, adjoint=True))  # [N, b^2*e, t]

        # Get initial guesses
        Li = L0
        Ri = R0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            Li.requires_grad_()
            Ri.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(L, R):
                # Perform L-, R- DC updates
                L = self.dc_update_L(L, R, A, ATy, BlockOp)
                R = self.dc_update_R(R, L, A, ATy, BlockOp)

                # Perform L-, R- CNN updates
                L = self.cnn_update_L(L, i)
                R = self.cnn_update_R(R, i)

                return L, R

            return update_fn

        # Iteratively update L, R basis functions
        for i in range(self.num_unrolls):
            if self.training and self.do_checkpoint:
                Li, Ri = cp.checkpoint(update(i), Li, Ri)
            else:
                Li, Ri = update(i)(Li, Ri)

        # Compose L, R into image
        xi = self.compose(Li, Ri, BlockOp)

        return xi


class AltMinCGv2(UnrolledLRNet):
    """
    Implementation of alternating minimization solver to solve the following
    low-rank optimization problem:

        argmin_{L,R} || Y - A(LR^H) ||_F^2 + Psi(L) + Phi(R)
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_cg_iter = config.MODEL.PARAMETERS.DSLR.NUM_CG_STEPS

    def dc_update_L(self, L_update, R_fixed, A, ATy, BlockOp):
        """
        Apply data consistency to L using conjugate gradient while keeping R fixed.
        """
        # Define normal equations for L
        def model_normal(L):
            x = BlockOp(A(A(self.compose(L, R_fixed, BlockOp)), adjoint=True))
            return torch.bmm(x, R_fixed)  # [N, b^2*e, nt] x [N, nt, nr] = [N, b^2*e, nr]

        # Run DC update (using CG solver)
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        L_update = cg_solve(L_update, torch.bmm(ATy, R_fixed))

        return L_update

    def dc_update_R(self, R_update, L_fixed, A, ATy, BlockOp):
        """
        Apply data consistency to R using conjugate gradient while keeping L fixed.
        """
        # Define normal equations for R
        def model_normal(R):
            x = BlockOp(A(A(self.compose(L_fixed, R, BlockOp)), adjoint=True))
            return torch.bmm(self.btranspose(x), L_fixed)  # [N, nt, b^2*e] x [N, b^2*e, nr] = [N, nt, nr]

        # Run DC update (using CG solver)
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        R_update = cg_solve(R_update, torch.bmm(self.btranspose(ATy), L_fixed))

        return R_update

    def forward(self, y, A, BlockOp, L0, R0):
        # Pre-compute ATy for convenience
        ATy = BlockOp(A(y, adjoint=True))  # [N, b^2*e, t]

        # Get initial guesses
        Li = L0
        Ri = R0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            Li.requires_grad_()
            Ri.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(L, R):
                # Update L
                L = self.dc_update_L(L, R, A, ATy, BlockOp)
                L = self.cnn_update_L(L, i)

                # Update R
                R = self.dc_update_R(R, L, A, ATy, BlockOp)
                R = self.cnn_update_R(R, i)

                return L, R

            return update_fn

        # Iteratively update L, R basis functions
        for i in range(self.num_unrolls):
            if self.training and self.do_checkpoint:
                Li, Ri = cp.checkpoint(update(i), Li, Ri)
            else:
                Li, Ri = update(i)(Li, Ri)

        # Compose L, R into image
        xi = self.compose(Li, Ri, BlockOp)

        return xi


class AltMinMoDLv1(UnrolledLRNet):
    """
    Implementation of alternating minimization solver to solve the following
    low-rank optimization problem:

        argmin_{L,R} || Y - A(LR^H) ||_F^2 + lambda_l * ||CNN(L) - L|| + lambda_r * ||CNN(R) - R||

    We can split this up into two convex sub-problems:

       (1) argmin_{L} || Y - A(LR^H) ||_F^2 + lambda_l * ||CNN(L) - L||
       (2) argmin_{R} || Y - A(LR^H) ||_F^2 + lambda_r * ||CNN(R) - R||

    Each sub-problem is repeatedly solved using MoDL (Aggarwal, et al. IEEE TMI, 2017)
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_cg_iter = config.MODEL.PARAMETERS.DSLR.NUM_CG_STEPS
        self.lambda_l = nn.Parameter(torch.tensor([1.0], dtype=torch.float32),
                                     requires_grad=(not self.fix_step_size))
        self.lambda_r = nn.Parameter(torch.tensor([2.0], dtype=torch.float32),
                                     requires_grad=(not self.fix_step_size))

    def dc_update_L(self, L_update, L_cnn, R_fixed, A, ATy, BlockOp):
        """

        """
        # Define normal equations for L
        def model_normal(L):
            x = BlockOp(A(A(self.compose(L, R_fixed, BlockOp)), adjoint=True))
            return torch.baddbmm(self.lambda_l * L, x, R_fixed)  # lam*L + x @ R

        # Solve for L
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        rhs = torch.baddbmm(self.lambda_l * L_cnn, ATy, R_fixed)  # lam*L + ATy @ R
        L_update = cg_solve(L_update, rhs)

        return L_update

    def dc_update_R(self, R_update, R_cnn, L_fixed, A, ATy, BlockOp):
        """

        """
        # Define normal equations for R
        def model_normal(R):
            x = BlockOp(A(A(self.compose(L_fixed, R, BlockOp)), adjoint=True))
            return torch.baddbmm(self.lambda_r * R, self.btranspose(x), L_fixed)  # lam*R + x^H @ L

        # Solve for R
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        rhs = torch.baddbmm(self.lambda_r * R_cnn, self.btranspose(ATy), L_fixed)  # lam*R + ATy^H @ L
        R_update = cg_solve(R_update, rhs)

        return R_update

    def forward(self, y, A, BlockOp, L0, R0):
        # Pre-compute ATy for convenience
        ATy = BlockOp(A(y, adjoint=True))  # [N, b^2*e, t]

        # Get initial guesses
        Li = L0
        Ri = R0

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            Li.requires_grad_()
            Ri.requires_grad_()

        # Define update rule
        def update(i):
            def update_fn(L, R):
                # Perform DC and CNN updates on L
                zL = self.cnn_update_L(L, i)
                L = self.dc_update_L(L, zL, R, A, ATy, BlockOp)

                # Perform DC and CNN updates on R
                zR = self.cnn_update_R(R, i)
                R = self.dc_update_R(R, zR, L, A, ATy, BlockOp)

                return L, R

            return update_fn

        # Iteratively update L, R basis functions
        for i in range(self.num_unrolls):
            if self.training and self.do_checkpoint:
                Li, Ri = cp.checkpoint(update(i), Li, Ri)
            else:
                Li, Ri = update(i)(Li, Ri)

        # Compose L, R into image
        xi = self.compose(Li, Ri, BlockOp)

        return xi


class AltMinMoDLv2(UnrolledLRNet):
    """
    Implementation of alternating minimization solver to solve the following
    low-rank optimization problem:

        argmin_{L,R} || Y - A(LR^H) ||_F^2 + lambda_l * ||CNN(L) - L|| + lambda_r * ||CNN(R) - R||

    We can split this up into two convex sub-problems:

       (1) argmin_{L} || Y - A(LR^H) ||_F^2 + lambda_l * ||CNN(L) - L||
       (2) argmin_{R} || Y - A(LR^H) ||_F^2 + lambda_r * ||CNN(R) - R||

    Each sub-problem is repeatedly solved using MoDL (Aggarwal, et al. IEEE TMI, 2017)
    """

    def __init__(self, config):
        super().__init__(config)

        self.num_cg_iter = config.MODEL.PARAMETERS.DSLR.NUM_CG_STEPS
        self.lambda_l = nn.Parameter(torch.tensor([5e-3], dtype=torch.float32),
                                     requires_grad=(not self.fix_step_size))
        self.lambda_r = nn.Parameter(torch.tensor([5e-3], dtype=torch.float32),
                                     requires_grad=(not self.fix_step_size))
        self.lambda_scale = 1e2  # increases learning rate for lambda parameters

    def dc_update_L(self, L_update, L_cnn, R_fixed, A, ATy, BlockOp):
        """

        """
        # Get penalty parameter
        lamda = self.lambda_scale * torch.clamp(self.lambda_l, min=0.0, max=None)

        # Define normal equations for L
        def model_normal(L):
            x = BlockOp(A(A(self.compose(L, R_fixed, BlockOp)), adjoint=True))
            return torch.baddbmm(lamda * L, x, R_fixed)  # lam*L + x @ R

        # Solve for L
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        rhs = torch.baddbmm(lamda * L_cnn, ATy, R_fixed)  # lam*L + ATy @ R
        L_update = cg_solve(L_update, rhs)

        return L_update

    def dc_update_R(self, R_update, R_cnn, L_fixed, A, ATy, BlockOp):
        """

        """
        # Get penalty parameter
        lamda = self.lambda_scale * torch.clamp(self.lambda_r, min=0.0, max=None)

        # Define normal equations for R
        def model_normal(R):
            x = BlockOp(A(A(self.compose(L_fixed, R, BlockOp)), adjoint=True))
            return torch.baddbmm(lamda * R, self.btranspose(x), L_fixed)  # lam*R + x^H @ L

        # Solve for R
        cg_solve = ConjugateGradient(model_normal, self.num_cg_iter)
        rhs = torch.baddbmm(lamda * R_cnn, self.btranspose(ATy), L_fixed)  # lam*R + ATy^H @ L
        R_update = cg_solve(R_update, rhs)

        return R_update

    def forward(self, y, A, BlockOp, L0, R0):
        # Pre-compute ATy for convenience
        ATy = BlockOp(A(y, adjoint=True))  # [N, b^2*e, t]

        # Get initial guesses
        Li = L0
        Ri = R0
        zLi = torch.zeros_like(L0)
        zRi = torch.zeros_like(R0)

        # Necessary when gradient checkpointing is on
        if self.training and self.do_checkpoint:
            Li.requires_grad_()
            zLi.requires_grad_()
            Ri.requires_grad_()
            zRi.requires_grad()

        # Define update rule
        def update(i):
            def update_fn(L, zL, R, zR):
                # Perform DC and CNN updates on L
                if i == 0:
                    # For R_fixed - use the initial guess
                    L = self.dc_update_L(L, zL, R, A, ATy, BlockOp)
                else:
                    # For R_fixed - use output of CNN from the previous iteration
                    L = self.dc_update_L(L, zL, zR, A, ATy, BlockOp)
                zL = self.cnn_update_L(L, i)

                # Perform DC and CNN updates on R
                R = self.dc_update_R(R, zR, zL, A, ATy, BlockOp)
                zR = self.cnn_update_R(R, i)

                return L, zL, R, zR

            return update_fn

        # Iteratively update L, R basis functions
        for i in range(self.num_unrolls):
            if self.training and self.do_checkpoint:
                Li, zLi, Ri, zRi = cp.checkpoint(update(i), Li, zLi, Ri, zRi)
            else:
                Li, zLi, Ri, zRi = update(i)(Li, zLi, Ri, zRi)

        # Compose L, R into image
        xi = self.compose(zLi, zRi, BlockOp)

        return xi
