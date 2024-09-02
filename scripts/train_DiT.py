"""
Training script for MRI unrolled recon. - with SeResNet
"""
import random
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from copy import deepcopy
from collections import OrderedDict

# PyTorch lightning modules
import pytorch_lightning as L
#import lightning as L
from pytorch_lightning.callbacks import GradientAccumulationScheduler
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.profilers import AdvancedProfiler
from deepspeed.ops.adam import DeepSpeedCPUAdam
from pytorch_lightning.utilities.deepspeed import convert_zero_checkpoint_to_fp32_state_dict

# Custom dl_cs modules
from dl_cs.config import load_cfg
from dl_cs.models import unrolledDiT
from dl_cs.utils import metrics as metric
from dl_cs.mri import transforms as T
from dl_cs.data.dataset import Hdf5Dataset
from dl_cs.data.preprocess import CinePreprocess
from dl_cs.diffusion import create_diffusion
#from torch_ema import ExponentialMovingAverage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os

"""
def submask(mask, factor):
    _, _, F, H, W = mask.size()  # Assuming mask has shape (B, C, F, H, W)
    
    mask2 = mask
    for f in range(F):
        frame = mask2[:, :, f, :, :]  # Shape (B, C, H, W)
        frame_sum = torch.sum(frame, dim=1)  # Summing over channels, result is (B, H, W)
        ones_indices = frame_sum.nonzero(as_tuple=False)  # Indices of non-zero elements
        num_ones_to_remove = int(ones_indices.size(0) * factor)
        perm = torch.randperm(ones_indices.size(0))
        selected_indices = ones_indices[perm[:num_ones_to_remove]]  # Randomly select some indices
        
        for idx in selected_indices:
            b, h, w = idx  # Get batch, height, and width indices
            mask2[:, :, f, h, w] = 0  # Set the corresponding position in mask to 0

    return mask2
"""
@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        if decay == 0:
            ema_params[name] = param.data
        else: 
            ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)
       
    
    """
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            print(f"ema_param is {ema_param} and param is {param}")
            print(f"is ema_param equal to param? {torch.equal(ema_param.data, param.data)}")
            ema_param.data.copy_(ema_param.data * decay + (1 - decay)*param.data)
    """

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

class LitUnrolled(L.LightningModule):
    """
    Unrolled model inside a PyTorch Lightning module.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config
        if self.config.MODEL.STRATEGY == 'deepspeed':
            self.sync_dist = True
        else:
            self.sync_dist = False

        self.predict_xstart = True

        if config.MODEL.META_ARCHITECTURE == 'dlespirit':
            self.model = unrolledDiT.ProximalGradientDescent(config)
        elif config.MODEL.META_ARCHITECTURE == 'modl':
            self.model = unrolledDiT.HalfQuadraticSplitting(config)
        elif config.MODEL.META_ARCHITECTURE == 'DDPM_X':
            self.model = unrolledDiT.DataConsistency(config)
        elif config.MODEL.META_ARCHITECTURE == 'DDPM_E':
            self.model = unrolledDiT.DDPM(config)
            self.predict_xstart = False
        else:
            raise ValueError('Meta architecture in config file not recognized!')
        
        self.diffusion = create_diffusion(  timestep_respacing="", 
                                            noise_schedule=self.config.MODEL.PARAMETERS.NOISE_SCHED, 
                                            diffusion_steps=1000, 
                                            learn_sigma = self.config.MODEL.PARAMETERS.LEARN_SIGMA,
                                            predict_xstart = self.predict_xstart
                                            ) 
        
        self.diffusion2 = create_diffusion( timestep_respacing="", 
                                            noise_schedule=self.config.MODEL.PARAMETERS.NOISE_SCHED, 
                                            diffusion_steps=100, 
                                            learn_sigma = self.config.MODEL.PARAMETERS.LEARN_SIGMA,
                                            predict_xstart = self.predict_xstart 
                                            )
        #self.ema = ExponentialMovingAverage(self.encoder.parameters(), decay=0.995)
        #self.ema = torch.optim.swa_utils.AveragedModel(self.model, multi_avg_fn = torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
        self.ema = deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
        requires_grad(self.ema, False)

    def submask(self, mask, factor):

        _,_, F, _,_ = mask.size()
        mask_unsamp = mask.detach().clone()
        mask_inv_unsamp = mask.detach().clone()
        for f in range(F):
            
            frame = mask_unsamp[:,:,f,:,:].squeeze()
            frame = torch.sum(frame, 1)
            ones_indices = frame.nonzero(as_tuple=False)
            #print(f"ones_indices.shape is {ones_indices.shape}")
            num_remove = int(ones_indices.shape[0]*factor) 
            perm = torch.randperm(ones_indices.shape[0])
            ind = ones_indices[perm[:num_remove]]
            ind_inv = ones_indices[perm[num_remove:]]
            #print(f"selected indices.shape is {selected_indices.shape}")
            #print(f"selected_indices are {selected_indices}")
            #print(f"mask of selected indices is {mask[0,0,f,selected_indices,:]}")

            mask_unsamp[:,:,f,ind,:] = 0
            mask_inv_unsamp[:,:,f,ind_inv,:] = 0

        return mask_unsamp, mask_inv_unsamp
    
    def compute_metrics(self, prediction, target, is_training=True):
        # Initialize dictionary of metrics
        metrics = {}
        tag = 'Train' if is_training else 'Validate'

        print("in compute metrics")
        # Complex-based error metrics
        metrics[f'{tag}/complex_l1'] = metric.l1(target, prediction, config.MODEL.RECON_LOSS.LOSS_WEIGHT )
        metrics[f'{tag}/complex_l2'] = metric.l2(target, prediction, config.MODEL.RECON_LOSS.LOSS_WEIGHT )
        metrics[f'{tag}/complex_psnr'] = metric.psnr(target, prediction, config.MODEL.RECON_LOSS.LOSS_WEIGHT )
        

        if self.config.MODEL.RECON_LOSS.NAME == "complex_vggloss":
            metrics[f'{tag}/complex_vggloss'] = metric.vggloss(target, prediction)

        # Get magnitudes of each tensor
        mPrediction = prediction.abs()
        mTarget = target.abs()

        # Magnitude-based error metrics  
        metrics[f'{tag}/mag_l1'] = metric.l1(mTarget, mPrediction, config.MODEL.RECON_LOSS.LOSS_WEIGHT )
        metrics[f'{tag}/mag_l2'] = metric.l2(mTarget, mPrediction, config.MODEL.RECON_LOSS.LOSS_WEIGHT )
        metrics[f'{tag}/mag_psnr'] = metric.psnr(mTarget, mPrediction, config.MODEL.RECON_LOSS.LOSS_WEIGHT )

        if self.config.MODEL.RECON_LOSS.NAME == "mag_vggloss":
            metrics[f'{tag}/mag_vggloss'] = metric.vggloss(mTarget, mPrediction)

        #metrics[f'{tag}/Diff_loss'] = self.diffusion.training_losses(self.model, target, prediction, model_kwargs)
        return metrics

    def log_data(self, initial_guess, prediction, noise_im, target, mask, mask2, flag=0):
        
        if flag == 0:
            label = ["Magnitude", 'Phase', 'Magnitude Error', 'Mask']
        else:
            label = ["Resample Magnitude", 'Resample Phase', 'Resample Error', 'Resample Mask']

        # Helper function for logging images
        def save_image(image, tag):
            image -= image.min()
            image /= image.max()
            #image = image.repeat(1, 1, 3, 1, 1) #to convert it to "gray scale"
            self.logger.experiment.add_image(tag, image, global_step=self.global_step)

        # Helper function for logging video data
        def save_video(video, tag):
            video = video[..., None].permute(0, 1, 4, 3, 2)  # [1, t, y, x, 1] -> [1, t, 1, x, y]
            video -= video.min()
            video /= video.max()
            video = video.repeat(1, 1, 3, 1, 1) #to convert it to "gray scale"
            self.logger.experiment.add_video(tag, video, global_step=self.global_step)

        # Stack images from left-to-right: init, output, target
        images = torch.cat((initial_guess, prediction, noise_im, target), dim=3)  # [1, e, t, y, x]
        images = images[:, 0, :, :, :]  # take first ESPIRiT channel

        # Get error between output and target
        #Stack images from left-to-right: mag error from full k-space and mag error of noise_image
        mag_error = torch.abs(prediction[:, 0, :, :, :]) - torch.abs(target[:, 0, :, :, :])
        noise_error = torch.abs(prediction[:, 0, :, :, :]) - torch.abs(noise_im[:, 0, :, :, :])
        error = torch.cat((mag_error, noise_error), dim=2)

        mask_cat = torch.cat((mask[:, 0, :, :, -1], mask2[:, 0, :, :, -1]), dim=2)
       
        # Save images
        save_video(torch.abs(images), label[0])
        save_video(torch.angle(images), label[1])
        save_video(torch.abs(error), label[2])
        save_image(torch.abs(mask_cat), label[3])

        return


    def training_step(self, batch, batch_idx):
        # Load batch of input-output pairs
        #kspace, mask, maps, initial_guess, scale, target = batch
        _, mask, maps, initial_guess, scale, target = batch

        #t = torch.tensor([(100*self.global_step)%1000],device=self.device)
        t = torch.randint(0, self.diffusion.num_timesteps, (initial_guess.shape[0],), device=self.device)

        #print(f"time step t is {t}\n")
        #Dummy class to make things work
        class_label = torch.tensor([1],device=self.device)
        
        # Re-normalize data
        if self.config.MODEL.RECON_LOSS.RENORMALIZE_DATA:
            #pred *= scale
            initial_guess *= scale
            target *= scale

        if config.MODEL.META_ARCHITECTURE == 'DDPM_X':
            mask_r, mask_p = self.submask(mask, 0.9)
            model_kwargs = dict(A = T.SenseModel(maps, weights=mask_p), 
                                A_1 = T.SenseModel(maps, weights=1-mask_p), 
                                A_F = T.SenseModel(maps), 
                                A_S = T.SenseModel(maps, weights=mask_r),
                                fs = target, 
                                c=class_label)
            #loss_dict, pred, init_guess_noise = self.diffusion.training_kspace_loss(self.model, initial_guess, t, model_kwargs)
            loss_dict, pred, init_guess_noise = self.diffusion.training_kspace_loss(self.model, target, t, model_kwargs)
        
        elif config.MODEL.META_ARCHITECTURE == 'DDPM_E':
            mask_r = mask   #dummy to make the log consistent
            mask_p = mask   #dummy to make the log consistent
            model_kwargs = dict(A = T.SenseModel(maps, weights=mask), 
                                A_1 = T.SenseModel(maps, weights=1-mask), 
                                A_F = T.SenseModel(maps), 
                                fs = target,
                                c=class_label)
            #loss_dict, pred, init_guess_noise = self.diffusion.training_losses(self.model, initial_guess, t, model_kwargs)
            loss_dict, pred, init_guess_noise = self.diffusion.training_losses(self.model, target, t, model_kwargs) #Train on fully sampled data

        # Log metrics
        metrics = {}
        metrics["Train MSE"] = loss_dict["loss"]
        self.log_dict(metrics, sync_dist=self.sync_dist)
 
        # Log images
        if (self.global_step+1) % self.config.LOGGER.LOG_IMAGES_EVERY_N_STEPS == 0:
            self.log_data(initial_guess, pred, init_guess_noise, target, mask_p, mask_r)

        
        if (self.global_step+1) % self.config.LOGGER.LOG_PREDICTION_EVERY_N_STEPS == 0:
            with torch.no_grad():
                #Don't use submask for model prediction 
                model_kwargs["A"] = T.SenseModel(maps, weights=mask)
                model_kwargs["A_1"] = T.SenseModel(maps, weights=1-mask)
                gen_im = self.diffusion2.p_sample_loop_conditional(self.model, initial_guess.shape, initial_guess, clip_denoised=False, 
                                                model_kwargs=model_kwargs, progress=True, device=self.device
                )
            self.log_data(initial_guess, gen_im, init_guess_noise, target, mask_p, mask_r, flag=1)
           
        # Get training loss (specified by config file)
        loss = loss_dict["loss"]
        # loss = metrics[f'Train/{self.config.MODEL.RECON_LOSS.NAME}']
        
        return loss

    def validation_step(self, batch, batch_idx):
        # Load batch of input-output pairs

        _, mask, maps, initial_guess, scale, target = batch

        #t = torch.tensor([(100*self.global_step)%1000],device=self.device)
        t = torch.randint(0, self.diffusion.num_timesteps, (initial_guess.shape[0],), device=self.device)
    
        #print(f"time step t is {t}\n")
        #Dummy class to make things work
        class_label = torch.tensor([1],device=self.device)
        
        # Re-normalize data
        if self.config.MODEL.RECON_LOSS.RENORMALIZE_DATA:
            #pred *= scale
            initial_guess *= scale
            target *= scale

        if config.MODEL.META_ARCHITECTURE == 'DDPM_X':
            mask_r, mask_p = self.submask(mask, 0.9)
            model_kwargs = dict(A = T.SenseModel(maps, weights=mask_p), 
                                A_1 = T.SenseModel(maps, weights=1-mask_p), 
                                A_F = T.SenseModel(maps), 
                                A_S = T.SenseModel(maps, weights=mask_r), 
                                fs = target, 
                                c=class_label)
            loss_dict, _, _ = self.diffusion.training_kspace_loss(self.model, initial_guess, t, model_kwargs)
        
        elif config.MODEL.META_ARCHITECTURE == 'DDPM_E':
            mask_r = mask   #dummy to make the log consistent
            mask_p = mask   #dummy to make the log consistent
            model_kwargs = dict(A = T.SenseModel(maps, weights=mask), 
                                A_1 = T.SenseModel(maps, weights=1-mask), 
                                A_F = T.SenseModel(maps), 
                                fs = target, 
                                c=class_label)
            loss_dict, _, _ = self.diffusion.training_losses(self.model, initial_guess, t, model_kwargs)
        
        # Log metrics
        metrics = {}
        metrics["Validate MSE"] = loss_dict["loss"]
        self.log_dict(metrics, sync_dist=self.sync_dist)


    def configure_optimizers(self):

        if self.config.MODEL.STRATEGY == 'deepspeed':
            # DeepSpeedCPUAdam provides 5x to 7x speedup over torch.optim.adam(w)
            return DeepSpeedCPUAdam(self.parameters())
        
        else: 
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.OPTIMIZER.ADAM.LR)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.config.LR_SCHEDULER.STEP_SIZE,
                                                    gamma=self.config.LR_SCHEDULER.GAMMA)
            return [optimizer], [scheduler]

    def configure_callbacks(self):

        if self.config.MODEL.STRATEGY == 'deepspeed':
            #convert_zero_checkpoint_to_fp32_state_dict(self.config.OUTPUT_DIR, "last.ckpt")
            checkpoint = ModelCheckpoint(
                dirpath=self.config.OUTPUT_DIR,
                save_top_k=1,
                monitor='Validate MSE',
                #monitor=f'Validate/{self.config.MODEL.RECON_LOSS.NAME}',
                mode='min',
                verbose=True
            )
            return [checkpoint]
        else:
            # Configure checkpoint callback
            checkpoint = ModelCheckpoint(
                dirpath=self.config.OUTPUT_DIR,
                save_top_k=1,
                monitor='Validate MSE',
                #monitor=f'Validate/{self.config.MODEL.RECON_LOSS.NAME}',
                mode='min',
                verbose=True
            )
            return [checkpoint]
        

    def train_dataloader(self):
        print("In train dataloader\n")
        # Initialize transform function (performs pre-processing on each batch)
        print("In preprocess")
        preprocess = CinePreprocess(self.config, use_seed=False)

        # Intialize data loader
        train_data = Hdf5Dataset(root_directory=self.config.DATASET.TRAIN[0], transform=preprocess)
        loader = DataLoader(
            dataset=train_data,
            batch_size=self.config.DATALOADER.TRAIN_BATCH_SIZE,
            num_workers=self.config.DATALOADER.NUM_WORKERS,
            pin_memory=True,
            shuffle=True
        )
        return loader

    def val_dataloader(self):
        print("In Validate dataloader\n")
        # Initialize transform function (performs pre-processing on each batch)
        print("In preprocess")
        preprocess = CinePreprocess(self.config, use_seed=True)

        
        # Intialize data loader
        val_data = Hdf5Dataset(root_directory=self.config.DATASET.VAL[0], transform=preprocess)
        loader = DataLoader(
            dataset=val_data,
            batch_size=self.config.DATALOADER.VAL_BATCH_SIZE,
            num_workers=self.config.DATALOADER.NUM_WORKERS,
            pin_memory=True,
            shuffle=False
        )
        return loader
    
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        #self.ema.update_parameters(self.model)
        update_ema(self.ema, self.model)

    def on_train_start(self):
        print("On train Start Synchronizing EMA weights")
        #self.ema = deepcopy(self.model).to(self.device)  # Create an EMA of the model for use after training
        #requires_grad(self.ema, False)
        update_ema(self.ema, self.model, decay=0)  # Ensure EMA is initialized with synced weights
    
    """
    def on_before_batch_transfer(self, batch, dataloader_idx=0):
        print("on_before_batch_transfer")
        return batch 
    
    def on_after_batch_transfer(self, batch, dataloader_idx=0):
        print("on_after_batch_transfer")
        return batch
    
    def on_train_batch_start (self, batch, batch_idx, unused = 0):
        print("on_train_batch_start")
        return batch

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0):
        print("on_train_batch_end")
    
    def on_train_start(self):
        print("on_train_start")
    
    def on_train_end(self):
        print("on_train_end")

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        print("on_before_optimizer_step")

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        print("on_validation_batch_start")

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        print("on_validation_batch_end")

    def on_validation_start(self):
        print("on_validation_start")

    def on_validation_end(self):
        print("on_validation_end")

    def on_fit_start(self):
        print("on_fit_start")
    
    def on_fit_end(self):
        print("on_fit_end")

    def on_before_zero_grad(self, optimizer):
        print("on_before_zero_grad")

    #def on_before_zero_grad(self, *args, **kwargs):
     #   print("on_before_optimizer_step")
        #self.ema.update(self.model.parameters())

    def on_before_optimizer_step(self, optimizer, optimizer_idx=0):
        print("on_before_optimizer_step")

    def on_before_backward(self, loss):
        print("on_before_backward")

    def on_after_backward(self):
        print("on_after_backward")
"""
    
    
def main(config, device, ckpt_file=None):

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    
    # Initialize unrolled model
    unrolled = LitUnrolled(config)

    # GPU has tensor cores, so do the following: 
    torch.set_float32_matmul_precision('medium' or 'high')

    #Advanced profiler
    profiler = AdvancedProfiler(dirpath=config.OUTPUT_DIR, filename="profiler-logs")

    # Initialize logger (for writing summary to TensorBoard)
    tb_logger = TensorBoardLogger(save_dir=config.OUTPUT_DIR, name="exp")

    # Initialize a trainer
    if config.MODEL.STRATEGY == 'deepspeed':
        trainer = L.Trainer(devices=device,
                        logger=tb_logger,
                        max_epochs=config.OPTIMIZER.MAX_EPOCHS,
                        log_every_n_steps=config.LOGGER.LOG_METRICS_EVERY_N_STEPS,
                        check_val_every_n_epoch=config.EVAL.RUN_EVERY_N_EPOCHS,
                        profiler = profiler,
                        #strategy="deepspeed_stage_3_offload",
                        #accelerator="gpu",
                        strategy=DeepSpeedStrategy(
                            stage=3,
                            offload_optimizer=True,
                            offload_parameters=True,
                            offload_optimizer_device="cpu",
                            offload_params_device="cpu",
                        )
        )
    
    else:
        trainer = L.Trainer(
                        accelerator ="gpu", 
                        devices=device,
                        logger=tb_logger,
                        max_epochs=config.OPTIMIZER.MAX_EPOCHS,
                        log_every_n_steps=config.LOGGER.LOG_METRICS_EVERY_N_STEPS,
                        check_val_every_n_epoch=config.EVAL.RUN_EVERY_N_EPOCHS,
                        profiler = profiler,
        )  

    # Train the model ⚡
    if ckpt_file is not None:
        trainer.fit(unrolled, ckpt_path=ckpt_file)
    else:
        trainer.fit(unrolled)

    return


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Training script for unrolled MRI recon.")
    parser.add_argument('--config-file', type=str, required=True, help='Training config file (yaml)')
    parser.add_argument('--resume', action='store_true', help='Resume training from checkpoint')
    parser.add_argument('--ckpt', type=str, help='Checkpoint file to resume training from')
    parser.add_argument('--devices', type=int, nargs='+', help='GPU devices')
    parser.add_argument('--verbose', action='store_true', help='Turn on debug statements')
    args = parser.parse_args()

    # Load config file
    config = load_cfg(args.config_file)

    # Set random seeds
    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    # Get checkpoint file if resume flag is on
    ckpt_file = args.ckpt if args.resume else None

    main(config, args.devices, ckpt_file)
