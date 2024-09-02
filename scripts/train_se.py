"""
Training script for MRI unrolled recon. - with SeResNet
"""
import random
import logging
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

# PyTorch lightning modules
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Custom dl_cs modules
from dl_cs.config import load_cfg
from dl_cs.models import unrolledSE
from dl_cs.models import unrolled
from dl_cs.utils import metrics as metric
from dl_cs.mri import transforms as T
from dl_cs.data.dataset import Hdf5Dataset
from dl_cs.data.preprocess import CinePreprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LitUnrolled(pl.LightningModule):
    """
    Unrolled model inside a PyTorch Lightning module.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.MODEL.META_ARCHITECTURE == 'dlespirit':
            self.model = unrolledSE.ProximalGradientDescent(config)
        elif config.MODEL.META_ARCHITECTURE == 'modl':
            self.model = unrolledSE.HalfQuadraticSplitting(config)
        else:
            raise ValueError('Meta architecture in config file not recognized!')

    def compute_metrics(self, prediction, target, is_training=True):
        # Initialize dictionary of metrics
        metrics = {}
        tag = 'Train' if is_training else 'Validate'

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

        if self.config.MODEL.RECON_LOSS.NAME == "complex_vggloss":
            metrics[f'{tag}/mag_vggloss'] = metric.vggloss(mTarget, mPrediction)

        return metrics

    def log_data(self, initial_guess, prediction, target, mask):
        # Helper function for logging images
        def save_image(image, tag):
            image -= image.min()
            image /= image.max()
            self.logger.experiment.add_image(tag, image, global_step=self.global_step)

        # Helper function for logging video data
        def save_video(video, tag):
            video = video[..., None].permute(0, 1, 4, 3, 2)  # [1, t, y, x, 1] -> [1, t, 1, x, y]
            video -= video.min()
            video /= video.max()
            self.logger.experiment.add_video(tag, video, global_step=self.global_step)

        # Stack images from left-to-right: init, output, target
        images = torch.cat((initial_guess, prediction, target), dim=3)  # [1, e, t, y, x]
        images = images[:, 0, :, :, :]  # take first ESPIRiT channel

        # Get error between output and target
        mag_error = torch.abs(prediction[:, 0, :, :, :]) - torch.abs(target[:, 0, :, :, :])

        # Save images
        save_video(torch.abs(images), 'Magnitude')
        save_video(torch.angle(images), 'Phase')
        save_video(torch.abs(mag_error), 'MagnitudeError')
        save_image(torch.abs(mask[:, 0, :, :, -1]), 'Mask')

        return

    def training_step(self, batch, batch_idx):
        # Load batch of input-output pairs
        kspace, mask, maps, initial_guess, scale, target = batch

        # Perform forward pass through unrolled network
        pred = self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)

        # Re-normalize data before computing loss functions
        if self.config.MODEL.RECON_LOSS.RENORMALIZE_DATA:
            pred *= scale
            initial_guess *= scale
            target *= scale

        # Compute and log metrics
        metrics = self.compute_metrics(pred, target, is_training=True)
        self.log_dict(metrics)

        # Log images
        if (self.global_step+1) % self.config.LOGGER.LOG_IMAGES_EVERY_N_STEPS == 0:
            self.log_data(initial_guess, pred, target, mask)

        # Get training loss (specified by config file)
        loss = metrics[f'Train/{self.config.MODEL.RECON_LOSS.NAME}']

        return loss

    def validation_step(self, batch, batch_idx):
        # Load batch of input-output pairs
        kspace, mask, maps, initial_guess, scale, target = batch

        # Perform forward pass through unrolled network
        pred = self.model(y=kspace, A=T.SenseModel(maps, weights=mask), x0=initial_guess)

        # Re-normalize data before computing loss functions
        if self.config.MODEL.RECON_LOSS.RENORMALIZE_DATA:
            pred *= scale
            initial_guess *= scale
            target *= scale

        # Compute and log metrics
        metrics = self.compute_metrics(pred, target, is_training=False)
        self.log_dict(metrics)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.OPTIMIZER.ADAM.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=self.config.LR_SCHEDULER.STEP_SIZE,
                                                    gamma=self.config.LR_SCHEDULER.GAMMA)
        return [optimizer], [scheduler]

    def configure_callbacks(self):
        # Configure checkpoint callback
        checkpoint = ModelCheckpoint(
            dirpath=self.config.OUTPUT_DIR,
            save_top_k=1,
            monitor=f'Validate/{self.config.MODEL.RECON_LOSS.NAME}',
            mode='min',
            verbose=True
        )
        return [checkpoint]

    def train_dataloader(self):
        # Initialize transform function (performs pre-processing on each batch)
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
        # Initialize transform function (performs pre-processing on each batch)
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


def main(config, devices, ckpt_file=None):
    # Initialize unrolled model
    unrolled = LitUnrolled(config)

    # Initialize logger (for writing summary to TensorBoard)
    tb_logger = TensorBoardLogger(save_dir=config.OUTPUT_DIR, name="exp")

    # Initialize a trainer
    trainer = Trainer(gpus=devices,
                      logger=tb_logger,
                      max_epochs=config.OPTIMIZER.MAX_EPOCHS,
                      log_every_n_steps=config.LOGGER.LOG_METRICS_EVERY_N_STEPS,
                      check_val_every_n_epoch=config.EVAL.RUN_EVERY_N_EPOCHS,
                      accumulate_grad_batches=config.OPTIMIZER.GRAD_ACCUM_ITERS,
                      resume_from_checkpoint=ckpt_file
    )

    # Train the model ⚡
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
