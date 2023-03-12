import torch
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
from monai.transforms import (
    AsDiscrete,
    Compose,
)
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch


class UNetSegmentation(pl.LightningModule):
    """
    Unet model for skull segmentation.
    spatial_dim: int, spatial dimension for UNet model, default to 3
    channels: tuple, sequence of channels, default to (16, 32, 64, 128, 256)
    lr: float, learning rate, default to 1e-4
    step_size: int, learning rate step, default to 10000
    """

    def __init__(
        self,
        spatial_dim: int = 3,
        channels: tuple = (16, 32, 64, 128, 256),
        lr: float = 1e-4,
        step_size: int = 10000,
    ):
        super().__init__()
        self.unet_model = UNet(
            spatial_dims=spatial_dim,
            in_channels=1,
            out_channels=2,
            channels=channels,
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )
        self.loss_function = DiceLoss(to_onehot_y=True, softmax=True)
        self.lr = lr
        self.step_size = step_size
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=2)])
        self.post_label = Compose([AsDiscrete(to_onehot=2)])
        self.dice_metric = DiceMetric(include_background=False, reduction="mean")
        self.save_hyperparameters()

    def forward(self, x):
        prediction = self.unet_model(x)
        return prediction

    def configure_optimizers(self):
        self.optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=self.step_size, gamma=0.5)
        return {"optimizer": self.optimizer, "scheduler": self.scheduler}

    def training_step(self, train_batch):
        batch_data = train_batch
        inputs, labels = (
            batch_data["image"],
            batch_data["label"],
        )
        outputs = self.forward(inputs)
        train_outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        train_labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=train_outputs, y=train_labels)
        train_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        train_loss = self.loss_function(outputs, labels)
        logs = {
            "train_loss": train_loss,
            "train_dice": train_dice,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=inputs[0],
        )
        return train_loss

    def validation_step(self, validation_batch):
        batch_data = validation_batch
        inputs, labels = (
            batch_data["image"],
            batch_data["label"],
        )
        roi_size = (160, 160, 160)
        sw_batch_size = 1
        outputs = sliding_window_inference(
            inputs, roi_size, sw_batch_size, self.forward
        )
        val_outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        val_labels = [self.post_label(i) for i in decollate_batch(labels)]
        self.dice_metric(y_pred=val_outputs, y=val_labels)
        val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        val_loss = self.loss_function(outputs, labels)
        logs = {
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
        self.log_dict(
            logs,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=inputs[0],
        )
        return val_dice
