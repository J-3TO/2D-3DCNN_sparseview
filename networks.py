import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import lightning as L
#from torchvision.models.feature_extraction import get_graph_node_names
#from torchvision.models.feature_extraction import create_feature_extractor
import sklearn
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
from utils import *

class dummyModel(nn.Module):
    """
    Returns input. Sometimes useful to test pipeline.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

#Standard U-Net with batchnorm implementation from https://github.com/milesial/Pytorch-UNet
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        y = self.up1(x5, x4)
        y = self.up2(y, x3)
        y = self.up3(y, x2)
        y = self.up4(y, x1)
        y = self.outc(y)
        out = torch.add(x, y)
        return out
   
#-----------------------------------------------------------------------------

#Define Lightning Model

class LitModel(L.LightningModule):
    """
    Lightning Model for training the U-Net. 
    
    --------------------------------------------------------
    
    Parameters:
    
    unet: torch model 
    U-Net model to be trained

    optimizer_algo: string
    Type of optimizer to use. Currently implemented 'Adam' and 'AdamW'

    optimizer_params: dict
    Parameters, which are passed to the optimizer. 
    Learning rate must be passed seperately to the model, so that the lr_finder from lightnig works

    loss: function
    Function used for calculating the training/validation loss
    default: nn.MSELoss(reduction='mean')

    scheduler_algo: str 
    Choose the scheduler algorithm to be used. Options are: "CosineAnnealingWarmRestarts", "ReduceLROnPlateau", "StepLR"

    scheduler_params: dict 
    Parameters for scheduler algorithm
    
    """
    def __init__(self, unet=None, 
                 optimizer_algo=None, 
                 optimizer_params=None,
                 loss = nn.MSELoss(reduction='mean'), 
                 lr = None,
                 scheduler_algo=None,
                 scheduler_params=None
                ):
        
        super(LitModel, self).__init__()
        self.unet = unet
        self.optimizer_algo = optimizer_algo
        self.optimizer_params = optimizer_params
        self.lr = lr
        self.loss = loss
        self.scheduler_algo = scheduler_algo
        self.scheduler_params = scheduler_params
        self.save_hyperparameters(ignore=["unet", "loss"])


    def forward(self, x):
        pred = self.unet(x.float())
        return pred

    def training_step(self, batch, batch_idx):
        batch_inpt, batch_target, labels = batch
        batch_pred = self(batch_inpt)

        loss = self.loss(batch_pred, batch_target)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss.float()

    def validation_step(self, batch, batch_idx):
        batch_inpt, batch_target, labels = batch
        batch_pred = self(batch_inpt)
        loss = self.loss(batch_pred, batch_target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss.float()

    def configure_optimizers(self):
        if self.optimizer_algo == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, **self.optimizer_params)
        if self.optimizer_algo == "AdamW":
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, **self.optimizer_params)

        if self.scheduler_algo == "CosineAnnealingWarmRestarts":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "ReduceLROnPlateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **self.scheduler_params)
        if self.scheduler_algo == "StepLR":
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **self.scheduler_params)
        
        return {
           'optimizer': optimizer,
           'lr_scheduler': scheduler, 
           'monitor': 'val_loss'}

