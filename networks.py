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

#3D decoder inspired by https://github.com/AghdamAmir/3D-UNet
class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    in_channels -> number of input channels
    out_channels -> desired number of output channels
    input -> input Tensor to be convolved
    returns -> Tensor
    """

    def __init__(self, in_channels, out_channels, norm='GN', norm_params={}) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1)
        self.relu = nn.ReLU()

        if norm=="BN":
            self.bn = nn.BatchNorm3d(num_features=out_channels)
        elif norm=="GN":
            self.bn = nn.GroupNorm(num_channels=out_channels, **norm_params)

    
    def forward(self, input):
        res = self.relu(self.bn(self.conv1(input)))
        res = self.relu(self.bn(self.conv2(res)))
        return res

class EncodeFeatures(nn.Module):
    """
    The basic block for pooling the z direction of input features x1-x5 by strided 3x1x1 convolutions.
    -- __init__()
    in_channels, 
    out_channels, norm='GN', norm_params={}, nr_pooling=4
    input -> input Tensor to be convolved
    returns -> Tensor
    """

    def __init__(self, in_channels, norm='GN', norm_params={}, nr_pooling=4, kernel_size=(3,1,1), padding=(1, 0, 0), stride=(2, 1, 1)) -> None:
        super(EncodeFeatures, self).__init__()
        self.nr_pooling = nr_pooling
        self.conv = nn.Conv3d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, padding=padding, stride=stride, groups=in_channels)
        self.relu = nn.ReLU()

        if norm=="BN":
            self.bn = nn.BatchNorm3d(num_features=in_channels, **norm_params)
        elif norm=="GN":
            self.bn = nn.GroupNorm(num_channels=in_channels, **norm_params)

    
    def forward(self, x):
        for k in range(self.nr_pooling):
            x = self.relu(self.bn(self.conv(x)))
        return x
        
class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :in_channels -> number of input channels
    :out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :param norm -> to use either BN or GN
    :upsample_z -> if True, the first dimension gets also upsampled, if not, this dimension stays the same
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels=0, last_layer=False, num_classes=1, bilinear=True, norm="GN", norm_params={}, upsample_z=False) -> None:
        super(UpConv3DBlock, self).__init__()
        if upsample_z:
            stride = (2, 2, 2)
            kernel_size = (2, 2, 2)
            padding = (0, 0, 0)
            scale_factor = (2, 2, 2)
        else:
            stride = (1, 2, 2)
            kernel_size=(3, 2, 2)
            padding=(1, 0, 0)
            scale_factor = (1, 2, 2)

        if bilinear:
            self.upconv1 = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='trilinear'),
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3, 3), padding=(1,1,1)))
        else:
            self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)

        self.relu = nn.ReLU()
        if norm=="BN":
            self.bn = nn.BatchNorm3d(num_features=out_channels, **norm_params)
        elif norm=="GN":
            self.bn = nn.GroupNorm(num_channels=out_channels, **norm_params)
            
        self.conv1 = nn.Conv3d(in_channels=out_channels*2, out_channels=out_channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.conv2 = nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=(1,1,1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=out_channels, out_channels=num_classes, kernel_size=(1,1,1))
            
        
    def forward(self, input, residual=None):
        #print("inside upconv - size inpt", input.size())
        #print("inside upconv - size residual", residual.size())
        out = self.upconv1(input)
        #print("inside upconv - size out upconv", out.size())
        out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: 
            out = self.conv3(out)
        return out

class Decoder3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256, 512, 512], bilinear=True, norm='GN', norm_params={}, residual=True) -> None:
        super(Decoder3D, self).__init__()
        self.bottleNeck = Conv3DBlock(in_channels=level_channels[4], out_channels=level_channels[4], norm=norm, norm_params=norm_params)
        self.s_block4 = UpConv3DBlock(in_channels=level_channels[4], out_channels=level_channels[3], bilinear=bilinear, norm=norm, norm_params=norm_params)
        self.s_block3 = UpConv3DBlock(in_channels=level_channels[3], out_channels=level_channels[2], bilinear=bilinear, norm=norm, norm_params=norm_params)
        self.s_block2 = UpConv3DBlock(in_channels=level_channels[2], out_channels=level_channels[1], bilinear=bilinear, norm=norm, norm_params=norm_params)
        self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], out_channels=level_channels[0], num_classes=num_classes, last_layer=True, bilinear=bilinear, norm=norm, norm_params=norm_params)
        self.residual = residual
    
    def forward(self, inpt, x1, x2, x3, x4, x5):

        #Synthesis path forward feed
        out = self.bottleNeck(x5)
        out = self.s_block4(out, x4)
        out = self.s_block3(out, x3)
        out = self.s_block2(out, x2)
        out = self.s_block1(out, x1)
        if self.residual:
            #print("decod", out.type())
            out = torch.add(out, inpt)
            #print("decod",out.type())
        return out

class Decoder3D_EncodeZ(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels 
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """
    
    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256, 512, 512], bilinear=True, norm='GN', norm_params={}, residual=True, zencoder_params={}) -> None:
        super(Decoder3D_EncodeZ, self).__init__()
        self.bilinear = bilinear
        self.norm = norm
        self.norm_params = norm_params
        self.level_channels = level_channels
        self.bottleNeck = Conv3DBlock(in_channels=level_channels[4], out_channels=level_channels[4], norm=norm, norm_params=norm_params)
        self.s_block4 = UpConv3DBlock(in_channels=level_channels[4], out_channels=level_channels[3], bilinear=bilinear, norm=norm, norm_params=norm_params, upsample_z=True)
        self.s_block3 = UpConv3DBlock(in_channels=level_channels[3], out_channels=level_channels[2], bilinear=bilinear, norm=norm, norm_params=norm_params, upsample_z=True)
        self.s_block2 = UpConv3DBlock(in_channels=level_channels[2], out_channels=level_channels[1], bilinear=bilinear, norm=norm, norm_params=norm_params, upsample_z=True)
        self.s_block1 = UpConv3DBlock(in_channels=level_channels[1], out_channels=level_channels[0], num_classes=num_classes, last_layer=True, bilinear=bilinear, norm=norm, norm_params=norm_params, upsample_z=True)
        
        self.encoder_layer5 = EncodeFeatures(in_channels=level_channels[4], norm=norm, norm_params=norm_params, nr_pooling=4, **zencoder_params)
        self.encoder_layer4 = EncodeFeatures(in_channels=level_channels[3], norm=norm, norm_params=norm_params, nr_pooling=3, **zencoder_params)
        self.encoder_layer3 = EncodeFeatures(in_channels=level_channels[2], norm=norm, norm_params=norm_params, nr_pooling=2, **zencoder_params)
        self.encoder_layer2 = EncodeFeatures(in_channels=level_channels[1], norm=norm, norm_params=norm_params, nr_pooling=1, **zencoder_params)
        
        self.residual = residual
    
    def forward(self, inpt, x1, x2, x3, x4, x5):

        #Synthesis path forward feed
        #print("x5:", x5.size())
        x5 = self.encoder_layer5(x5)
        #print("x5 encoded:", x5.size())
        out = self.bottleNeck(x5)
        #print("out bottleneck:", out.size())
        
        #print("x4:", x4.size()) 
        x4 = self.encoder_layer4(x4)
        #print("x4 encoded:", x4.size()) 
        out = self.s_block4(out, x4)
        #print("out block4:", out.size()) 
    
        x3 = self.encoder_layer3(x3)
        out = self.s_block3(out, x3)

        x2 = self.encoder_layer2(x2)
        out = self.s_block2(out, x2)
        
        out = self.s_block1(out, x1)
        if self.residual:
            #print("decod", out.type())
            out = torch.add(out, inpt)
            #print("decod",out.type())
        return out

#-----------------------------------------------------------------------------

#Define Lightning Model

class LitModel2D(L.LightningModule):
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
        
        super(LitModel2D, self).__init__()
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

class LitModel3D(L.LightningModule):
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
    def __init__(self, network=None, 
                 optimizer_algo=None, 
                 optimizer_params=None,
                 loss = nn.MSELoss(reduction='mean'), 
                 lr = None,
                 scheduler_algo=None,
                 scheduler_params=None
                ):
        
        super(LitModel3D, self).__init__()
        self.net = network
        self.optimizer_algo = optimizer_algo
        self.optimizer_params = optimizer_params
        self.lr = lr
        self.loss = loss
        self.scheduler_algo = scheduler_algo
        self.scheduler_params = scheduler_params
        self.save_hyperparameters(ignore=["network", "loss"])


    def forward(self, inpt, x1, x2, x3, x4, x5):
        pred = self.net(inpt, x1, x2, x3, x4, x5)
        return pred

    def training_step(self, batch, batch_idx):
        fullview, inpt, x1, x2, x3, x4, x5 = batch
        pred = self(inpt, x1, x2, x3, x4, x5)
        #print(inpt.type(), pred.type())
        loss = self.loss(pred, fullview)
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
           'monitor': 'train_loss'}
