import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys
import numpy as np
import torch 
import lightning
import matplotlib.pyplot as plt
from networks import *
from torchsummary import summary
from utils import *
import random
import tqdm
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
import pandas as pd
import time

#load model from checkpoint
unet = UNet(n_channels=1, n_classes=1, bilinear=True).float()
model = LitModel.load_from_checkpoint("/vault3/machine_learning/3CDNN_ArtifactReduction/model_weights/2DUNet/lightning_logs/version_3/checkpoints/epoch=38-step=34164.ckpt", unet=unet)
model.eval()
print()
ww = 3_000
wl = 0
batchsize = 32
model = model.cuda()
path = "/vault3/machine_learning/datasets/public/RSNA_PE/train_sorted_sparse/cone/128_angles/"
filenames = os.listdir(path)
path_save = "/data-pool/data_no_backup/ga63cun/PE/3DCNN/"


for filename in tqdm.tqdm(filenames[:150]):
    test_vol_sparse = np.load(f"/vault3/machine_learning/datasets/public/RSNA_PE/train_sorted_sparse/cone/128_angles/{filename}/{filename}.npy", mmap_mode="c")
    z = test_vol_sparse.shape[0]
    x1_list = []
    x2_list = []
    x3_list = []
    x4_list = []
    x5_list = []
    start = time.time()
    for k in tqdm.tqdm(range(0, z, batchsize)):
        inpt = torch.from_numpy(test_vol_sparse[k:k+batchsize, None]).cuda()
        inpt = window(inpt, wl=wl, ww=ww)
        with torch.no_grad():
            x1 = model.unet.inc(inpt)
            x2 = model.unet.down1(x1)
            x3 = model.unet.down2(x2)
            x4 = model.unet.down3(x3)
            x5 = model.unet.down4(x4)
    
        #y = model.unet.up4(model.unet.up3(model.unet.up2(model.unet.up1(x5, x4), x3), x2), x1)
        #final = model.unet.outc(y) + inpt
        #final = final*ww - ww/2
    
        x1_list.append(x1.detach().cpu().half())
        x2_list.append(x2.detach().cpu().half())
        x3_list.append(x3.detach().cpu().half())
        x4_list.append(x4.detach().cpu().half())
        x5_list.append(x5.detach().cpu().half())
        
        #fig, axes = plt.subplots(1, 3, figsize=(24, 12))
        #axes[0].imshow(test_vol[k], cmap='gray', vmin=-250, vmax=350)
        #axes[1].imshow(test_vol_sparse[k], cmap='gray', vmin=-250, vmax=350)
        #axes[2].imshow(final.detach().cpu().numpy().squeeze()[0], cmap='gray', vmin=-250, vmax=350)
    
    
    x1_stack = torch.swapaxes(torch.cat(x1_list, axis=0), 0, 1)
    x2_stack = torch.swapaxes(torch.cat(x2_list, axis=0), 0, 1)
    x3_stack = torch.swapaxes(torch.cat(x3_list, axis=0), 0, 1)
    x4_stack = torch.swapaxes(torch.cat(x4_list, axis=0), 0, 1)
    x5_stack = torch.swapaxes(torch.cat(x5_list, axis=0), 0, 1)
    
    if not os.path.exists(path_save + f"/{filename}/"):
        os.makedirs( path_save + f"/{filename}/")
                          
    np.save(path_save + f"/{filename}/{filename}_x1.npy", x1_stack)
    np.save(path_save + f"/{filename}/{filename}_x2.npy", x2_stack)
    np.save(path_save + f"/{filename}/{filename}_x3.npy", x3_stack)
    np.save(path_save + f"/{filename}/{filename}_x4.npy", x4_stack)
    np.save(path_save + f"/{filename}/{filename}_x5.npy", x5_stack)