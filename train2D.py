import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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
import pandas as pd
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from argparse import ArgumentParser

def main():    
    # ----- init dataset -----
    dataset_train = SparseDataset2D(df = df_train, 
                 path_sparse = path_sparse, 
                 path_gt = path_gt, 
                 augmentation = True, 
                 image_size=patch_size, 
                 ww=ww, 
                 wl=wl
                             )
    dataset_val = SparseDataset2D(df = df_val, 
                     path_sparse = path_sparse, 
                     path_gt = path_gt, 
                     augmentation = False, 
                     image_size=patch_size, 
                     ww=ww, 
                     wl=wl
                               )
    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, num_workers=4, shuffle=True)
    dataloader_val = DataLoader(dataset_val, batch_size=batchsize, num_workers=4, shuffle=False)
    
    
    # ----- init model -----
    unet = UNet(n_channels=1, n_classes=1, bilinear=True).float()
    model = LitModel2D(unet=unet, 
                 optimizer_algo=optimizer_algo, 
                 optimizer_params=optimizer_params,
                 loss = nn.MSELoss(reduction='mean'), 
                 lr = lr,
                 scheduler_algo="StepLR",
                 scheduler_params=scheduler_params
                   )
    
    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')
    tblogger = TensorBoardLogger(save_path)
    csvlogger = CSVLogger(save_path, version=tblogger.version)
    checkpoint = ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=3)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", patience=7)
    
    trainer = L.Trainer(logger=[csvlogger, tblogger], 
                        callbacks=[lr_monitor, checkpoint, early_stopping], 
                        max_epochs=400)
        
    trainer.fit(model, dataloader_train, dataloader_val)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batchsize", type=int, default=16)
    parser.add_argument("--patch_size", nargs='+', type=int, default=(256, 256))
    parser.add_argument("--optimizer_algo", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    
    #initialize training parameters
    args = parser.parse_args()
    batchsize = args.batchsize
    patch_size = tuple(args.patch_size)
    optimizer_algo = args.optimizer_algo
    lr = float(args.lr)
    weight_decay = args.weight_decay
    optimizer_algo = args.optimizer_algo
    optimizer_params={"weight_decay": weight_decay}
    scheduler_algo = "StepLR"
    scheduler_params = {"step_size":4, "gamma":0.9}

    #initialize dataset parameters
    path_sparse = '/data-pool/data_no_backup/ga63cun/PE/64/'
    path_gt = '/data-pool/data_no_backup/ga63cun/PE/4095/'
    save_path = "./model_weights/2DUNet/"
    df_train = pd.read_csv("./Test_allImages.csv") #I just put in the test csv path here
    df_val = pd.read_csv("./Test_allImages.csv") #I just put in the test csv path here
    dataloader_train_params={"batch_size":batchsize, "num_workers":4, "shuffle":True}
    dataloader_val_params={"batch_size":batchsize, "num_workers":4, "shuffle":False}
    ww = 3_000
    wl = 0
    main()
