import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "100"
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
torch.set_float32_matmul_precision("medium")

def main():    
    #initialize datasets
    batchsize=1
    path = '/data-pool/data_no_backup/ga63cun/PE/3DCNN//'
    save_path = "./model_weights/3Decoder/GN/"
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    df_train = pd.read_csv("./train.csv") 
    df_train_filtered = df_train[df_train["nr_slices"] > 48]
    #initialize training parameters
    lr = 1e-4
    weight_decay = 1e-2
    optimizer_algo = "AdamW"
    optimizer_params={"weight_decay": weight_decay}
    scheduler_algo = "ReduceLROnPlateau"
    scheduler_params = {"patience":3, "factor":0.5}
    ww = 3_000
    wl = 0
    print()
    
    dataset_train = SparseDataset3D(df = df_train_filtered, 
                     path = path, 
                     augmentation = False, 
                     ww=ww, 
                     wl=wl,
                     downsample=False,
                     size=(48, 512, 512))
    
    dataloader_train = DataLoader(dataset_train, batch_size=batchsize, num_workers=0, shuffle=True)
    
        # ----- init model -----
    decoder3D = Decoder3D(in_channels=1, num_classes=1, bilinear=False, norm="GN", norm_params={"num_groups":4}, residual=True)
    
    model = LitModel3D(network=decoder3D, 
                     optimizer_algo=optimizer_algo, 
                     optimizer_params=optimizer_params,
                     loss = nn.MSELoss(reduction='mean'), 
                     lr = lr,
                     scheduler_algo=scheduler_algo,
                     scheduler_params=scheduler_params
                       )

    lr_monitor = L.pytorch.callbacks.LearningRateMonitor(logging_interval='epoch')
    tblogger = TensorBoardLogger(save_path)
    csvlogger = CSVLogger(save_path, version=tblogger.version)
    checkpoint = ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=3)
    early_stopping = EarlyStopping(monitor="train_loss", mode="min", patience=5)
    
    trainer = L.Trainer(logger=[csvlogger, tblogger], 
                        callbacks=[lr_monitor, checkpoint, early_stopping], 
                        max_epochs=400, precision="bf16-mixed", accumulate_grad_batches=4)
    
    trainer.fit(model, dataloader_train)

if __name__ == "__main__": 
    main()
