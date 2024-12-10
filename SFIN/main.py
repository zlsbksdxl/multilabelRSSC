import timm
import os
import random
import math
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import shutil

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from torchvision import models
from datagen import DataGeneratorML

import torch.nn.functional as F
import copy
from metric import *
import argparse
parser = argparse.ArgumentParser(description='PyTorchUCM Training')
parser.add_argument('--tem', default=0.1,type=float,required=True)

args = parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_all(123)
pretrained_size = 256
pretrained_means = [0.3980, 0.4097, 0.3696]
pretrained_stds= [0.1468, 0.1340, 0.1303]

train_data_transform = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.RandomRotation(5),
                           transforms.RandomHorizontalFlip(0.5),
                           transforms.RandomCrop(pretrained_size, padding = 10),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

test_data_transform = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.CenterCrop(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean = pretrained_means, 
                                                std = pretrained_stds)
                       ])

datadir = '/media/admin324/000101FC000161AF/multilabeldataset/UC_Merced'
train_dataGen = DataGeneratorML(datadir=datadir,
                                dataset='UCM',
                                imgExt='tif',
                                imgTransform=train_data_transform,
                                phase='train')
test_dataGen = DataGeneratorML(datadir=datadir,
                              dataset='UCM',
                              imgExt='tif',
                              imgTransform=test_data_transform,
                              phase='test')
train_data_loader = DataLoader(train_dataGen,
                               batch_size=32,
                               num_workers=8,
                               shuffle=True,
                               pin_memory=False)
val_data_loader = DataLoader(test_dataGen,
                             batch_size=32,
                             num_workers=8,
                             shuffle=False,
                             pin_memory=False)
from modelend import SFIN
model = SFIN(net='resnet18', numclass=17, h=4).cuda()
EPOCHS = 100
FOUND_LR = 4e-5
optimizer = torch.optim.Adam(model.parameters(), lr = FOUND_LR)


multiLabelLoss = torch.nn.BCEWithLogitsLoss().cuda()

from mulconloss import mulConLoss
multiconlLoss = mulConLoss(torch.device("cuda"),temperature=args.tem)

best = 0
beste = 0
for epoch in tqdm(range(EPOCHS)):
    model.train()
    losses = MetricTracker()
    for idx, data in enumerate(train_data_loader):

        imgs = data['img'].cuda()
        multiHots = data['multiHot'].to(torch.device("cuda"))

        g1, g2, sf, logits = model(imgs)

 
        loss_ce = multiLabelLoss(logits, multiHots)
        loss_sup1 = multiLabelLoss(g1, multiHots)
        loss_sup2 = multiLabelLoss(g2, multiHots)
        loss_con = multiconlLoss(sf, multiHots)
        

        loss = loss_ce + loss_sup1 + loss_sup2 + 0.1*loss_con
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), imgs.size(0))
    if epoch ==(50):
        for group in optimizer.param_groups:
            group['lr'] = group['lr']*0.1    

    print(losses.avg)
# model = torch.load('modelucm.pth').cuda()
with torch.no_grad():
    model.eval()
    probas, targets = [], []
    for idx, data in enumerate(tqdm(val_data_loader, desc="test")):
        imgs = data['img'].to(torch.device("cuda"))
        multiHots = data['multiHot'].to(torch.device("cuda"))

        _, _, _, pre = model(imgs)
        proba  = torch.sigmoid(pre).data

        probas  += [proba.detach().to("cpu")]
        targets += [multiHots.detach().to("cpu")]

    targets = torch.cat(targets, dim=0).to(torch.int).numpy()
    probas = torch.cat(probas, dim=0).numpy()
    preds = (probas > 0.5).astype('int')
    metric = compute_metrics(targets, preds)
    print(metric)

content = f'\n{args.tem}\nmetric:\n{metric}\n'
result_dir = f'./results/ucm_tem'  
with open(result_dir,'a') as f:
    f.write(content)
