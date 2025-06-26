 
import torch.nn as nn

import os
import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torch.optim import lr_scheduler
from src.helper_functions.helper_functions import ModelEma, add_weight_decay
from src.loss_functions.losses import CosLoss
'''models'''
from src.models.ResCAM import ImageClassifier
from src.models.CPRFL import MyModel
from src.models.SGRE.sgre import MLIC
from src.models.MLTran import MLtran
from src.models.Resyyxz1 import model77
''''''
import torch.nn as nn
from utils.LT_engine_grouplr import *
from src.data_loader.datasets import build_dataset
from src.loss_functions.dbl import *
from src.loss_functions.asl import *
import traceback
import warnings
import sys

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

parser = argparse.ArgumentParser(description='PyTorch MS_COCO Training')
parser.add_argument('--modelname', help='model name', default='CPRFL', choices=['ResCAM', 'CPRFL', 'SGRE','MLTRAN','model77'],)
parser.add_argument('--dataset', default='coco-lt', type=str, choices=['voc-lt', 'coco-lt'], help='dataset name')
parser.add_argument('--image-size', default=448, type=int,
                    metavar='N', help='input image size (default: 448)')
parser.add_argument('--lr', default=1e-5, type=float)
parser.add_argument('--backbone', default='resnet101')
parser.add_argument('--pretrained', default='/data/pretrain_models/resnet/resnet101-63fe2227.pth', type=str)
parser.add_argument('--num-classes', default=80, type=int)
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 16)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='mini-batch size (default: 16)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--loss_function', default='asl', type=str, choices=['asl', 'bce', 'dbl', 'mls', 'FL', 'CBloss', 'DBloss-noFocal', 'DBloss'], help='loss function')
parser.add_argument('--threshold', default=0.8, type=float,
                    metavar='N', help='threshold value')
parser.add_argument('--max_epoch', default=80, type=int,
                    metavar='N', help='train epoch')
parser.add_argument('--flag',  action='store_true',
                    help='是否建立语义联系')
def main(cfg):
    if cfg.dataset == 'coco-lt':
        dataset_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic_light', 'fire_hydrant', 'stop_sign',
            'parking_meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports_ball', 'kite', 'baseball_bat', 'baseball_glove', 'skateboard',
            'surfboard', 'tennis_racket', 'bottle', 'wine_glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot_dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted_plant', 'bed', 'dining_table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy_bear', 'hair_drier', 'toothbrush'
            ]
        train_dataset = build_dataset(dataset=cfg.dataset, split='train')

        val_dataset = build_dataset(dataset=cfg.dataset, split='test')
    
    elif cfg.dataset == 'voc-lt':
        dataset_classes = [
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
            'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse','motorbike', 
            'person', 'pottedplant', 'sheep', 'sofa', 'train','tvmonitor'
            ]
        train_dataset = build_dataset(dataset=cfg.dataset, split='train')
        val_dataset = build_dataset(dataset=cfg.dataset, split='test')

    # Pytorch Data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.workers, pin_memory=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.workers, pin_memory=False)

    # loss functions
    if cfg.dataset == 'coco-lt':
        freq_file = '/home/drq/dr7/LTMLC/data/coco/class_freq.pkl'
    elif cfg.dataset == 'voc-lt':
        freq_file='/home/drq/dr7/LTMLC/data/voc/class_freq.pkl'

    if cfg.loss_function == 'bce':
        loss_function = nn.BCEWithLogitsLoss()
    if cfg.loss_function == 'mls':
        loss_function = nn.MultiLabelSoftMarginLoss()
    if cfg.loss_function == 'FL':
        if cfg.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func=None,
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
        elif cfg.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func=None,
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0, freq_file=freq_file
            )
    if cfg.loss_function == 'CBloss': #CB
        if cfg.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='CB',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                loss_weight=10.0, freq_file=freq_file
            )
        elif cfg.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='CB',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                loss_weight=10.0, freq_file=freq_file
            )
    if cfg.loss_function == 'DBloss-noFocal': # DB-0FL
        if cfg.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=False, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=0.5, freq_file=freq_file
            )
        elif cfg.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=False, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=0.5, freq_file=freq_file
            )
    if cfg.loss_function == 'asl':
        loss_function = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    if cfg.loss_function == 'dbl':
        if cfg.dataset == 'coco-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=2.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.2),
                loss_weight=1.0, freq_file=freq_file
            )
        elif cfg.dataset == 'voc-lt':
            loss_function = ResampleLoss(
                use_sigmoid=True,
                reweight_func='rebalance',
                focal=dict(focal=True, balance_param=2.0, gamma=2),
                logit_reg=dict(neg_scale=5.0, init_bias=0.05),
                map_param=dict(alpha=0.1, beta=10.0, gamma=0.3),
                loss_weight=1.0, freq_file=freq_file
            )


    # hyper-parameters
    print("len(train_dataset)): ", len(train_dataset))
    print("len(val_dataset)): ", len(val_dataset))
    
    
    # load model
    print('creating model {}'.format(cfg.modelname))
    models = nn.ModuleList([])
    if cfg.evaluate:
        if cfg.modelname == 'ResCAM':
            model = ImageClassifier(cfg)
        elif cfg.modelname == 'CPRFL':
            model = MyModel(cfg,classnames=dataset_classes)
        elif cfg.modelname == 'SGRE':
            model = MLIC(cfg.backbone, 2048, cfg)
        elif cfg.modelname == 'MLTRAN':
            model = MLtran(cfg,classnames=dataset_classes)
        elif cfg.modelname == 'model77':
            model = model77(cfg,classnames=dataset_classes)
        models.append(model)
    else:
        if cfg.modelname == 'ResCAM':
            regular_model = ImageClassifier(cfg).cuda()
        elif cfg.modelname == 'CPRFL':
            regular_model = MyModel(cfg,classnames=dataset_classes).cuda()
        elif cfg.modelname == 'SGRE':
            regular_model = MLIC(cfg.backbone, 2048, cfg).cuda()
        elif cfg.modelname == 'MLTRAN':
            regular_model = MLtran(cfg,classnames=dataset_classes).cuda()
        elif cfg.modelname == 'model77':
            regular_model = model77(cfg,classnames=dataset_classes).cuda()
        ema_model = ModelEma(regular_model, 0.998)  # 0.9997^641=0.82
        models.append(regular_model)
        models.append(ema_model)



    # set optimizer
    Epochs = 80
    weight_decay = 1e-4
    criterion = nn.ModuleList([])
    criterion.append(loss_function)
    # parameters = add_weight_decay(models[0], weight_decay)

    optimizer = torch.optim.AdamW(params=models[0].parameters(), lr=cfg.lr) # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=cfg.lr, steps_per_epoch=steps_per_epoch, epochs=Epochs,
                                        pct_start=0.1)

    engine = MultiLabelEngine(cfg)
    engine.learning(models, train_loader, val_loader, criterion, optimizer, scheduler)


if __name__ == '__main__':
    args = parser.parse_args()
    cfg = prepare_env(args, sys.argv)

    try:
        main(cfg)
    except (Exception, KeyboardInterrupt):
        print(traceback.format_exc())
        if not os.path.exists(cfg.ckpt_ema_best_path):
            clear_exp(cfg.exp_dir)


