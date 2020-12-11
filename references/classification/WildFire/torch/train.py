#!usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (c) Pyronear contributors.
# This file is dual licensed under the terms of the CeCILL-2.1 and AGPLv3 licenses.
# See the LICENSE file in the root of this repository for complete details.

import torch
import torch.nn as nn
from torch import optim
import torchvision
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import PIL
import random
import os

import holocron
from holocron.trainer.core import ClassificationTrainer
from pyronear.models.utils import cnn_model
from holocron.models.utils import load_pretrained_params


def target_transform(target):

    target = torch.tensor(target, dtype=torch.float32)

    return target.unsqueeze(dim=0)


def set_seed(seed):
    """Set the seed for pseudo-random number generations
    Args:
        seed (int): seed to set for reproducibility
    """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(args):
    """Training Script"""
    if args.deterministic:
        set_seed(42)

    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            args.device = 'cuda:0'
        else:
            args.device = 'cpu'

    # Create Dataloaders

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    size = args.s
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(size=size, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(size=(size)),
        transforms.CenterCrop(size=size),
        transforms.ToTensor(),
        normalize
    ])

    dsTrain = ImageFolder(args.DB + '/train/', train_transforms, target_transform=target_transform)
    dsVal = ImageFolder(args.DB + '/val/', val_transforms, target_transform=target_transform)

    train_loader = DataLoader(dsTrain, batch_size=args.bs, shuffle=True)
    val_loader = DataLoader(dsVal, batch_size=args.bs, shuffle=True)

    # Create Model

    # Get backbone
    base = holocron.models.__dict__[args.backbone](args.pretrained)
    # Change head
    if args.concat_pool:
        model = cnn_model(base, int(args.cut), nb_features=int(args.nb_features),
                          num_classes=int(args.num_classes))
    else:
        model = base
    # Load Weight
    if args.resume is not None:
        load_pretrained_params(model, args.resume)
    # Move to gpu
    if args.device == 'cuda:0' and torch.cuda.is_available():
        model = model.to('cuda:0')

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Create the contiguous parameters.
    model_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = holocron.optim.RAdam(model_params, args.lr, betas=(0.95, 0.99), eps=1e-6, weight_decay=args.wd)

    #Create Trainer
    trainer = ClassificationTrainer(model, train_loader, val_loader, criterion, optimizer, 0,
                                    output_file=args.checkpoint)

    # Fit n epochs
    trainer.fit_n_epochs(args.epochs, args.lr, args.freeze)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='PyroNear Classification Training',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Input / Output
    parser.add_argument('--DB', default='./DB', help='dataset root folder')
    parser.add_argument('--resume', default=None, help='checkpoint file to resume from')
    parser.add_argument('--checkpoint', default=None, type=str, help='name of output file')

    # Architecture
    parser.add_argument('--backbone', default='resnet18', type=str, help='backbone model architecture')
    parser.add_argument('--nb-class', default=2, type=int, help='number of classes')
    parser.add_argument('--nb_features', type=int, help='number of ouput feature for backbone')
    parser.add_argument('--cut', type=int, help='where you should cut the head')
    parser.add_argument("--concat_pool", dest="concat_pool",
                        help="replaces AdaptiveAvgPool2d with AdaptiveConcatPool2d",
                        action="store_true")
    parser.add_argument("--pretrained", dest="pretrained",
                        help="use ImageNet pre-trained parameters",
                        action="store_true")
    # Device
    parser.add_argument('--device', default=None, help='device')

    # Loader
    parser.add_argument('-b', '--batch-size', default=32, type=int, help='batch size')
    parser.add_argument('-s', '--resize', default=224, type=int, help='image size after resizing')

    # Optimizer
    parser.add_argument('--lr', default=3e-4, type=float, help='maximum learning rate')
    parser.add_argument('--epochs', default=20, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--wd', '--weight-decay', default=1e-2, type=float,
                        metavar='W', help='weight decay',
                        dest='weight_decay')
    parser.add_argument("--freeze", default=None, dest="freeze", help="should all layers be unfrozen",
                        action="store_true")

    args = parser.parse_args()

    main(args)
