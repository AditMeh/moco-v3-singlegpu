#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Multiple Instance Learning (MIL) classification using patch-based attention.
Images are split into patches, embedded by a frozen backbone, and aggregated
via attention for classification.
"""

import argparse
import math
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as torchvision_models
import wandb

import vits
from mil import MILClassifier, create_backbone

torchvision_model_names = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

model_names = ['vit_small', 'vit_base', 'vit_conv_small', 'vit_conv_base'] + torchvision_model_names

parser = argparse.ArgumentParser(description='MIL Classification (Single GPU)')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 64, smaller due to patch processing)')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use (default: 0)')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')
parser.add_argument('--input-size', default=224, type=int,
                    help='input image size (default: 224)')
parser.add_argument('--crop-size', default=32, type=int,
                    help='crop/patch size for MIL (default: 32, must divide input-size)')
parser.add_argument('--num-classes', default=1000, type=int,
                    help='number of classes (default: 1000)')
parser.add_argument('--projection-dim', default=512, type=int,
                    help='dimension to project patch embeddings to (default: 512)')
parser.add_argument('--attention-hidden-dim', default=512, type=int,
                    help='hidden dimension for attention network (default: 512)')
parser.add_argument('--mlp-hidden-dim', default=512, type=int,
                    help='hidden dimension for classification MLP (default: 512)')
parser.add_argument('--clear-cache-freq', default=10, type=int,
                    help='clear GPU cache every N iterations (default: 10)')
parser.add_argument('--no-pin-memory', action='store_true',
                    help='disable pin_memory in DataLoader (uses less RAM but slower)')

# wandb configs
parser.add_argument('--wandb-project', default='moco-v3-mil', type=str,
                    help='wandb project name (default: moco-v3-mil)')
parser.add_argument('--wandb-name', default=None, type=str,
                    help='wandb run name (default: None)')
parser.add_argument('--no-wandb', action='store_true',
                    help='disable wandb logging')

parser.add_argument('--val-freq', default=5, type=int,
                    help='validate every N epochs (default: 5)')

best_acc1 = 0


def main():
    global best_acc1
    args = parser.parse_args()

    # Validate crop size
    if args.input_size % args.crop_size != 0:
        raise ValueError(f"input-size ({args.input_size}) must be divisible by crop-size ({args.crop_size})")
    
    num_patches = (args.input_size // args.crop_size) ** 2
    print(f"=> MIL config: {args.input_size}x{args.input_size} images, "
          f"{args.crop_size}x{args.crop_size} patches, {num_patches} patches per image")

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            config=vars(args)
        )

    # Set GPU
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available. This script requires GPU support.')
    
    torch.cuda.set_device(args.gpu)
    print("Use GPU: {} for training".format(args.gpu))

    # create backbone model
    print("=> creating backbone model '{}'".format(args.arch))
    if args.arch.startswith('vit'):
        backbone = vits.__dict__[args.arch]()
        linear_keyword = 'head'
    else:
        backbone = torchvision_models.__dict__[args.arch]()
        linear_keyword = 'fc'

    # load pretrained weights into backbone before removing fc
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename moco pre-trained keys
            state_dict = checkpoint['state_dict']
            for k in list(state_dict.keys()):
                # Handle both 'module.base_encoder.' and 'base_encoder.' prefixes
                if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.%s' % linear_keyword):
                    state_dict[k[len("module.base_encoder."):]] = state_dict[k]
                elif k.startswith('base_encoder') and not k.startswith('base_encoder.%s' % linear_keyword):
                    state_dict[k[len("base_encoder."):]] = state_dict[k]
                del state_dict[k]

            msg = backbone.load_state_dict(state_dict, strict=False)
            print(f"=> loaded pre-trained model (missing keys: {msg.missing_keys})")
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # Remove fc layer and get embedding dimension
    backbone, embed_dim = create_backbone(backbone, args.arch)
    print(f"=> backbone embedding dimension: {embed_dim}")

    # Create MIL model
    model = MILClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=args.num_classes,
        patch_size=args.crop_size,
        input_size=args.input_size,
        attention_hidden_dim=args.attention_hidden_dim,
        mlp_hidden_dim=args.mlp_hidden_dim,
        projection_dim=args.projection_dim
    )

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 64

    # Move model to GPU
    model = model.cuda(args.gpu)
    print(model)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # Only optimize the trainable parameters (attention + classifier)
    parameters = model.get_trainable_parameters()
    num_params = sum(p.numel() for p in parameters)
    print(f"=> training {num_params:,} parameters (attention + classifier)")

    optimizer = torch.optim.AdamW(parameters, init_lr,
                                  weight_decay=args.weight_decay)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # Use fixed resize to input_size (no random crop - MIL handles spatial info)
    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    num_workers = min(args.workers, 8)
    if args.workers > 8:
        print(f"Warning: Capping workers at 8 (requested {args.workers}) to prevent RAM issues")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=not args.no_pin_memory, drop_last=True,
        persistent_workers=num_workers > 0)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize((args.input_size, args.input_size)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=not args.no_pin_memory)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # Clear cache after each epoch
        torch.cuda.empty_cache()

        # evaluate on validation set every val_freq epochs (and always on last epoch)
        is_val_epoch = (epoch + 1) % args.val_freq == 0 or epoch == args.epochs - 1
        if is_val_epoch:
            acc1 = validate(val_loader, model, criterion, args)

            # remember best acc@1
            best_acc1 = max(acc1, best_acc1)

            # Log validation metrics to wandb (commit=False to not increment step)
            if not args.no_wandb:
                wandb.log({
                    'val_acc1': acc1,
                    'best_acc1': best_acc1,
                }, commit=False)

    if not args.no_wandb:
        wandb.finish()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # Set model to train mode (only affects attention + classifier, backbone stays frozen)
    model.train()

    end = time.time()
    iters_per_epoch = len(train_loader)
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0].item(), images.size(0))
        top5.update(acc5[0].item(), images.size(0))

        # Log to wandb (only at print_freq to reduce logging overhead)
        if not args.no_wandb and i % args.print_freq == 0:
            wandb.log({
                'train_loss': loss.item(),
                'train_acc1': acc1[0].item(),
                'train_acc5': acc5[0].item(),
                'epoch': epoch,
            })

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Clear memory periodically
        if (i + 1) % args.clear_cache_freq == 0:
            torch.cuda.empty_cache()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, args.num_classes)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0].item(), images.size(0))
            top5.update(acc5[0].item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
