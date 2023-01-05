#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import math
import os
import random
import shutil
import time
import warnings
import builtins
import torch.distributed as dist

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as torchvision_models
from thop import profile
from torch import nn
import torch.nn.functional as F

from torchvision.transforms import AutoAugment
import torch.multiprocessing as mp

from datasets import OrderTrainDataset, OrderTestDataset
from lstm import LSTM

from vit import ViT

torch.set_printoptions(precision=8)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--save-dir', default='save')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
# 必须比num_classes大
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on all nodes when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--num_classes1', default=100, type=int)
parser.add_argument('--num_classes2', default=181, type=int)
parser.add_argument('--len', default=5, type=int)
# parser.add_argument('--loss_alpha', default=0, type=float)
# parser.add_argument('--temp', default=0.05, type=float)
parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', default=0.1, type=float)
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--data_model_path', default='data_model/model_acc_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to moco pretrained checkpoint')


def main():
    args = parser.parse_args()
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()

    # create model
    print("=> creating model")
    backbone = torchvision_models.resnet18()

    model = ViT(
        backbone=backbone,
        num_classes1=args.num_classes1,
        num_classes2=args.num_classes2,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        pool='cls',
        len=5,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1
    )
    # model = LSTM(backbone=backbone,
    #              num_classes1=args.num_classes1,
    #              num_classes2=args.num_classes2,
    #              hidden_size=256,
    #              num_layers=6,
    #              len=5,
    #              dropout=0.1)

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            state_dict = checkpoint['state_dict']
            args.start_epoch = 0
            model.load_state_dict(state_dict, strict=False)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    model = model.cuda(args.gpu)
    print(model)

    # flops, params = profile(model, (torch.randn((1, 5, 3, 224, 224)).cuda(), torch.randn((1, 5, 181)).cuda()))
    # print('flops: ', flops, 'params: ', params)
    # print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / args.world_size)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

    best_acc1 = 0
    best_acc2 = 0

    if args.data_model_path:
        print("=> loading data_model checkpoint '{}'".format(args.data_model_path))
        checkpoint = torch.load(args.data_model_path, map_location="cpu")
        state_dict = checkpoint['state_dict']

        for k in list(state_dict.keys()):
            # 把module.换成data_model.
            if k.startswith('module.'):
                state_dict["data_model." + k[len("module."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        model.load_state_dict(state_dict, strict=False)
        print("=> loaded data_model '{}'".format(args.data_model_path))

    for name, param in model.named_parameters():
        if "data_model" in name:
            param.requires_grad = False

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            best_acc2 = checkpoint['best_acc2']
            print("best_acc1: " + str(best_acc1))
            print("best_acc2: " + str(best_acc2))
            # if args.gpu is not None:
            #     # best_acc1 may be from a checkpoint from a different GPU
            #     best_acc1 = best_acc1.to(args.gpu)
            #     best_acc2 = best_acc2.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # 降低CPU占用率的，在模型加载到GPU后使用
    torch.set_num_threads(1)
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop((224, 224)),
        AutoAugment(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = OrderTrainDataset(transform=train_transform, num_classes2=args.num_classes2, input_len=args.len)
    for i in range(0, 3):
        train_dataset += OrderTrainDataset(transform=train_transform, num_classes2=args.num_classes2, input_len=args.len)
    test_dataset = OrderTestDataset(transform=val_transform, num_classes2=args.num_classes2, input_len=args.len)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # train for one epoch
        loss = train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args)

        # evaluate on validation set
        target_acc1, angle_acc2 = validate(val_loader, model, args)

        # remember best acc@1 and save checkpoint
        target_is_best = target_acc1 > best_acc1
        best_acc1 = max(target_acc1, best_acc1)

        angle_is_best = angle_acc2 > best_acc2
        best_acc2 = max(angle_acc2, best_acc2)

        if not args.multiprocessing_distributed \
                or (args.multiprocessing_distributed and args.rank == 0):

            if os.path.exists(args.save_dir) is False:
                os.mkdir(args.save_dir)
            with open(args.save_dir + "/loss.txt", "a") as file1:
                file1.write(str(loss) + "\n")
            file1.close()
            with open(args.save_dir + "/target_acc.txt", "a") as file1:
                file1.write(str(target_acc1) + " " + str(best_acc1) + "\n")
            file1.close()
            with open(args.save_dir + "/angle_acc.txt", "a") as file1:
                file1.write(str(angle_acc2) + " " + str(best_acc2) + "\n")
            file1.close()

            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'best_acc2': best_acc2,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, target_is_best=target_is_best, angle_is_best=angle_is_best, args=args)


def train(train_loader, model, criterion, optimizer, lr_scheduler, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    target_top1 = AverageMeter('TargetAcc@1', ':6.2f')
    target_top5 = AverageMeter('TargetAcc@5', ':6.2f')
    angle_top1 = AverageMeter('AngleAcc@1', ':6.2f')
    angle_top5 = AverageMeter('AngleAcc@1', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, losses, target_top1, target_top5, angle_top1, angle_top5],
        prefix="Epoch: [{}]".format(epoch))
    total_loss = 0

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.train()

    end = time.time()
    for i, (images, angles, target, target_angle) in enumerate(train_loader):
        if args.gpu is not None:
            # b,len,3,224,224
            images = images.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)
            # b,len
            angles = angles.cuda(args.gpu, non_blocking=True).to(dtype=torch.long)
            # b
            target = target.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)
            # b
            target_angle = target_angle.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)

        # b,len->b,len,181
        angles = F.one_hot(angles, num_classes=args.num_classes2)
        angles = angles.to(dtype=torch.float32)

        # b,len,3,224,224+b,len,181
        output1, output2 = model(images, angles)
        loss = criterion(output1, target) + \
               criterion(output2, target_angle)

        # measure accuracy and record loss
        target_acc1, target_acc5 = accuracy(output1, target, topk=(1, 5))
        angle_acc1, angle_acc5 = accuracy(output2, target_angle, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        target_top1.update(target_acc1[0], images.size(0))
        target_top5.update(target_acc5[0], images.size(0))
        angle_top1.update(angle_acc1[0], images.size(0))
        angle_top5.update(angle_acc5[0], images.size(0))

        total_loss += loss.item()

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return float(total_loss / (i + 1))


def validate(val_loader, model, args):
    total_correct = 0
    total_correct_angle = 0
    total_samples = 0

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for i, (images, angles, target, target_angle) in enumerate(val_loader):
            if args.gpu is not None:
                # b,len,3,224,224
                images = images.cuda(args.gpu, non_blocking=True).to(dtype=torch.float32)
                # b,len
                angles = angles.cuda(args.gpu, non_blocking=True).to(dtype=torch.long)
                # b
                target = target.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)
                # b
                target_angle = target_angle.cuda(args.gpu, non_blocking=True).to(dtype=torch.int64)

            # b,len->b,len,181
            angles = F.one_hot(angles, num_classes=args.num_classes2)
            angles = angles.to(dtype=torch.float32)

            # b,len,3,224,224+b,len,181
            output1, output2 = model(images, angles)

            # _,是batch_size*概率，preds是batch_size*最大概率的列号
            _, preds = output1.max(1)
            num_correct = (preds == target).sum()
            num_samples = preds.size(0)
            total_correct += num_correct.item()
            total_samples += num_samples

            _, preds = output2.max(1)
            num_correct = (preds == target_angle).sum()
            total_correct_angle += num_correct.item()

    acc1 = float(total_correct / total_samples)
    acc2 = float(total_correct_angle / total_samples)
    print("Test: TargetAcc1 " + str(acc1))
    print("Test: AngleAcc2 " + str(acc2))
    return acc1, acc2


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


def save_checkpoint(state, target_is_best, angle_is_best, args):
    torch.save(state, args.save_dir + "/checkpoint.pth.tar")
    if target_is_best:
        shutil.copyfile(args.save_dir + "/checkpoint.pth.tar",
                        args.save_dir + "/model_target_best.pth.tar")
    if angle_is_best:
        shutil.copyfile(args.save_dir + "/checkpoint.pth.tar",
                        args.save_dir + "/model_angle_best.pth.tar")


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


if __name__ == '__main__':
    main()
