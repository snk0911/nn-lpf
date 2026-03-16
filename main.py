import argparse
import os
import random
import shutil
import time
import warnings
import sys
import numpy as np
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import aa_models
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='Evaluating Shift and Corruption-Robustness on CNNs with TinyImageNet(-C) dataset')

parser.add_argument('--data', metavar='DIR', default='./dataset/tiny-imagenet-200',
                    help='path to dataset')
parser.add_argument('--data_c', metavar='DIR', default='./dataset/tiny-imagenet-c',
                    help='path to corruption dataset (Tiny ImageNet-C)')

parser.add_argument('--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

parser.add_argument('--aa_type', type=str, default='blur', help='choose between aa methods (default: blur)')
parser.add_argument('--filter_size', type=int, default=4, choices=range(1, 8), help='Anti-aliasing filter size (1-7, default: 4)')
parser.add_argument('--pasa_group', dest='pasa_group', type=int, default='8', help='group number of pasa operation (default: 8)')
parser.add_argument('--wavelet_type', type=str, default='db2', help='Type of wavelet')

parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')

parser.add_argument('-ep', '--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')

parser.add_argument('--lr', '--learning_rate', default=0.2, type=float, metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_scheduler', default='cosine', choices=['step', 'cosine','exponential','plateau'], help='learning rate scheduler (default: cosine)')
parser.add_argument("--lr_warmup_epochs", default=5, type=int, help="the number of epochs to warmup (default: 0)")
parser.add_argument("--lr-warmup-method", default="linear", type=str, help="the warmup method (default: linear)")
parser.add_argument("--lr_warmup_decay", default=0.01, type=float, help="the decay for lr")
parser.add_argument("--lr_step_size", default=30, type=int, help="decrease lr every step-size epochs")
parser.add_argument("--lr_gamma", default=0.1, type=float, help="decrease lr by a factor of lr-gamma")
parser.add_argument("--lr_min", default=0.0, type=float, help="minimum lr of lr schedule (default: 0.0)")

parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')

parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')

parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')

# parser.add_argument('--pretrained', dest='pretrained', action='store_true',
#                    help='use pre-trained model')

parser.add_argument('--force_nonfinetuned', dest='force_nonfinetuned', action='store_true',
                    help='if pretrained, load the model that is pretrained from scratch (if available)')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
                    
parser.add_argument('--evaluate_save', dest='evaluate_save', action='store_true',
                    help='save validation images off')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')

parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')

parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')

parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')

parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')

parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--eval_test', dest='eval_test', action='store_true',
                    help='Switch evaluation to use the test directory instead of validation')

# Added functionality from PyTorch codebase
parser.add_argument('--no-data-aug', dest='no_data_aug', action='store_true',
                    help='no shift-based data augmentation')

parser.add_argument('--output', dest='out_dir', default='./out', type=str,
                    help='output directory')

parser.add_argument('-es', '--evaluate_shift', dest='evaluate_shift', action='store_true',
                    help='evaluate model on shift-invariance')
parser.add_argument('-ed', '--evaluate_diagonal', dest='evaluate_diagonal', action='store_true',
                    help='evaluate model on diagonal')
parser.add_argument('--evaluate_c', action='store_true',
                    help='Evaluate mCE on Tiny ImageNet-C')

parser.add_argument('--epochs-shift', default=5, type=int, metavar='N',
                    help='number of total epochs to run for shift-invariance test')

parser.add_argument('-ba', '--batch-accum', default=1, type=int,
                    metavar='N',
                    help='number of mini-batches to accumulate gradient over before updating (default: 1)')

parser.add_argument('--embed', dest='embed', action='store_true',
                    help='embed statement before anything is evaluated (for debugging)')

parser.add_argument('--val_debug', dest='val_debug', action='store_true',
                    help='debug by training on val set')

parser.add_argument('--weights', default=None, type=str, metavar='PATH',
                    help='path to pretrained model weights')

parser.add_argument('--save_weights', default=None, type=str, metavar='PATH',
                    help='path to save model weights')

parser.add_argument('--finetune', action='store_true', help='finetune from baseline model')

parser.add_argument('-mti', '--max-train-iters', default=np.inf, type=int,
                    help='number of training iterations per epoch before cutting off (default: infinite)')

parser.add_argument('--wandb', action='store_true', help='use wandb logging')

best_acc1 = 0


def main():
    args = parser.parse_args()
    print(f"lr: {args.lr}")
    print(f"lr_scheduler: {args.lr_scheduler}")
    print(f"lr_warmup_method: {args.lr_warmup_method}")
    print(f"lr_warmup_epochs: {args.lr_warmup_epochs}")  
    print(f"aa_type: {args.aa_type}")
    if args.aa_type == 'blur':
        print(f"filter_size: {args.filter_size}")
        subdir = f"{args.arch}_{args.aa_type}_filter{args.filter_size}"
    elif args.aa_type == 'soft':
        print(f"filter_size: {args.filter_size}")
        subdir = f"{args.arch}_{args.aa_type}_filter{args.filter_size}"
    elif args.aa_type == 'dab':
        print(f"filter_size: {args.filter_size}")
        subdir = f"{args.arch}_{args.aa_type}_filter{args.filter_size}"
    elif args.aa_type == 'pasa':
        print(f"filter_size: {args.filter_size}")
        print(f"pasa_group: {args.pasa_group}")
        subdir = f"{args.arch}_{args.aa_type}_filter{args.filter_size}_group{args.pasa_group}"
    elif args.aa_type == 'dwt':
        print(f"wavelet_type: {args.wavelet_type}")
        print(f"DEBUG wavelet_type: {args.wavelet_type}, Type: {type(args.wavelet_type)}")
        subdir = f"{args.arch}_{args.aa_type}_{args.wavelet_type}"
    elif args.aa_type == 'none_debug':
        subdir = f"{args.arch}_baseline_debug"
    elif args.aa_type == 'none':
        subdir = f"{args.arch}_baseline"
    
    args.out_dir = os.path.join(args.out_dir, subdir)
    if(not os.path.exists(args.out_dir)):
        os.makedirs(args.out_dir)

    # mentioning this in the Reproducibilty Chapter
    if args.seed is not None:
        # https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        random.seed(args.seed)
        np.random.seed(args.seed) # Zhang forgot to add this so validate_shift is deterministic and therefore reproducable
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        warnings.warn(
            '\nYou have chosen to seed training.\n'  # <-- Add \n here
            'This will turn on the CUDNN deterministic setting,\n'
            'which can slow down your training considerably!\n'
            'You may see unexpected behavior when restarting\n'
            'from checkpoints.'
        )

    if args.gpu is not None:
        warnings.warn('\nYou have chosen a specific GPU. \nThis will completely disable data parallelism.')

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
    global best_acc1
    args.gpu = gpu

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

    # create model
    print("=> creating model '{}'".format(args.arch))

    if args.aa_type == 'none_debug':
        print("=> torchvision baseline in use")
        model = models.__dict__[args.arch](num_classes=200)
        num_classes_in_model = model.fc.out_features
        print(f"Model has {num_classes_in_model} classes at output.")

        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(3, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        print("Replaced conv1 with 3x3 stride 1 for CIFAR/Tiny-ImageNet-Style.")
    else:
        model = aa_models.__dict__[args.arch](
            aa_type=args.aa_type,
            wavelet_type=args.wavelet_type,
            filter_size=args.filter_size,
            pasa_group=args.pasa_group,
            num_classes=200
        )

    # instrumentation
    if(args.wandb):
        import wandb
        wandb.init(project='cnn_aa_methods')
        wandb.config.update(args)
        wandb.watch(model)

    if args.finetune: 
        print("=> copying over pretrained weights from [%s]"%args.arch[:-5])
        model_baseline = models.__dict__[args.arch[:-5]](weights=True)
        aa_models.copy_params_buffers(model_baseline, model)

    if args.weights is not None:
        print("=> using saved weights [%s]"%args.weights)
        weights = torch.load(args.weights)
        model.load_state_dict(weights['state_dict'])

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int(args.workers / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(
        model.parameters(), 
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        # added for ResNet-18 training; probably args needed; mentioned in Tiny-ImageNet-Paper
        nesterov=True
    )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            if('optimizer' in checkpoint.keys()): # if no optimizer, then only load weights
                args.start_epoch = checkpoint['epoch']
                best_acc1 = checkpoint['best_acc1']
                if args.gpu is not None:
                    # best_acc1 may be from a checkpoint from a different GPU
                    best_acc1 = best_acc1.to(args.gpu)
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('  No optimizer saved')
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    train_dir = os.path.join(args.data, 'train')
    val_dir = os.path.join(args.data, 'val')
    test_dir = os.path.join(args.data, 'test')

    # "Normalization ensures that the input features have a similar distribution. 
    # This is crucial because it helps the gradient descent algorithm converge faster and more stably by conditioning the Hessian matrix (LeCun et al., 1998). 
    # Therefore, calculating the exact mean and standard deviation for a specific dataset like Tiny ImageNet—rather than relying on generic values—is considered a best practice for maximizing performance (Krizhevsky et al., 2012)."
    # Done to replicate the full ImageNet-1k scenario

    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2764, 0.2689, 0.2816]) # for Tiny-ImageNet

    # minimal training for Tiny-ImageNet with 64x64 images
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    crop_size = 72 if(args.evaluate_shift or args.evaluate_diagonal or args.evaluate_save) else 64
    args.batch_size = 1 if (args.evaluate_diagonal or args.evaluate_save) else args.batch_size

    if args.eval_test:
        chosen_eval_dir = test_dir
        print("Using test dataset for evaluation.")
    else:
        chosen_eval_dir = val_dir
        print("Using validation dataset for evaluation.")

    eval_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(chosen_eval_dir, transforms.Compose([
            transforms.Resize(72, interpolation=transforms.InterpolationMode.NEAREST), # InterpolationMode.NEAREST so no additional blur during eval 
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.val_debug:
        if args.eval_test:
            # This throws an error and stops the script immediately
            raise ValueError("Conflict: --val_debug cannot be used together with --eval_test.")
        
        # If we get here, it is safe to train on val
        print("DEBUG MODE: Training on Validation Set")
        train_loader = eval_loader

    if(args.embed):
        from IPython import embed
        embed()

    if args.save_weights is not None: # "deparallelize" saved weights
        print("=> saving 'deparallelized' weights [%s]"%args.save_weights)
        
        # Helper to safely unwrap the model
        def get_raw_model(model):
            if isinstance(model, (torch.nn.DataParallel, torch.nn.parallel.DistributedDataParallel)):
                return model.module
            return model

        raw_model = get_raw_model(model)
        
        # Handle specific VGG/AlexNet feature wrapping if present
        if hasattr(raw_model, 'features') and isinstance(raw_model.features, torch.nn.DataParallel):
            raw_model.features = raw_model.features.module

        torch.save({'state_dict': raw_model.state_dict()}, args.save_weights, _use_new_zipfile_serialization=False)
        return

    if args.evaluate:
        evaluate(eval_loader, model, criterion, args)
        return

    if(args.evaluate_shift):
        evaluate_shift(eval_loader, model, args)
        return

    if(args.evaluate_diagonal):
        evaluate_diagonal(eval_loader, model, args)
        return

    if args.evaluate_c:
        evaluate_c(eval_loader, model, criterion, args)
        return

    if(args.evaluate_save):
        evaluate_save(eval_loader, mean, std, args)
        return

    args.lr_scheduler = args.lr_scheduler.lower()
    
    if args.lr_scheduler == "step":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosine":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=args.lr_min
        )
    elif args.lr_scheduler == "exponential":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    elif args.lr_scheduler == "plateau":
        main_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.1, patience=10, threshold=0.0001
        )
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only step, cosine, exponential and plateau "
            "are supported."
        )

    if args.lr_warmup_epochs > 0 and args.lr_scheduler != "plateau":
        # Note: Warmup is usually not used with ReduceLROnPlateau because Plateau is metric-dependent
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        print("sequential lr started, lr_warmup_method > 0")
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        # Log current LR before training/eval
        if args.wandb:
            wandb.log({'learning_rate': optimizer.param_groups[0]['lr']}, step=epoch, commit=False)

        # Train
        train(train_loader, model, criterion, optimizer, epoch, args)

        # Evaluate
        acc1, val_loss = evaluate(eval_loader, model, criterion, args)

        # Step the scheduler
        if args.lr_scheduler == "plateau":
            # Plateau steps based on validation metric (val_loss or acc1)
            # Assuming we want to maximize acc1 (mode='max' in setup above)
            lr_scheduler.step(acc1) 
        else:
            # Cosine, Step, Exponential step every epoch
            lr_scheduler.step()

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': state_dict,
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(), # CRITICAL: Save scheduler state
            }, is_best, epoch, out_dir=args.out_dir)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    accum_track = 0
    optimizer.zero_grad()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        loss.backward()

        accum_track+=1
        if(accum_track==args.batch_accum):
            optimizer.step()
            accum_track = 0
            optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))

            if(args.wandb):
                import wandb
                global_step = i + (epoch * len(train_loader))
                wandb.log(
                    {
                        'train_loss': losses.val,
                        'train_avg_loss': losses.avg,
                        'train_acc@1': top1.val,
                        'train_avg_acc@1': top1.avg,
                        'train_acc@5': top5.val,
                        'train_avg_acc@5': top5.avg,
                        'epoch': 1.*global_step/len(train_loader), 
                    },
                    step=global_step)

        if(i > args.max_train_iters):
            break


def evaluate(eval_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    latency = AverageMeter()  # NEW

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(eval_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # measure network latency (single sample, no batching)  # NEW
            torch.cuda.synchronize()
            lat_start = time.time()                                  
            _ = model(input[:1])                                 
            torch.cuda.synchronize()                
            latency.update((time.time() - lat_start) * 1000)   

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                      'Latency {latency.val:.2f}ms ({latency.avg:.2f}ms)'.format(
                       i, len(eval_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5, latency=latency))

        if args.wandb:
            import wandb
            wandb.log(
                {
                    'val_avg_loss': losses.avg,
                    'val_avg_acc@1': top1.avg,
                    'val_avg_acc@5': top5.avg,
                    'val_avg_latency_ms': latency.avg
                },
                commit=False)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Latency {latency.avg:.2f}ms'
              .format(top1=top1, top5=top5, latency=latency))

    return top1.avg, losses.avg, latency.avg


def evaluate_shift(eval_loader, model, args):
    batch_time = AverageMeter()
    consist = AverageMeter()
    chord = AverageMeter()

    model.eval()

    with torch.no_grad():
        end = time.time()
        for ep in range(args.epochs_shift):
            for i, (input, target) in enumerate(eval_loader):
                if args.gpu is not None:
                    input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)

                off0 = np.random.randint(8, size=2)
                off1 = np.random.randint(8, size=2)

                output0 = model(input[:,:,off0[0]:off0[0]+64,off0[1]:off0[1]+64])
                output1 = model(input[:,:,off1[0]:off1[0]+64,off1[1]:off1[1]+64])

                # Binary shift consistency
                cur_agree = agreement(output0, output1).type(torch.FloatTensor).to(output0.device)
                consist.update(cur_agree.item(), input.size(0))

                # Chord Distance
                #
                # Measures prediction stability under spatial shifts by comparing
                # the full softmax probability distributions of two shifted crops.
                # Goes beyond top-1 agreement to capture changes in the entire
                # confidence profile.
                #
                # Given two probability vectors p0 and p1 from softmax outputs:
                #   1. L2-normalize both vectors onto the unit hypersphere:
                #      p0' = p0 / ||p0||_2,  p1' = p1 / ||p1||_2
                #
                #   2. Compute the Euclidean distance between the normalized vectors:
                #      d_chord(p0, p1) = ||p0' - p1'||_2 = sqrt(2 - 2*cos(theta))
                #      where theta is the angle between p0' and p1'.
                #
                # Chord distance is a proper metric on the unit hypersphere,
                # ranging from 0 (identical predictions) to sqrt(2) (maximally
                # different predictions). Lower values indicate more consistent
                # predictions under shift.
                prob0 = torch.nn.Softmax(dim=1)(output0)
                prob1 = torch.nn.Softmax(dim=1)(output1)
                prob0_norm = torch.nn.functional.normalize(prob0, p=2, dim=1)
                prob1_norm = torch.nn.functional.normalize(prob1, p=2, dim=1)
                cur_chord = torch.norm(prob0_norm - prob1_norm, p=2, dim=1).mean().item()
                chord.update(cur_chord, input.size(0))
                
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    print('Ep [{0}/{1}]:\t'
                          'Test: [{2}/{3}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Consist {consist.val:.4f} ({consist.avg:.4f})\t'
                          'Chord {chord.val:.4f} ({chord.avg:.4f})\t'.format(
                           ep, args.epochs_shift, i, len(eval_loader),
                           batch_time=batch_time, consist=consist, chord=chord))

        print(' * Consistency {consist.avg:.3f} Chord {chord.avg:.3f}'
              .format(consist=consist, chord=chord))

    return consist.avg, chord.avg

# Interessant — die Chord-Ähnlichkeit (83.2) ist höher als die Konsistenz (81.2).
# Das ergibt Sinn: Die Konsistenz zählt nur binär, ob die Top-1-Vorhersage übereinstimmt. Wenn sie nicht übereinstimmt, gibt es eine 0 — egal wie knapp es war.
# Aber bei der Chord-Ähnlichkeit kann es sein, dass das Netzwerk bei einem Ausschnitt knapp "Katze" und beim anderen knapp "Hund" vorhersagt. Die Konsistenz sagt dann 0, aber die Verteilungen sind trotzdem fast identisch (z.B. 49% Katze vs. 51% Katze), also bleibt die Chord-Ähnlichkeit hoch.
# Das heißt: Viele der Fälle, in denen die Top-1-Vorhersage kippt, sind knappe Entscheidungen nahe an der Decision Boundary. Das Netzwerk ist stabiler als die Konsistenz allein vermuten lässt — es ändert nicht grundlegend seine Meinung, sondern schwankt nur bei ohnehin unsicheren Vorhersagen.

def evaluate_diagonal(eval_loader, model, args):
    batch_time = AverageMeter()
    prob = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    D = 9
    diag_probs = np.zeros((len(eval_loader.dataset),D))
    diag_probs2 = np.zeros((len(eval_loader.dataset),D)) # save highest probability, not including ground truth
    diag_corrs = np.zeros((len(eval_loader.dataset),D))
    diag_preds = np.zeros((len(eval_loader.dataset),D))

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(eval_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            inputs = []
            for off in range(D):
                inputs.append(input[:,:,off:off+64,off:off+64])
            inputs = torch.cat(inputs, dim=0)
            probs = torch.nn.Softmax(dim=1)(model(inputs))
            preds = probs.argmax(dim=1).cpu().data.numpy()
            corrs = preds == target.item()
            outputs = 100.*probs[:,target.item()]
            
            acc1, acc5 = accuracy(probs, target.repeat(D), topk=(1, 5))

            probs[:,target.item()] = 0
            probs2 = 100.*probs.max(dim=1)[0].cpu().data.numpy()

            diag_probs[i,:] = outputs.cpu().data.numpy()
            diag_probs2[i,:] = probs2
            diag_corrs[i,:] = corrs
            diag_preds[i,:] = preds

            # measure agreement and record
            prob.update(np.mean(diag_probs[i,:]), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Prob {prob.val:.4f} ({prob.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(eval_loader), batch_time=batch_time, prob=prob, top1=top1, top5=top5))

    print(' * Prob {prob.avg:.3f} Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
          .format(prob=prob,top1=top1, top5=top5))

    np.save(os.path.join(args.out_dir,'diag_probs'),diag_probs)
    np.save(os.path.join(args.out_dir,'diag_probs2'),diag_probs2)
    np.save(os.path.join(args.out_dir,'diag_corrs'),diag_corrs)
    np.save(os.path.join(args.out_dir,'diag_preds'),diag_preds)


def evaluate_c(eval_loader, model, criterion, args):
    distortions = [
        'gaussian_noise', 'shot_noise', 'impulse_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
        'snow', 'frost', 'fog', 'brightness',
        'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
    ]

    error_rates = []

    # First get clean error on val set
    print('\nComputing clean error...')
    acc1, _, _ = evaluate(eval_loader, model, criterion, args)
    clean_error = 1. - acc1.item() / 100.
    print('Clean error: {:.2f}%'.format(100 * clean_error))

    model.eval()

    for distortion_name in distortions:
        severity_errors = []

        for severity in range(1, 6):
            top1 = AverageMeter()

            new_root = os.path.join(args.data_c, distortion_name, str(severity))
            eval_loader.dataset.root = new_root
            eval_loader.dataset.samples = datasets.ImageFolder(new_root).samples
            eval_loader.dataset.imgs = eval_loader.dataset.samples

            with torch.no_grad():
                for i, (input, target) in enumerate(eval_loader):
                    if args.gpu is not None:
                        input = input.cuda(args.gpu, non_blocking=True)
                    target = target.cuda(args.gpu, non_blocking=True)

                    output = model(input)
                    acc1, _ = accuracy(output, target, topk=(1, 5))
                    top1.update(acc1[0], input.size(0))

                    if i % args.print_freq == 0:
                        print('Distortion: {:20s} | Severity: [{:d}] [{:d}/{:d}]\t'
                              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                               distortion_name, severity, i, len(eval_loader), top1=top1))

            severity_errors.append(1. - top1.avg.item() / 100.)

        raw_err = np.mean(severity_errors)
        error_rates.append(raw_err)

        print('Distortion: {:20s} | Raw Error (%): {:.2f}'.format(
            distortion_name, 100 * raw_err))

    mce = 100 * np.mean(error_rates)

    print('\n * Clean Error:  {:.2f}%'.format(100 * clean_error))
    print(' * mCE:          {:.2f}%'.format(mce))

    if args.wandb:
        import wandb
        wandb.log({
            'clean_error': 100 * clean_error,
            'mCE':         mce,
        })

    return mce, clean_error * 100


def evaluate_save(eval_loader, mean, std, args):
    import matplotlib.pyplot as plt
    import os
    for i, (input, target) in enumerate(eval_loader):
        img = (255*np.clip(input[0,...].data.cpu().numpy()*np.array(std)[:,None,None] + mean[:,None,None],0,1)).astype('uint8').transpose((1,2,0))
        plt.imsave(os.path.join(args.out_dir,'%05d.png'%i),img)


# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
def save_checkpoint(state, is_best, epoch, out_dir='./'):
    torch.save(state, os.path.join(out_dir,'checkpoint.pth.tar'))
    if(epoch % 10 == 0):
        torch.save(state, os.path.join(out_dir,'checkpoint_%03d.pth.tar'%epoch))
    if is_best:
        shutil.copyfile(os.path.join(out_dir,'checkpoint.pth.tar'), os.path.join(out_dir,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def agreement(output0, output1):
    pred0 = output0.argmax(dim=1, keepdim=False)
    pred1 = output1.argmax(dim=1, keepdim=False)
    agree = pred0.eq(pred1)
    agree = 100.*torch.mean(agree.type(torch.FloatTensor).to(output0.device))
    return agree


if __name__ == '__main__':
    main()
