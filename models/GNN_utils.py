import argparse
import shutil
import os
import time
import random
from enum import Enum
import torch

import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts


import logging
logging.basicConfig()
logging.root.setLevel(logging.NOTSET)

def save_checkpoint(state, is_best, args,filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join('best_models',args.experiment_name+'_model_best.pth'))

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))
        
    def display_summary(self):
        entries = [self.prefix]
        entries += [meter.summary() for meter in self.meters]
        logging.info(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
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

    def all_reduce(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        # dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
    
    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)
        
        return fmtstr.format(**self.__dict__)
    

def train(train_dataset, model, criterion, optimizer, epoch, device, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    # data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')

    progress = ProgressMeter(
        train_dataset.snapshot_count,
        [batch_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()

    for i,snapshot in enumerate(train_dataset):
        
        snapshot = snapshot.to(device)
        optimizer.zero_grad()
        
        # compute prediction
        y_hat = model(snapshot.x, snapshot.edge_index)
        
        loss = criterion(y_hat, snapshot.y)
        losses.update(loss.item())
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i + 1)
    writer.add_scalar("Loss/train", losses.avg, epoch)
    return writer

def validate(val_dataset, model, criterion,device, writer, epoch, args):

    def run_validate(val_dataset, base_progress=0):
        with torch.no_grad():
            end = time.time()
            
            for i,snapshot in enumerate(val_dataset):
                i = base_progress + i
                snapshot = snapshot.to(device)
                # compute output
                y_hat = model(snapshot.x, snapshot.edge_index)
                loss = criterion(y_hat, snapshot.y)

                # measure accuracy and record loss
                losses.update(loss.item())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % args.print_freq == 0:
                    progress.display(i + 1)

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        val_dataset.snapshot_count,
        [batch_time, losses],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    run_validate(val_dataset)

    progress.display_summary()
    
    writer.add_scalar("Loss/Val", losses.avg, epoch)

    return losses.avg, writer
