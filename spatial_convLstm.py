#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 15:43:29 2021

@author: coco
"""

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
from torch.autograd import Variable
import torch.optim as optim

import pickle
import os
from PIL import Image
import time
from tqdm import tqdm
import shutil
from random import randint
import argparse

import torchvision.transforms as transforms
from torchvision.models import resnet18

import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
import labeler as lb
from VideoDataset import VideoDataset
from torch.utils.data import DataLoader

# from convlstm import ConvLSTM
from spatial_stream import ConvLSTM

# import dataloader
# from utils import *
# from network import *

parser = argparse.ArgumentParser(description='Toyota_smart_home spatial stream on CCLSTM')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=32, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--frames_path', default='frame_output/', type=str, help='path to the image frames' )
parser.add_argument('--csv_path', default='labels/', type=str, help='path to the csv data' )
parser.add_argument('--iter-size', default=5, type=int, metavar='I', help='iter size as in Caffe to reduce memory usage (default: 5)')
parser.add_argument('--print-freq', default=50, type=int, metavar='N', help='print frequency (default: 50)')
parser.add_argument('--chanels', default=3, type=int, help='input data number of chanels')


def main():
    global arg
    arg = parser.parse_args()
    print(arg)

    #Prepare the loader
    train = VideoDataset(frames_path=arg.frames_path, csv_path=arg.csv_path+"cross_view1/train.csv", frame_size=112, sequence_length=8)
    test = VideoDataset(frames_path=arg.frames_path, csv_path=arg.csv_path+"cross_view1/test.csv", frame_size=112, sequence_length=8)
    val = VideoDataset(frames_path=arg.frames_path, csv_path=arg.csv_path+"cross_view1/val.csv", frame_size=112, sequence_length=8)
    
    train_loader = DataLoader(train, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test, batch_size=16, shuffle=False, num_workers=4)
    val_loader = DataLoader(val, batch_size=16, shuffle=False, num_workers=4)
    
    #The Model 
    model = Spatial_ConvLstm(
                        num_classes = train.get_num_classes(),
                        nb_epochs=arg.epochs,
                        lr=arg.lr,
                        batch_size=arg.batch_size,
                        resume=arg.resume,
                        start_epoch=arg.start_epoch,
                        evaluate=arg.evaluate,
                        train_loader=train_loader,
                        test_loader=val_loader,
    )
    #Training
    model.run()

### Ici le encoder represent le resnet qui va permettre d'extraire dans un premier temps le features
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet18(pretrained=True).cuda()
        # here they have removed the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x

class Spatial_ConvLstm():
    def __init__(self, num_classes, nb_epochs, lr, batch_size, resume, start_epoch, evaluate, train_loader, test_loader):
        self.num_classes = num_classes
        self.nb_epochs=nb_epochs
        self.lr=lr
        self.batch_size=batch_size
        self.resume=resume
        self.start_epoch=start_epoch
        self.evaluate=evaluate
        self.train_loader=train_loader
        self.test_loader=test_loader
        self.best_prec1=0

    def build_model(self):
        print ('==> Build model and setup loss and optimizer')
        #build model
        self.model = ConvLSTM(self.num_classes)
        # ConvLSTM(
        #     input_dim=arg.chanels, hidden_dim=[64, 64, 128], kernel_size=(3, 3), num_layers=3,
        #     batch_first=True,
        #     bias=True,
        #     return_all_layers=False)
        
        self.model.cuda()
        # self.encoder = Encoder()
        # self.encoder.cuda()
        #Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.optimizer = torch.optim.SGD(self.model.parameters(), self.lr, momentum=0.9)
        self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=1,verbose=True)
    
    def resume_and_evaluate(self):
        if self.resume:
            if os.path.isfile(self.resume):
                print("==> loading checkpoint '{}'".format(self.resume))
                checkpoint = torch.load(self.resume)
                self.start_epoch = checkpoint['epoch']
                self.best_prec1 = checkpoint['best_prec1']
                self.model.load_state_dict(checkpoint['state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("==> loaded checkpoint '{}' (epoch {}) (best_prec1 {})"
                  .format(self.resume, checkpoint['epoch'], self.best_prec1))
            else:
                print("==> no checkpoint found at '{}'".format(self.resume))
        if self.evaluate:
            self.epoch = 0
            prec1, val_loss = self.validate_()
            return

    def run(self):
        self.build_model()
        self.resume_and_evaluate()
        cudnn.benchmark = True
        
        for self.epoch in range(self.start_epoch, self.nb_epochs):
            # training on one epoch
            self.train_()
            
            # evaluation on on epoch
            prec1 = 0.0
            if (self.epoch + 1) % arg.save_freq == 0:
                prec1 = self.validate_1epoch()
                
            is_best = prec1 > self.best_prec1
            self.best_prec1 = max(prec1, self.best_prec1)
            
            # save model
            if (self.epoch + 1) % arg.save_freq == 0:
                checkpoint_name = "%03d_%s" % (self.epoch + 1, "checkpoint.pth.tar")
                save_checkpoint({
                    'epoch': self.epoch,
                    'arch': arg.arch,
                    'state_dict': self.model.state_dict(),
                    'best_prec1': self.best_prec1,
                    'optimizer' : self.optimizer.state_dict()},is_best,'record/spatial/checkpoint.pth.tar','record/spatial/model_best.pth.tar')

    def train_(self):
        print('==> Epoch:[{0}/{1}][training stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        #switch to train mode
        self.model.train()    
        end = time.time()
        self.optimizer.zero_grad()
        
        loss_mini_batch = 0.0
        acc_mini_batch = 0.0
        # mini-batch training
        
        for i, (datas, target) in enumerate(self.train_loader):
        
            datas = datas.cuda()
            target = target.cuda()
            
            #b, d, c, h, w = input_var.size()
            # input_var = input_var.view(-1, c, h, w)
            # extracted_features = self.encoder(input_var).squeeze()
            
            # extracted_features = extracted_features.view(b, d, -1) # b, d, 512
            output = self.model(datas)
    
            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            acc_mini_batch += prec1.item()
            loss = self.criterion(output, target)
            loss = loss / arg.iter_size
            loss_mini_batch += loss
            loss.backward()
    
            if (i+1) % arg.iter_size == 0:
                # compute gradient and do SGD step
                self.optimizer.step()
                self.optimizer.zero_grad()
    
                # losses.update(loss_mini_batch/args.iter_size, input.size(0))
                # top1.update(acc_mini_batch/args.iter_size, input.size(0))
                losses.update(loss_mini_batch, v_input.size(0))
                top1.update(acc_mini_batch/arg.iter_size, v_input.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                loss_mini_batch = 0
                acc_mini_batch = 0
    
                if (i+1) % arg.print_freq == 0:
    
                    print('Epoch: [{0}][{1}/{2}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                           self.epoch, i+1, len(self.train_loader)+1, batch_time=batch_time, loss=losses, top1=top1))
                    

    def validate_(self):
        print('==> Epoch:[{0}/{1}][validation stage]'.format(self.epoch, self.nb_epochs))
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()
        
        end = time.end()

        for i, (datas, target) in enumerate(self.val_loader):
            datas = datas.cuda()
            target = target.cuda()
    
            # compute output
            output = self.model(datas)
            loss = self.criterion(output, target)
    
            # measure accuracy and record loss
            prec1, prec3 = accuracy(output.data, target, topk=(1, 3))
            losses.update(loss.data[0], input.size(0))
            top1.update(prec1[0], input.size(0))
            top3.update(prec3[0], input.size(0))
    
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
    
            if i % arg.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@3 {top3.val:.3f} ({top3.avg:.3f})'.format(
                       i, len(self.val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top3=top3))
    
        print(' * Prec@1 {top1.avg:.3f} Prec@3 {top3.avg:.3f}'
              .format(top1=top1, top3=top3))

        return top1.avg

def save_checkpoint(state, is_best, filename, resume_path):
    cur_path = os.path.join(resume_path, filename)
    best_path = os.path.join(resume_path, 'model_best.pth.tar')
    torch.save(state, cur_path)
    if is_best:
        shutil.copyfile(cur_path, best_path)

def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)
    
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
    
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
    
class AverageMeter(object):
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
    
    
    
if __name__ == '__main__':
    main()