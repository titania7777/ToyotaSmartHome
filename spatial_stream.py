#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 17:57:52 2020

@author: coco
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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

# import dataloader
# from utils import *
# from network import *

parser = argparse.ArgumentParser(description='Toyota_smart_home spatial stream on CCLSTM')
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs')
parser.add_argument('--batch-size', default=25, type=int, metavar='N', help='mini-batch size (default: 25)')
parser.add_argument('--lr', default=5e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')

parser.add_argument("--dataset_path", type=str, default="data/UCF-101-frames", help="Path to UCF-101 dataset")
#Parser.add_argument("--split_path", type=str, default="data/ucfTrainTestlist", help="Path to train/test split")
parser.add_argument("--split_number", type=int, default=1, help="train/test split number. One of {1, 2, 3}")
parser.add_argument("--sequence_length", type=int, default=40, help="Number of frames in each sequence")
parser.add_argument("--img_dim", type=int, default=224, help="Height / width dimension")
parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
parser.add_argument("--latent_dim", type=int, default=512, help="Dimensionality of the latent representation")
parser.add_argument("--checkpoint_model", type=str, default="", help="Optional path to checkpoint model")
parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval between saving model checkpoints")
opt = parser.parse_args()
print(opt)
        
    

############################################################################################
#convLSTM
############################################################################################


### Ici le encoder represent le resnet qui va permettre d'extraire dans un premier temps le features
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = resnet18(pretrained=True).cuda()
        # here they have removed the last fully connected layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        # self.final = nn.Sequential(nn.Linear(resnet.fc.in_features, latent_dim))
        
    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return x

###__LSTM__###
# le but est mettre en place le lstm
# puisque le Convlstm est au fait du lstm dans lequel lon met du cnn, donc dans
# ce code ils ont dabord mis en place le ccn quils ont nomme encoder et ensuite dans 
# l'implementation du lstm ils ont just fait appel au cnn construt avec le decoder
class LSTM(nn.Module):
    def __init__(self, latent_dim, num_layers, hidden_dim, bidirectional):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_state = None
        
    def reset_hidden_state(self):
        self.hidden_state = None
    
    def forward(self, x):
        x, self.hidden_state = self.lstm(x, self.hidden_state)
        return x


###___The attention module___###
#Le module attetention. Tu peux ne pas l'inclure dans toon code
class Attention(nn.Module):
    def __init__(self, latent_dim, hidden_dim, attention_dim):
        super(Attention, self).__init__()
        self.latent_attention = nn.Linear(latent_dim, attention_dim)
        self.hidden_attention = nn.Linear(hidden_dim, attention_dim)
        self.joint_attention = nn.Linear(attention_dim, 1)

    def forward(self, latent_repr, hidden_repr):
        if hidden_repr is None:
            hidden_repr = [
                Variable(
                    torch.zeros(latent_repr.size(0), 1, self.hidden_attention.in_features), requires_grad=False
                ).float()
            ]
        h_t = hidden_repr[0]
        latent_att = self.latent_attention(latent_att)
        hidden_att = self.hidden_attention(h_t)
        joint_att = self.joint_attention(F.relu(latent_att + hidden_att)).squeeze(-1)
        attention_w = F.softmax(joint_att, dim=-1)
        return attention_w

### The convLSTM ###
class ConvLSTM(nn.Module):
    def __init__(self, num_classes, latent_dim=512, lstm_layers=1, hidden_dim=1024, bidirectional=True, attention=True):
        super(ConvLSTM, self).__init__()
        self.encoder = Encoder()
        self.lstm = LSTM(latent_dim, lstm_layers, hidden_dim, bidirectional)
        self.output_layers = nn.Sequential(
            nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim, momentum=0.01),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        self.attention = attention
        self.attention_layer = nn.Linear(2 * hidden_dim if bidirectional else hidden_dim, 1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        x = self.encoder(x).squeeze()
        x = x.view(batch_size, seq_length, -1)
        x = self.lstm(x) # batch, seq, feature
        if self.attention:
            attention_w = F.softmax(self.attention_layer(x).squeeze(-1), dim=-1)
            x = torch.sum(attention_w.unsqueeze(-1) * x, dim=1)
        else:
            x = x[:, -1]
        return self.output_layers(x)




        

