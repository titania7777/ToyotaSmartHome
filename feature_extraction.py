import torch
import os
import argparse
import torch.nn as nn
import numpy as np
import torchvision
from torch.utils.data import DataLoader
from VideoDataset import VideoDataset
import os
import csv
import json
from torch.backends import cudnn

if torch.cuda.is_available():
    cudnn.benchmark = True
    device = torch.device(f"cuda:0")
else:
    device = torch.device(f"cpu")

parser = argparse.ArgumentParser(description="Toyota Smart Home spatial stream on ConvLSTM")
parser.add_argument("--frames-path", default="./Data/mp4_frames/", type=str)
parser.add_argument("--csv-path", default="./Data/Labels/cross_subject/", type=str)
parser.add_argument("--cross-view", action="store_true")
parser.add_argument("--frame-size", default=224, type=int)
parser.add_argument("--sequence-length", default=16, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--batch-size", default=64, type=int)
parser.add_argument("--lr", default=1e-3, type=float)
parser.add_argument("--gpu-number", default=0, type=int)

args = parser.parse_args()

spatialmodel = torch.load("model_save/spatial_model_03_09.pth")

class new_model(nn.Module):
    def __init__(self, model, listLayer):
        super().__init__()
        self.pretrained = model
        self.encodeblock = listLayer[0]
        self.lstmblock = listLayer[1]
        self.children_list = []
        for name, children in self.pretrained.named_children():
            self.children_list.append(children)
            if name == self.lstmblock:
                break
        
        self.net = nn.Sequential(*self.children_list)
        self.pretrained = None
    
    def forward(self,x):

        for n, mod in self.net.named_children():
            if n == self.encodeblock:
                b, d, c, h, w = x.shape
                x = x.view(b * d, c, h, w)

            x = mod(x)

            if n == self.lstmblock:
                x = x.view(b, d, -1)
                x = mod(x)[0][:, -1]
            
        return x

# Function return list of features and label
# It is supposed to have all the model of the streams (spatial, opticalflow and skeleton), but for the test purpose i only take the spatial model into account
def get_output(spatialModel:nn.Module, dataloader):
    listrgbfeature = []; listlabels = []

    for i, (rgb_data, labels) in enumerate(dataloader):
         # Here I have tried to change the dimension of the input data after getting dimentional error
         # however the error remains. I think because of the lstm. For that reason I have created the the class new_model and tried to manipulate the dimension of the input
         # Expected 4-dimensional input for 4-dimensional weight [64, 3, 7, 7], but got 5-dimensional input of size [64, 16, 3, 224, 224] instead
         b, d, c, h, w = rgb_data.shape
         rgb_data = rgb_data.view(b*d, c, h, w)
         rgb = spatialModel(rgb_data.to(device))
         listrgbfeature.append(rgb)
         listlabels.append(labels)
    
    return listrgbfeature, listlabels

def get_Features(spatialModel:nn.Module, dataloader):
    spatialModel = nn.Sequential(*list(spatialModel.children())[:-1])

    for param in spatialModel.parameters():
        param.requires_grad = False
    
    spatiallsttfeat, listlabels = get_output(spatialModel, dataloader)

    return spatiallsttfeat, listlabels


train = VideoDataset(frames_path=args.frames_path, csv_path=args.csv_path + "train.csv", frame_size=args.frame_size, sequence_length=args.sequence_length)
val = VideoDataset(frames_path=args.frames_path, csv_path=os.path.join(args.csv_path, "val.csv" if args.cross_view else "test.csv"), frame_size=args.frame_size, sequence_length=args.sequence_length)
train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)

#get the features from the class
listfeature = []
model = new_model(spatialmodel, ["encoder", "lstm"])
model.to(device)

for i, (data, label) in enumerate(train_loader):
    out = model(data.to(device))
    listfeature.append(out)

#get the features from the functions
listspafeatures, listlabels = get_Features(spatialmodel, train_loader)
