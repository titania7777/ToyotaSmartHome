# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 16:24:25 2021

@author: rubijade
"""

import os
import csv
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import json

# VideoDataset load for video frames
# frames path: root path of frames
# you must be follow our csv format when using this loader
# csv file header => sub directory file path, index, category
class SkeletonDataset(Dataset):
    def __init__(self, rootpath:str, pose3D_path:str):

        self.rootpath = rootpath
        self.pose3D_path = pose3D_path
        # read a csv file
        with open(pose3D_path, "r") as f:
            reader = csv.reader(f)
            self.labels = {}
            self.categories = {}
            for rows in reader:
                subfilepath = rows[0]; index = int(rows[1]); category = rows[2]
                self.labels[subfilepath] = index # {sub directory file path: index}
                if index not in self.categories:
                    self.categories[index] = category # {index: category}
        self.subfilespath = list(self.labels)
 
    
    def __len__(self) -> int:
        return len(self.subfilespath)
    
    def get_category(self, index:int) -> str:
        return self.categories[index]

    def _get_poselist(self, path_frames:str) -> list:
        f = open(path_frames)
        donne = json.load(f)
        frame = donne['frames']
        pose_data = list()

        for i in range(len(frame)):
            for dic in frame[i]:
                pose_data.append(dic.get('pose3d'))
        
        return list(pose_data)
        

    def __getitem__(self, index):
        subfilepath = self.subfilespath[index]

        path = self.frames_path + subfilepath + '.json'
        
        #getting the 3poses of each frame as a list
        poses3D = self._get_poselist(path)
        labels = self.labels[subfilepath]

        return poses3D, labels

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        poses3D = [item[0].unsqueeze(0) for item in batch]
        pose3D = torch.cat(pose3D, dim=0)
        labels = [item[1] for item in batch]
        labels = pad_sequence(labels, batch_first=False, padding_value=self.pad_idx)

        return pose3D, labels
    
    def getloader(self, root_folder, annotation_file, pose3D, transform, batch_size=32, num_workers = 8, shuffle = True, pin_memory = True):
        data = SkeletonDataset(root_folder, pose3D)
        pad_idx = data.index(data[0])

        loader = DataLoader(dataset=data, batch_size=batch_size, num_workers=num_workers, 
        shuffle=shuffle, pin_memory=pin_memory, collate_fn=MyCollate(pad_idx=pad_idx))

        return loader, data