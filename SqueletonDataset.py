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
import torchvision.transforms as transforms
import json

# VideoDataset load for video frames
# frames path: root path of frames
# you must be follow our csv format when using this loader
# csv file header => sub directory file path, index, category
class SkeletonDataset(Dataset):
    def __init__(self, frames_path:str, csv_path:str, train:bool,
    # for self._index_sampler
    sequence_length:int = 16, max_interval:int = 7, random_interval:bool = False, random_start_position:bool = False, uniform_frame_sample:bool = True,
    # for self._add_pads
    random_pad_sample:bool = False):

        self.frames_path = frames_path
        self.train = train
        self.sequence_length = sequence_length

        # values for self._add_pads
        self.random_pad_sample = random_pad_sample
        # read a csv file
        with open(csv_path, "r") as f:
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
    
    def _add_pads(self, length_of_frames:int) -> list:
        # length to list
        sequence = np.arange(length_of_frames)

        if self.random_pad_sample:
            # random sampled of pad
            add_sequence = np.random.choice(sequence, self.sequence_length - length_of_frames)
        else:
            # repeated first pad
            add_sequence = np.repeat(sequence[0], self.sequence_length - length_of_frames)

        # sorting the list
        sequence = sorted(np.append(sequence, add_sequence, axis=0))

        return list(sequence)
        

    def __getitem__(self, index):
        subfilepath = self.subfilespath[index]

        path = self.frames_path + subfilepath + '.json'
        frames_pose = self._get_poselist(path)
        label = self.labels[subfilepath]

        leng_poses = len(frames_pose)

        indices = self._add_pads(leng_poses)

        frames_pose = frames_pose[indices]
        # load frames
        # dans le frame load tu affecte au data la valeur de la pose de te meme que le lavbel
        data = torch.stack([torch.tensor(frame_pose) for frame_pose in frames_pose], dim=0)

        return data, label, self.train