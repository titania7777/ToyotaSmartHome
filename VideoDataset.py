import os
import csv
import numpy as np
import torch
from PIL import Image # pillow-simd
from torch.utils.data import Dataset
from glob import glob
import torchvision.transforms as transforms

# VideoDataset load for video frames
# frames path: root path of frames
# you must be follow our csv format when using this loader
# csv file header => sub directory file path, index, category
class VideoDataset(Dataset):
    def __init__(self, frames_path:str, csv_path:str, frame_size:int = 112,
    # for self._index_sampler
    sequence_length:int = 16, max_interval:int = 7, random_interval:bool = False, random_start_position:bool = False, uniform_frame_sample:bool = True,
    # for self._add_pads
    random_pad_sample:bool = False):

        self.frames_path = frames_path
        self.sequence_length = sequence_length

        # values for self._index_sampler
        self.max_interval = max_interval
        self.random_interval = random_interval
        self.random_start_position = random_start_position
        self.uniform_frame_sample = uniform_frame_sample

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

        # simple transformer for 3D ResNets, customizing is needed if you want to using other models
        # Normalize => https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/mean.py
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.CenterCrop(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4345, 0.4051, 0.3775], std=[0.2768, 0.2713, 0.2737]),
        ])
        
    def __len__(self) -> int:
        return len(self.subfilespath)
    
    def get_category(self, index:int) -> str:
        return self.categories[index]
    
    def get_num_classes(self) -> int:
        print(self.categories)
        return len(self.categories)

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

    def _index_sampler(self, length_of_frames:int) -> list:
        # set the default interval
        interval = length_of_frames // self.sequence_length
        interval = 1 if interval == 0 else interval
        if self.max_interval != -1 and interval > self.max_interval:
            interval = self.max_interval
        
        # set the interval with randomlly(default is default interval)
        if self.random_interval:
            interval = np.random.permutation(np.arange(start=1, stop=interval + 1))[0]

        # set the start position(default is 0), this may be helpful when video has a large frames
        # required number of sample => (i + 1) x s - i, when i is interval and s is sequence length
        # ex) frames: 1, 2, 3, 4, 5, 6, 7 ,8, 9, 10
        # when sequence length is 3, interval is 2 and start position is 3 then you will get 3, 6, 9 frame and for this you need (2 + 1) x 3 - 2 = 7 samples
        range_of_start_position = length_of_frames - ((interval + 1)*self.sequence_length - interval)
        
        # interval resolving
        if range_of_start_position < 0:
            for interval in reversed(range(interval)):
                range_of_start_position = length_of_frames - ((interval + 1)*self.sequence_length - interval)
                if range_of_start_position > 0:
                    break

        # set the start position
        if self.random_start_position and range_of_start_position != 0:
            start_position = np.random.randint(0, range_of_start_position)
        else:
            start_position = 0

        # set the sampling strategy(default is uniform)
        # i'm not recommend using the random sampling
        if self.uniform_frame_sample:
            sampled_index = list(range(start_position, length_of_frames, interval + 1))[:self.sequence_length]
        else:
            sampled_index = sorted(np.permutation(np.arange(length_of_frames)))[:self.sequence_length]

        return list(sampled_index)

    def __getitem__(self, index):
        subfilepath = self.subfilespath[index]

        # get frames path
        images_path = np.array(sorted(glob(os.path.join(self.frames_path, subfilepath, "*"))))

        # get index of samples
        length_of_frames = len(images_path)
        assert length_of_frames != 0, f"'{subfilepath}' is not exists or empty."

        if length_of_frames >= self.sequence_length:
            indices = self._index_sampler(length_of_frames)
        else:
            indices = self._add_pads(length_of_frames)
            
        images_path = images_path[indices]

        # load frames
        data = torch.stack([self.transform(Image.open(image_path)) for image_path in images_path], dim=0)
        label = self.labels[subfilepath]

        return data, label