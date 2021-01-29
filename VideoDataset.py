import os
import csv
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from glob import glob
import torchvision.transforms as transforms

"""
[args]
frame_size:int = size of input frames (default: 112)
channels_first:bool = True(for 3D CNN), False(for 2D CNN)
sequence_length:int = number of frames to be used for training or validation per video
max_interval:int = this value can impose limits on frame sampling strategy, no limitation(-1)
random_interval:bool = True(floating interval), False(fixed interval)
random_start_position:bool = True(floating start position), False(start position is 0)
uniform_frame_sample:bool = True(uniform sampling based on "random_interval", "random_start_position" options), False(just random sampling)
random_pad_sample:bool = True(random samples of frames to be used for pad), False(duplicate first frame and repeat)
"""
class VideoDataset(Dataset):
    def __init__(self, frames_path:str, csv_path:str, frame_size:int = 112, channels_first:bool = False,
    # for self._index_sampler
    sequence_length:int = 16, max_interval:int = -1, random_interval:bool = False, random_start_position:bool = False, uniform_frame_sample:bool = True,
    # for self._add_pads
    random_pad_sample:bool = False):

        self.frames_path = frames_path
        self.channels_first = channels_first
        self.sequence_length = sequence_length

        # values for self._index_sampler
        self.max_interval = max_interval
        self.random_interval = random_interval
        self.random_start_position = random_start_position
        self.uniform_frame_sample = uniform_frame_sample

        # values for self._add_pads
        self.random_pad_sample = random_pad_sample

        # read a csv file
        self.labels = [] # [[sub directory file path, index]]
        self.categories = {} # {index: category}
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            for rows in reader:
                sub_file_path = rows[0]; index = int(rows[1]); category = rows[2]
                self.labels.append([sub_file_path, index])
                if index not in self.categories:
                    self.categories[index] = category
        self.num_classes = len(self.categories)
        print(self.categories)
        # transformer
        self.transform = transforms.Compose([
            transforms.Resize(frame_size),
            transforms.CenterCrop(frame_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.labels)
    
    def get_category(self, index:int) -> str:
        return self.categories[index]
    
    def _add_pads(self, length_of_frames:int) -> list:
        # length to list
        sequence = np.arange(length_of_frames)

        if self.random_pad_sample:
            # random samples of frames to be used for pad
            add_sequence = np.random.choice(sequence, self.sequence_length - length_of_frames)
        else:
            # duplicate first frame and repeat
            add_sequence = np.repeat(sequence[0], self.sequence_length - length_of_frames)

        # sorting of the array
        sequence = sorted(np.append(sequence, add_sequence, axis=0))

        return list(sequence)

    def _index_sampler(self, length_of_frames:int) -> list:
        # sampling strategy(uniformly / randomly)
        if self.uniform_frame_sample:
            # set the default interval
            interval = (length_of_frames // self.sequence_length) -1
            if self.max_interval != -1 and interval > self.max_interval:
                interval = max_interval
            
            # "random interval" is select an interval with randomly
            if self.random_interval:
                interval = np.random.permutation(np.arange(start=1, stop=interval + 1))[0]
            
            # "require frames" is number of frames to sampling with specified interval
            require_frames = ((interval + 1) * self.sequence_length - interval)

            # "range of start position" is will be used for select to start position
            range_of_start_position = length_of_frames - require_frames

            # "random start position" is select a start position with randomly
            if self.random_start_position and range_of_start_position > 0:
                start_position = np.random.randint(0, range_of_start_position + 1)
            else:
                start_position = 0
            
            sampled_index = list(range(start_position, require_frames + start_position, interval + 1))
        else:
            sampled_index = sorted(np.permutation(np.arange(length_of_frames)))[:self.sequence_length]

        return list(sampled_index)

    def __getitem__(self, index):
        sub_file_path, label = self.labels[index]

        # get a frames path
        images_path = np.array(sorted(glob(os.path.join(self.frames_path, sub_file_path, "*"))))

        # get a index of samples
        length_of_frames = len(images_path)
        assert length_of_frames != 0, f"'{subfilepath}' is not exists or empty."

        if length_of_frames >= self.sequence_length:
            indices = self._index_sampler(length_of_frames)
        else:
            indices = self._add_pads(length_of_frames)
            
        images_path = images_path[indices]

        # load a frames
        data = torch.stack([self.transform(Image.open(image_path)) for image_path in images_path], dim=1 if self.channels_first else 0)

        return data, label