# ToyotaSmartHome
sample codes for toyota smart home dataset

## Requirement
*   pytorch
*   torchvision
*   scikit-learn
*   pillow-simd[optional]
*   ffmpeg-python

## preparation
1. request a dataset from [Toyota Smart Home Official Site](https://project.inria.fr/toyotasmarthome/)
2. run a "./Data/data_preprocessing.sh" to prepare the dataset
```
directory structure of "./Data/" before the run a "./Data/data_preprocessing.sh"

Data
   L ./FrameExtractor
   L ./toyota_smarthome_mp4.tar.gz
   L ./toyota_smarthome_depth.tar.gz
   L ./toyota_smarthome_skeleton.tar.gz
```

## training
```
python train.py
```

## testing