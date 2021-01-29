#!/bin/bash

# request a dataset => https://project.inria.fr/toyotasmarthome/

# RGB Videos(./mp4)
tar -zxvf ./toyota_smarthome_mp4.tar.gz
rm -rf ./toyota_smarthome_mp4.tar.gz

# Depth Videos(./depth)
tar -zxvf ./toyota_smarthome_depth.tar.gz
rm -rf ./toyota_smarthome_depth.tar.gz

# Skeleton jsons(./json)
tar -zxvf ./toyota_smarthome_skeleton.tar.gz
rm -rf ./toyota_smarthome_skeleton.tar.gz

# RGB frame extraction
python ./FrameExtractor/frame_extractor.py --videos-path ./mp4/ --frames-path ./mp4_frames/ --frame-size 240 --quality 0.8

# Depth frame extraction
python ./FrameExtractor/frame_extractor.py --videos-path ./depth/ --frames-path ./depth_frames/ --frame-size 240 --quality 0.8

# build a labels
python labeler.py

# final results
# Data
#    L ./FrameExtractor
#    L ./Labels
#    L ./mp4
#    L ./mp4_frames
#    L ./depth
#    L ./depth_frames
#    L ./json