#!/usr/bin/python3

##########################################
#### Load modules and necessary files ####
##########################################

# Standard libraries
import argparse
import time
import os
import sys

######################################
#### Command line argument parser ####
######################################

parser = argparse.ArgumentParser(description='Anomaly Detection')
parser.add_argument('-r', '--record', action='store_true', default=False,
    help='specify to record a video')
parser.add_argument('-s', '--source', action='store', default='./videos/traffic-short.mp4',
    help='specify the file path for a video to be split')
parser.add_argument('-t', '--tile_size', type=int, default=40,
    help='specify the tile size')
parser.add_argument('-o', '--output_path', default='./output/video.mp4',
    help='specify the file path to the output video')
parser.add_argument('-p', '--python', action='store_true', default=False,
    help='specify to use Python implementation')
parser.add_argument('-c', '--cython', action='store_true', default=False,
    help='Specify to use Cython implementation')
args = parser.parse_args()

# Load custom libraries
if args.python and not args.cython:
  from python_files.code.source import *
  from python_files.code.anomaly import *
  from python_files.code.output import *
elif args.cython:
  from cython_files.code.source import *
  from cython_files.code.anomaly import *
  from cython_files.code.output import *

#########################################
#### Initialize the global variables ####
#########################################

temp_time = init_time = time.time()
FRAME_RES = (1920, 1080)
TILE_SIZE = args.tile_size
INPUT_PATH = args.source
OUTPUT_PATH = args.output_path
video = Video()

############################
#### Begin main program ####
############################

## Collect raw data for program ----- using source.py
if args.record:
  print('RECORDING VIDEO')
  # Record video
  video_path = video.record(video_res=FRAME_RES, video_len=5, preview=True)
  print('--> elapsed time: {0}'.format(time.time() - temp_time)) 
  temp_time = time.time()

print('CREATING FRAMES')
if args.record:
  video.get_frames(video_path)
  FPS = video.fps
  raw_frames = video.raw_frames
else:
  video.get_frames(video_path=INPUT_PATH)
  FPS = video.fps
  raw_frames = video.raw_frames
  FRAME_RES = raw_frames[0].shape[:2]
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

# Define boundaries for different tiles
print('BUILDING TILES')
tile_arr = video.tessellate(tile_size=TILE_SIZE, frame_res=FRAME_RES)
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

print('COLLECTING FEATURES')
# Collect all features for all tiles across all frames
X = video.get_features(tile_arr)
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

## Build anomaly forests ----- using anomaly.py
print('TRAINING FORESTS')
# Build iForests for each tile
iForests = []
for tile_r in range(X.shape[0]):
  iForests_row = []
  for tile_c in range(X.shape[1]):
    iForests_row.append(IsoForest(X[tile_r, tile_c, ::]))
  iForests.append(iForests_row)
iForests = np.array(iForests)
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

print('EVALUATING FORESTS')
# Evaluating tiles of a dataset to determine anomalies 
anomalies = np.empty(X.shape[0:3])
for tile_r in range(X.shape[0]):
  for tile_c in range(X.shape[1]):
    anomalies[tile_r, tile_c, :] = iForests[tile_r, tile_c].find_anomalies(X[tile_r, tile_c, ::], 20)
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

print('ANNOTATING ANOMALIES')
raw_frames, anomaly_frames = anomaly_outline(anomalies, tile_arr, raw_frames)
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

print('ANNOTATING VELOCITY')
raw_frames = anomaly_speed(raw_frames, anomaly_frames, 2, vid_fps=FPS)
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

print('COMPILING VIDEO')
compile_video(raw_frames, fps=FPS, output=OUTPUT_PATH)
print('--> elapsed time: {0}'.format(time.time() - temp_time))
temp_time = time.time()

print('TOTAL ELAPSED TIME: {0}'.format(time.time() - init_time))
