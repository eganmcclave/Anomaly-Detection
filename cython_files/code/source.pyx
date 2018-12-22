
# Load necessary modules
from imutils.video import VideoStream
#from picamera.array import PiRGBArray
#from picamera import PiCamera
import numpy as np
import imutils
import time
import cv2
import os

# Defined functions and classes
class Video:
  def __init__(self):
    pass

#  def record(self, video_path='./video', video_res=(1920, 1080), video_len=2, preview=False):
#    ''' Record video from a Raspberry Pi and save to a path
#    Input:
#      - video_path (str): string specifying the file path to save the video at
#          (Defaults to current directory)
#      - video_res (tuple): tuple containing two integers specifying the resolution
#          size of the video (Defaults to 1920x1080)
#      - video_len (int): int specifying the duration of the video (Defaults to 
#          record for 2 seconds)
#      - preview (bool): bool specifying whether a preview of the video should be
#          enable (Defaults to False - primarily used for debugging purposes)
#    Output:
#      - video_path (str): file path specifying the location of the recorded video 
#    '''
#
#    video_path = video_path + '.h264'
#    camera = PiCamera()
#    camera.resolution = video_res
#    if preview:
#      camera.start_preview()
#    camera.start_recording(video_path)
#    camera.wait_recording(video_len)
#    camera.stop_recording()
#    if preview:
#      camera.stop_preview()
#    return video_path

  def get_frames(self, video_path):
    vid = cv2.VideoCapture(video_path)
    success, frame = vid.read()
    raw_frames = []
    raw_frames.append(frame)

    while success:
      success, frame = vid.read()
      if frame is not None:
        raw_frames.append(frame)
    self.tot_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    self.raw_frames = np.array(raw_frames)
    self.fps = vid.get(cv2.CAP_PROP_FPS)
    
  def tessellate(self, tile_size, frame_res):
    ''' Tessellate the resolution of the frame into different regions
    Input:
      - tile_size (int): integer specifying the size of the tile to use 
      - frame_res (tuple): tuple of integers representing the height and width
          of a frame from the video 
    Output:
      - tile_arr (numpy array): numpy array with tile_id as the index and value 
          is a tuple containing the pixels coordinate corners for that tile
    '''

    rows, cols = [val // tile_size for val in frame_res]
    tile_arr = []
    for r in range(rows):
      tile_row = []
      for c in range(cols):
        tile_row.append((r*tile_size, (r+1)*tile_size,
                         c*tile_size, (c+1)*tile_size))
      tile_arr.append(tile_row)
    
    return np.array(tile_arr)

  def get_features(self, tile_arr):
    ''' Form a 3D numpy-array of means and standard deviations for all RGB
        channels for all tiles for all frames
    Input:
      - tile_arr (list): list with tile_id as the index and value is a tuple
          containing the pixels coordinate corners for that tile
    Output:
      - X (numpy-array): 3D numpy array of features for different files
          (Tile id x Frame id x Features)
    '''

    X = []
    for tile_r in range(tile_arr.shape[0]):
      row_features = []
      for tile_c in range(tile_arr.shape[1]):
        tl, bl, tr, br = tile_arr[tile_r, tile_c]
        col_features = []
        for frame in self.raw_frames:
          col_features.append(self.__extract_features__(frame[tl:bl, tr:br]))
        row_features.append(col_features)
      X.append(row_features)
    X = np.array(X)

    return X

  def __extract_features__(self, tile):
    ''' Extract the necessary features from a tile
    Input:
      - tile (numpy-array): 3D numpy-array containing multiple pixels of RGB
    Output:
      - features (list): list containing mean and sd for all RGB channels 
          across all pixels in the tile
    '''

    R = tile[::,2]
    G = tile[::,1]
    B = tile[::,0]
    features = [np.mean(R), np.mean(G), np.mean(B),
      np.std(R), np.std(G), np.std(B)]

    return features

