
# Load necessary modules
import cv2
from queue import Queue
import numpy as np

#### Class to be used in future work
# Defined functions and classes
class anomaly_object:
  def __init__(self, obj_id, start_tile, start_frame):
    self.obj_id = obj_id
    self.tiles = [start_tile]
    self.frames = [start_frame]

  def new_observation(self, cur_tile, cur_frame):
    self.tiles.append(cur_tile)
    self.frames.append(cur_frame)

  def calculate_speed(self, tile_dist, vid_fps):
    speeds = []
    for i in range(1, len(self.tiles)):
      tile_diff = self.tiles[i] - self.tiles[i-1]
      frame_diff = self.frames[i] - self.frames[i-1]
      dist = tile_diff * dist
      time = frame_diff / vid_fps
      speeds.append(dist / time)

    return np.mean(speeds)

def anomaly_outline(anomalies, tile_arr, raw_frames):
  ''' Displays outline of anomalies in boxes across images
  Input:
    - anomalies (numpy array): numpy array of anomalies per frame
    - tile_arr (list): list with tile_id as the index and value is a tuple
        containing the pixels coordinate corners for that tile
    - raw_frames (numpy array): numpy array of raw images
  Output:
    - raw_frames (numpy array): numpy array of raw images edited to draw
        outline of anomalies
    - anomaly_frames (numpy array): numpy array of frame numbers where
        anomalies occur
  '''
 
  anomaly_frames = []
  for tile_r in range(tile_arr.shape[0]):
    anomaly_row = []
    for tile_c in range(tile_arr.shape[1]):
      i_frames = np.where(anomalies[tile_r, tile_c, :] >= 0.75)[0]
      anomaly_row.append(i_frames)
      (tl, bl, tr, br) = tile_arr[tile_r, tile_c]
      for i_frame in range(len(raw_frames)):
        if i_frame in i_frames:
          frame = raw_frames[i_frame,::]
          frame = cv2.rectangle(frame, (tr, tl), (br, bl), (0, 255, 0), 1)
          raw_frames[i_frame,::] = frame
    anomaly_frames.append(anomaly_row)
  anomaly_frames = np.array(anomaly_frames)

  return raw_frames, anomaly_frames

def anomaly_speed(raw_frames, anomaly_frames, tile_dist, vid_fps):
  ''' Computes the velocity of an object in a video comprised as X
  Input:
    - raw_frames (numpy array): numpy array of raw images containing
        outline of anomalies
    - anomaly_frames (numpy array): numpy array of frame numbers where
        anomalies occur
    - tile_dist (float): distance between tiles in real life
    - vid_fps (float): float representing the fps of the original video
  Output:
    - raw_frames (numpy array): numpy array of raw images edited for 
        both outline of anomalies and velocity of objects
  '''

  font = cv2.FONT_HERSHEY_SIMPLEX
  queues = [Queue() for _ in range(anomaly_frames.shape[0])]
  speed = 0
  for i_frame, frame in enumerate(raw_frames):
    for tile_r in range(anomaly_frames.shape[0]):
      if i_frame in anomaly_frames[tile_r, 0]:
        queues[tile_r].put(i_frame)
      elif not queues[tile_r].empty() and i_frame in anomaly_frames[tile_r, -1]:
        dist = (anomaly_frames.shape[1]) * tile_dist
        time = (i_frame - queues[tile_r].get()) / vid_fps
        speed = dist / time
    text = 'Velocity: {0} mph'.format(round(speed, 3))
    frame = cv2.putText(frame, text, (30, 100), font, 0.65, (255, 255, 255))
    raw_frames[i_frame,::] = frame

  return raw_frames

def compile_video(frames, fps, output='output.mp4'):
  ''' Compiles a video from a list of files containing images
  Input:
    - frame_paths (list): list of strings for file paths to frames of a video
    - fps (float): float containing the frames per second of the video
    - output (str): str specifying the filename of the output video (Default to
        a video in the current directory called 'output.mp4'
  '''

  # Define the codec and create VideoWriter object
  height, width = frames[0].shape[:2]
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
  out = cv2.VideoWriter(output, fourcc, fps, (width, height))
  
  # Combine frames to VideoWriter object
  for frame in frames:
    out.write(frame)

  # Clear video buffer
  out.release()

