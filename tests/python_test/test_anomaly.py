
# Load necessary modules
import sys
import math
import unittest

sys.path.insert(0, '../../')
from python_files.code.source import *
from python_files.code.anomaly import *

class TestAnomalyMethods(unittest.TestCase):
  ''' Test IsoTree and IsoForest classes and other functions '''
  def test_IsoTree_training(self):
    ''' Test IsoTree training '''

    # Setup for function
    VIDEO_PATH = '../../videos/traffic-short.mp4'
    video = Video()
    video.get_frames(video_path=VIDEO_PATH)
    RAW_FRAMES = video.raw_frames
    FRAME_RES = RAW_FRAMES[0].shape[:2]
    tile_arr = video.tessellate(tile_size=15, frame_res=FRAME_RES)
    X = video.get_features(tile_arr)
    iTree = IsoTree(X[0, 0])

    # Check that the iTree objects are created correctly
    self.assertTrue(isinstance(iTree, IsoTree))
    self.assertTrue(isinstance(iTree.l_node, IsoTree) or None)
    self.assertTrue(isinstance(iTree.r_node, IsoTree) or None)
    self.assertTrue(isinstance(iTree.num_obs, int)) 
    self.assertEqual(iTree.num_obs, len(RAW_FRAMES))
    self.assertEqual(iTree.num_obs, X.shape[2])

  def test_IsoForest_training(self):
    ''' Test IsoForest training '''

    # setup for function
    video_path = '../../videos/traffic-short.mp4'
    video = Video()
    video.get_frames(video_path=video_path)
    raw_frames = video.raw_frames
    frame_res = raw_frames[0].shape[:2]
    tile_arr = video.tessellate(tile_size=15, frame_res=frame_res)
    X = video.get_features(tile_arr)
    iForest_1 = IsoForest(X[0, 0])
    iForest_2 = IsoForest(X[0, 0], num_trees=20)

    # Check that iForest objects are created correctly
    self.assertEqual(len(iForest_1.trees), 100)
    self.assertEqual(len(iForest_2.trees), 20)

    for iTree in iForest_1.trees:
      self.assertTrue(isinstance(iTree, IsoTree))
    for iTree in iForest_2.trees:
      self.assertTrue(isinstance(iTree, IsoTree))

  def test_IsoForest_anomaly_score(self):
    ''' Test IsoForest anomaly_score method '''

    # setup for function
    video_path = '../../videos/traffic-short.mp4'
    video = Video()
    video.get_frames(video_path=video_path)
    raw_frames = video.raw_frames
    FRAME_RES = raw_frames[0].shape[:2]
    tile_arr = video.tessellate(tile_size=15, frame_res=FRAME_RES)
    X = video.get_features(tile_arr)
    iForest = IsoForest(X[0, 0])
    x = np.array([0, 0, 0, 1, 1, 1])
    s_score = iForest.anomaly_score(x, hlim=20)

    # Checking values of anomaly_score
    self.assertTrue(isinstance(s_score, float))
    self.assertTrue(0 <= s_score and s_score <= 1)

  def test_IsoForest_find_anomalies(self):
    ''' Test IsoForest find_anomalies method '''

    # setup for function
    video_path = '../../videos/traffic-short.mp4'
    video = Video()
    video.get_frames(video_path=video_path)
    raw_frames = video.raw_frames
    FRAME_RES = raw_frames[0].shape[:2]
    tile_arr = video.tessellate(tile_size=15, frame_res=FRAME_RES)
    X = video.get_features(tile_arr)
    iForest = IsoForest(X[0, 0])
    anomalies_1 = iForest.find_anomalies(X[0,0], hlim=20)
    anomalies_2 = iForest.find_anomalies(X[0,0], hlim=20)

    self.assertEqual(anomalies_1, anomalies_2)
    self.assertTrue(isinstance(anomalies_1, list))

  def test_adjust_score(self):
    ''' Test adjust_score function '''
    self.assertEqual(adjust_score(1000), 2 * (math.log(999) + 0.5772156649) - 2 * 999 / 1000)
    self.assertEqual(adjust_score(2), 1)
    self.assertEqual(adjust_score(1), 0)

if __name__ == '__main__':
  unittest.main()

