
# Load necessary modules
import sys
import unittest
sys.path.insert(0, '../../')
from cython_files.code.source import *

# If testing with a Raspberry Pi then uncomment first function

class TestVideoMethods(unittest.TestCase):
#  def test_record(self):
#    video = Video()
#    video_path = video.record()
#    self.assertEqual(os.path.exists(video_path='./video.mjpeg'), video_path)

  def test_get_frames(self):
    ''' Test get_frames method of the Video class '''
    video = Video()
    RES = (112, 384)
    VIDEO_PATH = '../../videos/traffic-short.mp4'
    video.get_frames(video_path=VIDEO_PATH)

    # Check that the corresponding shape is correct
    self.assertEqual(video.raw_frames.shape, (video.tot_frames,) + RES + (3,))
    self.assertTrue((video.raw_frames >= 0).all())
    self.assertTrue((video.raw_frames <= 255).all())

    # Check that the fps is non-zero
    self.assertTrue(video.fps > 0)

  def test_tessellate(self):
    ''' Tests tessellate method of the Video class '''
    video = Video()
    RES = (112, 384)
    tile_arr_1 = video.tessellate(tile_size=15, frame_res=RES)
    tile_arr_2 = video.tessellate(tile_size=20, frame_res=RES)

    # Check that the arrays are numpy arrays
    self.assertTrue(isinstance(tile_arr_1, np.ndarray))
    self.assertTrue(isinstance(tile_arr_2, np.ndarray))

    # Check the dimensionality of the tile locations array is correct
    self.assertEqual(tile_arr_1.shape, (7, 25, 4))
    self.assertEqual(tile_arr_2.shape, (5, 19, 4))

    # Check that tile locations are within the boundaries
    self.assertTrue((tile_arr_1 >= 0).all())
    self.assertTrue((tile_arr_2 >= 0).all())

    for i_row in range(tile_arr_1.shape[0]):
      for i_col in range(tile_arr_1.shape[1]):
        tl, bl, tr, br = tile_arr_1[i_row, i_col]
        self.assertTrue(tl < 112 and bl < 112 and tr < 384 and br < 384)

    for i_row in range(tile_arr_2.shape[0]):
      for i_col in range(tile_arr_2.shape[1]):
        tl, bl, tr, br = tile_arr_1[i_row, i_col]
        self.assertTrue(tl < 112 and bl < 112 and tr < 384 and br < 384)

if __name__ == '__main__':
  unittest.main()
