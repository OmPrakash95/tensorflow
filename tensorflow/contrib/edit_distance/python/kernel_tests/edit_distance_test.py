from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


tf.contrib.edit_distance.Load()


class EditDistanceTest(tf.test.TestCase):

  def testEditDistance(self):
    with self.test_session() as sess:
      ref = [1, 2, 3, 4, 5, 6, 0, 9, 9, 9, 9, 9, 9, 9]
      ref = [[x] for x in ref]
      hyp = [   2, 3, 4, 6, 6, 7, 0, 8, 8, 8, 8, 8, 8, 8]
      hyp = [[x] for x in hyp]

      edit_dist, ref_len = tf.contrib.edit_distance.edit_distance_list(ref, hyp, 0)

      self.assertEqual(edit_dist.eval(), 3)
      self.assertEqual(ref_len.eval(), 6)


if __name__ == "__main__":
  tf.test.main()
