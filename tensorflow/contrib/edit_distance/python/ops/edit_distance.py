"""RNN helpers for TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import threading

from tensorflow.contrib.edit_distance.ops.gen_edit_distance_ops import *

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
#
#
#from tensorflow.python.framework import dtypes
#from tensorflow.python.framework import ops
#from tensorflow.python.framework import tensor_shape
#from tensorflow.python.framework import tensor_util
#from tensorflow.python.ops import logging_ops


EDIT_DISTANCE_OPS_FILE = "_edit_distance_ops.so"

_edit_distance_ops = None
_ops_lock = threading.Lock()


@ops.RegisterShape("EditDistanceList")
def _EditDistanceListShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  return [tensor_shape.TensorShape([batch_size]),
          tensor_shape.TensorShape([batch_size])]


def Load(library_base_dir=''):
  """Load the inference ops library and return the loaded module."""
  with _ops_lock:
    global _edit_distance_ops
    if not _edit_distance_ops:
      data_files_path = os.path.join(library_base_dir,
                                     tf.resource_loader.get_data_files_path())
      _edit_distance_ops = tf.load_op_library(os.path.join(
          data_files_path, EDIT_DISTANCE_OPS_FILE))

      assert _edit_distance_ops, 'Could not load %s' % EDIT_DISTANCE_OPS_FILE
  return _edit_distance_ops
