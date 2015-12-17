import sys
import tensorflow.python.platform
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import common_shapes
# pylint: disable=wildcard-import
# 'Constant' gets imported in the module 'array_ops'.
from tensorflow.python.ops.constant_op import constant
from tensorflow.python.ops.gen_array_ops import *

@ops.RegisterGradient("Gru")
def _GruGrad(op, grad):
  return [None] + GruGrad(cell_size=op.get_attr("cell_size"),
      sequence_len_max=op.get_attr("sequence_len_max"),
      sequence_len=op.inputs[0],
      wxr=op.inputs[1],
      whr=op.inputs[2],
      wxz=op.inputs[3],
      whz=op.inputs[4],
      wxh=op.inputs[5],
      whh=op.inputs[6],
      xs=op.inputs[7],
      rs=op.inputs[8],
      zs=op.inputs[9],
      rhs=op.inputs[10],
      gs=op.inputs[11],
      hs=op.inputs[12],
      drs=grad[0],
      dzs=grad[1],
      drhs=grad[2],
      dgs=grad[3],
      dhs=grad[4])
