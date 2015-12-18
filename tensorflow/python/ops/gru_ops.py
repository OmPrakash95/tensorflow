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
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import gen_gru_ops


def gru(cell_size, sequence_len, xs, name=None, scope=None):
  r"""gru

  args:
    sequence_len: a `tensor` of type `int64`.
    cell_size: an `int`.
    xs: a list of at least 1 `tensor` objects of type `float32`.
    name: a name for the operation (optional).

  returns:
    a tuple of `tensor` objects (rs, zs, rhs, gs, hs).
    rs: a list with the same number of `tensor` objects as `xs` of `tensor` objects of type `float32`.
    zs: a list with the same number of `tensor` objects as `xs` of `tensor` objects of type `float32`.
    rhs: a list with the same number of `tensor` objects as `xs` of `tensor` objects of type `float32`.
    gs: a list with the same number of `tensor` objects as `xs` of `tensor` objects of type `float32`.
    hs: a list with the same number of `tensor` objects as `xs` of `tensor` objects of type `float32`.
  """
  with vs.variable_scope(scope or "Gru"):
    input_size = xs[0].get_shape()[1].value

    wxr = vs.get_variable("wxr", [input_size, cell_size])
    whr = vs.get_variable("whr", [cell_size, cell_size])
    wxz = vs.get_variable("wxz", [input_size, cell_size])
    whz = vs.get_variable("whz", [cell_size, cell_size])
    wxh = vs.get_variable("wxh", [input_size, cell_size])
    whh = vs.get_variable("whh", [cell_size, cell_size])

    outputs = gen_gru_ops._gru(cell_size=cell_size, sequence_len=sequence_len,
        wxr=wxr, whr=whr, wxz=wxz, whz=whz, wxh=wxh, whh=whh, xs=xs, name=name)
    for output in outputs:
      gen_gru_ops.sink(output)
    return outputs


@ops.RegisterShape("Gru")
def _GruShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  cell_size = op.get_attr("cell_size")

  return [tensor_shape.TensorShape([batch_size, cell_size])] * ((len(op.inputs) - 7) * 5)


@ops.RegisterShape("Sink")
def _SinkShape(op):
  return []

# @ops.RegisterGradient("Gru")
# def _GruGrad(op, grad):
#   return [None] + GruGrad(cell_size=op.get_attr("cell_size"),
#       sequence_len_max=op.get_attr("sequence_len_max"),
#       sequence_len=op.inputs[0],
#       wxr=op.inputs[1],
#       whr=op.inputs[2],
#       wxz=op.inputs[3],
#       whz=op.inputs[4],
#       wxh=op.inputs[5],
#       whh=op.inputs[6],
#       xs=op.inputs[7],
#       rs=op.inputs[8],
#       zs=op.inputs[9],
#       rhs=op.inputs[10],
#       gs=op.inputs[11],
#       hs=op.inputs[12],
#       drs=grad[0],
#       dzs=grad[1],
#       drhs=grad[2],
#       dgs=grad[3],
#       dhs=grad[4])
