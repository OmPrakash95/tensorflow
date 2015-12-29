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


def attention_mask(attention_states_sequence_len, input, name=None):
  r"""AttentionMask

  Args:
    attention_states_sequence_len: A `Tensor` of type `int64`.
    input: A `Tensor` of type `float32`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return gen_gru_ops._attention_mask(
      attention_states_sequence_len=attention_states_sequence_len, input=input,
      fill_value=-np.finfo(np.float32).max, name=name)

def token_sample(ground_truth, token_distribution, sample_prob, seed=0, name=None):
  return gen_gru_ops.token_sample(
      ground_truth, token_distribution, sample_prob, seed=seed, name=name)

@ops.RegisterShape("TokenSample")
def _TokenSampleShape(op):
  return [op.inputs[0].get_shape()]

ops.NoGradient("TokenSample")

@ops.RegisterShape("AttentionMask")
def _AttentionMaskShape(op):
  return [op.inputs[1].get_shape()]


@ops.RegisterGradient("AttentionMask")
def _AttentionMaskGrad(op, *grad):
  attention_mask_grad = gen_gru_ops._attention_mask(
      attention_states_sequence_len=op.inputs[0], input=grad[0], fill_value=0.0)
  return [None] + [attention_mask_grad]


def gru_cell(cell_size, sequence_len, h_prev, x, name=None, scope=None, time_idx=None):
  r"""GRU Cell

  Args:
    sequence_len: A `Tensor` of type `int64`.
    wxr: A `Tensor` of type `float32`.
    whr: A `Tensor` of type `float32`.
    wxz: A `Tensor` of type `float32`.
    whz: A `Tensor` of type `float32`.
    wxh: A `Tensor` of type `float32`.
    whh: A `Tensor` of type `float32`.
    h_prev: A `Tensor` of type `float32`.
    x: A `Tensor` of type `float32`.
    cell_size: An `int`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (r, z, rh, g, h).
    r: A `Tensor` of type `float32`.
    z: A `Tensor` of type `float32`.
    rh: A `Tensor` of type `float32`.
    g: A `Tensor` of type `float32`.
    h: A `Tensor` of type `float32`.
  """
  with vs.variable_scope(scope or "GruCell"):
    input_size = x.get_shape()[1].value

    wxr = vs.get_variable("wxr", [input_size, cell_size])
    whr = vs.get_variable("whr", [cell_size, cell_size])
    wxz = vs.get_variable("wxz", [input_size, cell_size])
    whz = vs.get_variable("whz", [cell_size, cell_size])
    wxh = vs.get_variable("wxh", [input_size, cell_size])
    whh = vs.get_variable("whh", [cell_size, cell_size])

    return gen_gru_ops._gru_cell(cell_size=cell_size, sequence_len=sequence_len,
        wxr=wxr, whr=whr, wxz=wxz, whz=whz, wxh=wxh, whh=whh, h_prev=h_prev,
        x=x, name=name, time_idx=time_idx)


@ops.RegisterShape("GruCell")
def _GruCellShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  cell_size = op.get_attr("cell_size")

  return [tensorflow.TensorShape([batch_size, cell_size])] * 5


@ops.RegisterGradient("GruCell")
def _GruCellGrad(op, *grad):
  gru_grads = gen_gru_ops._gru_cell_grad(cell_size=op.get_attr("cell_size"),
      sequence_len=op.inputs[0],
      wxr=op.inputs[1],
      whr=op.inputs[2],
      wxz=op.inputs[3],
      whz=op.inputs[4],
      wxh=op.inputs[5],
      whh=op.inputs[6],
      h_prev=op.inputs[7],
      x=op.inputs[8],
      r=op.outputs[0],
      z=op.outputs[1],
      rh=op.outputs[2],
      hh=op.outputs[3],
      h=op.outputs[4],
      dh=grad[4],
      time_idx=op.get_attr("time_idx"))

  gru_grads_ = [None]
  for gru_grad in gru_grads:
    if isinstance(gru_grad, list):
      gru_grads_ += gru_grad
    else:
      gru_grads_ += [gru_grad]
  return gru_grads_


@ops.RegisterShape("GruCellGrad")
def _GruCellGradShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  input_size = op.inputs[1].get_shape()[0].value
  cell_size = op.get_attr("cell_size")

  return [tensor_shape.TensorShape([input_size, cell_size]),
          tensor_shape.TensorShape([cell_size, cell_size])] * 3 + [
          tensor_shape.TensorShape([batch_size, cell_size])] + [
          tensor_shape.TensorShape([batch_size, input_size])]


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

    return gen_gru_ops._gru(cell_size=cell_size, sequence_len=sequence_len,
        wxr=wxr, whr=whr, wxz=wxz, whz=whz, wxh=wxh, whh=whh, xs=xs, name=name)


@ops.RegisterShape("Gru")
def _GruShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  cell_size = op.get_attr("cell_size")

  return [tensor_shape.TensorShape([batch_size, cell_size])] * ((len(op.inputs) - 7) * 5)


@ops.RegisterShape("Sink")
def _SinkShape(op):
  return []


@ops.RegisterGradient("Gru")
def _GruGrad(op, *grad):
  outputs = zip(*[iter(op.outputs)] * (len(op.outputs) / 5))
  grads = zip(*[iter(grad)] * (len(grad) / 5))
  gru_grads = gen_gru_ops._gru_grad(cell_size=op.get_attr("cell_size"),
      sequence_len=op.inputs[0],
      wxr=op.inputs[1],
      whr=op.inputs[2],
      wxz=op.inputs[3],
      whz=op.inputs[4],
      wxh=op.inputs[5],
      whh=op.inputs[6],
      xs=op.inputs[7:],
      rs=outputs[0],
      zs=outputs[1],
      rhs=outputs[2],
      gs=outputs[3],
      hs=outputs[4],
      dhs=grads[4])

  gru_grads_ = [None]
  for gru_grad in gru_grads:
    if isinstance(gru_grad, list):
      gru_grads_ += gru_grad
    else:
      gru_grads_ += [gru_grad]
  return gru_grads_


@ops.RegisterShape("GruGrad")
def _GruGradShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  input_size = op.inputs[1].get_shape()[0].value
  cell_size = op.get_attr("cell_size")

  return [tensor_shape.TensorShape([input_size, cell_size]),
          tensor_shape.TensorShape([cell_size, cell_size])] * 3 + [
          tensor_shape.TensorShape([batch_size, input_size])] * ((len(op.inputs) - 7) / 7)


def gru_fused(cell_size, sequence_len, xs, name=None, scope=None):
  r"""gru

  args:
    sequence_len: a `tensor` of type `int64`.
    cell_size: an `int`.
    xs: a list of at least 1 `tensor` objects of type `float32`.
    name: a name for the operation (optional).

  returns:
    a tuple of `tensor` objects (rzs, rhs, gs, hs).
    rzs: a list with the same number of `tensor` objects as `xs` of `tensor` objects of type `float32`.
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

    return gen_gru_ops._gru_fused(cell_size=cell_size, sequence_len=sequence_len,
        wxr=wxr, whr=whr, wxz=wxz, whz=whz, wxh=wxh, whh=whh, xs=xs, name=name)


@ops.RegisterShape("GruFused")
def _GruFusedShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  cell_size = op.get_attr("cell_size")
  sequence_len_max = len(op.inputs) - 7

  return [tensor_shape.TensorShape([batch_size, cell_size * 2])] * sequence_len_max + [tensor_shape.TensorShape([batch_size, cell_size])] * (sequence_len_max * 3)


@ops.RegisterGradient("GruFused")
def _GruFusedGrad(op, *grad):
  outputs = zip(*[iter(op.outputs)] * (len(op.outputs) / 4))
  grads = zip(*[iter(grad)] * (len(grad) / 4))
  gru_grads = gen_gru_ops._gru_fused_grad(cell_size=op.get_attr("cell_size"),
      sequence_len=op.inputs[0],
      wxr=op.inputs[1],
      whr=op.inputs[2],
      wxz=op.inputs[3],
      whz=op.inputs[4],
      wxh=op.inputs[5],
      whh=op.inputs[6],
      xs=op.inputs[7:],
      rzs=outputs[0],
      rhs=outputs[1],
      gs=outputs[2],
      hs=outputs[3],
      dhs=grads[3])

  gru_grads_ = [None]
  for gru_grad in gru_grads:
    if isinstance(gru_grad, list):
      gru_grads_ += gru_grad
    else:
      gru_grads_ += [gru_grad]
  return gru_grads_


@ops.RegisterShape("GruFusedGrad")
def _GruFusedGradShape(op):
  batch_size = op.inputs[0].get_shape()[0].value
  input_size = op.inputs[1].get_shape()[0].value
  cell_size = op.get_attr("cell_size")

  return [tensor_shape.TensorShape([input_size, cell_size]),
          tensor_shape.TensorShape([cell_size, cell_size])] * 3 + [
          tensor_shape.TensorShape([batch_size, input_size])] * ((len(op.inputs) - 7) / 6)
