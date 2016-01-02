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
from tensorflow.python.ops import gen_attention_mask_ops


def attention_mask(attention_states_sequence_len, input, name=None):
  r"""AttentionMask

  Args:
    attention_states_sequence_len: A `Tensor` of type `int64`.
    input: A `Tensor` of type `float32`.
    fill_value: A `float`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return gen_attention_mask_ops._attention_mask(
      attention_states_sequence_len=attention_states_sequence_len, input=input,
      fill_value=-np.finfo(np.float32).max, name=name)

@ops.RegisterShape("AttentionMask")
def _AttentionMaskShape(op):
  return [op.inputs[1].get_shape()]


@ops.RegisterGradient("AttentionMask")
def _AttentionMaskGrad(op, *grad):
  attention_mask_grad = gen_attention_mask_ops._attention_mask(
      attention_states_sequence_len=op.inputs[0], input=grad[0], fill_value=0.0)
  return [None] + [attention_mask_grad]

def attention_mask_median(attention_states_sequence_len, input, prev,
                          window_l=None, window_r=None, name=None):
  r"""AttentionMaskMedian

  Args:
    attention_states_sequence_len: A `Tensor` of type `int64`.
    input: A `Tensor` of type `float32`.
    prev: A `Tensor` of type `float32`.
    fill_value: A `float`.
    window_l: An optional `int`. Defaults to `10`.
    window_r: An optional `int`. Defaults to `200`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `float32`.
  """
  return gen_attention_mask_ops._attention_mask_median(
      attention_states_sequence_len=attention_states_sequence_len, input=input,
      prev=prev, fill_value=-np.finfo(np.float32).max, window_l=window_l,
      window_r=window_r, name=name)

@ops.RegisterShape("AttentionMaskMedian")
def _AttentionMaskMedianShape(op):
  return [op.inputs[1].get_shape()]


@ops.RegisterGradient("AttentionMaskMedian")
def _AttentionMaskMedianGrad(op, *grad):
  attention_mask_grad = gen_attention_mask_ops._attention_mask_median(
      attention_states_sequence_len=op.inputs[0], input=grad[0],
      prev=op.inputs[2], fill_value=0.0, window_l=op.get_attr("window_l"),
      window_r=op.get_attr("window_r"))
  return [None] * 2 + [attention_mask_grad]
