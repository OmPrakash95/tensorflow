"""RNN helpers for TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.rnn.core.ops import rnn_cell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import variable_scope as vs


@ops.RegisterShape("LSTMBlock")
def _LSTMBlockShape(op):
  batch_size = op.inputs[0].get_shape().with_rank(1)[0].value
  cell_size = op.get_attr("cell_size")
  sequence_len_max = op.get_attr("sequence_len_max")

  return [tensor_shape.TensorShape([batch_size, cell_size])] * sequence_len_max + [
          tensor_shape.TensorShape([batch_size, cell_size * 7])] * sequence_len_max


@ops.RegisterShape("LSTMBlockGrad")
def _LSTMBlockGradShape(op):
  batch_size = op.inputs[0].get_shape().with_rank(1)[0].value
  input_size = op.inputs[2].get_shape().with_rank(2)[1].value
  cell_size = op.get_attr("cell_size")
  sequence_len_max = op.get_attr("sequence_len_max")

  return [tensor_shape.TensorShape([batch_size, input_size])] * sequence_len_max + [
          tensor_shape.TensorShape([input_size + cell_size, cell_size * 4])] + [
          tensor_shape.TensorShape([cell_size * 4])]


@ops.RegisterGradient("LSTMBlock")
def _LSTMBlockGrad(op, *grad):
  cell_size = op.get_attr("cell_size")
  sequence_len_max = op.get_attr("sequence_len_max")

  assert len(op.inputs) == sequence_len_max + 4
  assert len(op.outputs) == sequence_len_max * 2
  assert len(grad) == sequence_len_max * 2

  sequence_len = op.inputs[0]
  initial_state = op.inputs[1]
  x = op.inputs[2:sequence_len_max + 2]
  w = op.inputs[sequence_len_max + 2]
  b = op.inputs[sequence_len_max + 3]
  states = op.outputs[sequence_len_max:sequence_len_max * 2]
  h_grad = grad[0:sequence_len_max]

  lstm_block_grads = gen_nn_ops.lstm_block_grad(
      sequence_len, initial_state, x, w, b, states, h_grad, cell_size=cell_size)

  return [None] + [None] + lstm_block_grads[0] + [lstm_block_grads[1]] + [lstm_block_grads[2]]


def lstm_block(inputs, cell_size, sequence_length=None, initial_state=None,
               forget_bias=1.0, scope=None):
  r"""Computes the LSTM forward propagation for N time steps.

  This implementation uses 1 weight matrix and 1 bias vector, there is no
  diagonal peephole connection. The computation of this op is dynamic as a
  function of sequence_len. We compute N = max(sequence_len) timesteps.

  Args:
    sequence_len: A `Tensor` of type `int64`.
      A vector of batch_size containing the sequence length.
    initial_state: A `Tensor` of type `float32`. Initial state of the LSTM.
    x: A list of at least 1 `Tensor` objects of type `float32`.
      The list of inputs to the LSTM.
    w: A `Tensor` of type `float32`. The weight matrix.
    b: A `Tensor` of type `float32`. The bias vector.
    cell_size: An `int`. The LSTM cell size.
    forget_bias: An optional `float`. Defaults to `1`. The forget gate bias.
    name: A name for the operation (optional).
    scope: VariableScope for the created subgraph.

  Returns:
    A tuple of `Tensor` objects (h, states).
    h: A list with the same number of `Tensor` objects as `x` of `Tensor` objects of type `float32`. The list of outputs h of the LSTM.
    states: A list with the same number of `Tensor` objects as `x` of `Tensor` objects of type `float32`. The list of states (it is the concatenated vector of c an h).
  """
  with vs.variable_scope(scope or "LSTM"):
    batch_size = inputs[0].get_shape()[0].value
    input_size = inputs[0].get_shape()[1].value
    w = vs.get_variable("W", [input_size + cell_size, cell_size * 4])
    b = vs.get_variable("b", [w.get_shape()[1]],
                        initializer=init_ops.constant_initializer(0.0))

    if sequence_length is None:
      sequence_length = array_ops.constant(
          len(inputs), dtype=dtypes.int64, shape=[batch_size])

    if initial_state is None:
      initial_state = array_ops.constant(
          0.0, dtype=dtypes.float32, shape=[batch_size, cell_size * 7])

    return gen_nn_ops.lstm_block(
        sequence_length, initial_state, inputs, w, b, cell_size=cell_size,
        forget_bias=forget_bias)
