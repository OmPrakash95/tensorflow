"""Tests for rnn module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import timeit

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf


tf.contrib.rnn.Load()


def _flatten(list_of_lists):
  return [x for y in list_of_lists for x in y]


class LSTMTest(tf.test.TestCase):

  def setUp(self):
    self._seed = 23489
    np.random.seed(self._seed)

  def _testLSTMBasicToBlockRNN(self, use_gpu, use_sequence_length):
    time_steps = 8
    num_units = 3
    num_proj = 4
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)

    if use_sequence_length:
      sequence_length = np.random.randint(0, time_steps, size=batch_size)
    else:
      sequence_length = None

    ########### Step 1: Run BasicLSTMCell
    initializer = tf.random_uniform_initializer(-0.01, 0.01, seed=self._seed)
    basic_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)

    (values_basic, state_value_basic, basic_grad_values,
     basic_individual_grad_values,
     basic_individual_var_grad_values) = run_static_rnn(
         self, time_steps, batch_size, input_size, basic_lstm_cell,
         sequence_length, input_values, use_gpu, check_states=False,
         initializer=initializer)

    ########### Step 1: Run LSTMCellBlock
    lstm_cell_block_cell = tf.contrib.rnn.LSTMCellBlock(num_units)

    (values_block, state_value_block, block_grad_values,
     block_individual_grad_values,
     block_individual_var_grad_values) = run_static_rnn(
         self, time_steps, batch_size, input_size, lstm_cell_block_cell,
         sequence_length, input_values, use_gpu, check_states=False,
         initializer=initializer)

    ######### Step 3: Comparisons
    self.assertEqual(len(values_basic), len(values_block))
    for (value_basic, value_block) in zip(values_basic, values_block):
      self.assertAllClose(value_basic, value_block)

    self.assertAllClose(basic_grad_values, block_grad_values)

    self.assertEqual(len(basic_individual_grad_values),
                     len(block_individual_grad_values))
    self.assertEqual(len(basic_individual_var_grad_values),
                     len(block_individual_var_grad_values))

    for i, (a, b) in enumerate(zip(basic_individual_grad_values,
                                   block_individual_grad_values)):
      tf.logging.info("Comparing individual gradients iteration %d" % i)
      self.assertAllClose(a, b)

    for i, (a, b) in enumerate(reversed(zip(basic_individual_var_grad_values,
                                   block_individual_var_grad_values))):
      tf.logging.info(
          "Comparing individual variable gradients iteraiton %d" % i)
      self.assertAllClose(a, b)

  def testLSTMBasicToBlockRNN(self):
    self._testLSTMBasicToBlockRNN(use_gpu=False, use_sequence_length=False)
    self._testLSTMBasicToBlockRNN(use_gpu=False, use_sequence_length=True)
    self._testLSTMBasicToBlockRNN(use_gpu=True, use_sequence_length=False)
    self._testLSTMBasicToBlockRNN(use_gpu=True, use_sequence_length=True)


def run_static_rnn(test, time_steps, batch_size, input_size, cell,
                   sequence_length, input_values, use_gpu, check_states=True,
                   initializer=None):
  with test.test_session(use_gpu=use_gpu, graph=tf.Graph()) as sess:
    concat_inputs = tf.placeholder(tf.float32,
                                   shape=(time_steps, batch_size, input_size))
    inputs = tf.unpack(concat_inputs)

    with tf.variable_scope("dynamic_scope", initializer=initializer):
      outputs_static, state_static = tf.nn.rnn(
          cell, inputs, sequence_length=sequence_length, dtype=tf.float32)

    feeds = {concat_inputs: input_values}

    # Initialize
    tf.initialize_all_variables().run(feed_dict=feeds)

    # Generate gradients of sum of outputs w.r.t. inputs
    if check_states:
      static_gradients = tf.gradients(
          outputs_static + [state_static], [concat_inputs])
    else:
      static_gradients = tf.gradients(
          outputs_static, [concat_inputs])

    # Generate gradients of individual outputs w.r.t. inputs
    if check_states:
      static_individual_gradients = _flatten([
          tf.gradients(y, [concat_inputs])
          for y in [outputs_static[0],
                    outputs_static[-1],
                    state_static]])
    else:
      static_individual_gradients = _flatten([
          tf.gradients(y, [concat_inputs])
          for y in [outputs_static[0],
                    outputs_static[-1]]])

    # Generate gradients of individual variables w.r.t. inputs
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    assert len(trainable_variables) > 1, (
        "Count of trainable variables: %d" % len(trainable_variables))
    # pylint: disable=bad-builtin
    if check_states:
      static_individual_variable_gradients = _flatten([
          tf.gradients(y, trainable_variables)
          for y in [outputs_static[0],
                    outputs_static[-1],
                    state_static]])
    else:
      static_individual_variable_gradients = _flatten([
          tf.gradients(y, trainable_variables)
          for y in [outputs_static[0],
                    outputs_static[-1]]])

    # Test forward pass
    values_static = sess.run(outputs_static, feed_dict=feeds)
    if check_states:
      (state_value_static,) = sess.run((state_static,), feed_dict=feeds)
    else:
      state_value_static = None

    # Test gradients to inputs and variables w.r.t. outputs & final state
    static_grad_values = sess.run(static_gradients, feed_dict=feeds)

    static_individual_grad_values = sess.run(
        static_individual_gradients, feed_dict=feeds)

    static_individual_var_grad_values = sess.run(
        static_individual_variable_gradients, feed_dict=feeds)

    return (values_static, state_value_static, static_grad_values,
            static_individual_grad_values, static_individual_var_grad_values)


if __name__ == "__main__":
  tf.test.main()
