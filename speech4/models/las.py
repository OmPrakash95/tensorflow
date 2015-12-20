#!/usr/bin/env python

import numpy as np
import os.path
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gru_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('device', 0,
                            """The GPU device to use (set to negative to use CPU).""")

tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of utterances to process in a batch.""")

tf.app.flags.DEFINE_integer('features_width', 123,
                            """Size of each feature.""")

tf.app.flags.DEFINE_integer('features_len_max', 2560,
                            """Maximum number of features in an utterance.""")
tf.app.flags.DEFINE_integer('tokens_len_max', 256,
                            """Maximum number of tokens in an utterance.""")

tf.app.flags.DEFINE_integer('vocab_size', 64,
                            """Token vocabulary size.""")
tf.app.flags.DEFINE_integer('embedding_size', 32,
                            """Token vocabulary size.""")

tf.app.flags.DEFINE_integer('encoder_cell_size', 512,
                            """Encoder cell size.""")
tf.app.flags.DEFINE_integer('decoder_cell_size', 512,
                            """Decoder cell size.""")
tf.app.flags.DEFINE_integer('attention_embedding_size', 512,
                            """Attention embedding size.""")

tf.app.flags.DEFINE_float('max_gradient_norm', 1000,
                           """Maximum gradient norm.""")

tf.app.flags.DEFINE_float('learning_rate', 0.1,
                           """Learning rate.""")

tf.app.flags.DEFINE_string('logdir', '/tmp',
                           """Path to our outputs and logs.""")

class LASModel(object):
  def __init__(self, dataset, batch_size, features_width, features_len_max,
      vocab_size, embedding_size, tokens_len_max, encoder_cell_size,
      decoder_cell_size, attention_embedding_size, max_gradient_norm,
      learning_rate):
    self.dataset = dataset
    self.batch_size = batch_size
    
    self.features_width = features_width
    self.features_len_max = features_len_max
    self.vocab_size = vocab_size

    self.embedding_size = embedding_size
    self.tokens_len_max = tokens_len_max
    self.encoder_cell_size = encoder_cell_size
    self.decoder_cell_size = decoder_cell_size
    self.attention_embedding_size = attention_embedding_size

    self.max_gradient_norm = max_gradient_norm
    self.learning_rate = learning_rate

    self.step_total = 0
    self.step_time_total = 0

    self.global_step = tf.Variable(0, trainable=False)

    # Create the inputs.
    self.create_graph_inputs()

    # Create the encoder-encoder.
    self.create_encoder()
    self.create_decoder()

    # Create the loss.
    self.create_loss()

    # Create the optimizer.
    self.create_optimizer()

    self.saver = tf.train.Saver(tf.all_variables())


  def create_graph_inputs(self):
    dataset_map = {}
    if self.dataset == 'train_si284':
      self.dataset = 'speech4/data/train_si284.tfrecords'
      self.dataset_size = 37416
    elif self.dataset == 'test_dev93':
      self.dataset = 'speech4/data/test_dev93.tfrecords'
      self.dataset_size = 503
    elif self.dataset == 'test_eval92':
      self.dataset = 'speech4/data/test_eval92.tfrecords'
      self.dataset_size = 333
    filename_queue = tf.train.string_input_producer([self.dataset])

    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    serialized = tf.train.shuffle_batch(
        [serialized], batch_size=self.batch_size, num_threads=2, capacity=self.batch_size * 4 + 512,
        min_after_dequeue=512, seed=1000)
    
    # Parse the batched of serialized strings into the relevant utterance features.
    self.features, self.features_len, _, self.text, self.tokens, self.tokens_len, self.tokens_weights, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.features_len_max, tokens_len_max=self.tokens_len_max + 1)

    # Add the shape to the features.
    for feature in self.features:
      feature.set_shape([self.batch_size, self.features_width])
    for token in self.tokens:
      token.set_shape([self.batch_size])


  def create_encoder(self):
    start_time = time.time()

    self.encoder_states = [[self.features, self.features_len]]

    self.create_encoder_layer()
    self.create_encoder_layer()
    self.create_encoder_layer(subsample_input=2)
    self.create_encoder_layer(subsample_input=2)

    print('create_encoder graph time %f' % (time.time() - start_time))


  def create_encoder_layer(self, subsample_input=1, use_monolithic=True):
    with vs.variable_scope('encoder_layer_%d' % (len(self.encoder_states))):
      sequence_len_factor = tf.constant(subsample_input, shape=[self.batch_size], dtype=tf.int64)
      sequence_len = tf.div(self.encoder_states[-1][1], sequence_len_factor)
      if use_monolithic:
        self.encoder_states.append([gru_ops.gru(
            cell_size=self.encoder_cell_size,
            sequence_len=sequence_len,
            xs=self.encoder_states[-1][0][0::subsample_input])[-1], sequence_len])
      else:
        self.encoder_states.append([rnn.rnn(
            rnn_cell.GRUCell(self.encoder_cell_size),
            self.encoder_states[-1][0][0::subsample_input], dtype=tf.float32,
            sequence_length=sequence_len)[0], sequence_len])

      for encoder_state in self.encoder_states[-1][0]:
        encoder_state.set_shape([self.batch_size, self.encoder_cell_size])


  def create_decoder(self):
    start_time = time.time()

    self.decoder_states = []
    self.decoder_states.append(self.tokens[:-1])

    attention_states = []
    if len(self.encoder_states) > 1:
      attention_states = [
          array_ops.reshape(e, [-1, 1, self.encoder_cell_size])
          for e in self.encoder_states[-1][0]]
      attention_states = array_ops.concat(1, attention_states)

    self.create_decoder_layer(attention_states=attention_states)
    self.create_decoder_layer(output_projection=True)

    print('create_decoder graph time %f' % (time.time() - start_time))


  def create_decoder_layer(
      self, attention_states=None, output_projection=None, scope=None):
    with vs.variable_scope('decoder_layer_%d' % (len(self.decoder_states))):
      decoder_initial_state = tf.constant(
          0, shape=[self.batch_size, self.decoder_cell_size], dtype=tf.float32)
      decoder_initial_state.set_shape([self.batch_size, self.decoder_cell_size])

      # cell = rnn_cell.GRUCell(self.decoder_cell_size)
      cell = rnn_cell.GRUCellv2(self.decoder_cell_size, sequence_len=self.tokens_len)
      if output_projection == True:
        cell = rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)

      if attention_states:
        if len(self.decoder_states) == 1:
          self.decoder_states.append(seq2seq.embedding_attention_decoder(
              self.decoder_states[-1], decoder_initial_state, attention_states,
              cell, self.embedding_size, self.vocab_size,
              attention_states_sequence_len=self.encoder_states[-1][1],
              sequence_length=self.tokens_len)[0])
        else:
          self.decoder_states.append(seq2seq.attention_decoder(
              self.decoder_states[-1], decoder_initial_state, attention_states,
              cell, attention_states_sequence_len=self.encoder_states[-1][1],
              sequence_length=self.tokens_len)[0])
      else:
        if len(self.decoder_states) == 1:
          self.decoder_states.append(seq2seq.embedding_rnn_decoder(
              self.decoder_states[-1], decoder_initial_state, cell,
              self.embedding_size, self.vocab_size,
              sequence_length=self.tokens_len)[0])
        else:
          self.decoder_states.append(seq2seq.rnn_decoder(
              self.decoder_states[-1], decoder_initial_state, cell,
              sequence_length=self.tokens_len)[0])


  def create_loss(self):
    start_time = time.time()

    self.losses = []

    logits = self.decoder_states[-1]
    targets = self.tokens[:-1]
    weights = self.tokens_weights[:-1]
    with tf.op_scope(logits + targets + weights, "sequence_loss"):
      log_perp_list = []
      for idx, (logit, target, weight) in enumerate(zip(logits, targets, weights)):
        indices = target + self.vocab_size * math_ops.range(self.batch_size)
        with tf.device("/cpu:0"):
          target_dense = sparse_ops.sparse_to_dense(
              indices,
              array_ops.expand_dims(self.batch_size * self.vocab_size, 0),
              1.0, 0.0)
        target = array_ops.reshape(target_dense, [self.batch_size, self.vocab_size])
        xent = nn_ops.softmax_cross_entropy_with_logits(
            logit, target, name="CrossEntropyLoss{0}".format(idx))
        log_perp_list.append(xent * weight)
      log_perps = math_ops.add_n(log_perp_list) / (math_ops.add_n(weights) + 1e-12)
      log_perps = math_ops.reduce_sum(log_perps) / math_ops.cast(self.batch_size, tf.float32)
      self.losses.append(log_perps)

    print('create_loss graph time %f' % (time.time() - start_time))


  def create_optimizer(self):
    start_time = time.time()

    params = tf.trainable_variables()
    for param in params:
      print(param.name + ': ' + str(param.get_shape()))

    self.gradient_norms = []
    self.updates = []
    # opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)

    gradients = tf.gradients(self.losses, params)
    clipped_gradients, norm = tf.clip_by_global_norm(
        gradients, self.max_gradient_norm)

    self.gradient_norms.append(norm)
    self.updates.append(opt.apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step))

    print('create_optimizer graph time %f' % (time.time() - start_time))


  def step(self, sess, forward_only, epoch=False):
    if epoch:
      steps_per_epoch = self.dataset_size // self.batch_size

      for s in range(steps_per_epoch):
        self.step(sess, forward_only, epoch=False)

    start_time = time.time()

    if forward_only:
      uttid, text, logperp = sess.run([self.uttid, self.text] + self.losses)
    else:
      uttid, text, logperp, _ = sess.run([self.uttid, self.text] + self.losses + self.updates)

    print logperp
    tf.scalar_summary('logperp', logperp)

    step_time = time.time() - start_time

    self.step_total += 1
    self.step_time_total += step_time


def create_model(sess, dataset, forward_only):
  start_time = time.time()

  model = LASModel(
      dataset, FLAGS.batch_size, FLAGS.features_width, FLAGS.features_len_max,
      FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.tokens_len_max,
      FLAGS.encoder_cell_size, FLAGS.decoder_cell_size,
      FLAGS.attention_embedding_size, FLAGS.max_gradient_norm,
      FLAGS.learning_rate)

  sess.run(tf.initialize_all_variables())
  tf.train.start_queue_runners(sess=sess)

  print('create_model graph time %f' % (time.time() - start_time))

  return model


def run_train():
  device = '/gpu:%d' % FLAGS.device if FLAGS.device >= 0 else '/cpu:0'
  with tf.device(device):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      model = create_model(sess, 'train_si284', False)

      summary_op = tf.merge_all_summaries()

      summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
      summary_writer.flush()

      model.step(sess, False, epoch=True)

      summary_str = sess.run(summary_op)
      summary_writer.add_summary(summary_str, 0)


def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
