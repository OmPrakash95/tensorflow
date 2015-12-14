#!/usr/bin/env python

import numpy as np
import os.path
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
import time


FLAGS = tf.app.flags.FLAGS

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

tf.app.flags.DEFINE_integer('encoder_cell_size', 256,
                            """Encoder cell size.""")
tf.app.flags.DEFINE_integer('decoder_cell_size', 256,
                            """Decoder cell size.""")
tf.app.flags.DEFINE_integer('attention_embedding_size', 256,
                            """Attention embedding size.""")

tf.app.flags.DEFINE_float('max_gradient_norm', 1000,
                           """Maximum gradient norm.""")

tf.app.flags.DEFINE_string('logdir', '/tmp',
                           """Path to our outputs and logs.""")

class LASModel(object):
  def __init__(self, batch_size, features_width, features_len_max, vocab_size, embedding_size, tokens_len_max,
      encoder_cell_size, decoder_cell_size, attention_embedding_size, max_gradient_norm):
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
    filename_queue = tf.train.string_input_producer(['speech4/data/train_si284.tfrecords'])

    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    serialized = tf.train.shuffle_batch(
        [serialized], batch_size=self.batch_size, num_threads=2, capacity=self.batch_size * 4 + 512,
        min_after_dequeue=512)
    
    # Parse the batched of serialized strings into the relevant utterance features.
    self.features, self.features_len, _, self.text, self.tokens, self.tokens_len, self.tokens_weights, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.features_len_max, tokens_len_max=self.tokens_len_max + 1)

    # Add the shape to the features.
    for feature in self.features:
      feature.set_shape([self.batch_size, self.features_width])


  def create_encoder(self):
    start_time = time.time()

    self.encoder_states = [[self.features, self.features_len]]

    self.create_encoder_layer()
    self.create_encoder_layer()
    self.create_encoder_layer(subsample_input=2)
    self.create_encoder_layer(subsample_input=2)

    # encoder_embedding_projection = tf.get_variable(
    #     "encoder_embedding_projection", [self.encoder_cell_size, self.attention_embedding_size])
    # self.encoder_embeddings = [
    #     math_ops.matmul(state, encoder_embedding_projection) for state in self.encoder_states[-1][0]]

    print('create_encoder graph time %f' % (time.time() - start_time))


  def create_encoder_layer(self, subsample_input=1):
    with vs.variable_scope('encoder_layer_%d' % (len(self.encoder_states))):
      if subsample_input == 1:
        self.encoder_states.append([rnn.rnn(
            rnn_cell.GRUCell(self.encoder_cell_size), self.encoder_states[-1][0], dtype=tf.float32,
                             sequence_length=self.encoder_states[-1][1])[0], self.encoder_states[-1][1]])
      else:
        two = tf.constant(2, shape=[self.batch_size, 1], dtype=tf.int64)
        self.encoder_states.append([rnn.rnn(
            rnn_cell.GRUCell(self.encoder_cell_size), self.encoder_states[-1][0][0::subsample_input], dtype=tf.float32,
                             sequence_length=self.encoder_states[-1][1])[0], tf.div(self.encoder_states[-1][1], two)])


  def create_decoder(self):
    start_time = time.time()

    self.decoder_states = []
    self.decoder_states.append(self.tokens[:-1])

    attention_states = [array_ops.reshape(e, [-1, 1, self.encoder_cell_size]) for e in self.encoder_states[-1][0]]
    attention_states = array_ops.concat(1, attention_states)

    self.create_decoder_layer(attention_states=attention_states)
    self.create_decoder_layer(output_projection=True)

    print('create_decoder graph time %f' % (time.time() - start_time))


  def create_decoder_layer(self, attention_states=None, output_projection=None, scope=None):
    with vs.variable_scope('decoder_layer_%d' % (len(self.decoder_states))):
      decoder_initial_state = tf.constant(0, shape=[self.batch_size, self.decoder_cell_size], dtype=tf.float32)
      decoder_initial_state.set_shape([self.batch_size, self.decoder_cell_size])

      cell = rnn_cell.GRUCell(self.decoder_cell_size)
      if output_projection == True:
        cell = rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)

      if attention_states:
        if len(self.decoder_states) == 1:
          self.decoder_states.append(seq2seq.embedding_attention_decoder(
              self.decoder_states[-1], decoder_initial_state, attention_states, cell, self.embedding_size, self.vocab_size)[0])
        else:
          self.decoder_states.append(seq2seq.attention_decoder(
              self.decoder_states[-1], decoder_initial_state, attention_states, cell)[0])
      else:
        self.decoder_states.append(seq2seq.rnn_decoder(
            self.decoder_states[-1], decoder_initial_state, cell)[0])


  def create_loss(self):
    self.losses = []

    self.losses.append(seq2seq.sequence_loss(
        self.decoder_states[-1], self.tokens[1:], self.tokens_weights[1:], self.vocab_size))


  def create_optimizer(self):
    params = tf.trainable_variables()

    self.gradient_norms = []
    self.updates = []
    opt = tf.train.GradientDescentOptimizer(0.001)

    gradients = tf.gradients(self.losses, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.max_gradient_norm)

    self.gradient_norms.append(norm)
    self.updates.append(opt.apply_gradients(
        zip(clipped_gradients, params), global_step=self.global_step))


  def step(self, sess, forward_only):
    ret = [None] * 3
    if forward_only:
      ret = sess.run(self.losses)
    else:
      temp = sess.run(self.updates + self.gradient_norms + self.losses)
      ret[0] = temp[0:len(self.updates)]
      ret[1] = temp[len(ret[0]):len(ret[0]) + len(self.gradient_norms)]
      ret[2] = temp[len(ret[0]) + len(ret[1]):len(ret[0]) + len(ret[1]) + len(self.losses)]

    return ret


def create_model(sess, forward_only):
  model = LASModel(
      FLAGS.batch_size, FLAGS.features_width, FLAGS.features_len_max, FLAGS.vocab_size, FLAGS.embedding_size,
      FLAGS.tokens_len_max, FLAGS.encoder_cell_size, FLAGS.decoder_cell_size, FLAGS.attention_embedding_size,
      FLAGS.max_gradient_norm)

  sess.run(tf.initialize_all_variables())
  tf.train.start_queue_runners(sess=sess)

  return model


def run_train():
  with tf.Session() as sess:
    model = create_model(sess, False)

    writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)

    _, _, losses = model.step(sess, False)
    print losses


def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
