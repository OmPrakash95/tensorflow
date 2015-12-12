#!/usr/bin/env python

import numpy as np
import os.path
import tensorflow as tf
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
import tensorflow.python.platform
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of utterances to process in a batch.""")

tf.app.flags.DEFINE_integer('features_len_max', 2560,
                            """Maximum number of features in an utterance.""")
tf.app.flags.DEFINE_integer('tokens_len_max', 256,
                            """Maximum number of tokens in an utterance.""")

tf.app.flags.DEFINE_integer('encoder_cell_size', 256,
                            """Encoder cell size.""")
tf.app.flags.DEFINE_integer('decoder_cell_size', 256,
                            """Decoder cell size.""")
tf.app.flags.DEFINE_integer('attention_embedding_size', 256,
                            """Attention embedding size.""")

class LASModel(object):
  def __init__(self, batch_size, features_len_max, tokens_len_max,
      encoder_cell_size, decoder_cell_size, attention_embedding_size):
    self.batch_size = batch_size
    self.features_len_max = features_len_max
    self.tokens_len_max = tokens_len_max
    self.encoder_cell_size = encoder_cell_size
    self.decoder_cell_size = decoder_cell_size

    self.global_step = tf.Variable(0, trainable=False)

    # Create the inputs.
    self.create_graph_inputs()

    # Create the encoder-encoder.
    self.create_encoder()
    self.create_decoder()

    self.saver = tf.train.Saver(tf.all_variables())


  def create_graph_inputs(self):
    filename_queue = tf.train.string_input_producer(['speech4/data/train_si284.tfrecords'])

    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    serialized = tf.train.shuffle_batch(
        [serialized], batch_size=self.batch_size, num_threads=2, capacity=self.batch_size * 4 + 512, min_after_dequeue=512)
    
    # Parse the batched of serialized strings into the relevant utterance features.
    self.features, self.features_len, self.text, self.tokens, self.tokens_len, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.features_len_max, tokens_len_max=self.tokens_len_max)


  def create_encoder(self):
    two = tf.constant([2], dtype=tf.float32)

    self.encoder_states = [[self.features, self.features_len]]

    self.encoder_states.append([rnn.rnn(
        rnn_cell.GRUCell(self.encoder_cell_size), self.encoder_states[-1][0], dtype=tf.float32,
                         sequence_length=self.encoder_states[-1][1]), self.encoder_states[-1][1]])
    self.encoder_states.append([rnn.rnn(
        rnn_cell.GRUCell(self.encoder_cell_size), self.encoder_states[-1][0], dtype=tf.float32,
                         sequence_length=self.encoder_states[-1][1]), self.encoder_states[-1][1]])

    self.encoder_states.append([rnn.rnn(
      rnn_cell.GRUCell(self.encoder_cell_size), self.encoder_states[-1][0][0::2], dtype=tf.float32,
                         sequence_length=self.encoder_states[-1][1]), tf.div(self.encoder_states[-1][1], two)])

    self.encoder_states.append([rnn.rnn(
      rnn_cell.GRUCell(self.encoder_cell_size), self.encoder_states[-1][0][0::2], dtype=tf.float32,
                         sequence_length=self.encoder_states[-1][1]), tf.div(self.encoder_states[-1][1], two)])

    encoder_embedding_projection = tf.get_variable("encoder_embedding_projection", [self.encoder_cell_size, self.attention_embedding_size])
    self.encoder_embeddings = [math_ops.matmul(state, encoder_embedding_projection) for state in self.encoder_states[-1][0]]


  def create_decoder(self):
    self.attention_states = array_ops.concat(1, self.encoder_states[-1][0])

    decoder_initial_state = tf.constant(0, shape=[self.batch_size, self.decoder_cell_size], dype=float32)
    self.decoder_states = []
    self.decoder_stats.append(seq2seq.embedding_attention_decoder(
        self.tokens, decoder_cell_size, self.attention_states,
        rnn_cell.GRUCell(self.decoder_cell_size), self.embedding_symbols))


  def step(self, sess, forward_only):
    return sess.run(self.text)


def create_model(sess, forward_only):
  model = LASModel(FLAGS.batch_size, FLAGS.features_len_max, FLAGS.tokens_len_max, FLAGS.encoder_cell_size, FLAGS.decoder_cell_size, FLAGS.attention_embedding_size)
  sess.run(tf.initialize_all_variables())
  tf.train.start_queue_runners(sess=sess)
  return model


def run_train():
  with tf.Session() as sess:
    model = create_model(sess, False)

    x = model.step(sess, False)

    print x



def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
