import math
import numpy as np
import os.path
import time

import tensorflow as tf
from tensorflow.core.framework import speech4_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import gru_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
from tensorflow.python.platform import gfile
from speech4.models import las_model
from speech4.models import utterance


class Decoder(object):
  def __init__(self, sess, dataset, dataset_size, logdir, ckpt, decoder_params, model_params):
    self.model_params = model_params

    self.model_params.tokens_len_max = 2
    self.model_params.input_layer = "placeholder"

    self.dataset = dataset
    self.dataset_size = dataset_size

    self.model = las_model.LASModel(
        sess, dataset, logdir, ckpt, True, decoder_params.beam_width,
        self.model_params, 0.0, 0.0)

    print self.dataset
    tf.train.string_input_producer([dataset])
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([self.dataset])
    _, serialized = reader.read(filename_queue)
    serialized = tf.train.batch(
        [serialized], batch_size=1, num_threads=2, capacity=2)

    self.features, self.features_len, _, self.text, _, _, _, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.model_params.features_len_max, tokens_len_max=self.model_params.tokens_len_max + 1)

  # We read one utterance and return it in a dict.
  def read_utterance(self, sess):
    utt = utterance.Utterance()

    print 'wtf...'
    utt.text = sess.run(self.text)

    print utt.text


  def decode_utterance(self, sess, utt):
    feed_dict = {}


  def decode(self, sess):
    for idx in range(self.dataset_size):
      self.read_utterance(sess)
