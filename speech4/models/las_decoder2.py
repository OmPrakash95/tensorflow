###############################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
###############################################################################


import google
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
from speech4.models import token_model
from speech4.models import utterance


class Decoder2(object):
  def __init__(
      self, sess, dataset, dataset_size, logdir, ckpt, decoder_params,
      model_params):
    self.decoder_params = decoder_params

    if self.decoder_params.token_model:
      self.token_model = token_model.TokenModel(
          self.decoder_params.token_model)
    else:
      self.token_model = token_model.TokenModel(
          "speech4/conf/token_model_character_simple.pbtxt")

    self.model_params = model_params
    self.model_params.attention_params.type = "median"
    self.model_params.attention_params.median_window_l = 10
    self.model_params.attention_params.median_window_r = 100
    self.model_params.input_layer = "decoder"

    self.dataset = dataset
    self.dataset_size = dataset_size

    self.logdir = logdir

    with tf.variable_scope("model"):
      self.model = las_model.LASModel(
          sess, dataset, logdir, ckpt, True, self.decoder_params.beam_width,
          self.model_params)

    # Graph to read 1 utterance.
    tf.train.string_input_producer([dataset])
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([self.dataset])
    _, serialized = reader.read(filename_queue)
    serialized = tf.train.batch(
        [serialized], batch_size=1, num_threads=2, capacity=2)

    self.features, _, self.features_len, _, _, self.text, _, _, _, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.model_params.features_len_max,
        tokens_len_max=1)

  # We read one utterance and return it in a dict.
  def read_utterance(self, sess):
    targets = {}
    targets['features'] = self.features
    targets['features_len'] = self.features_len
    targets['text'] = self.text
    targets['uttid'] = self.uttid

    fetches = self.model.run_graph(sess, targets)

    utt = utterance.Utterance()
    utt.features = fetches['features']
    utt.features_len = fetches['features_len']
    utt.text = fetches['text'][0]
    utt.uttid = fetches['uttid'][0]

    return utt

  def decode(self, sess):
    cer = speech4_pb2.EditDistanceResultsProto()
    wer = speech4_pb2.EditDistanceResultsProto()
    utts = []
    for idx in range(self.dataset_size):
      utt = self.read_utterance(sess)
      self.decode_utterance(sess, utt)

      cer.edit_distance += utt.proto.cer.edit_distance
      cer.ref_length += utt.proto.cer.ref_length
      cer.error_rate = float(cer.edit_distance) / float(cer.ref_length)

      wer.edit_distance += utt.proto.wer.edit_distance
      wer.ref_length += utt.proto.wer.ref_length
      wer.error_rate = float(wer.edit_distance) / float(wer.ref_length)

      print("accum wer: %f (%d / %d); cer: %f; (%d / %d)" % (wer.error_rate, wer.edit_distance, wer.ref_length, cer.error_rate, idx, self.dataset_size))
      utts.append(utt)

    with open(os.path.join(self.logdir, "decode_results_cer.pbtxt"), "w") as proto_file:
      proto_file.write(str(cer.error_rate))
    with open(os.path.join(self.logdir, "decode_results_wer.pbtxt"), "w") as proto_file:
      proto_file.write(str(wer.error_rate))

    with open(os.path.join(self.logdir, "decode_results_details.pbtxt"), "w") as proto_file:
      for utt in utts:
        proto_file.write(str(utt.proto))

  def decode_utterance(self, sess, utt):
    self.run_model(sess, utt)

  def run_model(self, sess, utt):
    feed_dict = {}

    # Encoder features (i.e., fbanks).
    for idx in range(len(self.model.features)):
      feed_dict[self.model.features[idx]] = np.tile(
          utt.features[idx][0], [self.model.batch_size, 1])
    feed_dict[self.model.features_len] = np.array(
        np.tile(utt.features_len, self.model.batch_size), dtype=np.int64)

    # We should have only 1 input token.
    feed_dict[self.model.tokens[0]] = np.asarray(
        [self.token_model.proto.token_sos] * self.model.batch_size, dtype=np.int32)

    feed_dict[self.model.tokens_len] = np.array(
        [self.model_params.tokens_len_max] * self.model.batch_size, dtype=np.int64)

    fetches = {}
    for idx, logit in enumerate(self.model.decoder_states):
      fetches['logit_%03d' % idx] = logit
    fetches = self.model.run_graph(sess, fetches, feed_dict=feed_dict)

    hyp_text = ""
    for logit in sorted(fetches)[1:]:
      token = np.argmax(fetches[logit])

      if token == self.token_model.proto.token_eos:
        break
      hyp_text += self.token_model.token_to_string[token]
    utt.hypothesis_complete.append(utterance.Hypothesis(hyp_text))
    utt.create_proto(self.token_model)
