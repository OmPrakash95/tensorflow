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


class Decoder(object):
  def __init__(self, sess, dataset, dataset_size, logdir, ckpt, decoder_params, model_params):
    self.token_model = token_model.TokenModel(
        "speech4/conf/token_model_character_simple.pbtxt")

    self.decoder_params = decoder_params

    self.model_params = model_params
    self.model_params.tokens_len_max = 1
    self.model_params.input_layer = "placeholder"

    self.dataset = dataset
    self.dataset_size = dataset_size

    with tf.variable_scope("model"):
      self.model = las_model.LASModel(
          sess, dataset, logdir, ckpt, True, self.decoder_params.beam_width,
          self.model_params)

    tf.train.string_input_producer([dataset])
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer([self.dataset])
    _, serialized = reader.read(filename_queue)
    serialized = tf.train.batch(
        [serialized], batch_size=1, num_threads=2, capacity=2)

    self.features, self.features_len, _, self.text, _, _, _, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.model_params.features_len_max,
        tokens_len_max=self.model_params.tokens_len_max + 1)

  # We read one utterance and return it in a dict.
  def read_utterance(self, sess):
    utt = utterance.Utterance()

    targets = {}
    targets['features'] = self.features
    targets['features_len'] = self.features_len
    targets['text'] = self.text
    targets['uttid'] = self.uttid

    fetches = self.model.run_graph(sess, targets)

    utt.features = fetches['features']
    utt.features_len = fetches['features_len']
    utt.text = fetches['text'][0]
    utt.uttid = fetches['uttid'][0]

    return utt

  def decode_utterance(self, sess, utt):
    # Run the encoder.
    self.run_encoder(sess, utt)

    # Create the empty hypothesis.
    hyp = utterance.Hypothesis()
    hyp.text = ""
    hyp.state_prev = self.create_decoder_states_zero()
    hyp.state_prev = self.run_decoder_step(sess, utt, hyp.state_prev, [0])['decoder_states_stack']
    hyp.feed_token = self.token_model.proto.token_sos

    # Add the empty hypothesis to the utterance.
    utt.hypothesis_partial.append(hyp)

    # Decode until we have no more partial hypothesis.
    while utt.hypothesis_partial:
      hypothesis_partial_next = []

      # We need to batch up the utterances.
      for start_idx in range(0, len(utt.hypothesis_partial), self.model.batch_size):
        hyps = utt.hypothesis_partial[start_idx:start_idx + self.model.batch_size]

        feed_dict = self.batch_hyps(utt, hyps)

        targets = {}
        targets['decoder_states_stack'] = self.model.decoder_states_stack
        targets['logprob'] = self.model.logprob

        fetches = self.model.run_graph(sess, targets, feed_dict)

        for idx in range(len(hyps)):
          hyps[idx].state_next = [state[idx:idx+1,:] for state in fetches['decoder_states_stack']]
          hyps[idx].logprobs = [logit[idx,:] for logit in fetches['logprob']][0]
          partials, completed = hyps[idx].expand(self.token_model, self.decoder_params.beam_width)

          hypothesis_partial_next.extend(partials)
          utt.hypothesis_complete.append(completed)

      hypothesis_partial_next = sorted(hypothesis_partial_next, key=lambda hyp: hyp.logprob, reverse=True)
      hypothesis_partial_next = hypothesis_partial_next[:self.decoder_params.beam_width]
      if hypothesis_partial_next:
        print hypothesis_partial_next[0].text

      utt.hypothesis_partial = hypothesis_partial_next
    utt.hypothesis_complete = sorted(utt.hypothesis_complete, key=lambda hyp: hyp.logprob / len(hyp.text), reverse=True)
    print utt.text
    print utt.hypothesis_complete[0].text
    print utt.hypothesis_complete[0].logprob

  def batch_inputs(self, x):
    return np.asarray(x + [np.zeros(x[0].shape, dtype=x[0].dtype)] * (self.model.batch_size - len(x)))
    #return np.asarray(x)

  def batch_hyps(self, utt, hyps):
    # Encoder states.
    feed_dict = {}
    for idx in range(len(self.model.encoder_states[-1][0])):
      feed_dict[self.model.encoder_states[-1][0][idx]] = utt.encoder_states[idx]
    feed_dict[self.model.features_len] = np.array(
        np.tile(utt.features_len, self.model.batch_size), dtype=np.int64)
    # Tokens.
    for idx in range(len(self.model.tokens)):
      feed_dict[self.model.tokens[idx]] = np.array(
          [hyp.feed_token for hyp in hyps] + [0] * (self.model.batch_size - len(hyps)), dtype=np.int32)
      feed_dict[self.model.tokens_weights[idx]] = np.array(
          [1.0] * self.model.batch_size, dtype=np.float32)
    feed_dict[self.model.tokens_len] = np.array(
        [self.model_params.tokens_len_max] * self.model.batch_size, dtype=np.int64)
    # Initial state.
    for idx in range(len(self.model.decoder_states_initial)):
      feed_dict[self.model.decoder_states_initial[idx]] = self.batch_inputs(
          [hyp.state_prev[idx][0] for hyp in hyps])

    return feed_dict


  # Run and cache the encoder part of the model.
  # run_encoder is always operated on one utterance.
  def run_encoder(self, sess, utt):
    feed_dict = {}
    for idx in range(len(self.model.features)):
      feed_dict[self.model.features[idx]] = self.batch_inputs(
          [utt.features[idx][0]])
    feed_dict[self.model.features_len] = np.array(
        np.tile(utt.features_len, self.model.batch_size), dtype=np.int64)
    for idx in range(len(self.model.tokens)):
      feed_dict[self.model.tokens[idx]] = np.array(
          [self.token_model.proto.token_sos] * self.model.batch_size, dtype=np.int32)
      feed_dict[self.model.tokens_weights[idx]] = np.array(
          [0.0] * self.model.batch_size, dtype=np.float32)
    feed_dict[self.model.tokens_len] = np.array(
        [self.model_params.tokens_len_max] * self.model.batch_size, dtype=np.int64)

    utt.encoder_states = sess.run(self.model.encoder_states[-1][0], feed_dict)

  def create_decoder_states_zero(self):
    initial_state = []
    for state in self.model.decoder_states_initial:
      shape = state.get_shape().as_list()
      shape[0] = 1
      initial_state.append(np.zeros(shape))
    return initial_state

  def run_decoder_step(self, sess, utt, decoder_states_initial, tokens):
    assert hasattr(utt, 'encoder_states')
    tokens = tokens + [0] * (self.model.batch_size - len(tokens))

    feed_dict = {}
    for idx in range(len(self.model.encoder_states[-1][0])):
      feed_dict[self.model.encoder_states[-1][0][idx]] = utt.encoder_states[idx]
    feed_dict[self.model.features_len] = np.array(
        np.tile(utt.features_len, self.model.batch_size), dtype=np.int64)
    for idx in range(len(self.model.tokens)):
      feed_dict[self.model.tokens[idx]] = np.array(
          [self.token_model.proto.token_sos] * self.model.batch_size, dtype=np.int32)
      feed_dict[self.model.tokens_weights[idx]] = np.array(
          tokens, dtype=np.float32)
    feed_dict[self.model.tokens_len] = np.array(
        [self.model_params.tokens_len_max] * self.model.batch_size, dtype=np.int64)
    for idx in range(len(self.model.decoder_states_initial)):
      if decoder_states_initial[idx].shape[0] == 1:
        feed_dict[self.model.decoder_states_initial[idx]] = np.tile(
            decoder_states_initial[idx], [self.model.batch_size, 1])
      else:
        feed_dict[self.model.decoder_states_initial[idx]] = decoder_states_initial[idx]

    #decoder_states = sess.run(self.model.decoder_states_stack, feed_dict)
    targets = {}
    targets['decoder_states_stack'] = self.model.decoder_states_stack
    targets['logprob'] = self.model.logprob

    fetches = self.model.run_graph(sess, targets, feed_dict)

    return fetches

  def decode(self, sess):
    for idx in range(self.dataset_size):
      utt = self.read_utterance(sess)
      self.decode_utterance(sess, utt)
