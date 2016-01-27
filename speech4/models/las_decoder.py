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
    self.model_params.tokens_len_max = 1
    self.model_params.input_layer = "placeholder"

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
    # Run the encoder.
    self.run_encoder(sess, utt)

    # Create the empty hypothesis.
    hyp = utterance.Hypothesis()
    hyp.state_prev = self.create_decoder_states_zero()
    hyp.alignment_prev = self.create_decoder_alignments_zero()
    hyp.attention_prev = self.create_decoder_attentions_zero()

    # Run the decoder for 1 step (needed because we have <sos> <sos> utterance <eos>.
    self.run_decoder_step(sess, utt, [hyp])
    hyp.state_prev = hyp.state_next
    hyp.alignment_prev = hyp.alignment_next
    hyp.attention_prev = hyp.attention_next
    hyp.state_next = None
    hyp.alignment_next = None
    hyp.attention_next = None
    hyp.logprobs = None
    hyp.text = ""
    utt.hypothesis_partial.append(hyp)

    while utt.hypothesis_partial:
      hypothesis_partial_next = []

      for start_idx in range(0, len(utt.hypothesis_partial), self.model.batch_size):
        hyps = utt.hypothesis_partial[start_idx:start_idx + self.model.batch_size]

        self.run_decoder_step(sess, utt, hyps)

        for hyp in hyps:
          partials, completed = hyp.expand(
              self.token_model, self.decoder_params.beam_width)
          hypothesis_partial_next.extend(partials)
          utt.hypothesis_complete.append(completed)

      hypothesis_partial_next = sorted(
          hypothesis_partial_next, key=lambda hyp: hyp.logprob, reverse=True)
      hypothesis_partial_next = hypothesis_partial_next[:self.decoder_params.beam_width]
      utt.hypothesis_partial = hypothesis_partial_next

    # Sort the completed hypothesis.
    utt.hypothesis_complete = sorted(
        utt.hypothesis_complete, key=lambda hyp: hyp.logprob / len(hyp.text), reverse=True)
    utt.hypothesis_complete = utt.hypothesis_complete[:self.decoder_params.beam_width]
    print 'ground_truth: %s' % utt.text
    print 'top hyp     : %s' % utt.hypothesis_complete[0].text
    utt.create_proto(self.token_model)
    print 'wer         : %f' % utt.proto.wer.error_rate

  def run_encoder(self, sess, utt):
    feed_dict = {}
    for idx in range(len(self.model.features)):
      feed_dict[self.model.features[idx]] = np.tile(
          utt.features[idx][0], [self.model.batch_size, 1])
    feed_dict[self.model.features_len] = np.array(
        np.tile(utt.features_len, self.model.batch_size), dtype=np.int64)

    utt.encoder_states = sess.run(self.model.encoder_states[-1][0], feed_dict=feed_dict)

  def create_decoder_states_zero(self):
    initial_state = []
    for state in self.model.decoder_states_initial:
      shape = state.get_shape().as_list()
      shape[0] = 1
      initial_state.append(np.zeros(shape, dtype=np.float32))
    return initial_state

  def create_decoder_alignments_zero(self):
    initial_state = []
    for state in self.model.decoder_alignments_initial:
      shape = state.get_shape().as_list()
      shape[0] = 1
      state = np.zeros(shape, dtype=np.float32)
      state[0,0] = 1
      initial_state.append(state)
    return initial_state

  def create_decoder_attentions_zero(self):
    initial_state = []
    for state in self.model.decoder_attentions_initial:
      shape = state.get_shape().as_list()
      shape[0] = 1
      initial_state.append(np.zeros(shape, dtype=np.float32))
    return initial_state

  def run_decoder_step(self, sess, utt, hyps):
    pad_length = self.model.batch_size - len(hyps)
    feed_dict = {}

    # Encoder states.
    for idx in range(len(self.model.encoder_states[-1][0])):
      feed_dict[self.model.encoder_states[-1][0][idx]] = utt.encoder_states[idx]
    feed_dict[self.model.features_len] = np.array(
        np.tile(utt.features_len, self.model.batch_size), dtype=np.int64)

    # Feed token.
    feed_dict[self.model.tokens[0]] = np.array(
        [hyp.feed_token(self.token_model) for hyp in hyps] + [0] * pad_length, dtype=np.int32)
    feed_dict[self.model.tokens_len] = np.array(
        [1] * self.model.batch_size, dtype=np.int64)

    # Decoder states.
    for idx in range(len(self.model.decoder_states_initial)):
      state_padding = [np.zeros([
          pad_length, hyps[0].state_prev[idx].shape[1]], dtype=np.float32)]
      feed_dict[self.model.decoder_states_initial[idx]] = np.vstack(
          [hyp.state_prev[idx] for hyp in hyps] + state_padding)

    # Attention context states.
    for idx in range(len(self.model.decoder_attentions_initial)):
      state_padding = [np.zeros([
          pad_length, hyps[0].attention_prev[idx].shape[1]], dtype=np.float32)]
      feed_dict[self.model.decoder_attentions_initial[idx]] = np.vstack(
          [hyp.attention_prev[idx] for hyp in hyps] + state_padding)

    for idx in range(len(self.model.decoder_alignments_initial)):
      state_padding = [np.zeros([
          pad_length, hyps[0].alignment_prev[idx].shape[1]], dtype=np.float32)]
      feed_dict[self.model.decoder_alignments_initial[idx]] = np.vstack(
          [hyp.alignment_prev[idx] for hyp in hyps] + state_padding)

    # Fetch the next state and the log prob.
    fetches = {}
    fetches["decoder_state_last"] = self.model.decoder_state_last
    fetches["decoder_alignments_last"] = self.model.decoder_alignment_last
    fetches["decoder_attentions_last"] = self.model.decoder_attention_last
    fetches["logprob"] = self.model.logprob

    fetches = self.model.run_graph(sess, fetches, feed_dict=feed_dict)

    for idx in range(len(hyps)):
      hyps[idx].state_next = [state[idx:idx+1,:] for state in fetches['decoder_state_last']]
      hyps[idx].alignment_next = [state[idx:idx+1,:] for state in fetches['decoder_alignments_last']]
      hyps[idx].attention_next = [state[idx:idx+1,:] for state in fetches['decoder_attentions_last']]
      hyps[idx].logprobs = [logprob[idx,:] for logprob in fetches['logprob']][0]
