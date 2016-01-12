###############################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
###############################################################################


import numpy as np

from tensorflow.core.framework import speech4_pb2
from speech4.models import las_utils


class Hypothesis(object):
  def __init__(self):
    self.text = ""
    self.logprob = 0.0
    self.state_prev = None
    self.state_next = None
    self.attention_prev = None
    self.attention_next = None
    self.alignment_prev = None
    self.alignment_next = None
    self.logprobs = None

  def feed_token(self, token_model):
    if self.text == "":
      return token_model.proto.token_sos
    return token_model.string_to_token[self.text[-1]]

  def expand(self, token_model, beam_width):
    completed = Hypothesis()
    completed.text = self.text
    completed.logprob = self.logprob + self.logprobs[token_model.proto.token_eos]

    candidates = zip(range(self.logprobs.size), self.logprobs.tolist())
    candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    candidates = candidates[:beam_width]

    partials = []
    if len(self.text) < 256:
      for token, logprob in candidates:
        if token != token_model.proto.token_eos and token in token_model.token_to_string:
          partial = Hypothesis()
          partial.text = self.text + token_model.token_to_string[token]
          partial.logprob = self.logprob + logprob
          partial.state_prev = self.state_next
          partial.alignment_prev = self.alignment_next
          partial.attention_prev = self.attention_next

          partials.append(partial)

    return partials, completed

class Utterance(object):
  def __init__(self):
    self.features = None
    self.features_len = None
    self.text = None
    self.uttid = None

    self.encoder_states = []
    self.feed_dict = {}

    self.hypothesis_partial = []
    self.hypothesis_complete = []

  def compute_word_distance(self, proto):
    ref = proto.ref.split(' ')
    hyp = proto.hyp.split(' ')

    proto.wer.edit_distance = las_utils.LevensteinDistance(ref, hyp)
    proto.wer.ref_length = len(ref)
    proto.wer.hyp_length = len(hyp)

    proto.wer.error_rate = float(proto.wer.edit_distance) / float(proto.wer.ref_length)

  def compute_character_distance(self, proto):
    ref = list(proto.ref)
    hyp = list(proto.hyp)

    proto.cer.edit_distance = las_utils.LevensteinDistance(ref, hyp)
    proto.cer.ref_length = len(ref)
    proto.cer.hyp_length = len(hyp)

    proto.cer.error_rate = float(proto.cer.edit_distance) / float(proto.cer.ref_length)

  def create_proto(self, token_model):
    proto = speech4_pb2.UtteranceResultsProto()
    proto.uttid = self.uttid
    proto.ref = self.text
    proto.hyp = self.hypothesis_complete[0].text

    if token_model.proto.remove_eow:
      proto.ref = ''.join(proto.ref.split(token_model.proto.token_string_eow))
      proto.hyp = ''.join(proto.hyp.split(token_model.proto.token_string_eow))

    self.compute_word_distance(proto)
    self.compute_character_distance(proto)

    self.proto = proto
