###############################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
###############################################################################


import numpy as np


class Hypothesis(object):
  def __init__(self):
    self.text = ""
    self.logprob = 0.0
    self.state_prev = None
    self.state_next = None
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
