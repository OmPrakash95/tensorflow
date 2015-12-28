###############################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
###############################################################################


import google

from tensorflow.core.framework import token_model_pb2

class TokenModel(object):
  def __init__(self, token_model_path):
    self.proto = token_model_pb2.TokenModelProto()
    with open(token_model_path, "r") as proto_file:
      google.protobuf.text_format.Merge(proto_file.read(), self.proto)

    self.token_to_string = {}
    self.string_to_token = {}

    for token in self.proto.tokens:
      self.token_to_string[token.token_id] = token.token_string
      self.string_to_token[token.token_string] = token.token_id

  def string_to_tokens(self, string):
    return [self.proto.token_sos] + [self.token_to_string[w] for w in string]
