import string

from tensorflow.contrib.speech.data import token_model_pb2
from google.protobuf import text_format


class TokenModel(object):

  def __init__(self):
    self._token_model_proto = token_model_pb2.TokenModelProto()
    self._token_id_str_map = {}
    self._token_str_id_map = {}

  def load_proto(self, p):
    pass

  def save_proto(self, p):
    pass

  def token_model_proto(self):
    return self._token_model_proto

  def add_token(self, token_str, token_id=None):
    if not token_id:
      token_id = len(self._token_id_str_map)

    assert token_id not in self._token_id_str_map
    assert token_str not in self._token_str_id_map

    self._token_id_str_map[token_id] = token_str
    self._token_str_id_map[token_str] = token_id

    token_proto = self._token_model_proto.tokens.add()
    token_proto.token_id = token_id
    token_proto.token_str = token_str

  def create_basic_model(self):
    self.add_token("<sos>", 0)
    self.add_token("<eos>", 1)
    self.add_token(" ", 2)
    self.add_token("<unk>", 3)
    self.add_token("<blank>", 4)

    for c in string.ascii_uppercase:
      self.add_token(c)

    for n in range(10):
      self.add_token(str(n))
