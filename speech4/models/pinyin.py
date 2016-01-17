###############################################################################
# Copyright 2016 William Chan <williamchan@cmu.edu>.
###############################################################################


import codecs
import re
import tensorflow.core.framework.token_model_pb2 as token_model_pb2


class Pinyin(object):
  def __init__(self, dict_path="speech4/conf/dictionary/cedict_1_0_ts_utf-8_mdbg.txt"):
    self._simplified_to_pinyin = {}
    self._traditional_to_pinyin = {}
    with codecs.open(dict_path, "r", "utf-8") as dict_file:
      lines = dict_file.read().splitlines()
      for line in lines:
        if line.startswith("#"):
          continue
        cols = re.split("\[|\]", line)
        chinese = cols[0].strip().split(" ")
        assert len(chinese) == 2
        traditional = chinese[0]
        simplified  = chinese[1]
        pinyin = cols[1].strip().lower()
        self._simplified_to_pinyin[simplified] = pinyin
        self._traditional_to_pinyin[traditional] = pinyin
    self.create_token_model_proto()
    for c in "abcdefghijklmnopqrstuvwxyz0123456789":
      self.add_token(c)

  def create_token_model_proto(self):
    token_model_proto = token_model_pb2.TokenModelProto()
    self._token_model_proto = token_model_proto
    self._tokens = {}

    token_model_proto.token_sos = 0
    token_model_proto.token_string_sos = "<S>"
    self.add_token(token_model_proto.token_string_sos)
    token_model_proto.token_eos = 1
    token_model_proto.token_string_eos = "</S>"
    self.add_token(token_model_proto.token_string_eos)
    token_model_proto.token_eow = 2
    token_model_proto.token_string_eow = " "
    self.add_token(token_model_proto.token_string_eow)
    token_model_proto.token_unk = 3
    token_model_proto.token_string_unk = "<UNK>"
    self.add_token(token_model_proto.token_string_unk)

  def add_token(self, token_string):
    token_id = len(self._token_model_proto.tokens)
    token = self._token_model_proto.tokens.add()
    token.token_string = token_string
    token.token_id     = token_id
    self._tokens[token_string] = True

  def is_english(self, word):
    for c in word:
      if c not in "abcdefghijklmnopqrstuvwxyz0123456789":
        return False
    return True

  def simplified_to_pinyin(self, simplified):
    assert ":" not in simplified
    simplified = simplified.replace("[LAUGHTER]", " ")
    simplified = simplified.replace("[VOCALIZED-NOISE]", " ")
    simplified = simplified.lower()
    oov = False
    pinyin = ""

    for w in simplified.split(" "):
      if w == " " or w == "":
        pinyin += " "
      elif w in self._simplified_to_pinyin:
        pinyin += self._simplified_to_pinyin[w] + " "
      elif w in self._traditional_to_pinyin:
        pinyin += self._traditional_to_pinyin[w] + " "
      elif self.is_english(w):
        pinyin += w + " "
      else:
        for c in w:
          if c in self._simplified_to_pinyin:
            pinyin += self._simplified_to_pinyin[c] + " "
          elif c in self._traditional_to_pinyin:
            pinyin += self._traditional_to_pinyin[c] + " "
          else:
            pinyin += c
            oov = True
    pinyin = pinyin.replace(":", "")
    pinyin = " ".join(pinyin.split(" "))
    if oov:
      print pinyin
    return pinyin

def main():
  pinyin = Pinyin()
  lines_train = codecs.open("/data-local/wchan/kaldi/egs/gale_mandarin/s5/data/train/text", "r", "utf-8").read().splitlines()
  lines_dev = codecs.open("/data-local/wchan/kaldi/egs/gale_mandarin/s5/data/dev/text", "r", "utf-8").read().splitlines()
  lines = lines_train + lines_dev
  for line in lines:
    cols = line.split(" ", 1)
    simplified = cols[1]
    line = pinyin.simplified_to_pinyin(simplified)
    
    for c in line:
      if c not in pinyin._tokens:
        pinyin.add_token(c)

  with open("speech4/conf/gale_mandarin_pinyin.pbtxt", "w") as proto_file:
    proto_file.write(str(pinyin._token_model_proto))

if __name__ == "__main__":
  main()
