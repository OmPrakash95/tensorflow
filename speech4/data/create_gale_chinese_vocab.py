#!/usr/bin/env python

################################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
################################################################################

import argparse
import codecs
import google
import os
import operator
import sys
import tensorflow.core.framework.token_model_pb2 as token_model_pb2


def main():
  parser = argparse.ArgumentParser(description="SPEECH4 (C) 2015 William Chan <williamchan@cmu.edu>")
  parser.add_argument("--text", type=str, default="/data-local/wchan/kaldi/egs/gale_mandarin/s5/data/train/text")
  parser.add_argument("--token_model_pbtxt", type=str)
  args = vars(parser.parse_args())

  utterances = {}
  with codecs.open(args['text'], "r", "utf-8") as text_file:
    for line in text_file.read().splitlines():
      cols = line.split(" ", 1)
      uttid = cols[0]
      utterance = cols[1]

      utterances[uttid] = utterance

  vocab = {}
  for uttid, utterance in utterances.iteritems():
    for c in utterance:
      if c not in vocab:
        vocab[c] = 0
      vocab[c] = vocab[c] + 1
  vocab = sorted(vocab.items(), key=operator.itemgetter(1), reverse=True)

  token_model_proto = token_model_pb2.TokenModelProto()
  token_model_proto.token_sos = 0
  token_model_proto.token_string_sos = "<S>"
  token_model_proto.token_eos = 1
  token_model_proto.token_string_eos = "</S>"
  token_model_proto.token_eow = 2
  token_model_proto.token_string_eow = " "
  token_model_proto.token_unk = 3
  token_model_proto.token_string_unk = "<UNK>"

  token_id = 4

  for v, c in vocab:
    token = token_model_proto.tokens.add()
    if v == " ":
      token.token_id = token_model_proto.token_eow
    else:
      token.token_id = token_id
      token_id = token_id + 1
    token.token_string = v
    token.token_count = c

  with open(args['token_model_pbtxt'], "w") as proto_file:
    proto_file.write(str(token_model_proto))

if __name__ == '__main__':
  main()
