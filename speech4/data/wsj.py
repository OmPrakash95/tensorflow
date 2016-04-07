#!/usr/bin/env python

################################################################################
# Copyright 2016 William Chan <williamchan@cmu.edu>.
################################################################################

import argparse
import google
import itertools
import kaldi_io
import matplotlib
matplotlib.use('Agg')
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import os
import sys
import tensorflow as tf
import tensorflow.core.framework.token_model_pb2 as token_model_pb2


def main():
  parser = argparse.ArgumentParser(description='SPEECH3 (C) 2015 William Chan <williamchan@cmu.edu>')
  parser.add_argument('--kaldi_scp', type=str)
  parser.add_argument('--kaldi_txt', type=str)
  parser.add_argument('--tf_records', type=str)
  args   = vars(parser.parse_args())

  convert(args['kaldi_scp'], args['kaldi_txt'], args['tf_records'])


def normalize_text_wsj(line):
  cols = line.split(' ', 1)
  assert len(cols) >= 1 and len(cols) <= 2
  uttid = cols[0]
  if len(cols) == 1:
    utt = ''
  elif len(cols) == 2:
    utt = cols[1]

  assert '#' not in utt

  # Normalize to uppercase.
  utt = utt.upper()

  utt = utt.replace('<NOISE>', '')

  utt = utt.replace('&AMPERSAND', 'AMPERSAND')
  utt = utt.replace(')CLOSE_PAREN', 'CLOSEPAREN')
  utt = utt.replace(')CLOSE-PAREN', 'CLOSEPAREN')
  utt = utt.replace('\"CLOSE-QUOTE', 'CLOSEQUOTE')
  utt = utt.replace(':COLON', "COLON")
  utt = utt.replace(',COMMA', 'COMMA')
  utt = utt.replace('-DASH', 'DASH')
  utt = utt.replace('"DOUBLE-QUOTE', 'DOUBLEQUOTE')
  utt = utt.replace('"END-QUOTE', 'ENDQUOTE')
  utt = utt.replace('"END-OF-QUOTE', 'ENDOFQUOTE')
  utt = utt.replace(')END-OF-PAREN', 'ENDOFPAREN')
  utt = utt.replace(')END-THE-PAREN', 'ENDTHEPAREN')
  utt = utt.replace('!EXCLAMATION-POINT', 'EXCLAMATIONPOINT')
  utt = utt.replace('-HYPHEN', 'HYPHEN')
  utt = utt.replace('\"IN-QUOTES', 'INQUOTES')
  utt = utt.replace('(IN-PARENTHESIS', 'INPARENTHESIS')
  utt = utt.replace('{LEFT-BRACE', 'LEFTBRACE')
  utt = utt.replace('(LEFT-PAREN', 'LEFTPAREN')
  utt = utt.replace('(PAREN', 'PAREN')
  utt = utt.replace(')PAREN', 'PAREN')
  utt = utt.replace('?QUESTION-MARK', 'QUESTIONMARK')
  utt = utt.replace('"QUOTE', 'QUOTE')
  utt = utt.replace(')RIGHT-PAREN', 'RIGHTPAREN')
  utt = utt.replace('}RIGHT-BRACE', 'RIGHTBRACE')
  utt = utt.replace('\'SINGLE-QUOTE', 'SINGLEQUOTE')
  utt = utt.replace('/SLASH', 'SLASH')
  utt = utt.replace(';SEMI-COLON', "SEMICOLON")
  utt = utt.replace(')UN-PARENTHESES', 'UNPARENTHESES')
  utt = utt.replace('"UNQUOTE', 'UNQUOTE')
  utt = utt.replace('.PERIOD', "PERIOD")

  utt = re.sub(r'\([^)]*\)', '', utt)
  utt = re.sub(r'<[^)]*>', '', utt)

  utt = utt.replace('.', '')
  utt = utt.replace('-', '')
  utt = utt.replace('!', '')
  utt = utt.replace(':', '')
  utt = utt.replace(';', '')
  utt = utt.replace('*', '')
  utt = utt.replace('`', '\'')
  utt = utt.replace('~', '')

  assert '~' not in utt
  assert '`' not in utt
  assert '-' not in utt
  assert '_' not in utt
  assert '.' not in utt
  assert ',' not in utt
  assert ':' not in utt
  assert ';' not in utt
  assert '!' not in utt
  assert '?' not in utt
  assert '<' not in utt
  assert '(' not in utt
  assert ')' not in utt
  assert '[' not in utt
  # assert '\'' not in utt
  assert '"' not in utt
  assert '*' not in utt

  # Remove double spaces.
  utt = ' '.join(filter(bool, utt.split(' ')))

  return [uttid, utt]


def token_model_add_token(token_model_proto, token_id, token_string):
  token = token_model_proto.tokens.add()
  token.token_id = int(token_id)
  token.token_string = str(token_string)


def create_phone_token_model(phones_txt):
  token_model_proto = token_model_pb2.TokenModelProto()
  
  token_model_proto.token_sos = 0
  token_model_proto.token_string_sos = "<S>"
  token_model_add_token(
      token_model_proto, token_model_proto.token_sos,
      token_model_proto.token_string_sos)

  token_model_proto.token_eos = 1
  token_model_proto.token_string_eos = "</S>"
  token_model_add_token(
      token_model_proto, token_model_proto.token_eos,
      token_model_proto.token_string_eos)

  token_model_proto.token_eow = 2
  token_model_proto.token_string_eow = " "
  token_model_add_token(
      token_model_proto, token_model_proto.token_eow,
      token_model_proto.token_string_eow)


  token_model_proto.token_unk = 3
  token_model_proto.token_string_unk = "<UNK>"
  token_model_add_token(
      token_model_proto, token_model_proto.token_unk,
      token_model_proto.token_string_unk)

  token_model_proto.token_blank = 4
  token_model_proto.token_string_blank = "<BLANK>"
  token_model_add_token(
      token_model_proto, token_model_proto.token_blank,
      token_model_proto.token_string_blank)

  token_id = 5
  token_map = {}

  phones = load_phones(phones_txt)
  for phone in phones:
    token_model_add_token(token_model_proto, token_id, phone)
    token_map[phone] = token_id

    token_id = token_id + 1

  with open("speech4/conf/timit/token_model.pbtxt", "w") as proto_file:
    proto_file.write(str(token_model_proto))

  return token_model_proto, token_map


def is_vowel(c):
  vowels = ["A", "E", "I", "O", "U", "Y"]
  return c in vowels


def count_words(s):
  return s.count(" ")


def count_vowels(s):
  vowels = 0
  for c in s:
    vowels += is_vowel(c)
  return vowels


def convert(
    kaldi_scp, kaldi_txt, tf_records):
  # Load the token model.
  token_model_proto = token_model_pb2.TokenModelProto()

  character_to_token_map = {}
  token_model_pbtxt = "/data-local/wchan/speech4/speech4/conf/token_model_character_simple.pbtxt"
  with open(token_model_pbtxt, "r") as proto_file:
    google.protobuf.text_format.Merge(proto_file.read(), token_model_proto)

  for token in token_model_proto.tokens:
    character_to_token_map[token.token_string] = token.token_id

  text_map = {}
  lines = [line.strip() for line in open(kaldi_txt, 'r')]
  sorted_uttids = []
  for line in lines:
    [uttid, utt] = normalize_text_wsj(line)
    text_map[uttid] = utt

  tf_record_writer = tf.python_io.TFRecordWriter(tf_records)
  kaldi_feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(kaldi_scp)

  time_factor = 2
  utterance_count = 0
  features_width = 0
  features_len_total = 0
  features_len_max = 0
  tokens_len_total = 0
  tokens_len_max = 0
  feature_token_ratio_min = 10
  vowel_count = 0
  word_count = 0
  pad_min4 = 1e8
  pad_max4 = 0
  pad_sum4 = 0
  examples = []
  for uttid, feats in kaldi_feat_reader:
    text = text_map[uttid]
    tokens = [character_to_token_map[c] for c in text]

    features_len = feats.shape[0]
    features_len_max = max(features_len_max, features_len)
    features_len_total += features_len
    features_width = feats.shape[1]

    tokens_len = len(tokens)
    tokens_len_max = max(tokens_len_max, tokens_len)
    tokens_len_total += tokens_len

    if tokens_len:
      feature_token_ratio_min = min(feature_token_ratio_min, feats.shape[0] / tokens_len)

    vowel_count += count_vowels(text)
    word_count += count_words(text)

    pad4 = features_len / time_factor - tokens_len
    if pad4 <= 0:
      pad4 = features_len / time_factor - tokens_len
    if pad4 <= 0:
      print "skipping %s %s" % (uttid, text)

    if pad4 > 0:
      pad_min4 = min(pad_min4, pad4)
      pad_max4 = max(pad_max4, pad4)
      pad_sum4 += pad4

      example = tf.train.Example(features=tf.train.Features(feature={
          'features_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats.shape[0]])),
          'features': tf.train.Feature(float_list=tf.train.FloatList(value=feats.flatten('C').tolist())),
          'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
          'uttid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(uttid)])),
          'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))}))
    tf_record_writer.write(example.SerializeToString())

    utterance_count += 1
  print("utterance_count: %d" % utterance_count)
  print("features_width: %d" % features_width)
  print("features_len_avg: %f" % (float(features_len_total) / float(utterance_count)))
  print("features_len_total: %d" % features_len_total)
  print("features_len_max: %d" % features_len_max)
  print("tokens_len_total: %d" % tokens_len_total)
  print("tokens_len_avg: %f" % (float(tokens_len_total) / float(utterance_count)))
  print("tokens_len_max: %d" % tokens_len_max)
  print("feature_token_ratio_min: %f" % feature_token_ratio_min)
  print("vowel_count: %d" % vowel_count)
  print("vowel_ratio: %f" % (float(vowel_count) / float(features_len_total / time_factor)))
  print("word_count: %d" % word_count)
  print("word_ratio: %f" % (float(word_count) / float(features_len_total / time_factor)))
  print("pad_min4: %d" % pad_min4)
  print("pad_max4: %d" % pad_max4)
  print("pad_sum4: %d" % pad_sum4)
  print("pad_avg4: %f" % (float(pad_sum4) / float(utterance_count)))
  print("pad_ratio: %f" % (float(pad_sum4) / float(features_len_total / time_factor)))

if __name__ == '__main__':
  main()
