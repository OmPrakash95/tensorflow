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
  parser.add_argument('--kaldi_alignment', type=str)
  parser.add_argument('--transid_to_phoneid', type=str)
  parser.add_argument('--phones', type=str)
  parser.add_argument('--remap', type=str)
  parser.add_argument('--sort', action="store_true")
  parser.add_argument('--tf_records', type=str)
  args   = vars(parser.parse_args())

  convert(
      args['kaldi_scp'], args['kaldi_txt'], args['kaldi_alignment'],
      args['transid_to_phoneid'], args['phones'], args['remap'], args['sort'],
      args['tf_records'])


def load_transid_to_phoneid(transid_to_phoneid_txt):
  transid_to_phoneid = {}
  if transid_to_phoneid_txt:
    lines = [line.strip() for line in open(transid_to_phoneid_txt, 'r')]
    for line in lines:
      cols = line.split(" ")
      assert len(cols) == 2

      transid = int(cols[0])
      phoneid = int(cols[1]) - 1  # offset by 1.

      transid_to_phoneid[transid] = phoneid
  return transid_to_phoneid


def load_phones(phones_txt):
  lines = [line.strip() for line in open(phones_txt, 'r')]
  phones = []
  for line in lines:
    cols = line.split(" ")
    assert len(cols) == 2

    phone = cols[0]
    phones.append(phone)
  return phones


def load_remap(remap_txt):
  lines = [line.strip() for line in open(remap_txt, 'r')]
  remap_map = {}
  for line in lines:
    cols = line.split(" ")
    for phone in cols[1:]:
      remap_map[phone] = cols[0]
  return remap_map


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


def is_vowel(phone):
  vowels = ["aa", "ae", "ax", "ah", "ao", "eh", "ih", "iy", "uh", "uw", "ay", "aw", "ey", "ow", "oy", "axr", "er"]
  return phone in vowels


def count_vowels(phones):
  count = 0
  for phone in phones:
    if is_vowel(phone):
      count += 1
  return count

def load_text_map(kaldi_txt, phone_map):
  lines = [line.strip() for line in open(kaldi_txt, 'r')]
  text_map = {}
  for line in lines:
    cols = line.split(" ", 1)
    uttid = cols[0]
    text = cols[1]
    text_map[uttid] = text
  return text_map


def load_alignment(kaldi_alignment):
  alignment_map = {}
  if kaldi_alignment:
    kaldi_ali_reader = kaldi_io.SequentialInt32VectorReader(kaldi_alignment)
    for uttid, alignment in kaldi_ali_reader:
      alignment_map[uttid] = alignment.tolist()
  return alignment_map


def remap_alignment_to_phones(alignment_map, transid_to_phoneid_map):
  alignment_in_phones_map = {}
  for uttid, alignment in alignment_map.iteritems():
    # phone_ali is in kaldi indexing
    phone_ali = [transid_to_phoneid_map[x] for x in alignment]
    # token_ali is in speech4 indexing
    token_ali = [x + 5 for x in phone_ali]
    alignment_in_phones_map[uttid] = token_ali
  return alignment_in_phones_map


def visualize(feats):
  fig = plt.figure()
  ax = fig.add_subplot(1, 1, 1)
  ax.set_axis_off()
  ax.imshow(feats.transpose()[:23,:], interpolation="none")
  fig.savefig("fig.png")

  raise Exception("visualize once")


def convert(
    kaldi_scp, kaldi_txt, kaldi_alignment, transid_to_phoneid_txt, phones_txt, remap_txt, sort, tf_records):
  token_model_proto, phone_map = create_phone_token_model(phones_txt)
  # remap_map = load_remap(remap_txt)
  text_map = load_text_map(kaldi_txt, phone_map)

  # We only have this if we have GMM model (for training).
  transid_to_phoneid_map = load_transid_to_phoneid(transid_to_phoneid_txt)
  alignment_map = load_alignment(kaldi_alignment)
  alignment_in_phones_map = remap_alignment_to_phones(alignment_map, transid_to_phoneid_map)

  tf_record_writer = tf.python_io.TFRecordWriter(tf_records)
  kaldi_feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(kaldi_scp)

  features_width = 0
  features_len_total = 0
  features_len_max = 0
  tokens_len_total = 0
  tokens_len_max = 0
  feature_token_ratio_min = 10
  pad_min4 = 1e8
  pad_max4 = 0
  examples = []
  for uttid, feats in kaldi_feat_reader:
    text = text_map[uttid]
    phones = text.split(" ")
    phone_ali = None
    if alignment_in_phones_map:
      phone_ali = alignment_in_phones_map[uttid]
    tokens = [phone_map[phone] for phone in phones]
    vowel_count = count_vowels(phones)
    sil_count = tokens.count(5) + tokens.count(15) + tokens.count(48) + tokens.count(22)
    # tokens = [token_model_proto.token_sos] * 2 + tokens + [token_model_proto.token_eos]
    tokens = [int(token) for token in tokens]
    features_len = feats.shape[0]
    features_len_max = max(features_len_max, features_len)
    features_len_total += features_len
    features_width = feats.shape[1]
    tokens_len = len(tokens)
    tokens_len_max = max(tokens_len_max, tokens_len)
    tokens_len_total += tokens_len
    feature_token_ratio_min = min(feature_token_ratio_min, feats.shape[0] / tokens_len)
    pad_min4 = min(pad_min4, features_len / 4 - tokens_len - vowel_count)
    pad_max4 = max(pad_max4, features_len / 4 - tokens_len - vowel_count)

    # Sanity.
    if phone_ali:
      assert len(phone_ali) == features_len
      tokens_collapsed = [x[0] for x in itertools.groupby(tokens)]
      phone_ali_collapsed = [x[0] for x in itertools.groupby(phone_ali)]
      assert tokens_collapsed == phone_ali_collapsed

    if phone_ali:
      example = tf.train.Example(features=tf.train.Features(feature={
          'features_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats.shape[0]])),
          'features': tf.train.Feature(float_list=tf.train.FloatList(value=feats.flatten('C').tolist())),
          'alignment' : tf.train.Feature(int64_list=tf.train.Int64List(value=phone_ali)),
          'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
          'uttid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(uttid)])),
          'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))}))
    else:
      example = tf.train.Example(features=tf.train.Features(feature={
          'features_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats.shape[0]])),
          'features': tf.train.Feature(float_list=tf.train.FloatList(value=feats.flatten('C').tolist())),
          'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
          'uttid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(uttid)])),
          'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))}))
    examples.append([tokens_len, example])
  print("features_width: %d" % features_width)
  print("features_len_max: %d" % features_len_max)
  print("tokens_len_max: %d" % tokens_len_max)
  print("feature_token_ratio_min: %f" % feature_token_ratio_min)
  print("features_len_total: %d" % features_len_total)
  print("tokens_len_total: %d" % tokens_len_total)
  print("pad_min4: %d" % pad_min4)
  print("pad_max4: %d" % pad_max4)
  if sort:
    examples = sorted(examples, key=lambda x: x[0])
  for tokens_len, example in examples:
    tf_record_writer.write(example.SerializeToString())

if __name__ == '__main__':
  main()
