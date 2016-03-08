#!/usr/bin/env python

################################################################################
# Copyright 2016 William Chan <williamchan@cmu.edu>.
################################################################################

import argparse
import google
import kaldi_io
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
  parser.add_argument('--phones', type=str)
  parser.add_argument('--remap', type=str)
  parser.add_argument('--tf_records', type=str)
  args   = vars(parser.parse_args())

  convert(
      args['kaldi_scp'], args['kaldi_txt'], args['phones'], args['remap'],
      args['tf_records'])


def load_phones(phones_txt):
  lines = [line.strip() for line in open(phones_txt, 'r')]
  phones = []
  for line in lines:
    cols = line.split(" ")
    assert len(cols) == 2

    phone = cols[0]
    phones.append(phone)
  return phones


def create_phone_token_model(phones_txt):
  token_model_proto = token_model_pb2.TokenModelProto()
  token_model_proto.token_sos = 0
  token_model_proto.token_string_sos = "<S>"
  token_model_proto.token_eos = 1
  token_model_proto.token_string_eos = "</S>"
  token_model_proto.token_eow = 2
  token_model_proto.token_string_eow = " "
  token_model_proto.token_unk = 3
  token_model_proto.token_string_unk = "<UNK>"
  token_model_proto.token_blank = 4
  token_model_proto.token_string_blank = "<BLANK>"

  token_id = 5
  token_map = {}

  phones = load_phones(phones_txt)
  for phone in phones:
    token = token_model_proto.tokens.add()
    token.token_id = token_id
    token.token_string = phone

    token_map[token.token_string] = token.token_id

    token_id = token_id + 1

  with open("speech4/conf/timit/token_model.pbtxt", "w") as proto_file:
    proto_file.write(str(token_model_proto))

  return token_map


def load_text_map(kaldi_txt, phone_map):
  lines = [line.strip() for line in open(kaldi_txt, 'r')]
  text_map = {}
  for line in lines:
    cols = line.split(" ", 1)
    uttid = cols[0]
    text = cols[1]
    text_map[uttid] = text
  return text_map


def extract_text_line(line):
  cols = line.split(" ")
  uttid = cols[0]


def convert(
    kaldi_scp, kaldi_txt, phones_txt, remap_txt, tf_records):
  phone_map = create_phone_token_model(phones_txt)
  text_map = load_text_map(kaldi_txt, phone_map)

  tf_record_writer = tf.python_io.TFRecordWriter(tf_records)
  kaldi_feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(kaldi_scp)

  for uttid, feats in kaldi_feat_reader:
    text = text_map[uttid]
    tokens = [phone_map[phone] for phone in text.split(" ")]

    example = tf.train.Example(features=tf.train.Features(feature={
        'features_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats.shape[0]])),
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=feats.flatten('C').tolist())),
        'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        'uttid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(uttid)])),
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))}))
    tf_record_writer.write(example.SerializeToString())

if __name__ == '__main__':
  main()
