#!/usr/bin/env python

################################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
################################################################################

import argparse
import codecs
import google
import kaldi_io
import numpy as np
import os
import re
import sys
import tensorflow as tf
import tensorflow.core.framework.token_model_pb2 as token_model_pb2

def main():
  parser = argparse.ArgumentParser(description='SPEECH3 (C) 2015 William Chan <williamchan@cmu.edu>')
  parser.add_argument('--kaldi_cmvn_scp', type=str)
  parser.add_argument('--kaldi_scp', type=str)
  parser.add_argument('--kaldi_txt', type=str)
  parser.add_argument('--kaldi_utt2spk', type=str)
  parser.add_argument('--remove_space', dest='remove_space', action='store_true')
  parser.add_argument('--sort', dest='sort', action='store_true')
  parser.add_argument('--tf_records', type=str)
  parser.add_argument('--token_model_pbtxt', type=str, default='speech4/conf/token_model_gale_chinese.pbtxt')
  parser.add_argument('--tokens_max', type=int, default=1000)
  args   = vars(parser.parse_args())

  convert(
      args['kaldi_scp'], args['kaldi_txt'], args['tf_records'],
      args['token_model_pbtxt'], args['tokens_max'], args['remove_space'],
      args['sort'])


def convert(
    kaldi_scp, kaldi_txt, tf_records, token_model_pbtxt, tokens_max_filter,
    remove_space, sort):
  # Load the token model.
  token_model_proto = token_model_pb2.TokenModelProto()

  character_to_token_map = {}
  with open(token_model_pbtxt, 'r') as proto_file:
    google.protobuf.text_format.Merge(proto_file.read(), token_model_proto)

  for token in token_model_proto.tokens:
    character_to_token_map[token.token_string] = token.token_id

  tokens_max = 0
  tokens_total = 0
  # Read the text file into utterance_map.
  utterance_map = {}
  lines = [line.strip() for line in codecs.open(kaldi_txt, 'r', 'utf-8')]
  sorted_uttids = []
  for line in lines:
    cols = line.split(" ", 1)
    uttid = cols[0]
    utt = cols[1]

    # Remove the space -- only do this for chinese!
    if remove_space:
      utt_normalized = ''.join(utt.split(' '))
      utt = utt_normalized

    tokens_max = max(tokens_max, len(utt))
    tokens_total = tokens_total + len(utt)

    utterance_map[uttid] = utt
    sorted_uttids.append((uttid, utt))
  sorted_uttids = sorted(sorted_uttids, key=lambda x: len(x[1]))

  # Process the utterances.
  tf_record_writer = tf.python_io.TFRecordWriter(tf_records)
  utt_count = 0
  kaldi_feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(kaldi_scp)
  frames_max = 0
  frames_total = 0
  tokens_frames_count_map = {}

  sorted_protos = {}
  utterance_count = 0
  for uttid, feats in kaldi_feat_reader:
    uttid = unicode(uttid, "utf-8")

    # Corresponding text transcript.
    text = utterance_map[uttid]

    tokens_count = len(text)
    if tokens_count not in tokens_frames_count_map:
      tokens_frames_count_map[tokens_count] = []
    tokens_frames_count_map[tokens_count].append(feats.shape[0])

    # Corresponding tokens.
    tokens = [token_model_proto.token_sos] * 2 + [character_to_token_map[c] for c in text] + [token_model_proto.token_eos]
    assert len(tokens)

    if tokens_count <= tokens_max_filter:
      frames_max = max(frames_max, feats.shape[0])
      frames_total = frames_total + feats.shape[0]

      text = text.encode("utf8")
      uttid = uttid.encode("ascii", "ignore")
      example = tf.train.Example(features=tf.train.Features(feature={
          'features_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats.shape[0]])),
          'features': tf.train.Feature(float_list=tf.train.FloatList(value=feats.flatten('C').tolist())),
          'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
          'uttid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(uttid)])),
          'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))}))
      utterance_count = utterance_count + 1
      if sort:
        if tokens_count not in sorted_protos:
          sorted_protos[tokens_count] = []
        sorted_protos[tokens_count].append(example.SerializeToString())
      else:
        tf_record_writer.write(example.SerializeToString())

    utt_count = utt_count + 1
    if utt_count % 100 == 0:
      print 'processed %d out of %d, real %d' % (utt_count, len(utterance_map), utterance_count)

  if sort:
    assert len(sorted_protos)
    for tokens_count, protos in sorted(sorted_protos.iteritems()):
      for proto in protos:
        tf_record_writer.write(proto)
  else:
    assert len(sorted_protos) == 0

  print 'tokens_max: %d' % tokens_max
  print 'frames_max: %d' % frames_max
  print 'tokens_total: %d' % tokens_total
  print 'frames_total: %d' % frames_total
  print 'utt_count: %d' % utt_count
  print 'utterance_count: %d' % utterance_count

if __name__ == '__main__':
  main()
