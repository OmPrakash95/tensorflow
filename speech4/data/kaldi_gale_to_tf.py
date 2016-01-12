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
  parser.add_argument('--sort', dest='sort', action='store_true')
  parser.add_argument('--tf_records', type=str)
  parser.add_argument('--token_model_pbtxt', type=str, default='speech4/conf/token_model_gale_chinese.pbtxt')
  args   = vars(parser.parse_args())

  convert(
      args['kaldi_scp'], args['kaldi_txt'], args['tf_records'],
      args['token_model_pbtxt'])


def convert(kaldi_scp, kaldi_txt, tf_records, token_model_pbtxt):
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
  for uttid, feats in kaldi_feat_reader:
    uttid = unicode(uttid, "utf-8")
    # CMVN.
    feats_normalized = feats

    frames_max = max(frames_max, feats.shape[0])
    frames_total = frames_total + feats.shape[0]

    # Corresponding text transcript.
    text = utterance_map[uttid]

    # Corresponding tokens.
    tokens = [token_model_proto.token_sos] * 2 + [character_to_token_map[c] for c in text] + [token_model_proto.token_eos]
    assert len(tokens)

    text = "no_text"
    uttid = uttid.encode("ascii", "ignore")
    example = tf.train.Example(features=tf.train.Features(feature={
        'features_len': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats_normalized.shape[0]])),
        'features': tf.train.Feature(float_list=tf.train.FloatList(value=feats_normalized.flatten('C').tolist())),
        'tokens': tf.train.Feature(int64_list=tf.train.Int64List(value=tokens)),
        'uttid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(uttid)])),
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))}))
    tf_record_writer.write(example.SerializeToString())

    utt_count = utt_count + 1
    print 'processed %d out of %d' % (utt_count, len(utterance_map))

  print 'tokens_max: %d' % tokens_max
  print 'frames_max: %d' % frames_max
  print 'tokens_total: %d' % tokens_total
  print 'frames_total: %d' % frames_total

if __name__ == '__main__':
  main()
