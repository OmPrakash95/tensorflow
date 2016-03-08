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


def load_phone_map(phones_txt):
  lines = [line.strip() for line in open(phones_txt, 'r')]
  phone_map = {}
  for line in lines:
    cols = line.split(" ")
    assert len(cols) == 2

    phone = cols[0]
    label = int(cols[1]) - 1
    assert label >= 0
    assert label < 48
    phone_map[phone] = label
  return phone_map


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
  phone_map = load_phone_map(phones_txt)
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
