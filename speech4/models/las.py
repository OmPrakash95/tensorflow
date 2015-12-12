#!/usr/bin/env python

import numpy as np
import os.path
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
import time


def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  s4_parse_utterance(
      serialized_example,
      features_len_max=2560,
      tokens_len_max=100)

def input():
  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(['speech4/data/train_si284.tfrecords'])

    read_and_decode(filename_queue)


def run_train():
  with tf.Graph().as_default():
    input()


def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
