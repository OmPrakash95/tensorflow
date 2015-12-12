#!/usr/bin/env python

import numpy as np
import os.path
import tensorflow as tf
import tensorflow.python.platform
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 8,
                            """Number of utterances to process in a batch.""")



class LASModel(object):
  def __init__(self, batch_size):
    self.batch_size = batch_size

    self.global_step = tf.Variable(0, trainable=False)

    self.create_graph_inputs()


    self.saver = tf.train.Saver(tf.all_variables())


  def create_graph_inputs(self):
    filename_queue = tf.train.string_input_producer(['speech4/data/train_si284.tfrecords'])
    features, features_len, text, tokens, tokens_len, uttid = read_and_decode(filename_queue)

    self.text = text


  def step(self, sess, forward_only):
    return sess.run(self.text)



def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)

  return s4_parse_utterance(serialized_example, features_len_max=2560, tokens_len_max=100)


def create_model(sess, forward_only):
  model = LASModel(FLAGS.batch_size)
  sess.run(tf.initialize_all_variables())
  tf.train.start_queue_runners(sess=sess)
  return model


def run_train():
  with tf.Session() as sess:
    model = create_model(sess, False)

    x = model.step(sess, False)

    print x



def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
