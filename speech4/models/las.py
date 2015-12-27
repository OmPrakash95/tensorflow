#!/usr/bin/env python

import math
import numpy as np
import os.path
import sys
import tempfile
import time

SPEECH4_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
sys.path.append(os.path.join(SPEECH4_ROOT))

import tensorflow as tf
from tensorflow.core.framework import speech4_pb2
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import gru_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
from tensorflow.python.platform import gfile
from speech4.models import las_decoder
from speech4.models import las_model


FLAGS = tf.app.flags.FLAGS


tf.app.flags.DEFINE_integer('device', 0,
                            """The GPU device to use (set to negative to use CPU).""")

tf.app.flags.DEFINE_string('ckpt', '',
                            """The GPU device to use (set to negative to use CPU).""")

tf.app.flags.DEFINE_integer('random_seed', 1000,
                            """Random seed.""")

tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of utterances to process in a batch.""")

tf.app.flags.DEFINE_integer('features_width', 123,
                            """Size of each feature.""")

tf.app.flags.DEFINE_integer('features_len_max', 2560,
                            """Maximum number of features in an utterance.""")
tf.app.flags.DEFINE_integer('tokens_len_max', 256,
                            """Maximum number of tokens in an utterance.""")

tf.app.flags.DEFINE_integer('vocab_size', 64,
                            """Token vocabulary size.""")
tf.app.flags.DEFINE_integer('embedding_size', 32,
                            """Token vocabulary size.""")

tf.app.flags.DEFINE_integer('encoder_cell_size', 384,
                            """encoder cell size.""")
tf.app.flags.DEFINE_integer('decoder_cell_size', 384,
                            """Decoder cell size.""")
tf.app.flags.DEFINE_integer('attention_embedding_size', 128,
                            """Attention embedding size.""")

tf.app.flags.DEFINE_float('max_gradient_norm', 1.0,
                           """Maximum gradient norm.""")

tf.app.flags.DEFINE_float('learning_rate', 0.001,
                           """Learning rate.""")

tf.app.flags.DEFINE_string('logdir', '',
                           """Path to our outputs and logs.""")


# DecoderParams
tf.app.flags.DEFINE_integer('beam_width', 1,
                            """Decoder beam width.""")


def create_model_params():
  model_params = speech4_pb2.ModelParamsProto()
  
  model_params.features_width = FLAGS.features_width
  model_params.features_len_max = FLAGS.features_len_max
  model_params.tokens_len_max = FLAGS.tokens_len_max
  model_params.vocab_size = FLAGS.vocab_size

  model_params.embedding_size = FLAGS.embedding_size
  model_params.encoder_cell_size = FLAGS.encoder_cell_size
  model_params.decoder_cell_size = FLAGS.decoder_cell_size
  model_params.attention_embedding_size = FLAGS.attention_embedding_size

  return model_params

def create_model(sess, ckpt, dataset, forward_only):
  start_time = time.time()

  #initializer = tf.random_normal_initializer(0.0, 0.1)
  initializer = tf.random_uniform_initializer(-0.1, 0.1)

  model_params = create_model_params()

  with open(os.path.join(FLAGS.logdir, 'model_params.pbtxt'), 'w') as proto_file:
    proto_file.write(str(model_params))

  with tf.variable_scope("model", initializer=initializer):
    model = las_model.LASModel(
        sess, dataset, FLAGS.logdir, ckpt, forward_only, FLAGS.batch_size,
        model_params, FLAGS.max_gradient_norm, FLAGS.learning_rate)

  tf.train.write_graph(sess.graph_def, FLAGS.logdir, "graph_def.pbtxt")

  # tf.add_check_numerics_ops()
  # threads = tf.train.start_queue_runners(sess=sess)

  print('create_model graph time %f' % (time.time() - start_time))

  return model


def run(mode, dataset):
  device = '/gpu:%d' % FLAGS.device if FLAGS.device >= 0 else '/cpu:0'

  ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
  if not ckpt:
    ckpt = FLAGS.ckpt

  if dataset == 'train_si284':
    dataset = 'speech4/data/train_si284.tfrecords'
    dataset_size = 37416
  elif dataset == 'test_dev93':
    dataset = 'speech4/data/test_dev93.tfrecords'
    dataset_size = 503
  elif dataset == 'test_eval92':
    dataset = 'speech4/data/test_eval92.tfrecords'
    dataset_size = 333

  with tf.device(device):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      if mode == 'train':
        model = create_model(sess, ckpt, dataset, False)
      elif mode == 'valid':
        model = create_model(sess, ckpt, dataset, True)

      coord = tf.train.Coordinator()
      if mode == 'train' or mode == 'valid':
        model.global_epochs = run.global_epochs
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
        summary_writer.flush()

        model.step_epoch(sess, forward_only=(mode != 'train'))

        if mode == 'train':
          model.global_epochs = run.global_epochs + 1

        summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
        summary_writer.flush()

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
      elif mode == 'test':
        model_params = speech4_pb2.ModelParamsProto()
        model_params_pbtxt = os.path.join(FLAGS.logdir, 'model_params.pbtxt')
        if os.path.isfile(model_params_pbtxt):
          with open(model_params_pbtxt, 'r') as proto_file:
            google.protobuf.text_format.Merge(proto_file.read(), model_params)
        else:
          model_params = create_model_params()

        decoder_params = speech4_pb2.DecoderParamsProto()
        decoder_params.beam_width = FLAGS.beam_width

        decoder = las_decoder.Decoder(
            sess, dataset, dataset_size, FLAGS.logdir, ckpt, decoder_params, model_params)

        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        decoder.decode(sess)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)
run.global_epochs = 0

def main(_):
  if not FLAGS.logdir:
    FLAGS.logdir = tempfile.mkdtemp()
  print("logdir: %s" % FLAGS.logdir)

  tf.set_random_seed(FLAGS.random_seed)

  while True:
    run('train', 'train_si284')
    run('valid', 'test_dev93')


if __name__ == '__main__':
  tf.app.run()
