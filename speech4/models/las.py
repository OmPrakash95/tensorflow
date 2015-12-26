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
from speech4.models import las_model

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('device', 0,
                            """The GPU device to use (set to negative to use CPU).""")

tf.app.flags.DEFINE_string('ckpt', '',
                            """The GPU device to use (set to negative to use CPU).""")

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
                            """Encoder cell size.""")
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


def create_model(sess, dataset, forward_only):
  start_time = time.time()

  #initializer = tf.random_normal_initializer(0.0, 0.1)
  initializer = tf.random_uniform_initializer(-0.1, 0.1)
  with tf.variable_scope("model", initializer=initializer):
    model = las_model.LASModel(
        dataset, FLAGS.logdir, FLAGS.batch_size, FLAGS.features_width,
        FLAGS.features_len_max, FLAGS.vocab_size, FLAGS.embedding_size,
        FLAGS.tokens_len_max, FLAGS.encoder_cell_size, FLAGS.decoder_cell_size,
        FLAGS.attention_embedding_size, FLAGS.max_gradient_norm,
        FLAGS.learning_rate)

  tf.train.write_graph(sess.graph_def, FLAGS.logdir, "graph_def.pbtxt")

  # tf.add_check_numerics_ops()
  if gfile.Exists(FLAGS.ckpt):
    print("Reading model parameters from %s" % FLAGS.ckpt)
    model.saver.restore(sess, ckpt)
  else:
    sess.run(tf.initialize_all_variables())
  tf.train.start_queue_runners(sess=sess)

  print('create_model graph time %f' % (time.time() - start_time))

  return model


def run_train():
  tf.set_random_seed(1000)

  if not FLAGS.logdir:
    FLAGS.logdir = tempfile.mkdtemp()

  print("logdir: %s" % FLAGS.logdir)

  device = '/gpu:%d' % FLAGS.device if FLAGS.device >= 0 else '/cpu:0'
  with tf.device(device):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      model = create_model(sess, 'train_si284', False)

      summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
      summary_writer.flush()

      model.step(sess, forward_only=False)

      while True:
        model.step_epoch(sess, forward_only=False)


def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
