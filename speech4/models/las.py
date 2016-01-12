#!/usr/bin/env python

import google
import math
import numpy as np
import os.path
import random
import string
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
                            """Checkpoint to recover from.""")

tf.app.flags.DEFINE_string('dataset_train', 'train_si284',
                            """Dataset.""")
tf.app.flags.DEFINE_string('dataset_valid', 'test_dev93',
                            """Dataset.""")
tf.app.flags.DEFINE_string('dataset_test', 'test_eval92',
                            """Dataset.""")

tf.app.flags.DEFINE_integer('random_seed', 1000,
                            """Random seed.""")

tf.app.flags.DEFINE_integer('batch_size', 16,
                            """Number of utterances to process in a batch.""")

tf.app.flags.DEFINE_boolean("shuffle", True,
                         """Shuffle the data.""");

tf.app.flags.DEFINE_integer('global_epochs', 0,
                            """Global epochs.""")
tf.app.flags.DEFINE_integer('global_epochs_max', 20,
                            """Global epochs max.""")

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
tf.app.flags.DEFINE_integer('decoder_cell_size', 256,
                            """Decoder cell size.""")
tf.app.flags.DEFINE_integer('attention_embedding_size', 128,
                            """Attention embedding size.""")

tf.app.flags.DEFINE_string("decoder_params", "", """decoder_params proto""")
tf.app.flags.DEFINE_string("model_params", "", """model_params proto""")
tf.app.flags.DEFINE_string("optimization_params", "", """model_params proto""")

# optimization_params
tf.app.flags.DEFINE_string('optimization_params_type', 'adam',
                           """adam, gd""")

tf.app.flags.DEFINE_float('optimization_params_adagrad_learning_rate', 0.1,
                           """Adagrad""")

tf.app.flags.DEFINE_float('optimization_params_adagrad_initial_accumulator_value', 0.1,
                           """Adagrad""")
tf.app.flags.DEFINE_boolean('optimization_params_adagrad_reset', False,
                           """Adagrad""")

tf.app.flags.DEFINE_float('optimization_params_adam_learning_rate', 0.001,
                           """Adam""")
tf.app.flags.DEFINE_float('optimization_params_adam_beta1', 0.9,
                           """Adam""")
tf.app.flags.DEFINE_float('optimization_params_adam_beta2', 0.999,
                           """Adam""")
tf.app.flags.DEFINE_float('optimization_params_adam_epsilon', 1e-08,
                           """Adam""")
tf.app.flags.DEFINE_boolean('optimization_params_adam_reset', False,
                           """Adam""")

tf.app.flags.DEFINE_float('optimization_params_max_gradient_norm', 1.0,
                           """Maximum gradient norm.""")

tf.app.flags.DEFINE_float('optimization_params_gd_learning_rate', 0.1,
                           """Gradient Descent""")

tf.app.flags.DEFINE_float('optimization_params_sample_prob', 0.1,
                           """Sample probability.""")

tf.app.flags.DEFINE_float('optimization_params_encoder_lm_loss_weight', 0.01,
                           """This loss should be perhaps decayed?""")

tf.app.flags.DEFINE_float('optimization_params_gaussian_noise_stddev', 0.0,
                           """Noise on the gradients.""")

tf.app.flags.DEFINE_string("visualization_params", "", """VisualizationParamsProto""")

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

  if os.path.isfile(FLAGS.model_params):
    with open(FLAGS.model_params, "r") as proto_file:
      google.protobuf.text_format.Merge(proto_file.read(), model_params)

  if model_params.attention_params.type == "median":
    if not model_params.attention_params.median_window_l:
      model_params.attention_params.median_window_l = 100
    if not model_params.attention_params.median_window_r:
      model_params.attention_params.median_window_r = 100

  return model_params

def create_optimization_params(global_epochs):
  optimization_params = speech4_pb2.OptimizationParamsProto()
  optimization_params.type = FLAGS.optimization_params_type
  optimization_params.epochs = 1
  optimization_params.shuffle = FLAGS.shuffle

  if optimization_params.type == "adagrad":
    optimization_params.adagrad.learning_rate = FLAGS.optimization_params_adagrad_learning_rate
    optimization_params.adagrad.initial_accumulator_value = FLAGS.optimization_params_adagrad_initial_accumulator_value
    if global_epochs == 0 and FLAGS.optimization_params_adagrad_reset:
      optimization_params.adagrad.reset = True
  elif optimization_params.type == "adam":
    optimization_params.adam.learning_rate = FLAGS.optimization_params_adam_learning_rate
    optimization_params.adam.beta1 = FLAGS.optimization_params_adam_beta1
    optimization_params.adam.beta2 = FLAGS.optimization_params_adam_beta2
    optimization_params.adam.epsilon = FLAGS.optimization_params_adam_epsilon
    if global_epochs == 0 and FLAGS.optimization_params_adam_reset:
      optimization_params.adam.reset = True
  elif optimization_params.type == "gd":
    optimization_params.gd.learning_rate = FLAGS.optimization_params_gd_learning_rate

  optimization_params.max_gradient_norm = FLAGS.optimization_params_max_gradient_norm
  if global_epochs > 0:
    optimization_params.sample_prob = FLAGS.optimization_params_sample_prob

  optimization_params.encoder_lm_loss_weight = FLAGS.optimization_params_encoder_lm_loss_weight
  optimization_params.gaussian_noise_stddev = FLAGS.optimization_params_gaussian_noise_stddev

  if os.path.isfile(FLAGS.optimization_params):
    with open(FLAGS.optimization_params, "r") as proto_file:
      google.protobuf.text_format.Merge(proto_file.read(), optimization_params)

  return optimization_params

def create_visualization_params():
  visualization_params = speech4_pb2.VisualizationParamsProto()
  if os.path.isfile(FLAGS.visualization_params):
    with open(FLAGS.visualization_params, "r") as proto_file:
      google.protobuf.text_format.Merge(proto_file.read(), visualization_params)
  return visualization_params

def create_model(
    sess, ckpt, dataset, forward_only, global_epochs, model_params=None,
    optimization_params=None):
  start_time = time.time()

  initializer = tf.random_uniform_initializer(-0.1, 0.1)

  if not model_params:
    model_params = create_model_params()
  if not optimization_params:
    optimization_params = create_optimization_params(global_epochs)
  visualization_params = create_visualization_params()

  with open(os.path.join(FLAGS.logdir, "model_params.pbtxt"), "w") as proto_file:
    proto_file.write(str(model_params))
  with open(os.path.join(FLAGS.logdir, "optimization_params.pbtxt"), "w") as proto_file:
    proto_file.write(str(optimization_params))

  with tf.variable_scope("model", initializer=initializer):
    model = las_model.LASModel(
        sess, dataset, FLAGS.logdir, ckpt, forward_only, FLAGS.batch_size,
        model_params, optimization_params=optimization_params,
        visualization_params=visualization_params)

  tf.train.write_graph(sess.graph_def, FLAGS.logdir, "graph_def.pbtxt")

  print("create_model graph time %f" % (time.time() - start_time))

  return model

def run(mode, dataset, global_epochs, model_params=None, optimization_params=None):
  # Default device.
  device = '/gpu:%d' % FLAGS.device if FLAGS.device >= 0 else '/cpu:0'

  # Load latest checkpoint or from flags.
  ckpt = tf.train.latest_checkpoint(FLAGS.logdir)
  if not ckpt:
    ckpt = FLAGS.ckpt

  # Dataset alias.
  if dataset == 'train_si284':
    dataset = 'speech4/data/train_si284.tfrecords'
    dataset_size = 37416
  elif dataset == 'test_dev93':
    dataset = 'speech4/data/test_dev93.tfrecords'
    dataset_size = 503
  elif dataset == 'test_eval92':
    dataset = 'speech4/data/test_eval92.tfrecords'
    dataset_size = 333
  elif dataset == "swbd":
    dataset = "speech4/data/swbd.tfrecords"
    dataset_size = 263775
  elif dataset == "eval2000":
    dataset = "speech4/data/eval2000.tfrecords"
    dataset_size = 4458
  elif dataset == "gale_mandarin_train":
    dataset = "speech4/data/gale_mandarin_train.tfrecords"
    dataset_size = 58058
  elif dataset == "gale_mandarin_dev":
    dataset = "speech4/data/gale_mandarin_dev.tfrecords"
    dataset_size = 5191
  else:
    raise Exception("Unknown dataset %s" % dataset)

  # Create our graph.
  with tf.device(device):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      if mode == 'train':
        model = create_model(
            sess, ckpt, dataset, False, global_epochs=global_epochs,
            model_params=model_params, optimization_params=optimization_params)
      elif mode == 'valid':
        model = create_model(
            sess, ckpt, dataset, True, global_epochs=global_epochs,
            model_params=model_params, optimization_params=optimization_params)

      coord = tf.train.Coordinator()
      if mode == 'train' or mode == 'valid':
        model.global_epochs = global_epochs
        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
        summary_writer.flush()

        model.step_epoch(sess, forward_only=(mode != 'train'))

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

        if os.path.isfile(FLAGS.decoder_params):
          with open(FLAGS.decoder_params, "r") as proto_file:
            google.protobuf.text_format.Merge(proto_file.read(), decoder_params)

        decoder = las_decoder.Decoder(
            sess, dataset, dataset_size, FLAGS.logdir, ckpt, decoder_params, model_params)

        threads = []
        for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
          threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

        decoder.decode(sess)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def main(_):
  if not FLAGS.logdir:
    # FLAGS.logdir = tempfile.mkdtemp()
    FLAGS.logdir = os.path.join(
        "exp", "speech4_" + "".join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(8)))
  FLAGS.logdir = os.path.abspath(FLAGS.logdir)
  try:
    os.makedirs(FLAGS.logdir)
  except:
    pass
  print("logdir: %s" % FLAGS.logdir)

  tf.set_random_seed(FLAGS.random_seed)

  for global_epochs in range(FLAGS.global_epochs, FLAGS.global_epochs_max):
    #run('train', FLAGS.dataset_train, global_epochs)
    #run('valid', FLAGS.dataset_valid, global_epochs)
    run('test', FLAGS.dataset_test, global_epochs)

if __name__ == '__main__':
  tf.app.run()
