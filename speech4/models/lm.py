#!/usr/bin/env python

import google
import numpy as np
import os.path
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gru_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.gen_user_ops import s4_parse_utterance
import tensorflow.core.framework.token_model_pb2 as token_model_pb2
import time


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('device', 0,
                            """The GPU device to use (set to negative to use CPU).""")

tf.app.flags.DEFINE_integer('batch_size', 32,
                            """Number of utterances to process in a batch.""")

tf.app.flags.DEFINE_integer('features_width', 123,
                            """Size of each feature.""")

tf.app.flags.DEFINE_integer('features_len_max', 2560,
                            """Maximum number of features in an utterance.""")
tf.app.flags.DEFINE_integer('tokens_len_max', 50,
                            """Maximum number of tokens in an utterance.""")

tf.app.flags.DEFINE_integer('vocab_size', 64,
                            """Token vocabulary size.""")
tf.app.flags.DEFINE_integer('embedding_size', 64,
                            """Token vocabulary size.""")

tf.app.flags.DEFINE_integer('encoder_cell_size', 256,
                            """Encoder cell size.""")
tf.app.flags.DEFINE_integer('decoder_cell_size', 512,
                            """Decoder cell size.""")
tf.app.flags.DEFINE_integer('attention_embedding_size', 256,
                            """Attention embedding size.""")

tf.app.flags.DEFINE_float('max_gradient_norm', 5.0,
                           """Maximum gradient norm.""")

tf.app.flags.DEFINE_float('learning_rate', 10.0,
                           """Learning rate.""")

tf.app.flags.DEFINE_string('logdir', '/tmp',
                           """Path to our outputs and logs.""")

class LASModel(object):
  def __init__(self, dataset, batch_size, features_width, features_len_max,
      vocab_size, embedding_size, tokens_len_max, encoder_cell_size,
      decoder_cell_size, attention_embedding_size, max_gradient_norm,
      learning_rate):
    self.dataset = dataset
    self.batch_size = batch_size
    
    self.features_width = features_width
    self.features_len_max = features_len_max
    self.vocab_size = vocab_size

    self.embedding_size = embedding_size
    self.tokens_len_max = tokens_len_max
    self.encoder_cell_size = encoder_cell_size
    self.decoder_cell_size = decoder_cell_size
    self.attention_embedding_size = attention_embedding_size

    self.max_gradient_norm = max_gradient_norm
    self.learning_rate = learning_rate

    self.step_total = 0
    self.step_time_total = 0

    self.global_step = tf.Variable(0, trainable=False)

    # Create the inputs.
    self.create_graph_inputs()

    # Create the decoder.
    self.create_decoder()

    # Create the loss.
    self.create_loss()

    # Create the optimizer.
    self.create_optimizer()

    self.saver = tf.train.Saver(tf.all_variables())


  def create_graph_inputs(self):
    dataset_map = {}
    if self.dataset == 'train_si284':
      self.dataset = 'speech4/data/train_si284.tfrecords'
      self.dataset_size = 37416
    elif self.dataset == 'test_dev93':
      self.dataset = 'speech4/data/test_dev93.tfrecords'
      self.dataset_size = 503
    elif self.dataset == 'test_eval92':
      self.dataset = 'speech4/data/test_eval92.tfrecords'
      self.dataset_size = 333
    elif self.dataset == 'ptb_train':
      self.dataset = 'speech4/data/ptb_train.tfrecords'
      self.dataset_size = 42068
    elif self.dataset == 'ptb_valid':
      self.dataset = 'speech4/data/ptb_valid.tfrecords'
      self.dataset_size = 3370
    elif self.dataset == 'ptb_test':
      self.dataset = 'speech4/data/ptb_test.tfrecords'
      self.dataset_size = 3761
    filename_queue = tf.train.string_input_producer([self.dataset])

    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    serialized = tf.train.shuffle_batch(
        [serialized], batch_size=self.batch_size, num_threads=2, capacity=self.batch_size * 4 + 512,
        min_after_dequeue=512, seed=1000)
    
    # Parse the batched of serialized strings into the relevant utterance features.
    self.features, self.features_len, _, self.text, self.tokens, self.tokens_len, self.tokens_weights, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.features_len_max,
        tokens_len_max=self.tokens_len_max + 1)

    # Add the shape to the features.
    for feature in self.features:
      feature.set_shape([self.batch_size, self.features_width])
    for token in self.tokens:
      token.set_shape([self.batch_size])


  def create_decoder(self):
    start_time = time.time()

    with vs.variable_scope("embedding" or scope):
      tokens = self.tokens[:-1]
      embeddings = []
      with tf.device("/cpu:0"):
        sqrt3 = np.sqrt(3)
        embedding = vs.get_variable(
            "embedding", [self.vocab_size, self.embedding_size],
            initializer=tf.random_uniform_initializer(-sqrt3, sqrt3))

        for token in tokens:
          # Create the embedding layer.
          emb = embedding_ops.embedding_lookup(embedding, token)
          emb.set_shape([self.batch_size, self.embedding_size])
          embeddings.append(emb)

    cell = rnn_cell.GRUCell(self.decoder_cell_size)
    cell = rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)
    self.decoder_states = rnn.rnn(
        cell, embeddings, dtype=tf.float32, sequence_length=self.tokens_len)[0]
    self.logits = self.decoder_states

    print('create_decoder graph time %f' % (time.time() - start_time))


  def create_loss(self):
    start_time = time.time()

    self.losses = []

    logits = self.decoder_states
    targets = self.tokens[1:]
    weights = self.tokens_weights[1:]

    log_perps = seq2seq.sequence_loss(logits, targets, weights, self.vocab_size)
    self.losses.append(log_perps)

    print('create_loss graph time %f' % (time.time() - start_time))


  def create_optimizer(self):
    start_time = time.time()

    params = tf.trainable_variables()
    for param in params:
      print('param: ' + param.name + ': ' + str(param.get_shape()))

    self.gradient_norms = []
    self.updates = []
    #opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    #opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)

    gradients = tf.gradients(self.losses, params)
    if self.max_gradient_norm:
      gradients, norm = tf.clip_by_global_norm(
          gradients, self.max_gradient_norm)
      self.gradient_norms.append(norm)

    self.updates.append(opt.apply_gradients(
        zip(gradients, params), global_step=self.global_step))

    print('create_optimizer graph time %f' % (time.time() - start_time))


  def run_graph(self, sess, targets):
    fetches = []
    for name, target in targets.iteritems():
      if isinstance(target, (list, tuple)):
        fetches.extend(target)
      else:
        fetches.append(target)
    r = sess.run(fetches)

    f = {}
    start = 0
    for name, target in targets.iteritems():
      length = 1
      if isinstance(target, (list, tuple)):
        length = len(target)
      end = start + length
      if isinstance(target, (list, tuple)):
        f[name] = r[start:end]
      else:
        f[name] = r[start]
      start = end
    return f


  def compute_accuracy(self, logits, targets, weights):
    assert len(logits) == len(targets)
    assert len(logits) == len(weights)
    correct = 0.0
    count = 0.0
    for logit, target, weight in zip(logits, targets, weights):
      correct = correct + (np.equal(logit.argmax(axis=1), np.array(target)).astype(float) * weight).sum()
      count = count + weight.sum()

    return correct / count

  def step(self, sess, forward_only, epoch=False):
    if epoch:
      steps_per_epoch = self.dataset_size // self.batch_size

      for s in range(steps_per_epoch):
        self.step(sess, forward_only, epoch=False)

    start_time = time.time()

    if forward_only:
      uttid, text, logperp = sess.run([self.uttid, self.text] + self.losses)
    else:
      targets = {}
      targets['uttid'] = self.uttid
      targets['text'] = self.text
      targets['tokens'] = self.tokens
      targets['tokens_weights'] = self.tokens_weights
      targets['logperp'] = self.losses
      targets['logits'] = self.logits
      targets['updates'] = self.updates
      targets['gradient_norms'] = self.gradient_norms

      fetches = self.run_graph(sess, targets)

      print fetches['gradient_norms']

      logperp = fetches['logperp'][0]
      accuracy = self.compute_accuracy(
          fetches['logits'], fetches['tokens'][1:], fetches['tokens_weights'][1:])

    perplexity = np.exp(logperp)
    tf.scalar_summary('perplexity', perplexity)

    step_time = time.time() - start_time
    self.step_total += 1
    self.step_time_total += step_time

    print 'step_total %d, step_time: %f, accuracy %f, perplexity %f' % (self.step_total, step_time, accuracy, perplexity)


def create_model(sess, dataset, forward_only):
  start_time = time.time()

  #initializer = tf.random_normal_initializer(0.0, 0.1)
  initializer = tf.random_uniform_initializer(-0.1, 0.1)
  with tf.variable_scope("model", initializer=initializer):
    model = LASModel(
        dataset, FLAGS.batch_size, FLAGS.features_width, FLAGS.features_len_max,
        FLAGS.vocab_size, FLAGS.embedding_size, FLAGS.tokens_len_max,
        FLAGS.encoder_cell_size, FLAGS.decoder_cell_size,
        FLAGS.attention_embedding_size, FLAGS.max_gradient_norm,
        FLAGS.learning_rate)

  tf.add_check_numerics_ops()
  sess.run(tf.initialize_all_variables())
  tf.train.start_queue_runners(sess=sess)

  print('create_model graph time %f' % (time.time() - start_time))

  return model


def run_train():
  tf.set_random_seed(1000)

  device = '/gpu:%d' % FLAGS.device if FLAGS.device >= 0 else '/cpu:0'
  with tf.device(device):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      model = create_model(sess, 'ptb_train', False)

      summary_op = tf.merge_all_summaries()

      summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
      summary_writer.flush()

      model.step(sess, forward_only=False, epoch=True)
      for update in updates:
        print updates

      summary_str = sess.run(summary_op)
      summary_writer.add_summary(summary_str, 0)


def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
