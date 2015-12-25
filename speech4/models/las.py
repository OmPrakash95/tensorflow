#!/usr/bin/env python

import math
import numpy as np
import os.path
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
import time


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

tf.app.flags.DEFINE_integer('encoder_cell_size', 256,
                            """Encoder cell size.""")
tf.app.flags.DEFINE_integer('decoder_cell_size', 256,
                            """Decoder cell size.""")
tf.app.flags.DEFINE_integer('attention_embedding_size', 128,
                            """Attention embedding size.""")

tf.app.flags.DEFINE_float('max_gradient_norm', 1.0,
                           """Maximum gradient norm.""")

tf.app.flags.DEFINE_float('learning_rate', 0.001,
                           """Learning rate.""")

tf.app.flags.DEFINE_string('logdir', '/tmp',
                           """Path to our outputs and logs.""")

class LASModel(object):
  def __init__(self, dataset, logdir, batch_size, features_width,
      features_len_max, vocab_size, embedding_size, tokens_len_max,
      encoder_cell_size, decoder_cell_size, attention_embedding_size,
      max_gradient_norm, learning_rate):
    self.dataset = dataset
    self.logdir = logdir

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
    self.create_input_layer()

    # Create the encoder-encoder.
    self.create_encoder()
    self.create_decoder()

    # Create the loss.
    self.create_loss()

    # Create the optimizer.
    self.create_optimizer()

    self.saver = tf.train.Saver(tf.all_variables())


  def create_input_layer(self):
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
    filename_queue = tf.train.string_input_producer([self.dataset])

    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    serialized = tf.train.shuffle_batch(
        [serialized], batch_size=self.batch_size, num_threads=2, capacity=self.batch_size * 4 + 512,
        min_after_dequeue=512, seed=1000)
    
    # Parse the batched of serialized strings into the relevant utterance features.
    self.features, self.features_len, _, self.text, self.tokens, self.tokens_len, self.tokens_weights, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.features_len_max, tokens_len_max=self.tokens_len_max + 1)

    # Add the shape to the features.
    for feature in self.features:
      feature.set_shape([self.batch_size, self.features_width])
    for token in self.tokens:
      token.set_shape([self.batch_size])


  def create_encoder(self):
    start_time = time.time()

    self.encoder_states = [[self.features, self.features_len]]

    self.create_encoder_layer()
    self.create_encoder_layer()
    self.create_encoder_layer(subsample_input=2)
    self.create_encoder_layer(subsample_input=2)

    print('create_encoder graph time %f' % (time.time() - start_time))


  def create_encoder_layer(self, subsample_input=1, use_monolithic=True):
    with vs.variable_scope('encoder_layer_%d' % (len(self.encoder_states))):
      sequence_len_factor = tf.constant(subsample_input, shape=[self.batch_size], dtype=tf.int64)
      sequence_len = tf.div(self.encoder_states[-1][1], sequence_len_factor)
      if use_monolithic:
        # xs = self.encoder_states[-1][0]
        # if subsample_input > 1:
        #   xs = [xs[i:i + subsample_input] for i in range(0, len(xs), subsample_input)]
        #   xs = [array_ops.concat(1, x) for x in xs]
        xs = self.encoder_states[-1][0][0::subsample_input]
        self.encoder_states.append([gru_ops.gru(
            cell_size=self.encoder_cell_size, sequence_len=sequence_len, xs=xs)[-1], sequence_len])
      else:
        self.encoder_states.append([rnn.rnn(
            rnn_cell.GRUCell(self.encoder_cell_size),
            self.encoder_states[-1][0][0::subsample_input], dtype=tf.float32,
            sequence_length=sequence_len)[0], sequence_len])

      for encoder_state in self.encoder_states[-1][0]:
        encoder_state.set_shape([self.batch_size, self.encoder_cell_size])

  def create_decoder(self):
    start_time = time.time()

    self.decoder_states = []
    self.decoder_states.append(self.tokens[:-1])

    attention_states = []
    if len(self.encoder_states) > 1:
      attention_states = [
          array_ops.reshape(
              e, [-1, 1, self.encoder_cell_size], name='reshape_%d' % idx)
          for idx, e in enumerate(self.encoder_states[-1][0])]
      attention_states = array_ops.concat(1, attention_states)

    #self.create_decoder_layer(attention_states=attention_states)
    #self.create_decoder_layer_v1(output_projection=True)
    self.create_decoder_sequence(attention_states=attention_states)

    print('create_decoder graph time %f' % (time.time() - start_time))


  def create_decoder_layer_v1(
      self, attention_states=None, output_projection=None, scope=None):
    with vs.variable_scope('decoder_layer_%d' % (len(self.decoder_states))):
      decoder_initial_state = tf.constant(
          0, shape=[self.batch_size, self.decoder_cell_size], dtype=tf.float32)
      decoder_initial_state.set_shape([self.batch_size, self.decoder_cell_size])

      cell = rnn_cell.GRUCell(self.decoder_cell_size)
      #cell = rnn_cell.GRUCellv2(self.decoder_cell_size, sequence_len=self.tokens_len)
      if output_projection == True:
        cell = rnn_cell.OutputProjectionWrapper(cell, self.vocab_size)

      if attention_states:
        if len(self.decoder_states) == 1:
          self.decoder_states.append(seq2seq.embedding_attention_decoder(
              self.decoder_states[-1], decoder_initial_state, attention_states,
              cell, self.embedding_size, self.vocab_size,
              attention_states_sequence_len=self.encoder_states[-1][1],
              sequence_length=self.tokens_len)[0])
        else:
          self.decoder_states.append(seq2seq.attention_decoder(
              self.decoder_states[-1], decoder_initial_state, attention_states,
              cell, attention_states_sequence_len=self.encoder_states[-1][1],
              sequence_length=self.tokens_len)[0])
      else:
        if len(self.decoder_states) == 1:
          self.decoder_states.append(seq2seq.embedding_rnn_decoder(
              self.decoder_states[-1], decoder_initial_state, cell,
              self.embedding_size, self.vocab_size,
              sequence_length=self.tokens_len)[0])
        else:
          self.decoder_states.append(seq2seq.rnn_decoder(
              self.decoder_states[-1], decoder_initial_state, cell,
              sequence_length=self.tokens_len)[0])


  def create_decoder_sequence(self, attention_states, scope=None):
    with vs.variable_scope("decoder_layer" or scope):
      batch_size = self.batch_size
      attn_length = attention_states.get_shape()[1].value
      attn_size = attention_states.get_shape()[2].value

      encoder_states = array_ops.reshape(
          attention_states, [-1, attn_length, 1, attn_size])
      with vs.variable_scope("encoder_embedding"):
        k = vs.get_variable("W", [1, 1, attn_size, self.attention_embedding_size])
      encoder_embedding = nn_ops.conv2d(encoder_states, k, [1, 1, 1, 1], "SAME")

      self.decoder_states = []
      states = []
      attentions = []
      for decoder_time_idx in range(len(self.tokens) - 1):
        if decoder_time_idx > 0:
          vs.get_variable_scope().reuse_variables()

        # RNN-Attention Decoder.
        (outputs, states, attentions) = self.create_decoder_cell(
            decoder_time_idx, states, attentions, encoder_states,
            encoder_embedding)
 
        # Logit.
        with vs.variable_scope("Logit"):
          logit = nn_ops.xw_plus_b(
              outputs[-1],
              vs.get_variable("Matrix", [outputs[-1].get_shape()[1].value, self.vocab_size]),
              vs.get_variable("Bias", [self.vocab_size]), name="Logit_%d" % decoder_time_idx)
          self.decoder_states.append(logit)


  def create_decoder_cell(
      self, decoder_time_idx, states, attentions, encoder_states,
      encoder_embedding, scope=None):
    batch_size = self.batch_size
    attention_embedding_size = self.attention_embedding_size
    decoder_cell_size = self.decoder_cell_size

    # Create the embedding layer.
    with tf.device("/cpu:0"):
      sqrt3 = np.sqrt(3)
      embedding = vs.get_variable(
          "embedding", [self.vocab_size, self.embedding_size],
          initializer=tf.random_uniform_initializer(-sqrt3, sqrt3))

    emb = embedding_ops.embedding_lookup(
        embedding, self.tokens[decoder_time_idx])
    emb.set_shape([batch_size, self.embedding_size])

    def create_attention(decoder_state):
      with vs.variable_scope("attention"):
        U = vs.get_variable("U", [decoder_cell_size, attention_embedding_size])
        b = vs.get_variable("b", [attention_embedding_size])
        v = vs.get_variable("v", [attention_embedding_size])

        # Decoder embedding.
        decoder_state.set_shape([batch_size, decoder_cell_size])
        d = nn_ops.xw_plus_b(decoder_state, U, b)
        d = array_ops.reshape(d, [batch_size, 1, 1, attention_embedding_size])

        # Energies.
        e = math_ops.reduce_sum(
            v * math_ops.tanh(encoder_embedding + d), [2, 3])
        e = gru_ops.attention_mask(self.encoder_states[-1][1], e)

        # Alignment.
        a = nn_ops.softmax(e)

        # Context.
        c = math_ops.reduce_sum(
            array_ops.reshape(a, [batch_size, len(self.encoder_states[-1][0]), 1, 1]) * encoder_states,
            [1, 2])
        c = array_ops.reshape(c, [batch_size, self.encoder_cell_size])
        return c

    new_states = [None]
    new_attentions = [None]
    new_outputs = [emb]
    def create_gru_cell(attention):
      stack_idx = len(new_states)

      # If empty, create new state.
      if len(states):
        state = states[stack_idx]
      else:
        state = array_ops.zeros([batch_size, decoder_cell_size], tf.float32)

      # The input to this layer is the output of the previous layer.
      x = new_outputs[stack_idx - 1]
      # If the previous timestep has an attention context, concat it.
      if attention:
        if len(attentions):
          x = array_ops.concat(1, [x, attentions[stack_idx]])
        else:
          x = array_ops.concat(1, [x, array_ops.zeros([
              batch_size, self.encoder_cell_size], tf.float32)])
        x.set_shape([batch_size, new_outputs[stack_idx - 1].get_shape()[1].value + self.encoder_cell_size])

      # Create our GRU cell.
      _, _, _, _, h = gru_ops.gru_cell(
          decoder_cell_size, self.tokens_len, state, x, time_idx=decoder_time_idx)
      h.set_shape([batch_size, decoder_cell_size])

      new_states.append(h)
      if attention:
        c = create_attention(h)
        new_attentions.append(c)
        h = array_ops.concat(1, [h, c])
        h.set_shape(
            [batch_size, self.decoder_cell_size + self.encoder_cell_size])
      else:
        new_attentions.append(None)
      new_outputs.append(h)

    with vs.variable_scope("1"):
      create_gru_cell(attention=True)
    with vs.variable_scope("2"):
      create_gru_cell(attention=False)

    return new_outputs, new_states, new_attentions


  def create_loss(self):
    start_time = time.time()

    self.losses = []

    self.logits = self.decoder_states
    targets = self.tokens[1:]
    weights = self.tokens_weights[1:]

    log_perps = seq2seq.sequence_loss(self.logits, targets, weights, self.vocab_size)
    self.losses.append(log_perps)

    print('create_loss graph time %f' % (time.time() - start_time))


  def create_optimizer(self):
    start_time = time.time()

    params = tf.trainable_variables()
    for idx, param in enumerate(params):
      print('param %d: %s %s' % (idx, param.name, str(param.get_shape())))

    self.updates = []

    grads = tf.gradients(self.losses, params)
    if self.max_gradient_norm:
      cgrads, norm = clip_ops.clip_by_global_norm(
          grads, self.max_gradient_norm, name="clip_gradients")
      self.gradient_norm = norm

    opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
    #opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
    self.updates.append(opt.apply_gradients(
        zip(cgrads, params), global_step=self.global_step))

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


  def step_epoch(self, sess, forward_only):
    steps_per_epoch = int(math.ceil(self.dataset_size // self.batch_size))
    for s in range(steps_per_epoch):
      self.step(sess, forward_only)
    self.epochs = self.step_total / steps_per_epoch

    self.saver.save(
        sess, os.path.join(self.logdir, 'ckpt'), global_step=self.epochs)

    print("*** epoch done %d ***" % self.epochs)
    print("step_total %d, avg_step_time: %f, accuracy %f, perplexity %f" % (
        self.step_total, avg_step_time, accuracy, perplexity))
    print("*** epoch done %d ***" % self.epochs)


  def step(self, sess, forward_only):
    steps_per_epoch = int(math.ceil(self.dataset_size // self.batch_size))

    start_time = time.time()

    steps_per_report = 100
    report = self.step_total % steps_per_report == 0

    targets = {}
    if forward_only or report:
      targets['uttid'] = self.uttid
      targets['text'] = self.text
      targets['tokens'] = self.tokens
      targets['tokens_weights'] = self.tokens_weights
      targets['logperp'] = self.losses
      targets['logits'] = self.logits

    if not forward_only or report:
      targets['gradient_norm'] = self.gradient_norm

    if not forward_only:
      targets['updates'] = self.updates

    fetches = self.run_graph(sess, targets)

    step_time = time.time() - start_time
    self.step_total += 1
    self.step_time_total += step_time
    self.epochs = self.step_total / steps_per_epoch

    avg_step_time = self.step_time_total / self.step_total

    if report:
      logperp = fetches['logperp'][0]

      perplexity = np.exp(logperp)
      gradient_norm = fetches['gradient_norm']
      
      accuracy = self.compute_accuracy(
          fetches['logits'], fetches['tokens'][1:], fetches['tokens_weights'][1:])

      self.saver.save(
          sess, os.path.join(self.logdir, 'ckpt'), global_step=self.epochs)

      print("step_total %d, avg_step_time: %f, accuracy %f, perplexity %f, gradient_norm %f" % (
          self.step_total, avg_step_time, accuracy, perplexity, gradient_norm))

    return fetches


def create_model(sess, dataset, forward_only):
  start_time = time.time()

  #initializer = tf.random_normal_initializer(0.0, 0.1)
  initializer = tf.random_uniform_initializer(-0.1, 0.1)
  with tf.variable_scope("model", initializer=initializer):
    model = LASModel(
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

  device = '/gpu:%d' % FLAGS.device if FLAGS.device >= 0 else '/cpu:0'
  with tf.device(device):
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      model = create_model(sess, 'train_si284', False)

      summary_writer = tf.train.SummaryWriter(FLAGS.logdir, sess.graph_def)
      summary_writer.flush()

      model.step(sess, forward_only=False)
      model.step_epoch(sess, forward_only=False)


def main(_):
  run_train()


if __name__ == '__main__':
  tf.app.run()
