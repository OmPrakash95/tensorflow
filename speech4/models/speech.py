#!/usr/bin/env python


import google
import math
import numpy as np
import os.path
import random
import string
import time

import tensorflow as tf
from tensorflow.core.framework import speech4_pb2
from tensorflow.python.ops import attention_mask_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_data_flow_ops
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


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer("device", 0,
                            """The GPU device to use.""")

tf.app.flags.DEFINE_string("ckpt", "",
                            """Checkpoint to recover from.""")

tf.app.flags.DEFINE_string("dataset", "wsj",
                            """Dataset.""")

tf.app.flags.DEFINE_integer("random_seed", 1000,
                            """Random seed.""")

tf.app.flags.DEFINE_string("decoder_params", "", """decoder_params proto""")
tf.app.flags.DEFINE_string("model_params", "", """model_params proto""")
tf.app.flags.DEFINE_string("optimization_params", "", """model_params proto""")

tf.app.flags.DEFINE_string("logdir", "",
                           """Path to our outputs and logs.""")

def try_load_proto(proto_path, proto):
  if isinstance(proto_path, str) and os.path.isfile(proto_path):
    with open(proto_path, "r") as proto_file:
      google.protobuf.text_format.Merge(proto_file.read(), proto)
  return proto


class SpeechModel(object):
  def __init__(self, sess, mode, dataset_params, model_params, optimization_params, batch_size=16, seed=1):
    dataset_params = try_load_proto(dataset_params, dataset_params)
    model_params = try_load_proto(model_params, model_params)
    optimization_params = try_load_proto(optimization_params, optimization_params)

    self.dataset_params = dataset_params
    self.model_params = model_params
    self.optimization_params = optimization_params

    self.batch_size = batch_size
    if mode == "test":
      self.batch_size = 1
    self.seed = seed
    tf.set_random_seed(self.seed)

    self.create_model(sess, mode)


  def create_model(self, sess, mode):
    print("creating model...")

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope("model", initializer=initializer):
      self.global_step = tf.Variable(0, trainable=False)

      self.create_input()
      self.create_encoder()
      self.create_decoder(mode)
      self.create_loss()
      if mode == "train": self.create_optimizer()

    print("initializing model...")
    sess.run(tf.initialize_all_variables())

    if gfile.Exists(self.model_params.ckpt):
      self.restore(sess)
    self.saver = tf.train.Saver(tf.all_variables())

  def create_input(self):
    if not os.path.isfile(self.dataset_params.path):
      raise Exception("Invalid dataset: " % str(self.dataset_params))
    assert os.path.isfile(self.dataset_params.path)
    filename_queue = tf.train.string_input_producer([self.dataset_params.path])

    reader = tf.TFRecordReader()
    _, serialized = reader.read(filename_queue)

    if not self.optimization_params or self.optimization_params.shuffle == False:
      serialized = tf.train.batch(
          [serialized], batch_size=self.batch_size, num_threads=2,
          capacity=self.batch_size * 4 + 512)
    else:
      serialized = tf.train.shuffle_batch(
          [serialized], batch_size=self.batch_size, num_threads=2,
          capacity=self.batch_size * 4 + 512, min_after_dequeue=512, seed=self.seed)

    assert self.model_params.features_width
    assert self.model_params.frame_stack
    self.features, _, self.features_len, _, _, self.text, self.tokens, self.tokens_pinyin, self.tokens_len, self.tokens_weights, self.tokens_pinyin_weights, self.uttid = s4_parse_utterance(
        serialized, features_len_max=self.model_params.features_len_max,
        tokens_len_max=self.model_params.tokens_len_max + 1,
        frame_stack=self.model_params.frame_stack,
        frame_skip=self.model_params.frame_skip)

    for feature in self.features:
      feature.set_shape([self.batch_size, self.model_params.features_width * self.model_params.frame_stack])
    for token in self.tokens:
      token.set_shape([self.batch_size])
    self.features_len.set_shape([self.batch_size])


  def create_encoder(self):
    # with vs.variable_scope("encoder"):
    #self.create_dynamic_encoder()
    self.create_monolithic_enocder()

    # Create the encoder embedding.
    encoder_state = self.encoder_states[-1][0]
    attn_len = encoder_state.get_shape()[1].value
    # encoder_state shape is [batch_size, attn_len, 1, attn_size]
    encoder_state = array_ops.reshape(
        encoder_state, [self.batch_size, attn_len, 1,
                        self.model_params.encoder_cell_size])

    # We create this outside the "encoder" vs to support legacy ckpts.
    with vs.variable_scope(self.model_params.encoder_embedding):
      k = vs.get_variable(
          "W", [
              1, 1, self.model_params.encoder_cell_size,
              self.model_params.attention_embedding_size])
    # encoder_embedding shape is [batch_size, attn_len, 1, embedding_size]
    encoder_embedding = nn_ops.conv2d(encoder_state, k, [1, 1, 1, 1], "SAME")
    encoder_embedding.set_shape([
        self.batch_size, attn_len, 1,
        self.model_params.attention_embedding_size])
    self.encoder_embedding = [encoder_embedding, self.encoder_states[-1][1]]


  def create_monolithic_enocder(self):
    print("creating monolithic encoder...")
    self.encoder_states = [[self.features, self.features_len]]

    # Create the encoder layers.
    assert self.model_params.encoder_layer
    for idx, s in enumerate(self.model_params.encoder_layer):
      with vs.variable_scope("encoder_%d" % (idx + 1)) as scope:
        self.create_monolithic_encoder_layer(input_time_stride=int(s), scope=scope)

    # Concat the encoder states.
    encoder_states = self.encoder_states[-1][0]
    encoder_states = [array_ops.reshape(
        e, [self.batch_size, 1, self.model_params.encoder_cell_size],
        name="reshape_%d" % idx)
        for idx, e in enumerate(encoder_states)]
    encoder_states = array_ops.concat(1, encoder_states)
    self.encoder_states.append([encoder_states, self.encoder_states[-1][1]])


  def create_monolithic_encoder_layer(self, input_time_stride=1, scope=None):
    input_time_stride_t = tf.constant(
        input_time_stride, shape=[self.batch_size], dtype=tf.int64)
    sequence_len = tf.div(self.encoder_states[-1][1], input_time_stride_t)

    xs = self.encoder_states[-1][0][0::input_time_stride]
    self.encoder_states.append([gru_ops.gru(
        cell_size=self.model_params.encoder_cell_size,
        sequence_len=sequence_len, xs=xs)[-1], sequence_len])

  def create_dynamic_encoder(self):
    print("creating dynamic encoder...")

    # Concat and reshape our features into 1 tensor of [time_max, batch_size, feature_size].
    self.features = [
        array_ops.reshape(f, [1, self.batch_size, f.get_shape()[1].value]) for f in self.features]
    self.features = array_ops.concat(0, self.features)
    self.features.set_shape(
        [self.model_params.features_len_max, self.batch_size,
         self.model_params.features_width * self.model_params.frame_stack])

    self.encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
        self.model_params.encoder_cell_size)

    # The first encoder state are the features.
    self.encoder_states = [[self.features, self.features_len]]

    # Create the encoder layers.
    assert self.model_params.encoder_layer
    for idx, s in enumerate(self.model_params.encoder_layer):
      with vs.variable_scope("encoder_%d" % idx) as scope:
        self.create_dynamic_encoder_layer(input_time_stride=int(s), scope=scope)

    # Transpose from time major to batch major.
    encoder_state = self.encoder_states[-1][0]
    attn_len = encoder_state.get_shape()[0].value
    encoder_state = array_ops.transpose(encoder_state, [1, 0, 2])
    encoder_state.set_shape([
        self.batch_size, attn_len, self.model_params.encoder_cell_size])
    self.encoder_states.append([encoder_state, self.encoder_states[-1][1]])


  def create_dynamic_encoder_layer(self, input_time_stride=1, scope=None):
    print("creating encoder layer...")
    assert self.encoder_states

    initial_state = self.encoder_cell.zero_state(self.batch_size, tf.float32)
    sequence = self.encoder_states[-1][0]
    sequence_len = self.encoder_states[-1][1]

    if input_time_stride > 1:
      input_time_stride_t = tf.constant(
          input_time_stride, shape=[self.batch_size], dtype=tf.int64)

      sequence = gen_data_flow_ops._tensor_array_subsample(
          stride=input_time_stride, input=sequence)
      sequence_len = tf.div(self.encoder_states[-1][1], input_time_stride_t)
    sequence_max_time = sequence.get_shape()[0].value

    output, state = rnn.dynamic_rnn(
        self.encoder_cell, sequence, sequence_len, initial_state=initial_state,
        time_major=True, scope=scope)
    # output shape is [time, batch_size, cell_size]
    output.set_shape([
        sequence_max_time, self.batch_size,
        self.model_params.encoder_cell_size])
    self.encoder_states.append([output, sequence_len])


  def create_decoder(self, mode):
    self.create_decoder_sequence_legacy(
        attention_states=None,
        encoder_states=self.encoder_states[-1][0],
        encoder_embedding=self.encoder_embedding[0])


  def create_decoder_new(self, mode):
    print("creating decoder...")
    with vs.variable_scope("decoder"):
      self.decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(
          self.model_params.decoder_cell_size)
      self.decoder_cell = tf.nn.rnn_cell.GRUCell(
          self.model_params.decoder_cell_size)

      batch_size = self.batch_size

      if hasattr(self, "encoder_embedding"):
        attn_length = self.encoder_embedding[0].get_shape()[1].value
        attn_size = self.model_params.attention_embedding_size

        encoder_state = array_ops.reshape(
            self.encoder_states[-1][0],
            [self.batch_size, attn_length, 1, self.model_params.encoder_cell_size])

      def attention(query, alignment):
        with vs.variable_scope("attention"):
          v = vs.get_variable("v", [attn_size])

          y = rnn_cell.linear(query, attn_size, True)
          y = array_ops.reshape(y, [batch_size, 1, 1, attn_size])
          # Attention mask is a softmax of v^T * tanh(...).
          e = math_ops.reduce_sum(
              v * math_ops.tanh(self.encoder_embedding[0] + y), [2, 3])

          # Mask it by sequence length.
          if alignment and self.model_params.attention_params.type == "median":
            window_l = self.model_params.attention_params.median_window_l
            window_r = self.model_params.attention_params.median_window_r
            e = attention_mask_ops.attention_mask_median(
                self.encoder_states[-1][1], e, alignment, window_l=window_l,
                window_r=window_r)
          else:
            e = attention_mask_ops.attention_mask(self.encoder_states[-1][1], e)

          a = nn_ops.softmax(e)
          # Now calculate the attention-weighted vector c.
          c = math_ops.reduce_sum(
              array_ops.reshape(a, [batch_size, attn_length, 1, 1]) * encoder_state,
              [1, 2])
          return array_ops.reshape(c, [batch_size, self.model_params.encoder_cell_size]), a

      def attention_rnn_cell(x, state, attn, alignment, sequence_len):
        if attn:
          x = rnn_cell.linear(
              [x] + [attn], self.decoder_cell.input_size, True,
              scope="input_projection")
        output, state = self.decoder_cell(x, state)

        if attn:
          attn, alignment = attention(state, alignment)
          output = rnn_cell.linear(
              [output] + [attn], self.decoder_cell.output_size, True,
              scope="output_projection")
        return output, state, attn, alignment

      logits = []
      probs = []
      alignment = None
      for decoder_time_idx in range(len(self.tokens) - 1):
        if decoder_time_idx > 0:
          vs.get_variable_scope().reuse_variables()

        # Get the input token.
        if mode == "test":
          if decoder_time_idx == 0:
            inp = self.tokens[decoder_time_idx]
          else:
            inp = math_ops.argmax(logits[-1], 1)
        else:
          inp = self.tokens[decoder_time_idx]

        # Embedding.
        with tf.device("/cpu:0"):
          sqrt3 = np.sqrt(3)
          embedding_matrix = vs.get_variable(
              "embedding",
              [self.model_params.vocab_size, self.model_params.embedding_size],
              initializer=tf.random_uniform_initializer(-sqrt3, sqrt3))

          inp = embedding_ops.embedding_lookup(embedding_matrix, inp)

        # Attention RNN cell.
        if decoder_time_idx == 0:
          state = array_ops.zeros([batch_size, self.decoder_cell.state_size])
          attn = array_ops.zeros([batch_size, self.model_params.encoder_cell_size])
        output, state, attn, alignment = attention_rnn_cell(
            inp, state, attn, alignment, self.tokens_len)

        # Logits / prob.
        logit = rnn_cell.linear([output], self.model_params.vocab_size, True)
        prob = nn_ops.softmax(logit)

        logits.append(logit)
        probs.append(prob)
      self.logits = logits
      self.probs = probs


  def create_decoder_sequence_legacy(
      self, attention_states, encoder_states, encoder_embedding, scope=None):
    attn_length = self.encoder_embedding[0].get_shape()[1].value
    self.encoder_state_legacy = array_ops.reshape(
        self.encoder_states[-1][0],
        [self.batch_size, attn_length, 1, self.model_params.encoder_cell_size])

    with vs.variable_scope(self.model_params.decoder_prefix or scope):
      self.decoder_states = []
      self.logits_pinyin = []
      self.prob = []
      self.logprob = []
      states = []
      attentions = []
      alignments = None
      prev_logit = None
      for decoder_time_idx in range(len(self.tokens) - 1):
        if decoder_time_idx > 0:
          vs.get_variable_scope().reuse_variables()

        # RNN-Attention Decoder.
        (outputs, states, attentions, alignments) = self.create_decoder_cell_legacy(
            decoder_time_idx, states, attentions, alignments, encoder_states,
            encoder_embedding, prev_logit)
 
        # Logit.
        with vs.variable_scope("Logit"):
          logit = nn_ops.xw_plus_b(
              outputs[-1],
              vs.get_variable("Matrix", [outputs[-1].get_shape()[1].value, self.model_params.vocab_size]),
              vs.get_variable("Bias", [self.model_params.vocab_size]), name="Logit_%d" % decoder_time_idx)
          prev_logit = logit
          self.decoder_states.append(logit)
          prob = tf.nn.softmax(logit, name="Softmax_%d" % decoder_time_idx)
          self.prob.append(prob)
          self.logprob.append(tf.log(prob, name="LogProb_%d" % decoder_time_idx))

        # Pinyin Logits.
        if self.model_params.pinyin_ext:
          assert self.model_params.pinyin_dim == 7
          with vs.variable_scope("LogitPinyin"):
            pinyin_logit = nn_ops.xw_plus_b(
                outputs[-1],
                vs.get_variable("Matrix", [outputs[-1].get_shape()[1].value, self.model_params.pinyin_vocab_size * 7]),
                vs.get_variable("Bias", [self.model_params.pinyin_vocab_size * 7]), name="Logit_%d" % decoder_time_idx)
            pinyin_logit = array_ops.reshape(pinyin_logit, [self.batch_size * 7, self.model_params.pinyin_vocab_size])
            self.logits_pinyin.append(pinyin_logit)

      self.decoder_state_last = states[1:]
      self.decoder_alignment_last = filter(None, alignments)
      self.decoder_attention_last = filter(None, attentions)
    self.logits = self.decoder_states

  def create_decoder_cell_legacy(
      self, decoder_time_idx, states, attentions, prev_alignments, encoder_states,
      encoder_embedding, prev_logit, scope=None):
    batch_size = self.batch_size
    attention_embedding_size = self.model_params.attention_embedding_size
    decoder_cell_size = self.model_params.decoder_cell_size

    # Create the embedding layer.
    with tf.device("/cpu:0"):
      sqrt3 = np.sqrt(3)
      embedding = vs.get_variable(
          "embedding", [self.model_params.vocab_size, self.model_params.embedding_size],
          initializer=tf.random_uniform_initializer(-sqrt3, sqrt3))

      if decoder_time_idx == 0 or self.model_params.input_layer == 'placeholder':
        emb = embedding_ops.embedding_lookup(
            embedding, self.tokens[decoder_time_idx])
      elif self.model_params.input_layer == "decoder" or self.optimization_params == None:
        # We want the arg max of the previous token here.
        assert prev_logit
        prev_token = math_ops.argmax(prev_logit, 1)
        emb = embedding_ops.embedding_lookup(embedding, prev_token)
      else:
        emb = embedding_ops.embedding_lookup(embedding, gru_ops.token_sample(
            self.tokens[decoder_time_idx], self.prob[-1],
            sample_prob=self.optimization_params.sample_prob,
            seed=len(self.prob)))
      emb.set_shape([batch_size, self.model_params.embedding_size])

    def create_attention(decoder_state, prev_alignment):
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
        if self.model_params.attention_params.type == "window":
          e = attention_mask_ops.attention_mask_window(
              self.encoder_states[-1][1], decoder_time_idx, e)
        elif prev_alignment and self.model_params.attention_params.type == "median":
          window_l = self.model_params.attention_params.median_window_l
          window_r = self.model_params.attention_params.median_window_r
          e = attention_mask_ops.attention_mask_median(
              self.encoder_states[-1][1], e, prev_alignment, window_l=window_l,
              window_r=window_r)
        else:
          e = attention_mask_ops.attention_mask(self.encoder_states[-1][1], e)

        # Alignment.
        a = nn_ops.softmax(e, name="alignment_%d" % (decoder_time_idx))

        # Context.
        attn_length = self.encoder_embedding[0].get_shape()[1].value
        c = math_ops.reduce_sum(
            array_ops.reshape(a, [batch_size, attn_length, 1, 1]) * self.encoder_state_legacy,
            [1, 2])
        c = array_ops.reshape(c, [batch_size, self.model_params.encoder_cell_size])
        return a, c

    new_states = [None]
    new_attentions = [None]
    new_alignments = [None]
    new_outputs = [emb]
    def create_gru_cell(attention):
      stack_idx = len(new_states)

      # If empty, create new state.
      if len(states):
        state = states[stack_idx]
      elif self.model_params.input_layer == 'placeholder':
        # During decoding, this is given to us.
        state = tf.placeholder(
            tf.float32, shape=(batch_size, decoder_cell_size))
        self.decoder_states_initial.append(state)
      else:
        state = array_ops.zeros([batch_size, decoder_cell_size], tf.float32)

      # The input to this layer is the output of the previous layer.
      x = new_outputs[stack_idx - 1]
      # If the previous timestep has an attention context, concat it.
      if attention:
        if self.model_params.input_layer == "placeholder":
          attention_placeholder = tf.placeholder(
              tf.float32, shape=[batch_size, self.model_params.encoder_cell_size],
              name="attention_%d" % stack_idx)
          self.decoder_attentions_initial.append(attention_placeholder)
          x = array_ops.concat(1, [x, attention_placeholder])
        elif len(attentions):
          x = array_ops.concat(1, [x, attentions[stack_idx]])
        else:
          x = array_ops.concat(1, [x, array_ops.zeros([
              batch_size, self.model_params.encoder_cell_size], tf.float32)])
        x.set_shape([batch_size, new_outputs[stack_idx - 1].get_shape()[1].value + self.model_params.encoder_cell_size])

      # Create our GRU cell.
      _, _, _, _, h = gru_ops.gru_cell(
          decoder_cell_size, self.tokens_len, state, x, time_idx=decoder_time_idx)
      h.set_shape([batch_size, decoder_cell_size])

      new_states.append(h)
      if attention:
        prev_alignment = None
        if self.model_params.input_layer == "placeholder":
          prev_alignment = tf.placeholder(
              tf.float32, shape=[batch_size, len(self.encoder_states[-1][0])],
              name="alignment_%d" % stack_idx)
          self.decoder_alignments_initial.append(prev_alignment)
        elif prev_alignments:
          prev_alignment = prev_alignments[stack_idx]
        a, c = create_attention(h, prev_alignment)
        new_attentions.append(c)
        new_alignments.append(a)
        h = array_ops.concat(1, [h, c])
        h.set_shape(
            [batch_size, self.model_params.decoder_cell_size + self.model_params.encoder_cell_size])
      else:
        new_attentions.append(None)
        new_alignments.append(None)
      new_outputs.append(h)

    if self.model_params.decoder_layer:
      for idx, s in enumerate(self.model_params.decoder_layer):
        with vs.variable_scope(str(idx + 1)):
          create_gru_cell(attention=(s == "attention"))
    else:
      with vs.variable_scope("1"):
        create_gru_cell(attention=True)
      with vs.variable_scope("2"):
        create_gru_cell(attention=False)

    return new_outputs, new_states, new_attentions, new_alignments


  def create_loss(self):
    print("creating loss...")
    self.losses = []
    if self.model_params.loss.log_prob:
      self.create_loss_log_prob()
    if self.model_params.loss.edit_distance:
      self.create_loss_edit_distance()


  def create_loss_log_prob(self):
    targets = self.tokens[1:]
    weights = self.tokens_weights[1:]

    log_perplexity = seq2seq.sequence_loss(
        self.logits, targets, weights, self.model_params.vocab_size)
    self.log_perplexity = log_perplexity
    self.losses.append(log_perplexity)

    # Warning, this doesn't take into account the weight!
    correct = []
    for idx, logit in enumerate(self.logits):
      correct.append(tf.nn.in_top_k(logit, targets[idx], 1))
    self.correct = correct


  def create_loss_edit_distance(self):
    ref = self.tokens[1:]
    hyp = [tf.to_int32(math_ops.argmax(logit, 1)) for logit in self.logits]

    self.edit_distance = gen_array_ops.edit_distance_list(
        ref, hyp, collapse_eow=self.dataset_params.collapse_eow)


  def create_optimizer(self):
    print("creating optimizer...")

    params = tf.trainable_variables()
    for idx, param in enumerate(params):
      print("%d: %s %s" % (idx, param.name, str(param.get_shape())))

    self.create_gradient(params)
    # Merge the shapes w/ the params (easier for debugging).
    for idx, grad in enumerate(self.grads):
      if isinstance(grad, tf.Tensor):
        grad.set_shape(grad.get_shape().merge_with(params[idx].get_shape()))

    if self.optimization_params.type == "adagrad":
      opt = tf.train.AdagradOptimizer(
          learning_rate=self.optimization_params.adagrad.learning_rate,
          initial_accumulator_value=self.optimization_params.adagrad.initial_accumulator_value)
    elif self.optimization_params.type == "adadelta":
      assert self.optimization_params.adadelta.learning_rate
      assert self.optimization_params.adadelta.decay_rate
      assert self.optimization_params.adadelta.epsilon
      opt = tf.train.AdadeltaOptimizer(
          learning_rate=self.optimization_params.adadelta.learning_rate,
          decay_rate=self.optimization_params.adadelta.decay_rate,
          epsilon=self.optimization_params.adadelta.epsilon)
    elif self.optimization_params.type == "adam":
      opt = tf.train.AdamOptimizer(
          learning_rate=self.optimization_params.adam.learning_rate,
          beta1=self.optimization_params.adam.beta1,
          beta2=self.optimization_params.adam.beta2,
          epsilon=self.optimization_params.adam.epsilon)
    elif self.optimization_params.type == "gd":
      opt = tf.train.GradientDescentOptimizer(
          learning_rate=self.optimization_params.gd.learning_rate)
    else:
      raise ValueError(
          "Unknown optimization type: %s" % str(self.optimization_params))

    self.updates = []
    self.updates.append(opt.apply_gradients(
        zip(self.grads, params), global_step=self.global_step))


  def create_gradient(self, params):
    print("creating gradient...")
    self.grads = tf.gradients(self.losses, params)

    if self.optimization_params.max_gradient_norm:
      cgrads, norm = clip_ops.clip_by_global_norm(
          self.grads, self.optimization_params.max_gradient_norm,
          name="clip_gradients")
      self.grads = cgrads
      self.grads_norm = norm


  def step_epoch(self, sess, update, results_proto, profile_proto):
    steps_per_report = 100
    if update == False:
      steps_per_report = 1
    for idx in range(self.dataset_params.size / self.batch_size):
      self.step(sess, update, results_proto, profile_proto)

      if (idx % steps_per_report) == 0:
        percentage = float(idx) / float(self.dataset_params.size) * float(self.batch_size)
        accuracy = float(results_proto.acc.pos) / float(results_proto.acc.count)
        edit_distance = float(results_proto.edit_distance.edit_distance) / float(results_proto.edit_distance.ref_length)
        step_time = profile_proto.secs / profile_proto.steps
        print "step: %.2f, step_time: %.2f, accuracy %.4f, edit_distance %.4f" % (percentage, step_time, accuracy, edit_distance)


  def restore(self, sess):
    assert os.path.isfile(self.model_params.ckpt)

    trainable_only = False
    if self.optimization_params is not None:
      if self.optimization_params.adagrad.reset:
        trainable_only = True
      elif self.optimization_params.adadelta.reset:
        trainable_only = True
      elif self.optimization_params.adam.reset:
        trainable_only = True

    variables = []
    if trainable_only:
      variables = tf.trainable_variables()
    else:
      variables = tf.all_variables()
    for idx, variable in enumerate(variables):
      print("%d: %s %s" % (idx, variable.name, str(variable.get_shape())))

    self.saver = tf.train.Saver(variables)
    self.saver.restore(sess, self.model_params.ckpt)


  def save(self, sess, prefix, results_proto=None):
    model_params = speech4_pb2.ModelParamsProto()
    model_params.CopyFrom(self.model_params)
    model_params.ckpt = prefix + ".ckpt"

    if results_proto:
      model_params.results.CopyFrom(results_proto)

    with open(prefix + ".model_params", "w") as proto_file:
      proto_file.write(str(model_params))
    ckpt_filepath = prefix + ".ckpt"
    self.saver.save(sess, ckpt_filepath)
    return ckpt_filepath


  def step(self, sess, update, results_proto, profile_proto):
    start_time = time.time()

    targets = {}
    targets["uttid"] = self.uttid
    targets["text"] = self.text

    targets["tokens"] = self.tokens[:-1]
    targets["tokens_weights"] = self.tokens_weights[:-1]
    if hasattr(self, "correct"):
      targets["correct"] = self.correct
    if hasattr(self, "edit_distance"):
      targets["edit_distance"] = self.edit_distance

    if update:
      targets['updates'] = self.updates

    fetches = self.run_graph(sess, targets)
    if hasattr(self, "correct"):
      self.compute_accuracy(fetches, results_proto.acc)
    if hasattr(self, "edit_distance"):
      self.compute_edit_distance(fetches, results_proto.edit_distance)
   
    step_time = time.time() - start_time
    profile_proto.secs = profile_proto.secs + step_time
    profile_proto.steps = profile_proto.steps + 1


  def compute_accuracy(self, fetches, acc_proto):
    assert "tokens_weights" in fetches
    assert "correct" in fetches

    assert len(fetches["tokens_weights"]) == len(fetches["correct"])

    for token_weight, correct in zip(fetches["tokens_weights"], fetches["correct"]):
      count = token_weight.astype(int).sum()
      weighted_correct = np.multiply(token_weight.astype(int), correct).sum()

      acc_proto.pos += weighted_correct
      acc_proto.count += count


  def compute_edit_distance(self, fetches, edit_distance_proto):
    assert "edit_distance" in fetches

    edit_distance = fetches["edit_distance"][0].sum()
    ref_length = fetches["edit_distance"][1].sum()

    edit_distance_proto.edit_distance += edit_distance
    edit_distance_proto.ref_length += ref_length


  def run_graph(self, sess, targets, feed_dict=None):
    fetches = []
    for name, target in targets.iteritems():
      if isinstance(target, (list, tuple)):
        fetches.extend(target)
      else:
        fetches.append(target)
    r = sess.run(fetches, feed_dict)

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


def create_dataset_params(name, path, size):
  params = speech4_pb2.DatasetParamsProto()
  params.name = name
  params.path = path
  params.size = size
  return params


def load_dataset_params(mode):
  if FLAGS.dataset == "wsj":
    wsj = {}
    wsj["train"] = create_dataset_params("train_si284", "speech4/data/train_si284.tfrecords", 37416)
    wsj["valid"] = create_dataset_params("test_dev93", "speech4/data/test_dev93.tfrecords", 503)
    wsj["test"] = create_dataset_params("test_eval92", "speech4/data/test_eval92.tfrecords", 333)
    return wsj[mode]
  elif FLAGS.dataset == "gale_mandarin":
    gale = {}
    gale["train"] = create_dataset_params("train", "speech4/data/gale_mandarin_sp_train_space.tfrecords", 58058)
    gale["valid"] = create_dataset_params("valid", "speech4/data/gale_mandarin_sp_dev_space.tfrecords", 5191)
    gale["test"] = create_dataset_params("test", "speech4/data/gale_mandarin_sp_dev_space.tfrecords", 5191)
    for key in gale:
      gale[key].collapse_eow = True
    return gale[mode]
  elif FLAGS.dataset == "timit":
    timit = {}
    timit["train"] = create_dataset_params("train", "/data-local/data/tfrecords/timit_train.tfrecords", 3696)
    timit["valid"] = create_dataset_params("dev", "/data-local/data/tfrecords/timit_dev.tfrecords", 400)
    timit["test"] = create_dataset_params("test", "/data-local/data/tfrecords/timit_test.tfrecords", 192)
    for key in timit:
      timit[key].feature_size = 69
    return timit[mode]

  raise Exception("Unknown dataset %s" % FLAGS.dataset)


def load_model_params(mode, dataset_params):
  model_params = speech4_pb2.ModelParamsProto()
  model_params.features_width = 123
  if dataset_params.features_width:
    model_params.features_width = dataset_params.features_width
  model_params.features_len_max = 2560
  model_params.frame_skip = 1
  model_params.frame_stack = 1
  model_params.tokens_len_max = 256
  model_params.vocab_size = 64
  model_params.embedding_size = 32
  model_params.encoder_cell_size = 384
  model_params.decoder_cell_size = 256
  model_params.attention_embedding_size = 128

  model_params.loss.log_prob = True
  model_params.loss.edit_distance = True

  model_params = try_load_proto(FLAGS.model_params, model_params)

  if len(model_params.encoder_layer) == 0:
    model_params.encoder_layer.append("1")
    model_params.encoder_layer.append("1")
    model_params.encoder_layer.append("2")
    model_params.encoder_layer.append("2")

  if not model_params.encoder_prefix:
    model_params.encoder_prefix = "encoder"
  if not model_params.encoder_embedding:
    model_params.encoder_embedding = "encoder_embedding"
  if not model_params.decoder_prefix:
    model_params.decoder_prefix = "decoder"

  return model_params


def load_optimization_params(mode):
  if mode == "train":
    optimization_params = speech4_pb2.OptimizationParamsProto()
    optimization_params.type = "adadelta"
    optimization_params.adadelta.learning_rate = 1.0
    optimization_params.adadelta.decay_rate = 0.95
    optimization_params.adadelta.epsilon = 1e-8
    optimization_params.adadelta.reset = True
    optimization_params.max_gradient_norm = 1.0

    optimization_params = try_load_proto(FLAGS.optimization_params, optimization_params)
    return optimization_params
  return None


def load_params(mode):
  dataset_params = load_dataset_params(mode)
  model_params = load_model_params(mode, dataset_params)
  optimization_params = load_optimization_params(mode)
  return dataset_params, model_params, optimization_params


def run(mode, epoch, ckpt=None):
  ckpt_filepath = None
  with tf.device("/gpu:%d" % FLAGS.device):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      dataset_params, model_params, optimization_params = load_params(mode)

      if ckpt:
        model_params.ckpt = ckpt

      speech_model = SpeechModel(
          sess, mode, dataset_params, model_params, optimization_params, seed=epoch)

      coord = tf.train.Coordinator()
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))

      results_proto = speech4_pb2.ResultsProto()
      profile_proto = speech4_pb2.ProfileProto()
      print("epoch: %d" % epoch)
      speech_model.step_epoch(sess, mode == "train", results_proto, profile_proto)

      if mode == "train":
        prefix = os.path.join(FLAGS.logdir, "%d" % (epoch + 1))
        ckpt_filepath = speech_model.save(sess, prefix, results_proto)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)
  return ckpt_filepath


def main(_):
  if not FLAGS.logdir:
    FLAGS.logdir = os.path.join(
        "exp", "speech4_" + "".join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(8)))
  FLAGS.logdir = os.path.abspath(FLAGS.logdir)
  try:
    os.makedirs(FLAGS.logdir)
  except:
    pass

  print "logdir: %s" % FLAGS.logdir

  ckpt_filepath = FLAGS.ckpt
  for epoch in range(20):
    ckpt_filepath = run("train", epoch, ckpt_filepath)
    run("valid", epoch, ckpt_filepath)
    run("test", epoch, ckpt_filepath)

if __name__ == '__main__':
  tf.app.run()
