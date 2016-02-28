import math
import numpy as np
import os.path
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


class SpeechModel(object):
  def __init__(self, sess, dataset_params, model_params, optimization_params):
    self.dataset_params = dataset_params
    self.model_params = model_params
    self.optimization_params = optimization_params

    self.batch_size = 16
    self.seed = 1

    self.create_model(sess)


  def create_model(self, sess):
    print("creating model...")
    self.global_step = tf.Variable(0, trainable=False)

    initializer = tf.random_uniform_initializer(-0.1, 0.1)
    with tf.variable_scope("model", initializer=initializer):
      self.create_input()
      self.create_encoder()
      self.create_decoder()
      self.create_loss()
      self.create_optimizer()

    print("initializing model...")
    sess.run(tf.initialize_all_variables())

    variables = tf.all_variables()
    self.saver = tf.train.Saver(variables)
    if gfile.Exists(self.model_params.ckpt):
      self.saver.restore(sess, self.model_params.ckpt)
    self.saver = tf.train.Saver(tf.all_variables())

  def create_input(self):
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
    with vs.variable_scope("encoder"):
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
    k = vs.get_variable(
        "encoder_embedding", [
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
      with vs.variable_scope("layer_%d" % (idx + 1)) as scope:
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
      with vs.variable_scope("layer_%d" % idx) as scope:
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


  def create_decoder(self):
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
              array_ops.reshape(a, [-1, attn_length, 1, 1]) * encoder_state,
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

    self.edit_distance = gen_array_ops.edit_distance_list(ref, hyp)


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
    for idx in range(self.dataset_params.size / self.batch_size):
      self.step(sess, update, results_proto, profile_proto)

      if idx % 10 == 0:
        percentage = float(idx) / float(self.dataset_params.size)
        accuracy = float(results_proto.acc.pos) / float(results_proto.acc.count)
        edit_distance = float(results_proto.edit_distance.edit_distance) / float(results_proto.edit_distance.ref_length)
        step_time = profile_proto.secs / profile_proto.steps
        print "step: %.2f, step_time: %.2f, accuracy %.2f, edit_distance %.2f" % (percentage, step_time, accuracy, edit_distance)


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


def main(_):
  with tf.device("/gpu:0"):
    with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
      dataset_params = speech4_pb2.DatasetParamsProto()
      dataset_params.path = "speech4/data/train_si284.tfrecords"
      dataset_params.size = 37416

      model_params = speech4_pb2.ModelParamsProto()
      model_params.features_width = 123
      model_params.features_len_max = 2560
      model_params.frame_skip = 1
      model_params.frame_stack = 1
      model_params.tokens_len_max = 256
      model_params.vocab_size = 64
      model_params.embedding_size = 32
      model_params.encoder_cell_size = 384
      model_params.decoder_cell_size = 256
      model_params.attention_embedding_size = 128

      model_params.encoder_layer.append("1")
      model_params.encoder_layer.append("1")
      model_params.encoder_layer.append("2")
      model_params.encoder_layer.append("2")

      model_params.loss.log_prob = True
      model_params.loss.edit_distance = True

      optimization_params = speech4_pb2.OptimizationParamsProto()
      optimization_params.type = "adadelta"
      optimization_params.adadelta.learning_rate = 1.0
      optimization_params.adadelta.decay_rate = 0.95
      optimization_params.adadelta.epsilon = 1e-8
      optimization_params.max_gradient_norm = 1.0

      speech_model = SpeechModel(sess, dataset_params, model_params, optimization_params)

      coord = tf.train.Coordinator()
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))


      results_proto = speech4_pb2.ResultsProto()
      profile_proto = speech4_pb2.ProfileProto()
      while True:
        speech_model.step_epoch(sess, True, results_proto, profile_proto)

      coord.request_stop()
      coord.join(threads, stop_grace_period_secs=10)


if __name__ == '__main__':
  tf.app.run()
