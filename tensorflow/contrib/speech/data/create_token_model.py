import tensorflow as tf

from tensorflow.contrib.speech.data import token_model


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("save_proto", None, "Path to save the proto.")


def main(unused_args):
  tm = token_model.TokenModel()
  tm.create_basic_model()
  if FLAGS.save_proto:
    tm.save_proto(FLAGS.save_proto)

if __name__ == "__main__":
  tf.app.run()
