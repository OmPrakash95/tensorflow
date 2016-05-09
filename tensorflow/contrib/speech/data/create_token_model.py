import tensorflow as tf

from tensorflow.contrib.speech.data import token_model


def main(unused_args):
  tm = token_model.TokenModel()
  tm.create_basic_model()
  print(str(tm.token_model_proto()))


if __name__ == "__main__":
  tf.app.run()
