class Utterance(object):
  def __init__(self):
    self.features = None
    self.features_len = None
    self.text = None
    self.uttid = None

    self.encoder_states = []
    self.feed_dict = {}

  def build_feed_dict(self):
    for idx, feature in enumerate(self.features):
      self.feed_dict['features_%d' % idx] = feature
    self.feed_dict['features_len'] = self.features_len
