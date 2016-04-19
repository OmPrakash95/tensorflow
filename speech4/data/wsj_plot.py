#!/usr/bin/env python

################################################################################
# Copyright 2016 William Chan <williamchan@cmu.edu>.
################################################################################

import argparse
import google
import itertools
import kaldi_io
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import os
import string
import sys
import tensorflow as tf
import tensorflow.core.framework.token_model_pb2 as token_model_pb2


def normalize_text_wsj(line):
  cols = line.split(' ', 1)
  assert len(cols) >= 1 and len(cols) <= 2
  uttid = cols[0]
  if len(cols) == 1:
    utt = ''
  elif len(cols) == 2:
    utt = cols[1]

  assert '#' not in utt

  # Normalize to uppercase.
  utt = utt.upper()

  utt = utt.replace('<NOISE>', '')

  utt = utt.replace('&AMPERSAND', 'AMPERSAND')
  utt = utt.replace(')CLOSE_PAREN', 'CLOSEPAREN')
  utt = utt.replace(')CLOSE-PAREN', 'CLOSEPAREN')
  utt = utt.replace('\"CLOSE-QUOTE', 'CLOSEQUOTE')
  utt = utt.replace(':COLON', "COLON")
  utt = utt.replace(',COMMA', 'COMMA')
  utt = utt.replace('-DASH', 'DASH')
  utt = utt.replace('"DOUBLE-QUOTE', 'DOUBLEQUOTE')
  utt = utt.replace('"END-QUOTE', 'ENDQUOTE')
  utt = utt.replace('"END-OF-QUOTE', 'ENDOFQUOTE')
  utt = utt.replace(')END-OF-PAREN', 'ENDOFPAREN')
  utt = utt.replace(')END-THE-PAREN', 'ENDTHEPAREN')
  utt = utt.replace('!EXCLAMATION-POINT', 'EXCLAMATIONPOINT')
  utt = utt.replace('-HYPHEN', 'HYPHEN')
  utt = utt.replace('\"IN-QUOTES', 'INQUOTES')
  utt = utt.replace('(IN-PARENTHESIS', 'INPARENTHESIS')
  utt = utt.replace('{LEFT-BRACE', 'LEFTBRACE')
  utt = utt.replace('(LEFT-PAREN', 'LEFTPAREN')
  utt = utt.replace('(PAREN', 'PAREN')
  utt = utt.replace(')PAREN', 'PAREN')
  utt = utt.replace('?QUESTION-MARK', 'QUESTIONMARK')
  utt = utt.replace('"QUOTE', 'QUOTE')
  utt = utt.replace(')RIGHT-PAREN', 'RIGHTPAREN')
  utt = utt.replace('}RIGHT-BRACE', 'RIGHTBRACE')
  utt = utt.replace('\'SINGLE-QUOTE', 'SINGLEQUOTE')
  utt = utt.replace('/SLASH', 'SLASH')
  utt = utt.replace(';SEMI-COLON', "SEMICOLON")
  utt = utt.replace(')UN-PARENTHESES', 'UNPARENTHESES')
  utt = utt.replace('"UNQUOTE', 'UNQUOTE')
  utt = utt.replace('.PERIOD', "PERIOD")

  utt = re.sub(r'\([^)]*\)', '', utt)
  utt = re.sub(r'<[^)]*>', '', utt)

  utt = utt.replace('.', '')
  utt = utt.replace('-', '')
  utt = utt.replace('!', '')
  utt = utt.replace(':', '')
  utt = utt.replace(';', '')
  utt = utt.replace('*', '')
  utt = utt.replace('`', '\'')
  utt = utt.replace('~', '')

  assert '~' not in utt
  assert '`' not in utt
  assert '-' not in utt
  assert '_' not in utt
  assert '.' not in utt
  assert ',' not in utt
  assert ':' not in utt
  assert ';' not in utt
  assert '!' not in utt
  assert '?' not in utt
  assert '<' not in utt
  assert '(' not in utt
  assert ')' not in utt
  assert '[' not in utt
  # assert '\'' not in utt
  assert '"' not in utt
  assert '*' not in utt

  # Remove double spaces.
  utt = ' '.join(filter(bool, utt.split(' ')))

  return [uttid, utt]



def load_text_map(kaldi_txt):
  text_map = {}
  lines = [line.strip() for line in open(kaldi_txt, "r")]
  for line in lines:
    [uttid, utt] = normalize_text_wsj(line)
    text_map[uttid] = utt
  return text_map


def main():
  parser = argparse.ArgumentParser(description="SPEECH4 (C) 2016 William Chan <williamchan@cmu.edu>")
  parser.add_argument("--kaldi_scp", type=str)
  parser.add_argument("--kaldi_txt", type=str)
  parser.add_argument("--uttid", type=str)
  parser.add_argument("--alignment", type=str)
  args   = vars(parser.parse_args())

  kaldi_scp = args["kaldi_scp"]
  kaldi_txt = args["kaldi_txt"]
  uttid = args["uttid"]
  alignment = list(args["alignment"])

  text = load_text_map(kaldi_txt)[uttid]

  # kaldi_feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(kaldi_scp)
  kaldi_feat_reader = kaldi_io.RandomAccessBaseFloatMatrixReader(kaldi_scp)
  feats = kaldi_feat_reader[uttid].transpose()

  #alignment = ["~"] * (feats.shape[1] / 2)
  print feats.shape
  print len(alignment)

  # Let us try and find the starting point of our audio using some simple
  # heuristic.
  feats_max = feats.max(0)
  s_min = 0
  s_max = 0
  for idx in range(len(feats_max)):
    if feats_max[idx] > 1.0:
      s_min = idx
      break
  for idx in range(len(feats_max) - 1, 0, -1):
    if feats_max[idx] > 1.0:
      s_max = idx
      break
  s_min = s_min / 2 + 8
  s_max = min(s_max / 2 + 4, len(feats_max) - 1)

  alignment[s_min] = "*"
  alignment[s_max] = "*"

  assert feats.shape[0] == 40

  fig = plt.figure(figsize=(40, 5))
  ax = fig.add_subplot(1, 1, 1)

  ax.set_title(text)

  cax = ax.imshow(feats, interpolation="none")
  # fig.colorbar(cax, orientation="horizontal")

  ax.get_yaxis().set_visible(False)
  ax.set_xticks(np.arange(0, feats.shape[1], feats.shape[1] / len(alignment)), minor=False)
  ax.set_xticklabels(alignment)

  fig.savefig(os.path.join("pngs", "%s.png" % uttid), dpi=100)


if __name__ == '__main__':
  main()
