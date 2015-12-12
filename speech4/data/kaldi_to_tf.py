#!/usr/bin/env python

################################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
################################################################################

import argparse
import google
import kaldi_io
import numpy
import os
import re
import os
import sys
import tensorflow as tf

def main():
  parser = argparse.ArgumentParser(description='SPEECH3 (C) 2015 William Chan <williamchan@cmu.edu>')
  parser.add_argument('--kaldi_cmvn_scp', type=str)
  parser.add_argument('--kaldi_scp', type=str)
  parser.add_argument('--kaldi_txt', type=str)
  parser.add_argument('--kaldi_utt2spk', type=str)
  parser.add_argument('--tf_records', type=str)
  parser.add_argument('--type', type=str, default='wsj')
  args   = vars(parser.parse_args())

  convert(args['kaldi_scp'], args['kaldi_txt'], args['tf_records'], args['kaldi_cmvn_scp'], args['kaldi_utt2spk'])


def convert(kaldi_scp, kaldi_txt, tf_records, kaldi_cmvn_scp=None, kaldi_utt2spk=None):
  # Read the text file into utterance_map.
  utterance_map = {}
  lines = [line.strip() for line in open(kaldi_txt, 'r')]
  for line in lines:
    [uttid, utt] = normalize_text_wsj(line)
    utterance_map[uttid] = utt

  # Read the speaker normalization.
  normalization_map = {}
  kaldi_cmvn_reader = kaldi_io.SequentialBaseFloatMatrixReader(kaldi_cmvn_scp)
  for [spkid, mean_var] in kaldi_cmvn_reader:
    normalization_map[spkid] = mean_var

  utt2spk_map = {}
  lines = [line.strip() for line in open(kaldi_utt2spk, 'r')]
  for line in lines:
    [uttid, spkid] = line.split(' ')
    utt2spk_map[uttid] = spkid

  # Process the utterances.
  kaldi_feat_reader = kaldi_io.SequentialBaseFloatMatrixReader(kaldi_scp)
  tf_record_writer = tf.python_io.TFRecordWriter(tf_records)
  utt_count = 0
  for uttid, feats in kaldi_feat_reader:
    # CMVN.
    spkid = utt2spk_map[uttid]
    mean_var = normalization_map[spkid]

    dim = mean_var.shape[1] - 1
    count = mean_var[0, dim]

    mean = mean_var[0, :-1] / count
    var = (mean_var[1, :-1] / count) - mean * mean

    scale = 1.0 / numpy.sqrt(var)
    offset = - mean * scale

    feats_normalized = feats * scale[numpy.newaxis, :] + offset

    # Corresponding text transcript.
    text = utterance_map[uttid]

    example = tf.train.Example(features=tf.train.Features(feature={
        'dim_t': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats_normalized.shape[0]])),
        'dim_w': tf.train.Feature(int64_list=tf.train.Int64List(value=[feats_normalized.shape[1]])),
        'features': tf.train.Feature(bytes_list=tf.train.BytesList(value=[feats_normalized.tostring()])),
        'uttid': tf.train.Feature(bytes_list=tf.train.BytesList(value=[uttid])),
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text]))}))
    tf_record_writer.write(example.SerializeToString())

    utt_count = utt_count + 1


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


def normalize_text_swbd(lines, utterance_list_proto):
  utterance_proto = utterance_list_proto.utterances.add()

  cols = line.split(' ', 1)
  assert len(cols) >= 1 and len(cols) <= 2
  uttid = cols[0]
  if len(cols) == 1:
    utt = ''
  elif len(cols) == 2:
    utt = cols[1]

  assert '#' not in utt

  utterance_proto.transcript = utt

  # Normalize to uppercase.
  utt = utt.upper()

  # Underscore / period.
  # These tokens don't exist in eval2000.
  utt = utt.replace('_', '')
  utt = utt.replace('.', '')

  # (%HESITATION)
  utt = utt.replace('(%HESITATION)', '#')

  # Everything in (...).
  utt = re.sub(r'\([^)]*\)', '', utt)

  # Special tokens.
  utt = utt.replace('[LAUGHTER]', '#')
  utt = utt.replace('[NOISE]', '#')
  utt = utt.replace('[VOCALIZED-NOISE]', '#')
  # utt = re.sub(r'\[[^)]*\]', '#', utt)

  # Replace dash with space.
  utt = utt.replace('-', ' ')

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
  assert '[' not in utt

  # Remove double spaces.
  utt = ' '.join(filter(bool, utt.split(' ')))

  utterance_proto.uttid = uttid
  utterance_proto.transcript_normalized = utt

  # for c in utt:
  #   if c == '#': c = '<UNK>'
  #   utterance_proto.tokens.append(c)


def normalize_text_eval2000(line):
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

  # (%HESITATION)
  utt = utt.replace('(%HESITATION)', '')

  # Everything in (...)
  utt = re.sub(r'\([^)]*\)', '', utt)

  # <B_ASIDE>, <E_ASIDE>
  utt = utt.replace('<B_ASIDE>', '')
  utt = utt.replace('<E_ASIDE>', '')

  # Replace dash with space.
  utt = utt.replace('-', ' ')

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
  assert '(' not in utt
  assert '[' not in utt

  # Remove double spaces.
  utt = ' '.join(filter(bool, utt.split(' ')))

  return [uttid, utt]


if __name__ == '__main__':
  main()
