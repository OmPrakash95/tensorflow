#!/usr/bin/env bash

################################################################################
# Copyright 2016 William Chan <williamchan@cmu.edu>.
################################################################################

echo "---------------------------------------------------------------------"
echo "SPEECH4 <williamchan@cmu.edu>"
echo "---------------------------------------------------------------------"

KALDI_ROOT=/data-local/wchan/kaldi
WSJ_ROOT=${KALDI_ROOT}/egs/wsj/s5

for dataset in train_si284 test_dev93 test_eval92; do
  data="${WSJ_ROOT}/data/${dataset}"
  feats="ark,s,cs:${KALDI_ROOT}/src/featbin/apply-cmvn --norm-vars=true --utt2spk=ark:${data}/utt2spk scp:${data}/cmvn.scp scp:${data}/feats.scp ark:- | ${KALDI_ROOT}/src/featbin/add-deltas ark:- ark:- |"

  python speech4/data/wsj.py \
      --kaldi_scp "${feats}" \
      --kaldi_txt ${data}/text \
      --tf_records wsj_${dataset}.tfrecords
done
