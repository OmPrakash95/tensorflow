#!/usr/bin/env bash

################################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
################################################################################

echo "---------------------------------------------------------------------"
echo "SPEECH4 <williamchan@cmu.edu>"
echo "---------------------------------------------------------------------"

KALDI_ROOT=/data-local/wchan/kaldi

# WSJ
for dataset in train_si284; do
  python speech4/data/kaldi_to_tf.py \
      --kaldi_cmvn_scp scp:${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/cmvn.scp \
      --kaldi_scp scp:${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/feats.scp \
      --kaldi_txt ${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/text \
      --kaldi_utt2spk ${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/utt2spk \
      --tf_records speech4/data/${dataset}.tfrecords \
      --type wsj
done

for dataset in test_dev93 test_eval92; do
  python speech4/data/kaldi_to_tf.py \
      --kaldi_cmvn_scp scp:${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/cmvn.scp \
      --kaldi_scp scp:${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/feats.scp \
      --kaldi_txt ${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/text \
      --kaldi_utt2spk ${KALDI_ROOT}/egs/wsj/s5/data/${dataset}/utt2spk \
      --sort \
      --tf_records speech4/data/${dataset}_sorted.tfrecords \
      --type wsj
done

# SWBD
data="${KALDI_ROOT}/egs/swbd/s5c/data/train"
feats="ark,s,cs:${KALDI_ROOT}/src/featbin/apply-cmvn --norm-vars=true --utt2spk=ark:${data}/utt2spk scp:${data}/cmvn.scp scp:${data}/feats.scp ark:- |"
python speech4/data/kaldi_to_tf.py \
    --kaldi_scp "${feats}" \
    --kaldi_txt ${data}/text \
    --tf_records speech4/data/swbd.tfrecords \
    --type swbd

# eval2000
data="${KALDI_ROOT}/egs/swbd/s5c/data/eval2000"
feats="ark,s,cs:${KALDI_ROOT}/src/featbin/apply-cmvn --norm-vars=true --utt2spk=ark:${data}/utt2spk scp:${data}/cmvn.scp scp:${data}/feats.scp ark:- |"
python speech4/data/kaldi_to_tf.py \
    --kaldi_scp "${feats}" \
    --kaldi_txt ${data}/text \
    --tf_records speech4/data/eval2000.tfrecords \
    --type eval2000
