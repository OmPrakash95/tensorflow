#!/usr/bin/env bash

################################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
################################################################################

echo "---------------------------------------------------------------------"
echo "SPEECH4 <williamchan@cmu.edu>"
echo "---------------------------------------------------------------------"

KALDI_ROOT=/data-local/wchan/kaldi
TIMIT_ROOT=${KALDI_ROOT}/egs/timit/s5

for dataset in dev test train; do
  data="${TIMIT_ROOT}/data/${dataset}"
  feats="ark,s,cs:${KALDI_ROOT}/src/featbin/apply-cmvn --norm-vars=true --utt2spk=ark:${data}/utt2spk scp:${data}/cmvn.scp scp:${data}/feats.scp ark:- | ${KALDI_ROOT}/src/featbin/add-deltas ark:- ark:- |"

  python speech4/data/timit.py \
      --kaldi_scp "${feats}" \
      --kaldi_txt ${data}/text \
      --kaldi_alignment ark:"gunzip -c ${TIMIT_ROOT}/exp/sgmm2_4_ali/ali.*.gz |" \
      --phones speech4/conf/timit/phones.txt \
      --remap speech4/conf/timit/remap.txt \
      --tf_records timit_${dataset}.tfrecords
done
