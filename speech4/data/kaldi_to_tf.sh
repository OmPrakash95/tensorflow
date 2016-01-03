#!/usr/bin/env bash

################################################################################
# Copyright 2015 William Chan <williamchan@cmu.edu>.
################################################################################

echo "---------------------------------------------------------------------"
echo "SPEECH4 <williamchan@cmu.edu>"
echo "---------------------------------------------------------------------"

KALDI_ROOT=/data-local/wchan/kaldi

if [ -z $KALDI_RECIPE_PATH ]; then
  KALDI_RECIPE_PATH=/data-local/wchan/kaldi/egs/wsj/s5
fi

for dataset in train_si284 test_dev93 test_eval92; do
  python speech4/data/kaldi_to_tf.py \
      --kaldi_cmvn_scp scp:${KALDI_RECIPE_PATH}/data/${dataset}/cmvn.scp \
      --kaldi_scp scp:${KALDI_RECIPE_PATH}/data/${dataset}/feats.scp \
      --kaldi_txt ${KALDI_RECIPE_PATH}/data/${dataset}/text \
      --kaldi_utt2spk ${KALDI_RECIPE_PATH}/data/${dataset}/utt2spk \
      --tf_records speech4/data/${dataset}.tfrecords \
      --type wsj
done

for dataset in train_si284 test_dev93 test_eval92; do
  python speech4/data/kaldi_to_tf.py \
      --kaldi_cmvn_scp scp:${KALDI_RECIPE_PATH}/data/${dataset}/cmvn.scp \
      --kaldi_scp scp:${KALDI_RECIPE_PATH}/data/${dataset}/feats.scp \
      --kaldi_txt ${KALDI_RECIPE_PATH}/data/${dataset}/text \
      --kaldi_utt2spk ${KALDI_RECIPE_PATH}/data/${dataset}/utt2spk \
      --sort \
      --tf_records speech4/data/${dataset}_sorted.tfrecords \
      --type wsj
done
