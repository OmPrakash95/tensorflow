#!/usr/bin/env bash

KALDI_ROOT=/data-local/wchan/kaldi

for dataset in train dev; do
  # /data-local/wchan/kaldi/egs/gale_mandarin
  data="${KALDI_ROOT}/egs/gale_mandarin/s5/data/${dataset}"
  feats="ark,s,cs:${KALDI_ROOT}/src/featbin/apply-cmvn --norm-vars=true --utt2spk=ark:${data}/utt2spk scp:${data}/cmvn.scp scp:${data}/feats.scp ark:- |"
  python speech4/data/kaldi_gale_to_tf.py \
      --kaldi_scp "${feats}" \
      --kaldi_txt ${data}/text \
      --tf_records speech4/data/gale_mandarin_${dataset}.tfrecords
done
