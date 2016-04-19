#!/usr/bin/env bash

################################################################################
# Copyright 2016 William Chan <williamchan@cmu.edu>.
################################################################################

echo "---------------------------------------------------------------------"
echo "SPEECH4 <williamchan@cmu.edu>"
echo "---------------------------------------------------------------------"

KALDI_ROOT=/data-local/wchan/kaldi
WSJ_ROOT=${KALDI_ROOT}/egs/wsj/s5

dataset=train_si284
data="${WSJ_ROOT}/data/${dataset}"
feats="ark,s,cs:${KALDI_ROOT}/src/featbin/apply-cmvn --norm-vars=true --utt2spk=ark:${data}/utt2spk scp:${data}/cmvn.scp scp:${data}/feats.scp ark:- |"

uttid=01fc020c
alignment="~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~F~~O~~R~~~~ ~~~~E~~~~X~~A~~~M~PL~~E~~~~ ~~~~~~M~R~~~~ ~~~~W~~E~~~I~~~~N~ST~~E~~~I~~~~N~~~~ ~~~~D~~I~~~SL~~I~~~K~~E~~~S~~~~ ~~~~T~H~~E~~~~ ~~~~~~W~~A~~Y~~~~ ~~~~~~~~MR~~~~ ~~~~H~~U~~~L~B~~E~~R~T~~~~ ~~~~C~~A~~~L~C~~U~~L~~A~~~T~~E~~~S~~~~ ~~~~E~~~~NT~R~~Y~~~~ ~~~~~~A~~~~N~D~~~~ ~~~~E~~~~X~~I~~~T~~~~ ~~~~PR~~I~~~C~~E~~~S~~~~~~~~~~~"
python speech4/data/wsj_plot.py \
    --kaldi_scp "${feats}" \
    --kaldi_txt ${data}/text \
    --uttid ${uttid} \
    --alignment "${alignment}"
