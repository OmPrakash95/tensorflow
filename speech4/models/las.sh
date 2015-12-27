#!/bin/bash

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:=0}

python speech4/models/las.py
