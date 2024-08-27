#!/bin/bash

source ~/.bashrc
conda activate LynD

module load gcc
module load cuda

cd /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/src 

# python analysis.py \
# 	--features onehot \
# 	--model MLP \
# 	--routine threshold \
# 	--variable_region 14 15 16 17 18 19 20 21 \
# 	--ckpt checkpoints/NNK7_rd3_epoch=33_val_accuracy=0.92.ckpt \
# 	--include_C \
# 	--sele /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/data/csv/NNK7/rd3/sele/ \
# 	--anti /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/data/csv/NNK7/rd3/anti/

