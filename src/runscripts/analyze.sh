#!/bin/bash
#SBATCH -J analyze
#SBATCH -t 2:00:00
#SBATCH --partition=volta-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source ~/.bashrc
mamba activate LynD

module load gcc
module load cuda

cd /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/src 

# calculate PPV curve for model and S score
python analysis.py \
	--features onehot \
	--model MLP \
	--routine threshold \
	--variable_region 14 15 16 17 18 19 20 21 \
	--ckpt model_weights/LynD_NNK7_rd3.ckpt \
	--include_C \
	--sele /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/sele/ \
	--anti /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/anti/

# screen N random peptides according to a threshold and Hamming distance constraint
python analysis.py \
	--features onehot \
	--model MLP \
	--routine screen \
	--variable_region 14 15 16 17 18 19 20 21 \
	--ckpt model_weights/LynD_NNK7_rd3.ckpt \
	--include_C \
    --thresh 0.97 \
    --hamming 2 \
	--sele /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/sele/ \
	--anti /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/anti/

# calculate epistasis matrix for a semi-randomized library
python analysis.py \
	--features onehot \
	--model MLP \
	--routine epistasis \
	--variable_region 14 15 16 17 18 19 20 21 \
	--ckpt model_weights/LynD_NNK7_rd3.ckpt \
	--include_C \
	--sele /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/sele/ \
	--anti /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/anti/
