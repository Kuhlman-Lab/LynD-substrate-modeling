#!/bin/bash
#SBATCH -J train
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

# NOTE: you will need to change the file paths to match your system and code/data install locations
repo_location=/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/src
data_location=/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/data/csv/NNK7/rd3/

sele=${data_location}/sele
anti=${data_location}/anti
cd $repo_location

python trainer.py \
	--sele $sele \
	--anti $anti \
	--epochs 50 \
	--features onehot \
	--run LynD_NNK7_rd3 \
	--model MLP \
	--seed 1234 \
	--variable_region 14 15 16 17 18 19 20 21 \
	--batch_size 2048
