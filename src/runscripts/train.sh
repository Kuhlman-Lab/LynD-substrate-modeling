#!/bin/bash
#SBATCH -J train
#SBATCH -t 12:00:00
#SBATCH --partition=volta-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=R-%x.%j.out
#SBATCH --error=R-%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu_access

source ~/.bashrc
# conda activate LynD
conda activate /nas/longleaf/home/dieckhau/miniconda3/envs/proteinMPNN

module load gcc
module load cuda

cd /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/src 

sele=/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/data/csv/NNK7/rd3/sele
anti=/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/data/csv/NNK7/rd3/anti

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
