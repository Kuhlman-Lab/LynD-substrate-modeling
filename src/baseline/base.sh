#!/bin/sh

#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32g
#SBATCH -t 24:00:00
#SBATCH -J LynD_base

source ~/.bashrc

loc=/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/src/baseline
cd $loc

conda activate /nas/longleaf/home/dieckhau/miniconda3/envs/sandbox 

python prep.py 
