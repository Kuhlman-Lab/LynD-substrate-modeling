export CUDA_VISIBLE_DEVICES=3

echo "Running on GPU $CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate thermoMPNN

python trainer.py \
	--sele /home/hdieckhaus/scripts/LynD-substrate-modeling/data/csv/A3_Sup_pos_LynD_R1_001.csv \
	--anti /home/hdieckhaus/scripts/LynD-substrate-modeling/data/csv/A5_Elu_pos_LynD_R1_001.csv \
	--epochs 50 \
	--features ECFP \
	--run Transf_ECFP \
	--model Transf \
	--seed 1234
