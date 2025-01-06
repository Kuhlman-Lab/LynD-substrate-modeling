# LynD-substrate-modeling

This repository contains the code, model weights, and raw data for our paper on LynD substrate specificity ([Steude et al., 2024](https://www.biorxiv.org/content/10.1101/2024.10.14.618330)).

## Citation

If this code or model proves useful for you, please include the following citation in your work:
```
@article {Steude2024.10.14.618330,
	author = {Steude, Emma G. and Dieckhaus, Henry and Pelton, Jarrett M. and Kuhlman, Brian and Bowers, Albert A.},
	title = {Assessing substrate scope of the cyclodehydratase LynD by mRNA display-enabled machine learning models},
	year = {2024},
	doi = {10.1101/2024.10.14.618330},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2024/10/14/2024.10.14.618330},
	eprint = {https://www.biorxiv.org/content/early/2024/10/14/2024.10.14.618330.full.pdf},
	journal = {bioRxiv}
}
```

Note: While the model training and nomination code is original to this repo, we make heavy use of analysis and visualization functions from the [LazDEF analysis](https://github.com/avngrdv/mRNA-display-deep-learning.git) repository by Vinogradov et al. We have attempted to note where these tools are used/adapted throughout the source code. We thank the authors of this work for making these useful tools open-source and well-organized.

## Repo Contents:

`datasets/`: Preprocessed sequencing data for selection rounds 1-3 in CSV format.\
`src/model_weights`: Trained ML model weights for LynD NNK7 semi-randomized library rounds 1-3.\
`src/notebooks/`: Jupyter notebook walkthroughs for exploratory data analysis (`eda.ipynb`) and peptide nomination (`nom.ipynb`).

`src/trainer.py`: Script for training new models.\
`src/analysis.py`: Script for analyzing patterns in substrate tolerance and epistasis.\
`src/pct_mod.py`: Script for assessing percent-modification levels for partially randomized libraries (used for epi analysis).\
`src/runscripts`: Example runscripts for the training and analysis code with default options used in the paper.

`src/environment.yml`: List of python dependencies for installation using conda.

## Installation

We use Mamba and git to install and manage the necessary python dependencies:
```
git clone git@github.com:Kuhlman-Lab/LynD-substrate-modeling.git
cd LynD-substrate-modeling

mamba create -n LynD python=3.10
mamba activate LynD
pip install .
```
Install PyTorch dependencies:
```
mamba install pytorch=2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```
Install other dependencies:
```
mamba install -c conda-forge omegaconf "numpy<2.0" wandb tqdm pandas torchmetrics pytorch-lightning matplotlib seaborn
```

## Training
To see a full list of training options, run ```python trainer.py -h```. Training for 50 epochs should take around 15-20 minutes on a V100 GPU with 8 CPU workers. To run with default settings:
```
cd src/runscripts
sh train.sh
```

## Inference (nomination)
Inference and peptide nomination/filtering code is provided in ```src/nomination.ipynb```.

## Analysis
Several analysis functions are included in ```src/analysis.py``` and example inputs are provided in ```src/runscripts/analyze.sh```

To screen a set of peptides based on activity threshold and Hamming distance constraints:
```
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
```

To calculate PPV curves for S score and model predictions on a given library:
```
python analysis.py \
--features onehot \
--model MLP \
--routine threshold \
--variable_region 14 15 16 17 18 19 20 21 \
--ckpt model_weights/LynD_NNK7_rd3.ckpt \
--include_C \
--seed 1234 \
--sele /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/sele/ \
--anti /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/anti/
```

To calculate epistasis matrices for a semi-randomized library:
```
python analysis.py \
	--features onehot \
	--model MLP \
	--routine epistasis \
	--variable_region 14 15 16 17 18 19 20 21 \
	--ckpt model_weights/LynD_NNK7_rd3.ckpt \
	--include_C \
	--sele /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/sele/ \
	--anti /proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd3/anti/
```

## Contact

General questions about the LynD manuscript should be directed to the corresponding author (Dr. Albert Bowers). However, questions regarding this code repository should be raised by opening a GitHub issue on this repo.

## License
This work is made available under an MIT license. See the LICENSE file for details.
