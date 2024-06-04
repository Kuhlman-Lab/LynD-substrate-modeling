# NOTE: taken from https://github.com/dohlee/abyssal-pytorch/

# set location to download/find ESM models
import os
os.environ['TORCH_HOME'] = '/proj/kuhl_lab/users/dieckhau/torch_hub/'

import torch

import pandas as pd
import numpy as np
import sys
sys.path.append('/proj/kuhl_lab/esmfold/esm-main/')

import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--input', '-i', type=str, default='data/mega.train.csv', help='input csv of seqs to use')
parser.add_argument('--output-dir', '-o', type=str, default='data/embeddings')
parser.add_argument('--batch-size', '-b', type=int, default=496)
parser.add_argument('--repr_layer', type=int, default=6)
parser.add_argument('--label', help='filename label for embeddings', type=str, default='emb')
parser.add_argument('--model', help='name of ESM model to load', default='esm2_t6_8M_UR50D', type=str)
args = parser.parse_args()

# prep dataframe - get list of sequences to featurize
df = pd.read_csv(args.input)
df = df.drop_duplicates(subset=['seq']).reset_index(drop=True)

print('Running embeddings for %s sequences with batch size %s' % (str(df.shape[0]), str(args.batch_size)))

# Load ESM-2 model
if args.model == 'esm2_t6_8M_UR50D':
    from esm.pretrained import esm2_t6_8M_UR50D
    model, alphabet = esm2_t6_8M_UR50D()
elif args.model == 'esm2_t33_650M_UR50D':
    from esm.pretrained import esm2_t33_650M_UR50D
    model, alphabet = esm2_t33_650M_UR50D()
else:
    raise ValueError("Invalid ESM model name %s provided!\tPlease use one of these options: (esm2_t6_8M_UR50D, esm2_t33_650M_UR50D)" % args.model)
batch_converter = alphabet.get_batch_converter()
model.eval()
model = model.cuda()
print('Loaded ESM model')

df['var_seq'] = df.seq.str[22:25] + df.seq.str[26:29]
# TODO: if running truncated embeddings, do this NOW before inference
# df['seq'] = df['var_seq']

# Save embeddings
for i in tqdm(range(0, len(df), args.batch_size)):
    batch = df.iloc[i:i + args.batch_size]

    # Get ESM2 embeddings.
    data = [(i, seq.strip('*')) for i, seq in enumerate(batch['seq'].values)]
    # print(data[0])
    _, _, tokens = batch_converter(data)
    tokens = tokens.cuda()
    with torch.no_grad():
        result = model(tokens, repr_layers=[args.repr_layer])
    
    # grab embeddings along variable region indices
    pos = (22, 29)
    h = result['representations'][args.repr_layer][range(len(batch)), pos[0]:pos[1]].cpu().numpy() # [B, L, EMBED_DIM]
    # print(h.shape, '**')
    
    # save embeddings in custom filetree based on var_seq
    variable_seqs = batch.var_seq
    
    for idx, vseq in enumerate(variable_seqs):
        # print(vseq, idx)
        cpath = args.output_dir
        # build filetree for each seq
        for aa in vseq:
            cpath = os.path.join(cpath, aa)
            # print(cpath)
            if not os.path.isdir(cpath):
                os.mkdir(cpath)
        emb = h[idx, ...] # [L, EMBED_DIM]
        fname = os.path.join(cpath, args.label + '.pt')
        torch.save(emb, fname)
        
    # quit()