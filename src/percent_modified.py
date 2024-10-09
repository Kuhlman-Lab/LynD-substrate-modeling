import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils import sample_random_peptides_wCys, sample_random_peptides, convert_seq_to_idx, sample_random_peptides_biased, sample_random_peptides_templated
from trainer import SequenceModelPL
from dataset import LibraryDataset, SequenceDataset
from analysis import screen_routine


def percent_modified(args):
    """
    Screen random library of N peptides, then filter by activity threshold
    """
    # load model
    model = SequenceModelPL.load_from_checkpoint(args.ckpt, args=args).model
    model = model.to('cuda')
    model.eval()
    # sample N random peptides
    from ecfp import constants
    np.random.seed(args.seed)
    
    # random sampling technique 1 (add cys on top)
    # library = sample_random_peptides_wCys(n=args.seqs, length=args.length, amino_acids=constants.aas)
    
    # random sampling technique 2 (filter cys later)
    # library = sample_random_peptides(n=args.seqs, length=args.length, amino_acids=constants.aas)
    # library = np.array([seq for seq in library if 'C' in seq])
    # print(f'Library filtered according down to {len(library)} sequences')

    # biased sampling technique (filter cys later but use biased sampling first based on QC dataset)
    ref = SequenceDataset(sele='/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd1/QC/', 
                          anti='/proj/kuhl_lab/users/dieckhau/LynD-substrate-modeling/LynD-substrate-modeling/datasets/rd1/QC/', 
                          features=args.features, variable_region=args.variable_region)

    ref.df = ref.df.drop_duplicates(subset=['var_seq'])
    library = ref.df.var_seq.values.astype('str')
    library = np.array([list(word) for word in library])
    library = convert_seq_to_idx(library)

    freqs = np.zeros((20, 8), dtype=float)
    for i in range(library.shape[1]):
        u, counts = np.unique(library[:, i], return_counts=True)
        counts = counts / library.shape[0]
        freqs[:, i] = counts

    library = sample_random_peptides_biased(n=args.seqs, length=args.length, amino_acids=constants.aas, bias=freqs)
    library = np.array([seq for seq in library if 'C' in seq])
    print(f'Library filtered according down to {len(library)} sequences')

    pred_list = []
    # batch peptides and run them through the network
    ds = LibraryDataset(library, features=args.features)
    loader = DataLoader(ds, num_workers=8, shuffle=False, batch_size=4096)
    for batch in tqdm(loader):
        batch = batch.to('cuda')
        preds = model(batch)
        preds = preds.squeeze(-1)
        pred_list.append(preds.cpu().detach().numpy())
  
    pred_list = np.concatenate(pred_list, axis=0)

    # calculate % passing specified threshold
    thresholds = [0.01, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.99]
    for thresh in thresholds:
        pct = 100 * (np.sum(pred_list > thresh) / pred_list.shape[0])
        print(f'{round(pct)}% predicted with at least {thresh} modification')


def avg_modified(args, peptide='XXXXXXXX'):
    """Generate many peptides, filter by a given motif, then calculate avg modification"""
    assert len(peptide) == args.length

    # load model
    model = SequenceModelPL.load_from_checkpoint(args.ckpt, args=args).model
    model = model.to('cuda')
    model.eval()
    # sample N random peptides
    from ecfp import constants
    np.random.seed(args.seed)
    
    # aas = constants.aas
    aas = [aa for aa in constants.aas if aa != 'C']
    # purely random sampling
    library = sample_random_peptides_templated(n=args.seqs, length=args.length, amino_acids=aas, peptide=peptide)
    library = np.array([seq for seq in library if 'C' in seq])
    # drop any early C options
    # library = np.array([seq for seq in library if not 'C' in seq[:4]])
    print(f'Library generated containing {len(library)} sequences')

    # run all through model and take avg modification score
    print(library.shape)
    pred_list = []
    # batch peptides and run them through the network
    ds = LibraryDataset(library, features=args.features)
    loader = DataLoader(ds, num_workers=8, shuffle=False, batch_size=4096)
    for batch in tqdm(loader):
        batch = batch.to('cuda')
        preds = model(batch)
        preds = preds.squeeze(-1)
        pred_list.append(preds.cpu().detach().numpy())
  
    pred_list = np.concatenate(pred_list, axis=0)

    mean_pred = np.mean(pred_list)
    print(f'{peptide} seeded library of size {args.seqs} has avg score {mean_pred}')

    pct = 100 * (np.sum(pred_list > 0.9) / pred_list.shape[0])
    print(f'{round(pct)}% predicted with at least {0.9} modification')


params = {
    'features': 'onehot', 
    'model': 'MLP', 
    'learning_rate': 0.01, 
    'ckpt': 'model_weights/LynD_NNK7_rd3.ckpt', 
    # 'seqs': 15000000, # for 1+Cys biased sampling
    # 'seqs': 10000000, # for wCys sampling
    # 'seqs': 30000000, # for 1+Cys sampling
    'seqs': 1000000, # for epi peptide testing
    'variable_region': [14, 15, 16, 17, 18, 19, 20, 21], 
    'length': 8,
    'seed': 1234,
    'include_C': True
}

params = OmegaConf.create(params)

peptide_sets = {
    'LSMQKCYS': [
        'XXXXXCXX', 
        'LXXXXCXX',
        'XSXXXCXX', 
        'XXMXXCXX', 
        'XXXQXCXX', 
        'XXXXKCXX', 
        'XXXXXCXX',
        'XXXXXCYX', 
        'XXXXXCXS',

        'XXMXKCXX', 
        'XXXXKCYX', 
        'XXMXXCYX',

        'LSMQKCYS', 
        'XXXXXXXX',
    ], 
    'DQFDLCMK': [
        'XXXXXCXX', 
        'DXXXXCXX',
        'XQXXXCXX', 
        'XXFXXCXX', 
        'XXXDXCXX', 
        'XXXXLCXX', 
        'XXXXXCXX',
        'XXXXXCMX', 
        'XXXXXCXK',

        'DQFDLCMK', 
        'XXXXXXXX',
     
        'XXXXLCXK', 
        'XXXXXCMK', 
        'XXXXLCMX', 
        'XXXDXCXK',

        'XXXXLCMK',
        'XXXDLCMK',
        'XXFDLCMK',
        'XQFDLCMK',

        'XXFXLCMK', 
        'XXFDXCMK', 
        'XXFDLCXK', 
        'XXFDLCMX', 
        'XXFXXCXK',
        'XXFDXCXK',
        'XXFXLCXK',
        'XXFXXCMK',
        'XQFXXCXK',
        'DXFXXCXK',
    ], 
    'FATFNCPL': [
        'XXXXXXXX', 
        'FATFNCPL', 
        'XXXXXCXX', 

        'FXXXXCXX', 
        'XAXXXCXX', 
        'XXTXXCXX',
        'XXXFXCXX', 
        'XXXXNCXX', 
        'XXXXXCXX', 
        'XXXXXCPX', 
        'XXXXXCXL',

        'FATFXCXL', 
        'XXXXNCPX',

        'FATFNCXL', 
        'FATFXCPL', 
        'FATFNCPL', 
        
        'XXXXNCPX', 
        'FXXXNCPX', 
        'XAXXNCPX', 
        'XXTXNCPX', 
        'XXXFNCPX',
        'XXXXNCPL',
    ], 
    'LWISWCCY': [
        'XXXXXXXX', 
        'XXXXXCXX', 
        'XXXXXXCX', 
        'XXXXXCCX', 
        'LWISWCCY', 
        
        'LXXXXCCX', 
        'XWXXXCCX', 
        'XXIXXCCX', 
        'XXXSXCCX', 
        'XXXXWCCX', 
        'XXXXXCCY', 

        'XXXSXCCY', 
        'XXXSWCCY',
        'XXXSWCCX',
        'XXXXWCCY',

        'XXISWCCY',
        'XWISWCCY',
        'LWISWCCY'

        'LWISWCCY',
        'LWISWXCY',
        'LWISWCXY',

        'LWISWCXY', 
        'XWISWCXY', 
        'LXISWCXY', 
        'LWXSWCXY', 
        'LWIXWCXY', 
        'LWISXCXY', 
        'LWISWCXX',

        'LXISWCCY',
        'LXISWXXY',
    ], 
    'NWPERCLC': [
        'NWPERCLC', 
        'XXXXXCXX', 
        'XXXXXXXC', 
        'XXXXXCXC',

        'NWPERCLX', 
        'NWPERXLC',

        'NXXXXCXC', 
        'XWXXXCXC',
        'XXPXXCXC',
        'XXXEXCXC',
        'XXXXRCXC',
        'XXXXXCLC',

        'NXXERCXC', 
        'XWPXXCXC', 
        'NWPERCXC', 
        'NXXERCLC', 
        'XWPXXCLC',
        'XXXXXCLX', 
        'XXXXXXLC'
    ]

}

peptides = peptide_sets['NWPERCLC']
for pep in peptides:
    avg_modified(params, pep)

# percent_modified(params)
