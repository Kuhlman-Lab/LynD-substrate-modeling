import torch
import pandas as pd
import numpy as np
import os
from glob import glob
import torch.nn.functional as F
from ecfp import dense_morgan


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset class for training on LynD Sequence Data"""
    def __init__(self, sele, anti, features='onehot', variable_region=None, filter_seq=None):
        # load dataframes and remove duplicates and overlap
        sele_df, anti_df = import_data(sele, anti)
        self.features = features

        # featurize and combine dataframes
        # variable_region = [[22, 25], [26, 29]] if 'LynD' in sele else None # use variable region to extract as needed
        self.df = featurize(sele_df, anti_df, vr=variable_region)

        if filter_seq is not None:
            self._filter_seq(filter_seq)

        if self.features == 'ECFP':
            # generate feature matrix
            self.matrix, _, _ = dense_morgan(4, False)
            self.matrix = torch.tensor(self.matrix)
        return
    
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index):
        """Retrieve a single sample and featurize it"""
        sample = self.df.iloc[index]
        seq = sample.var_seq
    
        # convert variable seq region to an array of AA indices     
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        idx_list = []
        for aa in seq:
            idx = alphabet.index(aa)
            idx_list.append(idx)
        
        idx_list = torch.flatten(torch.tensor(idx_list)) # [6]
        
        if self.features == 'onehot':
            features = F.one_hot(idx_list, num_classes=20).to(torch.float32) # [120]
        
        elif self.features == 'continuous':
            features = idx_list
        
        elif self.features == 'ECFP':
            idx_list = idx_list[:, None].repeat(1, self.matrix.shape[-1]) # reshape to [6, 208]
            features = torch.gather(self.matrix, 0, idx_list) # [6, 208]
            features = features.to(torch.float32) # [6, 208]

        else:
            raise ValueError(f"Invalid feature set {self.features} selected!")
        return features, sample.active

    def _filter_seq(self, filter_seq='C1-4'):
        """Remove any sequences matching filter seq criteria (e.g., all with Cys in positions 1-4)"""
        aa = filter_seq[0]
        pos_range = filter_seq[1:].split('-')
        pos_range = [int(p) for p in pos_range]
        mask = self.df['var_seq'].str[pos_range[0] - 1: pos_range[1]].str.contains(aa)
        self.df = self.df.loc[~mask].reset_index(drop=True)
        print('Dataset filtered to remove pattern %s, leaving %s data points' % (filter_seq, str(self.df.shape[0])))
        return


class LibraryDataset(torch.utils.data.Dataset):
    """Sequence library dataset for nomination purposes."""
    def __init__(self, x, features='onehot'):
                
        self.X = x  # should be array [N, L] where N is number of peptides and L is constant peptide length
        self.features = features

        if self.features == 'ECFP':
            # generate feature matrix
            self.matrix, _, _ = dense_morgan(4, False)
            self.matrix = torch.tensor(self.matrix)
        return
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, index):
        
        sample = self.X[index, :]
       
        # convert variable seq region to an array of AA indices     
        alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        idx_list = []
        for aa in sample:
            idx = alphabet.index(aa)
            idx_list.append(idx)
        
        idx_list = torch.flatten(torch.tensor(idx_list)) # [6]
        
        if self.features == 'onehot':
            features = F.one_hot(idx_list, num_classes=20).to(torch.float32) # [120]
        
        elif self.features == 'continuous':
            features = idx_list
        
        elif self.features == 'ECFP':
            idx_list = idx_list[:, None].repeat(1, self.matrix.shape[-1]) # reshape to [6, 208]
            features = torch.gather(self.matrix, 0, idx_list) # [6, 208]
            features = features.to(torch.float32) # [6 x 208]

        else:
            raise ValueError(f"Invalid feature set {self.features} selected!")
        return features
            
     
def import_data(sel='../../data/csv/A3_Sup_pos_LynD_R1_001.csv', 
                anti='../../data/csv/A5_Elu_pos_LynD_R1_001.csv', var_region=None):
    """Import and curate data for modeling use. Combines multiple CSVs (sets of reads) if compatible."""    
    def get_df(location, header=None):
        """Load file or, if dir, grab all files in all subdirectories recursively"""
        
        if os.path.isdir(location):
            files = [y for x in os.walk(location) for y in glob(os.path.join(x[0], '*.csv'))]
            if header is None:
                df = pd.concat([pd.read_csv(os.path.join(location, f), header=header) for f in files])
            else:
                df = pd.concat([pd.read_csv(os.path.join(location, f)) for f in files])
    
        else:
            if header is None:
                df = pd.read_csv(location, header=header)
            else:
                df = pd.read_csv(location)
    
        if header is None:
            df.columns = ['seq', None]
        return df.drop_duplicates(subset=['seq'])
    
    header = 1 if 'LynD' in sel else None

    sel = get_df(sel, header)
    anti = get_df(anti, header)

    # find intersecting seqs and remove them as unreliable
    intersect = np.intersect1d(anti.seq.values, sel.seq.values)
    sel = sel.loc[~sel.seq.isin(intersect)].reset_index(drop=True)
    anti = anti.loc[~anti.seq.isin(intersect)].reset_index(drop=True)
    
    print('Curated dataset sizes:\nSelection:\t', sel.shape[0], '\nAntiselection:\t', anti.shape[0])
    return sel, anti


def featurize(sel, anti, vr=None):
    """Combine and featurize dataset into model-compatible format"""
    if vr is None:
        sel['var_seq'] = sel['seq']
        anti['var_seq'] = anti['seq']
    else:
        for df in (sel, anti):
            seqs = []
            for full_str in df.seq.values:
                s = [full_str[v_i: v_i + 1] for v_i in vr]
                seqs.append(''.join(s))
            df['var_seq'] = seqs
            
    # label active/inactive splits
    sel['active'] = 1
    anti['active'] = 0
    sel = sel.loc[~sel['var_seq'].str.contains('X')]
    anti = anti.loc[~anti['var_seq'].str.contains('X')]

    # combine into a single dataset
    return pd.concat([sel, anti], axis=0).reset_index(drop=True)
