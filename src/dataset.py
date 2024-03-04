import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F


class SequenceDataset(torch.utils.data.Dataset):
    """Dataset class for LynD Sequence Data"""
    
    def __init__(self, sele, anti):
        
        # load dataframes and remove duplicates and overlap
        sele_df, anti_df = import_data(sele, anti)
        
        # featurize and combine dataframes
        self.df = featurize(sele_df, anti_df)

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
        one_hot = F.one_hot(idx_list, num_classes=20) # [120]
        return torch.flatten(one_hot).to(torch.float32), sample.active
    
     
def import_data(sel='../../data/csv/A3_Sup_pos_LynD_R1_001.csv', 
                anti='../../data/csv/A5_Elu_pos_LynD_R1_001.csv'):
    """Import and curate data for modeling"""
    sel = pd.read_csv(sel)
    sel = sel.drop_duplicates(subset=['seq'])
    
    anti = pd.read_csv(anti)
    anti = anti.drop_duplicates(subset=['seq'])

    # find intersecting seqs and remove them as unreliable
    intersect = np.intersect1d(anti.seq.values, sel.seq.values)
    sel = sel.loc[~sel.seq.isin(intersect)].reset_index(drop=True)
    anti = anti.loc[~anti.seq.isin(intersect)].reset_index(drop=True)
    
    # extract variable-region sequence for modeling
    sel['var_seq'] = sel.seq.str[22:25] + sel.seq.str[26:29]
    anti['var_seq'] = anti.seq.str[22:25] + anti.seq.str[26:29]
    
    print('Curated dataset sizes:\nSelection:\t', sel.shape[0], '\nAntiselection:\t', anti.shape[0])
    return sel, anti


def featurize(sel, anti):
    """Combine and featurize dataset into sklearn-compatible format"""
    
    # extract variable region sequence
    sel['var_seq'] = sel.seq.str[22:25] + sel.seq.str[26:29]
    anti['var_seq'] = anti.seq.str[22:25] + anti.seq.str[26:29]
    
    # label active/inactive splits
    sel['active'] = 1
    anti['active'] = 0
    
    # combine into a single dataset
    df = pd.concat([sel, anti], axis=0).reset_index(drop=True)
    return df

