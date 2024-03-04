import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd


from trainer import SequenceModelPL
from dataset import LibraryDataset, SequenceDataset
from utils import sample_random_peptides, hamming_distance, compute_pairwise_epistasis, convert_seq_to_idx
# TODO implement epistasis calculation

# TODO implement integrated gradients



def screen_routine(args, N=10000, thresh=0.5, H=2, save=False):
    """
    Screen random library of N peptides, then filter by activity threshold and Hamming distance
    """
    # load model
    model = SequenceModelPL.load_from_checkpoint(args.ckpt, args=args).model
    
    # sample N random peptides
    from ecfp import constants
    library = sample_random_peptides(n=N, length=6, amino_acids=constants.aas)
    
    pred_list = []
    # batch peptides and run them through the network
    ds = LibraryDataset(library, features=args.features)
    loader = DataLoader(ds, num_workers=8, shuffle=False, batch_size=2048)
    for batch in tqdm(loader):
        batch = batch.to('cuda') 
        preds = model(batch)
        preds = torch.squeeze(preds, -1)
        pred_list.append(preds)
    
    pred_list = torch.cat(pred_list, dim=0).to('cpu')

    # filter peptides by predicted activity
    if thresh > 0:
        library = library[pred_list > thresh, :]
        pred_list = pred_list[pred_list > thresh]
        print('%s peptides passed predicted activity threshold' % str(library.shape[0]))    

    # filter peptides by hamming distance - lower value is more similar
    if H > 0:
        # load training data
        ref = SequenceDataset(sele='../data/csv/A3_Sup_pos_LynD_R1_001.csv', 
                                anti='../data/csv/A5_Elu_pos_LynD_R1_001.csv', 
                                features='onehot')
        
        training_peptides = ref.df.var_seq.values.astype('str')
        training_peptides = np.array([ list(word) for word in training_peptides ])
        
        flag = []
        for pep in tqdm(library):
            # NOTE: most random peptides are w/in H=2 of something in the training set...
            matches = hamming_distance(P=training_peptides, pep=pep, h=H, cum=True, return_count=True)
            # matches.shape is number of matches
            flag.append(matches < 1)

        library = library[flag, :]
        pred_list = pred_list[flag]
        print('%s peptides passed Hamming distance threshold (H <= %s)' % (str(library.shape[0]), str(H)))

    if save:
        library = np.array([''.join(li) for li in library])
        df = pd.DataFrame({
            'seq': library, 
            'pred_list': pred_list.detach().numpy()
        })
        df.to_csv('library.csv', index_label='index')
    else:
        return library, pred_list.detach().numpy()


def main(args):
    
    # generate N random peptides and screen them through our classifier to select promising peptides
    if args.routine == 'screen':
        # Note: thresholding is quick, but Hamming cutoff is slow
        screen_routine(args, H=-1, N=100000, thresh=0.5, save=True)
        

    # generate N random peptides and use them to calculate epistasis constants
    elif args.routine == 'epistasis':
        library, preds = screen_routine(args, H=-1, N=1000000, thresh=0.5)
        print(library.shape, preds.shape)
        library = convert_seq_to_idx(library)
        epi, proba = compute_pairwise_epistasis(library, preds)
        
        # plot epistasis of positions vs positions
        # from Plotter import epistasis_bw_positions
        # epistasis_bw_positions(epi, 'epi_bw_pos') # collapses along AA dim to get epi over each position
        
        # plot epistasis of amino acids vs amino acids for 2 specific positions
        # from Plotter import positional_epistasis
        # positional_epistasis(epi, pos1=3, pos2=4, basename='epi_pos_3v4')
        
        # plot expected epistasis for a specific peptide
        from Plotter import pep_epistatic_interactions
        pep_epistatic_interactions(epi, pep=library[14], basename='example_pep_epi')
        
    return        



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, help='which features to use (onehot, continuous, or ECFP)', default='onehot')
    parser.add_argument('--model', type=str, help='Which model type to use (MLP, CNN, Transf)', default='MLP')
    parser.add_argument('--learning_rate', type=float, help='initial network learning rate', default=1e-3)

    # new analysis arguments
    parser.add_argument('--routine', type=str, help='analysis routine to run (screen, epistasis)', default='screen')
    parser.add_argument('--ckpt', type=str, help='checkpoint file from which to load weights', default='./ckpt.pt')
    main(parser.parse_args())




