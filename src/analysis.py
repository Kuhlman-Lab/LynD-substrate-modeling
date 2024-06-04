import argparse
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import pandas as pd


from trainer import SequenceModelPL
from dataset import LibraryDataset, SequenceDataset
from utils import sample_random_peptides, hamming_distance, compute_pairwise_epistasis, convert_seq_to_idx
from utils import sample_random_peptides_wCys

def screen_routine(args, N=10000, thresh=0.5, H=2, save=False, length=6, filter='C4'):
    """
    Screen random library of N peptides, then filter by activity threshold and Hamming distance
    """
    # load model
    model = SequenceModelPL.load_from_checkpoint(args.ckpt, args=args).model
    model = model.to('cuda')
    model.eval()
    # sample N random peptides
    from ecfp import constants
    np.random.seed(args.seed)
    # TODO constrain to always include 1 Cys
    if args.include_C: 
       library = sample_random_peptides_wCys(n=N, length=length, amino_acids=constants.aas)
    else:
        library = sample_random_peptides(n=N, length=length, amino_acids=constants.aas)

    # TODO filter based on filter arg if present
    if filter is not None:
        aa = filter[0]
        n = int(filter[1:]) - 1
        library = np.array([seq for seq in library if seq[n] == aa])
        print('Library filtered according to %s down to %s sequences' % (filter, str(len(library))))
        print(library[:5])

    pred_list = []
    # batch peptides and run them through the network
    ds = LibraryDataset(library, features=args.features)
    loader = DataLoader(ds, num_workers=8, shuffle=False, batch_size=1024)
    for batch in tqdm(loader):
        batch = batch.to('cuda') 
        preds = model(batch)
        preds = torch.squeeze(preds, -1)
        pred_list.append(preds) 
    pred_list = torch.cat(pred_list, dim=0).to('cpu')
    # filter peptides by predicted activity
    if thresh > 0:
        if args.activity == 'high':
            library = library[pred_list > thresh, :]
            pred_list = pred_list[pred_list > thresh]
        else:
            library = library[pred_list < thresh, :]
            pred_list = pred_list[pred_list < thresh]
        print('%s peptides passed predicted activity threshold' % str(library.shape[0]))    

    return library, pred_list

    # NOTE: custom filter for removing extra Cys or doing other custom filtering (e.g., Cys position)
    # custom filter to remove any w/more than N cys
    # n_C = 2
    # ncys_flags = [''.join(li).count('C') < n_C for li in library]

    # library = library[ncys_flags]
    # pred_list = pred_list[ncys_flags]
    # print('%s peptides passed N_cys filter' % str(library.shape[0]))
    
    # filter peptides by hamming distance - lower value is more similar
    # if H > 0:
    #     # load training data
    #     ref = SequenceDataset(sele=args.sele, 
    #                             anti=args.anti, 
    #                             features='onehot', 
    #                             variable_region=args.variable_region)
        
    #     training_peptides = ref.df.var_seq.values.astype('str')
    #     training_peptides = np.array([ list(word) for word in training_peptides ])
        
    #     flag = []
    #     min_H = []
    #     closest_seq = []
    #     for pep in tqdm(library):
    #         distances = hamming_distance(P=training_peptides, pep=pep, return_distance=True)
    #         print(training_peptides)
    #         print(pep)
    #         quit()
    #         # matches.shape array [N, ] of distances between query peptide pep and every member of dataset P
    #         min_dist = np.min(distances)
    #         flag.append(min_dist > H)
    #         min_H.append(min_dist)
    #         # TODO add most similar seq to each example
    #         sim_seq = training_peptides[np.argmin(distances), :]
    #         closest_seq.append(sim_seq)
            
    #     library = library[flag, :]
    #     pred_list = pred_list.detach().numpy()[flag]
    #     min_H = np.array(min_H)[flag]
    #     closest_seq = np.array(closest_seq)[flag]
    #     print('%s peptides passed Hamming distance threshold (H > %s)' % (str(library.shape[0]), str(H)))

    # if save:
    #     library = np.array([''.join(li) for li in library])
    #     closest_seq = np.array([''.join(cs) for cs in closest_seq])
    #     df = pd.DataFrame({
    #         'seq': library, 
    #         'predicted_activity': pred_list,
    #         'min_training_seq_hamming_dist': min_H, 
    #         'most_similar_training_seq': closest_seq
    #     })
    #     df.to_csv('library.csv', index_label='index')
    # else:
    #     return library, pred_list


def main(args):
    
    # generate N random peptides and screen them through our classifier to select promising peptides
    if args.variable_region is not None:
        VAR_LENGTH = len(args.variable_region)
    else:
        VAR_LENGTH = args.variable_length
    
    if args.routine == 'screen':
        # Note: thresholding is quick, but Hamming cutoff is slow
        screen_routine(args, H=args.hamming, N=100000, thresh=args.thresh, save=True, length=VAR_LENGTH, filter=args.filter_seq)
        
    # generate N random peptides and use them to calculate epistasis constants
    elif args.routine == 'epistasis':
        library, preds = screen_routine(args, H=-1, N=10000000, thresh=-1, length=VAR_LENGTH, filter=args.filter_seq)
        library = convert_seq_to_idx(library)
        epi, proba = compute_pairwise_epistasis(library, preds.detach().numpy())
        
        # plot epistasis of positions vs positions
        from Plotter import epistasis_bw_positions
        epistasis_bw_positions(epi, 'epi_bw_pos') # collapses along AA dim to get epi over each position
        
        # plot epistasis of amino acids vs amino acids for 2 specific positions
        # from Plotter import positional_epistasis
        # positional_epistasis(epi, pos1=2, pos2=4, basename='epi_pos_2v4')
        
        # plot expected epistasis for a specific peptide
        # from Plotter import pep_epistatic_interactions
        # pep_epistatic_interactions(epi, pep=np.array('ACLMCVTM'), basename='example_pep_epi')
        
    return        



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=str, help='which features to use (onehot, continuous, or ECFP)', default='onehot')
    parser.add_argument('--model', type=str, help='Which model type to use (MLP, CNN, Transf)', default='MLP')
    parser.add_argument('--learning_rate', type=float, help='initial network learning rate', default=1e-3)

    # new analysis arguments
    parser.add_argument('--routine', type=str, help='analysis routine to run (screen, epistasis)', default='screen')
    parser.add_argument('--ckpt', type=str, help='checkpoint file from which to load weights', default='./ckpt.pt')
    parser.add_argument('--variable_region', nargs='+', type=int, help='variable seq positions', default=None)
    parser.add_argument('--variable_length', default=6, type=int, help='only needed if variable_region is None')
    parser.add_argument('--filter_seq', type=str, default=None, help='Keep only those with this condition (e.g., C1)')
    
    # screening args
    parser.add_argument('--hamming', help='hamming distance cutoff to use for validation screen (keep only those peptides with MIN_SIMILARITY > H)', default=-1, type=int)
    parser.add_argument('--thresh', help='classifier threshold to use as screening filter (0.8-0.99)', type=float, default=0.97)
    parser.add_argument('--sele', help='location of selection csvs for hamming filter (optional)', default='../data/csv/NNK7/rd2/sele', type=str)
    parser.add_argument('--anti', help='location of antiselection csvs for hamming filter (optional)', default='../data/csv/NNK7/rd2/anti', type=str)
    parser.add_argument('--seed', help='random seed for replication of random peptide screens', default=1234, type=int)
    parser.add_argument('--include_C', help='force Cys inclusion in the peptide instead of random (implicit) inclusion (needed for NNK7 but not NNK6)', action='store_true')
    parser.add_argument('--activity', help='filter for or high activity?', default='high', type=str)
    main(parser.parse_args())




