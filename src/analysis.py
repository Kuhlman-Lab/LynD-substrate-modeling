import argparse
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt


from trainer import SequenceModelPL
from dataset import LibraryDataset, SequenceDataset
from utils import sample_random_peptides, hamming_distance, compute_pairwise_epistasis, convert_seq_to_idx, compute_pairwise_epistasis_exp
from utils import sample_random_peptides_wCys, sample_random_peptides_biased


def s_score(df, y_star):
    """
    S-score calculation for a specific peptide based on y_star scores for the library
    Formula obtained from https://pubs.acs.org/doi/10.1021/acscentsci.2c00223
    Serves as a (weak, naive) expression of peptide favorability, but ignores epistatic effects    
    """
    alphabet = 'AVLIMCSTNQGPFYWHKRDE'
    y_star = np.log2(y_star)
    seqs = df['var_seq']
    s_scores = []
    # iterate over seqs
    for s in tqdm(seqs):
        y_list = []
        # iterate over positions
        flag = True
        for idx, aa in enumerate(s):
            try:
                y = y_star[idx, alphabet.index(aa)]
                y_list.append(y)
            except ValueError:
                flag = False
                break
        if flag:
            s_scores.append(np.sum(y_list))
    return s_scores


def y_star_score(sel_df, anti_df):
    """
    y-star score for a library of NGS counts
    Formula obtained from https://pubs.acs.org/doi/10.1021/acscentsci.2c00223
    Serves as an expression of favorability of a specific amino acid / position based on SSM / NGS data
    """
    
    def y_star_single(sel_df, anti_df, p, aa):
        alphabet = 'AVLIMCSTNQGPFYWHKRDE'
        f_sel = sel_df.loc[sel_df['var_seq'].str[p] == alphabet[aa]]['count'].sum() / sel_df['count'].sum()
        f_anti = anti_df.loc[anti_df['var_seq'].str[p] == alphabet[aa]]['count'].sum() / anti_df['count'].sum()
        return f_sel / f_anti
    
    seqs = sel_df['var_seq'].values
    pos_idx = len(seqs[0])
    aa_idx = 20
    y_star = np.zeros((pos_idx, aa_idx))
    
    for p in tqdm(range(pos_idx)):
        for aa in range(aa_idx):
            y_star[p, aa] = y_star_single(sel_df, anti_df, p, aa)    
    return y_star


def screen_routine(args, N=10000, thresh=0.5, H=2, save=False, length=6, filter='C4', bias=None):
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
    elif bias is not None:
        library = sample_random_peptides_biased(n=N, length=length, amino_acids=constants.aas, bias=bias)
        print(library.shape, '***')
    else:
        library = sample_random_peptides(n=N, length=length, amino_acids=constants.aas)

    # TODO filter based on filter arg if present
    if filter is not None:
        aa = filter[0]
        n = int(filter[1:]) - 1
        library = np.array([seq for seq in library if seq[n] == aa])
        print('Library filtered according to %s down to %s sequences' % (filter, str(len(library))))
        print(library[:5])

    # NOTE: custom filter for removing extra Cys or doing other custom filtering (e.g., Cys position)
    # custom filter to remove any w/more than N cys
    # n_C = 0
    # ncys_flags = [''.join(li).count('C') > n_C for li in library]

    # library = library[ncys_flags]
    # print('%s peptides passed N_cys filter' % str(library.shape[0]))

    pred_list = []
    # batch peptides and run them through the network
    ds = LibraryDataset(library, features=args.features)
    loader = DataLoader(ds, num_workers=8, shuffle=False, batch_size=4096)
    for batch in tqdm(loader):
        batch = batch.to('cuda') 
        preds = model(batch)
        preds = torch.squeeze(preds, -1)
        pred_list.append(preds.cpu().detach().numpy())
        
    pred_list = np.concatenate(pred_list, axis=0)
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


def threshold_test(args):
    """Compare thresholding by model and S score"""
    print('Running thresholding routine for PPV calculation')
    full_dataset = SequenceDataset(sele=args.sele, anti=args.anti, features=args.features, variable_region=args.variable_region)
    
    # split dataset - replicate training splits
    # seed = 1234 # hardcoded, copied from training script
    # generator = torch.Generator()
    # generator = generator.manual_seed(seed)
    # train_dataset, val_dataset, test_dataset = random_split(full_dataset, lengths=(0.8, 0.1, 0.1), generator=generator)
    # test_loader = DataLoader(test_dataset, batch_size=2048, num_workers=8, shuffle=False)
    
    # # screen test set by model score
    # model = SequenceModelPL.load_from_checkpoint(args.ckpt, args=args).model
    # model = model.to('cuda')
    # model.eval()

    # pred_list = []
    # for batch in tqdm(test_loader):
    #     batch, _ = batch # drop true activity
    #     batch = batch.to('cuda') 
    #     preds = model(batch)
    #     preds = torch.squeeze(preds, -1)
    #     pred_list.append(preds.to('cpu')) 
    # pred_list = torch.cat(pred_list, dim=0).to('cpu')

    # thresholds = [0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
    # ppvs, passed = [], []
    # true_activity = torch.tensor([test_dataset.dataset.df.active[i] for i in test_dataset.indices]).to('cpu')

    # for thresh in thresholds:
    #     # calculate range of PPVs
    #     true_pos = torch.sum(true_activity[pred_list > thresh])
    #     pred_pos = (true_activity[pred_list > thresh]).numel()
    #     ppvs.append((true_pos / pred_pos).item())
    #     passed.append(pred_pos)

    # print(len(passed), len(thresholds), len(ppvs))

    # # get S score from training set and percentile convert it, then re-run PPV tests

    # train_df = train_dataset.dataset.df.iloc[train_dataset.indices]
    # # print(train_df.active.value_counts())
    # sel_df = train_df.loc[train_df.active.astype(bool)]
    # anti_df = train_df.loc[~train_df.active.astype(bool)]
    # # y_star = y_star_score(sel_df, anti_df) # fit to training set
    # # np.save('ystar.npy', y_star)

    # y_star = np.load('ystar.npy')

    # test_df = test_dataset.dataset.df.iloc[test_dataset.indices]
    # # ssc = s_score(test_df, y_star) # apply to test set
    # # np.save('sscore.npy', ssc)

    # test_activity = test_df.active
    # test_sscore = np.load('sscore.npy')
    # ppv_sscore, passed_sscore = [] , []
    # for thresh in thresholds:
    #     perc = np.percentile(test_sscore, q=thresh * 100.) 
    #     print(perc)
    #     pred_hits = test_activity[test_sscore > perc] # grab those with passed S score
    #     true_hits = pred_hits[test_activity.astype(bool)]
    #     print(pred_hits, true_hits)
    #     ppv_sscore.append(true_hits.shape[0] / pred_hits.shape[0])
    #     passed_sscore.append(pred_hits.shape[0])

    # disp  = pd.DataFrame({
    # 'Sequences Passed (Model)': passed, 
    # 'Sequences Passed (S Score)': passed_sscore,
    # 'Threshold': thresholds, 
    # 'PPV (Model)': ppvs, 
    # 'PPV (S Score)': ppv_sscore
    # })

    # disp.to_csv('disp.csv')
    # disp = pd.read_csv('disp.csv')
    # print(disp)

    # twin plot seq and threshold for PPV and Model
    # ax = disp.plot(x='Threshold', y='Sequences Passed (Model)', legend=False, color='steelblue', marker='s', linestyle='--')
    # ax = disp.plot(x='Threshold', y='Sequences Passed (S Score)', legend=False, color='darkslategrey', marker='o', ax=ax, linestyle='--')

    # ax2 = ax.twinx()
    # disp.plot(x='Threshold', y='PPV (Model)', ax=ax2, legend=False, color='olive', marker='s')
    # disp.plot(x='Threshold', y='PPV (S Score)', ax=ax2, legend=False, color='darkolivegreen', marker='o')

    # plt.title('NNK7 Library rd3\n')
    # ax.set_ylabel('Sequences Passed')
    # ax2.set_ylabel('PPV (%)')
    # ax.set_xlabel('Threshold')
    # plt.gcf().subplots_adjust(left=0.15)
    # ax.figure.legend(loc='lower center')
    # plt.savefig('NNK7-rd3-Sscore-PPV.pdf')
    # plt.show()

    # --------------- #
    # make S score distribution plot
    df = full_dataset.df
    sel_df = df.loc[df.active.astype(bool)]
    anti_df = df.loc[~df.active.astype(bool)]
    # # np.save('ystar_full.npy', y_star)

    y_star = np.load('ystar_full.npy')

    sel_ssc = s_score(sel_df, y_star) # apply to test set
    anti_ssc = s_score(anti_df, y_star) # apply to test set

    combined = np.concatenate([sel_ssc, anti_ssc])
    combined = pd.DataFrame({'S Score': combined, 
                            'Activity': np.concatenate([sel_df.active.values, anti_df.active.values])})
    combined['S Score Percentile'] = combined['S Score'].rank(pct=True)
    print(combined)
    combined.to_csv('Sscore_perc.csv')
    # make hist plots (maybe use percentile scale)

    import seaborn as sns
    # sns.histplot(combined, x='S Score', hue='Activity', stat='density')
    sns.histplot(combined, x='S Score Percentile', hue='Activity', stat='density')
    plt.savefig('SScore_dist_perc.pdf')
    return

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

        # TODO retrieve real library and use that for epi calc
        # REAL DATA ---------
        ref = SequenceDataset(sele=args.sele, anti=args.anti, features=args.features, variable_region=args.variable_region)
        library = ref.df.var_seq.values.astype('str')
        library = np.array([list(word) for word in library])
        library = convert_seq_to_idx(library)

        # TODO filter based on filter arg if present
        # filter = 'C5'
        # if filter is not None:
        #     aa = filter[0]
        #     n = int(filter[1:]) - 1
        #     filter_flag = np.array([seq[n] == aa for seq in library])
        #     library = library[filter_flag]
        #     preds = ref.df.active.values[filter_flag]
        #     print('Library filtered according to %s down to %s sequences' % (filter, str(len(library))))
        #     print(library[:5])

        # library = library[ref.df.active == 1] # keep only selection part of the library?

        print(library.shape)
        # get frequency tables out
        freqs = np.zeros((20, 8), dtype=float)
        for i in range(library.shape[1]):
            u, counts = np.unique(library[:, i], return_counts=True)
            counts = counts / library.shape[0]
            freqs[:, i] = counts

        # epi = compute_pairwise_epistasis_exp(library)

        # SIMULATED DATA --------
        N = 10000000
        library, preds = screen_routine(args, H=-1, N=N, thresh=-1, length=VAR_LENGTH, filter=args.filter_seq, bias=freqs) # use frequency table to make substrates!

        # check for at least one Cys
        flag = []
        i = 3 # C4
        for seq in range(library.shape[0]):
            # TODO filter for specific Cys position(s)
            tmp = 'C' in library[seq, :] and list(library[seq, :]).count('C') < 3
            tmp = tmp and library[seq, i] == 'C'
            flag.append(tmp)
        library = library[flag, :]
        preds = preds[flag]
        
        # TODO keep only those with high enough scores???
        # library = library[preds > 0.5, :]
        # preds = preds[preds > 0.5]

        library = convert_seq_to_idx(library)
        print('Library size for epi calculation:', preds.shape)

        epi, proba = compute_pairwise_epistasis(library, preds)
        print(epi.shape)
        np.save(args.npy, epi)

        # plot epistasis of positions vs positions
        from Plotter import epistasis_bw_positions
        epistasis_bw_positions(epi, 'epi_bw_pos') # collapses along AA dim to get epi over each position
        
        # plot epistasis of amino acids vs amino acids for 2 specific positions
        from Plotter import positional_epistasis
        positional_epistasis(epi, pos1=4, pos2=6, basename='epi_pos_5v7')
        
        # plot expected epistasis for a specific peptide
        # from Plotter import pep_epistatic_interactions
        # pep_epistatic_interactions(epi, pep=np.array('ACLMCVTM'), basename='example_pep_epi')
        
    elif args.routine == 'threshold':
        # load real library train/test splits
        # calculate true S score and plot hist
        threshold_test(args)
        # screen 5M random seqs
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

    # epi args
    parser.add_argument('--npy', help='npy filename for epi matrix', default='epi.npy', type=str)
    main(parser.parse_args())




