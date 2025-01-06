import numpy as np
from modules.ecfp import constants


def convert_seq_to_idx(library):
    """
    Convert a [N, L] library of sequence data to [N, L] library of AA indexes
    """
    n = library.shape[0]
    m = library.shape[1]
    # convert variable seq region to an array of AA indices
    
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    new_lib = np.empty_like(library, dtype=int)
    for ix in range(n):
        for iy in range(m):
            new_lib[ix, iy] = alphabet.index(library[ix, iy])
    return new_lib

def compute_pairwise_epistasis(X, y, return_proba=True):
    '''
    Compute pairwise epi scores. See the manuscript for details.
    Computation is not optimized, so it may take a while for large X.

        Parameters:
                    X:     P dataset subject to the computation 
                           (2D np array, dtype=int)
                           
                    y:     1D np.ndarray holding substrate fitness
                           of peptides from P as predicted by the model.
                           
         return_proba:     True/False. if True, the op will also return average
                           fitness array (same dimensions as epi, see below); every
                           entry corresponds to average fitness of a sublibrary 
                           that contains aa1 and aa2 in pos1 and pos2, respectively.
                         
        Returns:
                  epi:     4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library
    '''   
    
    size = X.shape[0]
    seq_len = X.shape[1]
    n_aas = len(constants.aas)
    
    epi = np.zeros((seq_len, seq_len, n_aas, n_aas), dtype=np.float32)
    proba = np.zeros((seq_len, seq_len, n_aas, n_aas), dtype=np.float32)

    #p_good is p(G)
    p_good = np.mean(y)

    #fill in iteratively
    for pos1 in range(seq_len):
        for pos2 in range(pos1 + 1, seq_len):
            for aa1 in constants.aas:
                for aa2 in constants.aas:
                    #compute all of the requisite probabilities
                    aa1_mask = X[:,pos1] == constants.aa_dict[aa1]
                    aa2_mask = X[:,pos2] == constants.aa_dict[aa2]
                    # print(np.sum(aa1_mask), np.sum(aa2_mask), np.sum([aa1_mask & aa2_mask])) # check each score sample size?
                    #p_aa1 is p(aa1)
                    p_aa1 = np.divide(np.sum(aa1_mask), size)
                    p_aa2 = np.divide(np.sum(aa2_mask), size)
                    p_aa12 = np.divide(X[aa1_mask & aa2_mask].shape[0], size)
                    
                    #p_good_c_aa1 is p(good|aa1) etc
                    p_good_c_aa1 = np.mean(y[aa1_mask])
                    p_good_c_aa2 = np.mean(y[aa2_mask])
                    p_good_c_aa12 = np.mean(y[aa1_mask & aa2_mask])
                    #compute epi
                    # print(p_good_c_aa12, p_aa12, p_good, '\n', p_good_c_aa1, p_good_c_aa2, p_aa1, p_aa2)
                    x = np.divide(p_good_c_aa12 * p_aa12 * p_good, 
                                  p_good_c_aa1 * p_good_c_aa2 * p_aa1 * p_aa2)
  
                    # print(np.log2(x))
                    epi[pos1, pos2, constants.aa_dict[aa1], constants.aa_dict[aa2]] = np.log2(x)
                    proba[pos1, pos2, constants.aa_dict[aa1], constants.aa_dict[aa2]] = p_good_c_aa12
            
            
            print(f'Pos{pos1+1}/pos{pos2+1} epi computed. . .')
    # np.save('epi.npy', epi)        
    if return_proba:
        return epi, proba
    return epi    


def compute_pairwise_epistasis_exp(X):
    '''
    Compute pairwise epi scores. See the manuscript for details.
    Computation is not optimized, so it may take a while for large X.

        Parameters:
                    X:     P dataset subject to the computation 
                           (2D np array, dtype=int)
                         
        Returns:
                  epi:     4D np.ndarray; shape=(X.shape[1], X.shape[1], n_aas, n_aas)
                           where X.shape[1] is peptide sequence length (number of positions),
                           and n_aas is the number of amino acid monomers in the library
    '''   
    
    size = X.shape[0]
    seq_len = X.shape[1]
    n_aas = len(constants.aas)
    
    epi = np.zeros((seq_len, seq_len, n_aas, n_aas), dtype=np.float32)

    #fill in iteratively
    for pos1 in range(seq_len):
        for pos2 in range(pos1 + 1, seq_len):
            for aa1 in constants.aas:
                for aa2 in constants.aas:
                    #compute all of the requisite probabilities
                    aa1_mask = X[:,pos1] == constants.aa_dict[aa1] # abundance of aa1
                    aa2_mask = X[:,pos2] == constants.aa_dict[aa2] # abundance of aa2
                    aa1_freq = np.sum(aa1_mask) / size
                    aa2_freq = np.sum(aa2_mask) / size
                    
                    aa1_aa2_freq = np.sum(aa1_mask & aa2_mask) / size

                    x = aa1_aa2_freq / (aa1_freq * aa2_freq)

                    # print(np.sum(aa1_mask), np.sum(aa2_mask), np.sum(aa1_mask & aa2_mask))
                    # print(np.log2(x))
                    # print('=' * 50)
                    epi[pos1, pos2, constants.aa_dict[aa1], constants.aa_dict[aa2]] = np.log2(x)
            
            print(f'Pos{pos1+1}/pos{pos2+1} epi computed. . .')
    # np.save('epi.npy', epi)        
    return epi    


def hamming_distance(P, pep, 
                     h=0,
                     cum=False,
                     return_count=False, 
                     return_index=False,
                     return_distance=False):

    '''
    Source: https://github.com/avngrdv/mRNA-display-deep-learning/blob/main/code/utils/misc.py
    A flexible Hamming distance calculator.
       
        Parameters:
                P:     P dataset subject to the computation (2D np array)
                pep:   peptide to compare against (1D np array)
                       pep.dtype should be the same as P.dtype
                     
                h:     int; Hamming distance spec. The op will return a view
                       of the original P dataset where for every peptide x in
                       the resulting dataset Hamming_distance(x, pep) = h
                 
                cum:   True/False; if True, all peptides from P at a Hamming
                       distance h or less from pep will be returned
                       
       return_count:   True/False; if True, return the number of peptides in
                       P which are at Hamming_distance=h from pep
                       
       return_index:   True/False; if True, return the indices of peptides in
                       P which are at Hamming_distance=h from pep                       

    return_distance:   True/False; if True, return an array of distances between
                       peptides in P and pep
    
        Returns:
                  H:   a slice of the original P array    
    '''    
    # P is reference set [N, L]
    # pep is query peptide [L]
    D = P == pep # D is a boolean matrix of matches [N, L]
    
    if return_distance:
        return np.sum(~D, axis=1) # returns matrix of Hamming distances [N]
    
    match = pep.size - h
    if cum:
        ind = np.sum(D, axis=1) >= match
    else:
        ind = np.sum(D, axis=1) == match
        
    H = P[ind]
    
    if return_count:
        return H.shape[0]
        
    elif return_index:
        return np.where(ind)[0]

    return H


def sample_random_peptides(n, length, amino_acids, bias=None):
    '''
    From https://github.com/avngrdv/mRNA-display-deep-learning/blob/main/code/utils/misc.py
    Generate an array of random peptide sequences. shape = (n_peptides, pep_len)
    '''
    P = np.random.choice(amino_acids, size=(n, length), replace=True, p=bias)    
    return P  


def sample_random_peptides_biased(n, length, amino_acids, bias=None):
    '''
    From https://github.com/avngrdv/mRNA-display-deep-learning/blob/main/code/utils/misc.py
    Generate an array of random peptide sequences. shape = (n_peptides, pep_len)
    '''
    p_list = []
    for le in range(length):
        P = np.random.choice(amino_acids, size=(n, ), replace=True, p=bias[:, le])  
        p_list.append(P)  
    return np.stack(p_list).T


def sample_random_peptides_wCys(n, length, amino_acids):
    '''
    From https://github.com/avngrdv/mRNA-display-deep-learning/blob/main/code/utils/misc.py
    Generate an array of random peptide sequences. shape = (n_peptides, pep_len)
    
    CHANGES:
    Modified to do constrained generation to make sure at least 1x Cys is present in each peptide
    '''
    
    # randomly choose cys position
    C_pos = np.random.choice(8, size=n, replace=True) # [N,]
    
    # make normal random array
    P = np.random.choice(amino_acids, size=(n, length), replace=True) # [N, L]
    
    # fill in C at each randomly selected position
    for idx in range(n):
        P[idx, C_pos[idx]] = 'C'

    return P  


def sample_random_peptides_templated(n, length, amino_acids, peptide=None):
    '''
    Generate an array of random peptide sequences. shape = (n_peptides, pep_len)
    '''
    assert len(peptide) == length

    P = np.empty((n, length), dtype=str)

    for pos in range(length):
        template_AA = peptide[pos]
        # for each position, either randomly or manually assign AA (X indicates random)
        if template_AA == 'X':
            P[:, pos] = np.random.choice(amino_acids, size=(n, ), replace=True)
        else:
            P[:, pos] = template_AA

    return P
