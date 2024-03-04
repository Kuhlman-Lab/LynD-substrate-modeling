import numpy as np

class constants:
    '''
    SOURCE: https://github.com/avngrdv/mRNA-display-deep-learning/
    Star symbol (*) is reserved for stop codons.
    Plus and underscore symbols (+ and _) are internally reserved tokens.
    Numerals (1234567890) are internally reserved for library design specification. 
    These symbols (123456790+_) should not be used to encode amino acids. 
    Other symbols are OK.
    
    Although the class holds multiple attributes, only
    the codon table should be edited. Everything else
    is inferred automatically from it.
    '''
    codon_table = {
                    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
                    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
                    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
                    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
                    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
                    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
                    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
                    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
                    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
                    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
                    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
                    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
                    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
                    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
                    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
                    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
                   }


    bases = ('T', 'C', 'A', 'G')
    complement_table = str.maketrans('ACTGN', 'TGACN')
    
    #probabilities of individual bases in randomized positions
    base_calls = {
                    'T': (1.00, 0.00, 0.00, 0.00),
                    'C': (0.00, 1.00, 0.00, 0.00),
                    'A': (0.00, 0.00, 1.00, 0.00),
                    'G': (0.00, 0.00, 0.00, 1.00),
                    'K': (0.50, 0.00, 0.00, 0.50),
                    'R': (0.00, 0.00, 0.50, 0.50),
                    'Y': (0.50, 0.50, 0.00, 0.00),
                    'S': (0.00, 0.50, 0.00, 0.50),
                    'W': (0.50, 0.00, 0.50, 0.00),
                    'M': (0.00, 0.50, 0.50, 0.00),
                    'D': (0.34, 0.00, 0.33, 0.33),
                    'H': (0.34, 0.33, 0.33, 0.00),
                    'V': (0.00, 0.34, 0.33, 0.33),                                
                    'B': (0.34, 0.33, 0.00, 0.33),
                    'N': (0.25, 0.25, 0.25, 0.25)
                    
                 }

    #Cys is represented as its IAA-alkylation reaction product
    aaSMILES = [    'N[C@@H](C)C(=O)',           
                    #'N[C@@H](CSCC(=O)N)C(=O)',  #IAA alkylation form for Cys: uncomment for LazBF
                    'N[C@@H](CS)C(=O)',        #regular Cys: uncomment for LazDEF
                    'N[C@@H](CC(=O)O)C(=O)',
                    'N[C@@H](CCC(=O)O)C(=O)',
                    'N[C@@H](Cc1ccccc1)C(=O)',
                    'NCC(=O)',
                    'N[C@@H](Cc1c[nH]cn1)C(=O)',
                    'N[C@@H]([C@H](CC)C)C(=O)',
                    'N[C@@H](CCCCN)C(=O)',        
                    'N[C@@H](CC(C)C)C(=O)',
                    'N[C@@H](CCSC)C(=O)',
                    'N[C@@H](CC(=O)N)C(=O)',
                    'O=C[C@@H]1CCCN1',  
                    'N[C@@H](CCC(=O)N)C(=O)',
                    'N[C@@H](CCCNC(=N)N)C(=O)',
                    'N[C@@H](CO)C(=O)',
                    'N[C@@H]([C@H](O)C)C(=O)',
                    'N[C@@H](C(C)C)C(=O)',
                    'N[C@@H](Cc1c[nH]c2c1cccc2)C(=O)',
                    'N[C@@H](Cc1ccc(O)cc1)C(=O)'
              ]


    global _reserved_aa_names
    _reserved_aa_names = ('_', '+', '*', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0')    
    
    aas = tuple(sorted(set(x for x in codon_table.values() if x not in _reserved_aa_names)))
    codons = tuple(sorted(set(x for x in codon_table.keys())))
    aa_dict = {aa: i for i,aa in enumerate(aas)}
    

def dense_morgan(r, w=True):
    '''
    SOURCE: https://github.com/avngrdv/mRNA-display-deep-learning/
    Create a feature matrix dims = (number of aas, number of features)
    For each amino acid looked up in constants, create a list of Morgan
    fingerprints. Number of features = number of unique fingerprints.
    
    bit_features and info output values can be ignored in most cases
    (used primarily for mapping integrated gradient attributions to 
     the chemical structure of the underlying peptide)
    
    Parameters
    ----------
    r : int; maximum fingerprint radius
    w : flag True, if the resulting matrix should written to an .npy file

    Returns
    -------
    F : feature matrix
    bit_features : fingerprint accesion number (internal RDkit repr)
    info : list of dicts; fingerprint information (internal RDkit repr)
    '''
    from rdkit.Chem import AllChem
    from rdkit import Chem
    import os


    aas = [Chem.MolFromSmiles(x) for x in constants.aaSMILES]
    
    #construct a list of all bit features
    bit_features = []
    for aa in aas:
        fingerprints = AllChem.GetMorganFingerprint(aa, r)
        keys = list(fingerprints.GetNonzeroElements().keys())
        for k in keys:
            bit_features.append(k)
            
    bit_features = list(set(bit_features))
        
    #assemble the F matrix, encoding fingerprints as a dense bit string
    F = np.zeros((len(constants.aaSMILES), len(bit_features)))
    info = []
    for i, aa in enumerate(aas):
        fp_info = {}
        fingerprints = AllChem.GetMorganFingerprint(aa, r, bitInfo=fp_info).GetNonzeroElements()
        for f in fingerprints:
            F[i,bit_features.index(f)] = 1

        info.append(fp_info)

    if w:
        if not os.path.isdir('../feature_matrices'):
            os.makedirs('../feature_matrices')
            
        fname = 'DENSE_Morgan_F_r=' + str(r) + '.npy'
        np.save(os.path.join('../feature_matrices', fname), F)

    return F, bit_features, info
