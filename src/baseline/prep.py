import pandas as pd
import numpy as np

import json

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    

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
    
    # featurize into arrays ready for sklearn processing
    print(df.shape, '\n', df.columns)
    
    # featurize x-data into N x L matrix of amino acid indices
    seqs = df.var_seq.values
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    
    X = np.zeros((seqs.shape[0], len(seqs[0])))
    for n, seq in enumerate(seqs):
        for ns, s in enumerate(seq):
            idx = alphabet.index(s)
            X[n, ns] = idx        

    # extract y-data (activity)
    Y = df.active.values
    return X, Y


def split(X, Y, test_size=0.1):
    """Split dataset for train/test use - TODO add cross-validation later"""
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, shuffle=True)
    
    return X_train, X_test, Y_train, Y_test


def metrics(Y_true, Y_pred):
    """Calculates useful classifier metrics"""
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

    met_dict = {}

    met_list = [accuracy_score, f1_score, precision_score, recall_score]
    label_list = ['Accuracy', 'F1_score', 'Precision', 'Recall']

    for met, label in zip(met_list, label_list):
        print(label + ':\t' + str(round(met(Y_true, Y_pred), 2)))
        met_dict[label] = met(Y_true, Y_pred)
        
    tn, fp, fn, tp = confusion_matrix(Y_true, Y_pred).ravel()
    for met, label in zip([tn, fp, fn, tp], ['TN', 'FP', 'FN', 'TP']):
        met_dict[label] = met
    print('TP:', tp, '\nFP:', fp, '\nTN:', tn, '\nFN:', fn)
    return met_dict


def supervised(X_train, X_test, Y_train, Y_test, model='logreg'):
    """Supervised learning algorithm sweep fxn"""
    models = {
        'logreg': LogisticRegression(), 
        'ridge': RidgeClassifier(), 
        'svc': LinearSVC(), 
        'rf': RandomForestClassifier(),      
        'gb': GradientBoostingClassifier(), 
        'ada': AdaBoostClassifier(), 
        'mlp': MLPClassifier(hidden_layer_sizes=(128, 128, 128,), verbose=True)
    }
    
    clf = models[model]
    clf.fit(X_train, Y_train)
    
    if model == 'logreg':
        feat_coef = clf.coef_
        np.save('coef.npz', feat_coef)

    Y_pred = clf.predict(X_test)
    met_dict = metrics(Y_test, Y_pred)
    return met_dict


def embed(X, fmt='onehot'):
    """Embed a matrix of N, L features of amino acid positions to N-dim space"""
    from torch.nn.functional import embedding
    import torch

    if fmt == 'onehot':
        X = OneHotEncoder().fit_transform(X) # from [N, 6] to [N, 120] one-hot vector
    elif fmt == 'embed':
        X = embedding(torch.tensor(X, dtype=torch.long), torch.randn(128, 20)) # [N, 6] to [N, 128] continuous embedding
        X = torch.reshape(X, (X.shape[0], X.shape[1] * X.shape[2]))
        X = X.numpy()

    return X



# ------------- Main ------------ #
def main():
    sel, anti = import_data()

    X, Y = featurize(sel, anti)

    # X = embed(X, fmt='embed')
    X = embed(X, fmt='onehot')

    X_train, X_test, Y_train, Y_test = split(X, Y)

    model_list = [
        'logreg', 
        'ridge', 
        'svc', 
        'rf', 
        'gb', 
        'ada', 
        'mlp'
    ]

    model_list = ['logreg']

    for model in model_list:
        print('Running model %s' % model)
        metrics_all = supervised(X_train, X_test, Y_train, Y_test, model)
        print(metrics_all)
        with open('%s.json' % model, 'w') as fp:
            json.dump(metrics_all, fp, cls=NpEncoder)

        print('Completed model %s' % model)
        print('=' * 50)

if __name__ == "__main__":
    main()
