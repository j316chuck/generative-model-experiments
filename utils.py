
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np


# In[77]:


def one_hot_encode(X):
    """
    Input: X is a list of DNA Sequences represented by the base pairs ACTG. 
        All DNA Sequences must be the same length
    Output: one hot encoded list of dna sequences
    Example: one_hot_encode(["ACT", "ACG"]) = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
                                                    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]
    """
    assert(len(X) > 0)
    assert(all([len(X[0]) == len(X[i]) for i in range(len(X))]))
    alphabet = ["A", "C", "T", "G"]
    alphabet_size = len(alphabet)
    alphabet_dict = dict(zip(alphabet, range(alphabet_size)))
    one_hot_matrix = np.zeros((len(X), alphabet_size * len(X[0]))) 
    for i, dna_sequence in enumerate(X):
        for j, base_pair in enumerate(dna_sequence):
            index = alphabet_dict[base_pair]
            one_hot_matrix[i, alphabet_size * j + index] = 1.0
    return one_hot_matrix

def one_hot_decode(X): 
    """
    Input: X is a one hot encoded list of DNA Sequences represented by the base pairs ACTG. 
        All DNA Sequences must be the same length
    Output: list of dna sequences
    Example: one_hot_decode([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], 
                            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]) = ["ACT", "ACG"]
    """
    assert(len(X) > 0)
    assert(all([len(X[0]) == len(X[i]) for i in range(len(X))]))
    alphabet = ["A", "C", "T", "G"]
    alphabet_size = len(alphabet)
    dna_sequences = []
    for i, one_hot_sequence in enumerate(X): 
        dna_sequence = []
        for j in range(0, len(one_hot_sequence), 4): 
            if one_hot_sequence[j]:
                dna_sequence.append("A")
            elif one_hot_sequence[j + 1]: 
                dna_sequence.append("C")
            elif one_hot_sequence[j + 2]: 
                dna_sequence.append("T")
            elif one_hot_sequence[j + 3]: 
                dna_sequence.append("G")
        dna_sequences.append("".join(dna_sequence))
    return np.array(dna_sequences)

def train_test_split(X, train_test_ratio = 0.8, shuffle = False): 
    """ splits data into train and test groups """
    assert(len(X) > 0 and 0 <= train_test_ratio and train_test_ratio <= 1)
    if shuffle: 
        np.random.shuffle(X)
    train_size = int(len(X) * 0.8)
    return X[0:train_size, :], X[train_size:, :]
    
    
    
def load_gfp_data(gfp_data_path = "./data/gfp_data.csv", train_test_ratio = 0.8, shuffle = False):
    """ one hot encodes gfp data into train and test set"""
    df = pd.read_csv(gfp_data_path, index_col = 0)
    one_hot_matrix = one_hot_encode(df["nucSequence"].values)  
    return train_test_split(one_hot_matrix, 
                            train_test_ratio = train_test_ratio, 
                            shuffle = shuffle)

