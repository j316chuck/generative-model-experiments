import pandas as pd
import numpy as np
import time 
import torch 

from sklearn.model_selection import train_test_split
from itertools import zip_longest
from Bio.Data import CodonTable
from Bio.Seq import translate, IUPAC


def normalize(X): 
    """ normalize an array """
    return (X - X.mean()) / X.std()

def get_wild_type_dna_sequence(): 
    return pd.read_csv("./data/gfp_data.csv")["nucSequence"].values[0]

def dna_to_amino_acid(dna_seq):
    return translate(dna_seq)

def get_all_amino_acids(): 
    return "*" + IUPAC.protein.letters
    
def get_wild_type_amino_acid_sequence(): 
    return dna_to_amino_acid(get_wild_type_dna_sequence())

def count_substring_mismatch(s1, s2): 
    """ returns the number of misaligned pairs within strings s1 and s2"""
    return sum([i != j for i, j in zip_longest(s1, s2)])

def generate_random_gfp_data_mutations(num_of_mutations_lst, num_per_mutation_count = 1000): 
    """
    Input: num_of_mutations_lst is a list defining the count of mutations from the base sequence we want, 
           num_per_mutation_count is an int defining how many of each mutation count do we want. Often set to the test size
    Output: a pandas dataframe comprised of two columns: the number of mutations and the mutation dna sequence
    """
    start_time = time.time()
    total_data_points = len(num_of_mutations_lst) * num_per_mutation_count
    assert(total_data_points < 200000)
    wild_type_sequence = get_wild_type_dna_sequence()
    wild_type_lst = list(wild_type_sequence)
    mutation_lst = np.vstack([wild_type_lst] * total_data_points)
    mutation_count_lst = np.array([[mutation_count for _ in range(num_per_mutation_count)] for mutation_count in num_of_mutations_lst]).flatten()
    
    np.testing.assert_array_equal(mutation_lst[0], mutation_lst[2])
    np.testing.assert_array_equal(mutation_lst[0], wild_type_lst)
    assert(len(mutation_lst) == total_data_points and len(mutation_count_lst) == len(mutation_lst))
    assert(mutation_count_lst[0] == 1 and mutation_count_lst[num_per_mutation_count] == 2 and mutation_count_lst[num_per_mutation_count - 1] == 1)
    
    bases = "ACTG"
    index = list(range(0, 4))
    base_index_map = dict(zip(bases, index))
    index_base_map = dict(zip(index, bases))

    for i, mutation_count in enumerate(mutation_count_lst):   
        mutation_index = np.random.choice(len(wild_type_sequence), mutation_count, replace = False).tolist()
        for j in mutation_index: 
            k = base_index_map[mutation_lst[i, j]]
            new_index = (k + np.random.randint(1, 4)) % 4
            mutation_lst[i, j] = index_base_map[new_index] 
    mutation_sequence_lst = np.array(["".join(lst) for lst in mutation_lst])
    mutated_df = pd.DataFrame.from_dict({'mutation_count' : mutation_count_lst, 'mutated_dna_sequence' : mutation_sequence_lst})
    mutated_df["mutated_amino_acid_sequence"] = mutated_df["mutated_dna_sequence"].apply(lambda x : dna_to_amino_acid(x))
    print(time.time() - start_time, "seconds to generate the mutated df")
    return mutated_df

def save_mutated_gfp_data(mutated_df, path = "./data/mutated_df.csv"):
    mutated_df.to_csv(path, index = None)
    
def load_saved_mutated_gfp_data(path = "./data/mutated_df.csv"): 
    return pd.read_csv(path)

def get_gfp_data(amino_acid = False, gfp_data_path = "./data/gfp_data.csv", x_feature = "nucSequence", y_feature = "medianBrightness", normalize_y = True, test_size = 0.2, shuffle = False):
    # returns gfp data in dna or amino acid form which is then split into train and test set
    df = pd.read_csv(gfp_data_path, index_col = 0)
    if amino_acid: 
        X = df[x_feature].apply(lambda x : dna_to_amino_acid(x)).values
    else: 
        X = df[x_feature].values
    y = df[y_feature].values
    if normalize_y: 
        y = normalize(y)
    return train_test_split(X, y, test_size = test_size, shuffle = shuffle)

def save_gfp_data(X_train, X_test, y_train, y_test, gfp_data_path): 
    np.save(gfp_data_path + "X_train.npy", X_train)
    np.save(gfp_data_path + "X_test.npy", X_test)
    np.save(gfp_data_path + "y_train.npy", y_train)
    np.save(gfp_data_path + "y_test.npy", y_test)

def load_gfp_data(gfp_data_path):
    X_train = np.load(gfp_data_path + "X_train.npy")
    X_test = np.load(gfp_data_path + "X_test.npy")
    y_train = np.load(gfp_data_path + "y_train.npy")
    y_test = np.load(gfp_data_path + "y_test.npy")
    return X_train, X_test, y_train, y_test

  
def one_hot_encode_dna_sequence(X):
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

def one_hot_decode_dna_sequence(X): 
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

def string_to_tensor(string, character_to_int):
    tensor = torch.zeros(len(string)).long()
    for i, c in enumerate(string): 
        tensor[i] = character_to_int[c]
    return tensor

def string_to_numpy(string, character_to_int):
    arr = np.zeros(len(string))
    for i, c in enumerate(string): 
        arr[i] = character_to_int[c]
    return arr
