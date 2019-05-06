import pandas as pd
import numpy as np
import time 
import torch 

from sklearn.model_selection import train_test_split
from itertools import zip_longest
from Bio.Seq import translate, IUPAC
from torch.nn import functional as F
import matplotlib.pyplot as plt


def normalize(X):
    """
    :param X: dataframe or numpy array to be normalized
    :return: dataframe or numpy array that is normalized
    >>> normalize(np.array([0, 1, 2]))
    array([-1.22474487,  0.        ,  1.22474487])
    """
    return (X - X.mean()) / X.std()


def get_wild_type_dna_sequence():
    """
    :return: string of wild type dna sequence from cached location
    """
    return pd.read_csv("./data/gfp_data.csv")["nucSequence"].values[0]


def dna_to_amino_acid(dna_seq):
    """
    :param dna_seq: string of dna sequence
    :return: dna sequence in amino acid form
    >>> dna_to_amino_acid("ACTGGCTAT")
    'TGY'
    """
    return translate(dna_seq)


def get_all_amino_acids():
    """
    :return: string of all amino acids + stop character
    >>> get_all_amino_acids()
    '*ACDEFGHIKLMNPQRSTVWY'
    >>> len(get_all_amino_acids())
    21
    """
    return "*" + IUPAC.protein.letters


def get_wild_type_amino_acid_sequence():
    """
    :return: string of wild type amino acid sequence from cached location
    """
    return dna_to_amino_acid(get_wild_type_dna_sequence())


def count_substring_mismatch(s1, s2):
    """
    :param s1: string one
    :param s2: string two
    :return: int of the number of mismatches between the two sequences
    >>> count_substring_mismatch('1', '2')
    1
    >>> count_substring_mismatch('ACT', 'ACGA')
    2
    """
    return sum([i != j for i, j in zip_longest(s1, s2)])


def get_gfp_data(amino_acid=False, gfp_data_path="./data/gfp_data.csv", x_feature="nucSequence", y_feature="medianBrightness", normalize_y=True, test_size=0.2, shuffle=False):
    """
    :param amino_acid:  amino acid format or DNA
    :param gfp_data_path: gfp data path
    :param x_feature: column to use for x data
    :param y_feature: column to use for y data
    :param normalize_y: normalize y or not
    :param test_size: size of test set
    :param shuffle: shuffle data or not
    :return: gfp data split across train and test set
    """
    df = pd.read_csv(gfp_data_path, index_col=0)
    if amino_acid: 
        X = df[x_feature].apply(lambda x: dna_to_amino_acid(x)).values
    else: 
        X = df[x_feature].values
    y = df[y_feature].values
    if normalize_y: 
        y = normalize(y)
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle)


def save_gfp_data(X_train, X_test, y_train, y_test, gfp_data_path):
    """
    save your gfp data in location
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training output
    :param y_test: testing output
    :param gfp_data_path: path to save data
    :return: None
    """
    np.save(gfp_data_path + "X_train.npy", X_train)
    np.save(gfp_data_path + "X_test.npy", X_test)
    np.save(gfp_data_path + "y_train.npy", y_train)
    np.save(gfp_data_path + "y_test.npy", y_test)


def load_gfp_data(gfp_data_path):
    """
    load your gfp data from location
    :param gfp_data_path: path of your gfp data
    :return: train test matrices with expected outputs
    """
    X_train = np.load(gfp_data_path + "X_train.npy")
    X_test = np.load(gfp_data_path + "X_test.npy")
    y_train = np.load(gfp_data_path + "y_train.npy")
    y_test = np.load(gfp_data_path + "y_test.npy")
    return X_train, X_test, y_train, y_test


def plot_mismatches_histogram(sequences, wild_type, save_fig_dir=None, show=False):
    """
    :param sequences: list of sampled sequences
    :param wild_type: base sequence to compare all other sequences against
    :param save_fig_dir: saves the histogram of mismatches
    :param show: shows the histogram of mismatches
    :return: list that counts the number of mismatches from the wild type
    >>> plot_mismatches_histogram(["ACT", "ACG"], "ACG", None, False)
    [1, 0]
    """
    assert(type(wild_type) == str and all(type(seq) == str for seq in sequences))
    assert(all([len(wild_type) == len(seq) for seq in sequences]))
    mismatches = [count_substring_mismatch(x, wild_type) for x in sequences]
    plt.figure(figsize=(15, 15))
    plt.title("mismatches from wild type", fontsize=15)
    plt.hist(mismatches, bins=15)
    plt.xlabel("mismatches", fontsize=12)
    plt.ylabel("counts", fontsize=12)
    if save_fig_dir:
        plt.savefig(save_fig_dir)
    if show:
        plt.show()
    plt.close()
    return mismatches


def one_hot_encode(X, alphabet):
    """
    one hot encode a list of strings
    :param X: list of sequences represented by the set of letters in alphabet
    :param alphabet:
    :return: one hot encoded list of X sequences
    """
    """
    Input: X is a list of sequences represented by the set of letters in alphabet
        All sequences must be the same length
    Output: one hot encoded list of X sequences
    Example: one_hot_encode(["ACT", "ACG"], "ACTG") = [[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                                              [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]
    """
    assert(len(X) > 0)
    assert(all([len(X[0]) == len(X[i]) for i in range(len(X))]))
    alphabet_size = len(alphabet)
    alphabet_dict = dict(zip(alphabet, range(alphabet_size)))
    one_hot_matrix = np.zeros((len(X), alphabet_size * len(X[0])))
    for i, sequence in enumerate(X):
        for j, letter in enumerate(sequence):
            if letter not in alphabet:
                raise KeyError("letter not in alphabet")
            index = alphabet_dict[letter]
            one_hot_matrix[i, alphabet_size * j + index] = 1.0
    return one_hot_matrix


def one_hot_decode(X, alphabet):
    """
    one hot decode a matrix
    :param X: one hot encoded list of DNA Sequences represented by the alphabet
    :param alphabet: all the letters in the vocabulary of X
    :return: a one hot decoded matrix in list of strings format
    >>> one_hot_decode([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], \
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]], "ACTG")
    ['ACT', 'ACG']
    """
    assert(len(X) > 0)
    assert(all([len(X[0]) == len(X[i]) for i in range(len(X))]))
    alphabet_size = len(alphabet)
    sequences_lst = []
    for i, one_hot_sequence in enumerate(X):
        sequence, sequence_len = [], len(one_hot_sequence)
        for j in range(0, sequence_len, alphabet_size):
            index = np.argmax(one_hot_sequence[j:j+alphabet_size])
            sequence.append(alphabet[index])
        sequences_lst.append("".join(sequence))
    return sequences_lst


def string_to_tensor(string, character_to_int):
    """
    converts string to tensor array
    :param string: string
    :param character_to_int: character to integer map
    :return: tensor representation of string
    """
    tensor = torch.zeros(len(string)).long()
    for i, c in enumerate(string): 
        tensor[i] = character_to_int[c]
    return tensor


def string_to_numpy(string, character_to_int):
    """
    converts string to numpy array
    :param string: string
    :param character_to_int: character to integer map
    :return: numpy representation of string
    """
    arr = np.zeros(len(string))
    for i, c in enumerate(string): 
        arr[i] = character_to_int[c]
    return arr


def to_tensor(x, device=torch.device("cpu")):
    """
    converts numpy array to tensor with specific device
    :param x: numpy array
    :param device: device to put tensor on
    :return: tensor
    """
    assert(type(x) == np.ndarray)
    return torch.from_numpy(x).float().to(device)


def sample_tensor_to_string(x, int_to_character, softmax=False):
    """
    Samples a tensor from the probability distribution of x.
    :param x: a 2d or 1d vector of shape (characters x vocabulary_size)
    :param int_to_character: maps indexes to character
    :param softmax: apply softmax layer or not before sampling
    :return: string format of sampled probability tensor
    >>> int_to_character = dict(zip(range(21), get_all_amino_acids()))
    >>> seed = torch.manual_seed(1)
    >>> sample_tensor_to_string(torch.randn(18, 21), int_to_character, softmax=True) 
    'YDNYCIYNINISLNPFPR'
    """
    num_characters = len(int_to_character)
    assert(type(x) == torch.Tensor)
    assert(x.shape[0] % num_characters == 0 or x.shape[1] % num_characters == 0)
    x = x.reshape(-1, num_characters)
    if softmax:
        x = F.softmax(x, dim = -1)
    string = []
    for dist in x: 
        index = torch.multinomial(dist, 1).item()
        string.append(int_to_character[index])
    return "".join(string)


def tensor_to_string(x, int_to_character):
    """
    Converts tensor to string
    :param x: tensor
    :param int_to_character: maps indexes to character
    :return: string format of tensor

    >>> int_to_character = dict(zip(range(4), "ACTG"))
    >>> tensor_to_string(torch.tensor([0, 0, 1, 0, 0, 0, 1, 0]), int_to_character)
    'TT'
    >>> tensor_to_string(torch.tensor([0.8, 0.15, 0.05, 0, 0, 0.9, 0.1, 0]), int_to_character)
    'AC'
    """
    num_characters = len(int_to_character)
    assert(type(x) == torch.Tensor)
    assert(len(x) % num_characters == 0)
    x = x.reshape(-1, num_characters)
    _, index = x.max(dim = 1)
    return "".join([int_to_character[i] for i in index.numpy()])

