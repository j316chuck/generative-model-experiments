import pandas as pd
import numpy as np
import time 
import torch
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from itertools import zip_longest
from Bio.Seq import translate, IUPAC
from torch.nn import functional as F
from scipy.stats import skewnorm


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


def save_data(X_train, X_test, y_train, y_test, data_path):
    """
    save your gfp data in location
    :param X_train: training data
    :param X_test: testing data
    :param y_train: training output
    :param y_test: testing output
    :param data_path: path to save data
    :return: None
    """
    np.save(data_path + "X_train.npy", X_train)
    np.save(data_path + "X_test.npy", X_test)
    np.save(data_path + "y_train.npy", y_train)
    np.save(data_path + "y_test.npy", y_test)


def load_data(data_path):
    """
    load your data from location
    :param data_path: path of your gfp data
    :return: train test matrices with expected outputs
    """
    X_train = np.load(data_path + "X_train.npy")
    X_test = np.load(data_path + "X_test.npy")
    y_train = np.load(data_path + "y_train.npy")
    y_test = np.load(data_path + "y_test.npy")
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


def generate_mutations_df(base_sequence, mutations_lst, mutation_count_lst, alphabet,
                          start_mutation_index=0, end_mutation_index=0, verbose=False):
    """
    Generates a mutated dataframe
    :param base_sequence: string, the sequence to be mutated
    :param mutations_lst: list, the number of mutations from the base sequence
    :param mutation_count_lst: list, defines the number of each mutation count to generate
    :param alphabet: string, the entire set of letters in the vocabulary
    :param start_mutation_index: int, how many characters to skip at the beginning of the mutation
    :param end_mutation_index: int, how many characters to skip at the end of the mutation
    :param verbose: bool, print data generation results?
    :return: a dataframe comprised of three columns: the mutated strings, the number of mutations, the base_sequence
    >>> base_sequence, mutations_lst = get_wild_type_amino_acid_sequence(), [1, 2, 3]
    >>> mutation_count_lst, alphabet = [100, 100, 100], get_all_amino_acids()
    >>> mutated_df = generate_mutations_df(base_sequence, mutations_lst, mutation_count_lst, alphabet)
    >>> for i, row in mutated_df.iterrows():
    ...     assert(count_substring_mismatch(row["mutated_string"], row["base_sequence"]) == row["mutation_count"])
    """
    start_time = time.time()
    # get index maps
    characters_to_index = dict(zip(alphabet, range(len(alphabet))))
    index_to_characters = dict(zip(range(len(alphabet)), alphabet))
    num_characters = len(alphabet)
    total_data_points = int(sum(mutation_count_lst))
    mutations_per_sequence_lst = []
    for mutation_count, num_mutations in zip(mutations_lst, mutation_count_lst):
        mutations_per_sequence_lst.extend([mutation_count for _ in range(int(num_mutations))])
    print(len(mutations_per_sequence_lst))
    assert(total_data_points < 200000)
    assert(len(mutations_lst) == len(mutation_count_lst))
    assert(len(mutations_per_sequence_lst) == total_data_points)
    mutated_strings_lst = []
    for i, mutation_count in enumerate(mutations_per_sequence_lst):
        mutated_string = get_mutation(string=base_sequence,
                                      num_mutations=mutation_count,
                                      num_characters=num_characters,
                                      characters_to_index=characters_to_index,
                                      index_to_characters=index_to_characters,
                                      start_mutation_index=start_mutation_index,
                                      end_mutation_index=end_mutation_index)
        mutated_strings_lst.append(mutated_string)
        if i % 1000 == 0 and verbose:
            print(
                "Generated {0}/{1} mutations in {2:.2f} seconds".format(i, total_data_points, time.time() - start_time))
    base_sequences_lst = [base_sequence] * total_data_points
    mutated_df = pd.DataFrame.from_dict({
        'mutated_string': mutated_strings_lst,
        'mutation_count': mutations_per_sequence_lst,
        'base_sequence': base_sequences_lst
    })
    if verbose:
        print("{0:.2f} seconds to generate the mutated df".format(time.time() - start_time))
    return mutated_df


def get_mutation(string, num_mutations, num_characters, characters_to_index, index_to_characters,
                 start_mutation_index=0, end_mutation_index=0):
    """
    get a mutation of a string
    :param string: the string to be mutated
    :param num_mutations: the number of mutations
    :param num_characters: the number of characters in alphabet
    :param characters_to_index: map from characters in alphabet to index
    :param index_to_characters: map from index in alphabet to characters
    :param start_mutation_index: how many characters to skip at the beginning of the mutation
    :param end_mutation_index: how many characters to skip at the end of the mutation
    :return: mutated string
    >>> get_mutation("AAC", num_mutations=1, num_characters=2, characters_to_index={"A": 0, "C": 1}, \
        index_to_characters={0: "A", 1: "C"}, start_mutation_index=1, end_mutation_index=1)
    'ACC'
    >>> np.random.seed(1)
    >>> get_mutation("ACTGA", num_mutations=2, num_characters=4, characters_to_index={"A": 0, "C": 1, "T": 2, "G": 3}, \
        index_to_characters={0: "A", 1: "C", 2: "T", 3: "G"}, start_mutation_index=1, end_mutation_index=1)
    'ATTAA'
    """
    mutation = list(string)
    indexes = np.random.choice(range(start_mutation_index, len(string) - end_mutation_index),
                            num_mutations, replace=False)
    for i in indexes:
        original_c = string[i]
        original_index = characters_to_index[original_c]
        # mutation by moving the character somewhere else along the possible sequences.
        new_index = (original_index + np.random.randint(1, num_characters)) % num_characters
        mutation[i] = index_to_characters[new_index]
    return "".join(mutation)


def generate_discrete_gaussian_distribution(num_values, mean=5.5, std=2, low=1, high=10):
    """
    generates a discrete gaussian distribution
    :param num_values: int, number of samples to take from discrete distribution
    :param mean: float, mean of gaussian
    :param std: float, standard deviation of gaussian
    :param low: int, lowest value the discrete distribution will have
    :param high: int, highest value the discrete distribution will have
    :return: gaussian_lst, list, list of values in distributions
            bins, dict, map between the values and counts in distribution
    >>> np.random.seed(1)
    >>> generate_discrete_gaussian_distribution(10, 0, 1, -4, 4)[0]
    [2.0, -1.0, -1.0, -1.0, 1.0, -2.0, 2.0, -1.0, 0.0, -0.0]
    >>> generate_discrete_gaussian_distribution(1000, 0, 1, -4, 4)[1]
    {-4: 0.0, -3: 8.0, -2: 53.0, -1: 217.0, 0: 393.0, 1: 267.0, 2: 57.0, 3: 4.0, 4: 1.0}
    """
    normal_lst = np.random.normal(loc=mean, scale=std, size=num_values)
    bins = dict(zip(list(range(low, high + 1)), np.zeros(high - low + 1)))
    gaussian_lst = []
    for x in normal_lst:
        if x <= low:
            gaussian_lst.append(low)
            bins[low] += 1
        elif x >= high:
            gaussian_lst.append(high)
            bins[high] += 1
        else:
            gaussian_lst.append(round(x))
            bins[round(x)] += 1
    return gaussian_lst, bins


def generate_discrete_skewed_gaussian_distribution(num_values, a=4, mean=5.5, std=2, low=1, high=10):
    """
    generates a discrete skewed gaussian distribution
    :param num_values: int, number of samples to take from discrete distribution
    :param a: float, skew of the distribution, positive is right leaning, negative is left learning
    :param mean: float, mean of gaussian
    :param std: int, standard deviation of gaussian
    :param low: int, lowest value the discrete distribution will have
    :param high: int, highest value the discrete distribution will have
    :return: skewed_gaussian_lst, list, list of values in distributions
            bins, dict, map between the values and counts in distribution
    >>> np.random.seed(1)
    >>> generate_discrete_skewed_gaussian_distribution(num_values=10, a=4, mean=0, std=1, low=-4, high=4)[0]
    [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 0.0, 0.0]
    >>> generate_discrete_skewed_gaussian_distribution(num_values=10000, a=3, mean=3, std=2, low=1, high=10)[1]
    {1: 7.0, 2: 296.0, 3: 2085.0, 4: 3059.0, 5: 2428.0, 6: 1308.0, 7: 573.0, 8: 176.0, 9: 55.0, 10: 13.0}
    """
    skewed_lst = skewnorm(a=a, loc=mean, scale=std).rvs(num_values)
    bins = dict(zip(list(range(low, high + 1)), np.zeros(high - low + 1)))
    skewed_gaussian_lst = []
    for x in skewed_lst:
        if x <= low:
            skewed_gaussian_lst.append(low)
            bins[low] += 1
        elif x >= high:
            skewed_gaussian_lst.append(high)
            bins[high] += 1
        else:
            skewed_gaussian_lst.append(round(x))
            bins[round(x)] += 1
    return skewed_gaussian_lst, bins


def generate_discrete_uniform_distribution(num_values, low=1, high=10):
    """
    generates a discrete uniform distribution
    :param num_values: int, number of samples to take from discrete distribution
    :param low: int, lowest value the discrete distribution will have
    :param high: int, highest value the discrete distribution will have
    :return: uniform_lst, list, list of values in distribution
            bins, dict, map between the values and counts in distribution
    >>> np.random.seed(1)
    >>> generate_discrete_uniform_distribution(num_values=10, low=1, high=10)[0]
    [6, 9, 10, 6, 1, 1, 2, 8, 7, 10]
    >>> generate_discrete_uniform_distribution(num_values=1000, low=1, high=10)[1]
    {1: 109.0, 2: 91.0, 3: 110.0, 4: 97.0, 5: 100.0, 6: 95.0, 7: 109.0, 8: 104.0, 9: 89.0, 10: 96.0}
    """
    uniform_lst = np.random.choice(range(low, high + 1), num_values, replace=True)
    bins = dict(zip(list(range(low, high + 1)), np.zeros(high - low + 1)))
    for x in uniform_lst:
        bins[x] += 1
    return uniform_lst.tolist(), bins


def load_base_sequence(name):
    """
    loads the base sequence of a specific dataset
    :param name: string, dataset name
    :return: string, base sequence
    >>> load_base_sequence("synthetic_unimodal_data_length_20_uniform")
    'GYSSASKIIFGSGTRLSIRP'
    >>> load_base_sequence("synthetic_unimodal_data_length_50_skewed_gaussian")
    'MDILLDLGWHFSNCDEDTFYSPVQNTEGDLLFFDHNLKTDRGHVERSVMD'
    >>> load_base_sequence("synthetic_unimodal_data_length_100_gaussian")
    'MQKPCKENEGKPKCSVPKREEKRPYGEFERQQTEGNFRQRLLQSLEEFKEDIDYRHFKDEEMTREGDEMERCLEEIRGLRKKFRALHSNHRHSRDRPYPI'
    >>> load_base_sequence("synthetic_multimodal_data_length_50_modes_2_uniform")
    ['MVAYWRQAGLSYIRYSQICAKAVRDALKTEFKANAEKTSGSNVKIVKVKKE', 'MSSHKTFTIKRFLAKKQKQNRPIPQWIQMKPGSKIRYNSKRRHWRRTKLGL']
    >>> load_base_sequence("synthetic_multimodal_data_length_50_modes_3_uniform")
    ['MVAYWRQAGLSYIRYSQICAKAVRDALKTEFKANAEKTSGSNVKIVKVKKE', 'MSSHKTFTIKRFLAKKQKQNRPIPQWIQMKPGSKIRYNSKRRHWRRTKLGL', 'MTSWPGGSFGPDPLLALLVVILLARLILWSCLGTYIDYRLAQRRPQKPKQD']
    >>> load_base_sequence("synthetic_multimodal_data_length_50_modes_5_uniform")
    ['MVAYWRQAGLSYIRYSQICAKAVRDALKTEFKANAEKTSGSNVKIVKVKKE', 'MSSHKTFTIKRFLAKKQKQNRPIPQWIQMKPGSKIRYNSKRRHWRRTKLGL', 'MTSWPGGSFGPDPLLALLVVILLARLILWSCLGTYIDYRLAQRRPQKPKQD', 'MVQECCSQSLYYEELHSYHIVPYASENAIYEMGYTSSHLEQNSQLLIYKMN', 'MSGPLSPVCSCPQLPFMLSPCHMHHHPGHVALSQTVSPASLLTQGLGLPQH']
    >>> try:
    ...     load_base_sequence("tmp_data_length_10")
    ... except AssertionError:
    ...       print("dataset not found")
    dataset not found
    """
    if "gfp" in name:
        return get_wild_type_amino_acid_sequence()
    elif "length_20" in name and "synthetic" in name and "unimodal" in name:
        return 'GYSSASKIIFGSGTRLSIRP'
    elif "length_50" in name and "synthetic" in name and "unimodal" in name:
        return 'MDILLDLGWHFSNCDEDTFYSPVQNTEGDLLFFDHNLKTDRGHVERSVMD'
    elif "length_100" in name and "synthetic" in name and "unimodal" in name:
        return 'MQKPCKENEGKPKCSVPKREEKRPYGEFERQQTEGNFRQRLLQSLEEFKEDIDYRHFKDEEMTREGDEMERCLEEIRGLRKKFRALHSNHRHSRDRPYPI'
    elif "length_50" in name and "synthetic" in name and "multimodal" in name and "modes_2" in name:
        return ['MVAYWRQAGLSYIRYSQICAKAVRDALKTEFKANAEKTSGSNVKIVKVKKE',
                'MSSHKTFTIKRFLAKKQKQNRPIPQWIQMKPGSKIRYNSKRRHWRRTKLGL']
    elif "length_50" in name and "synthetic" in name and "multimodal" in name and "modes_3" in name:
        return ['MVAYWRQAGLSYIRYSQICAKAVRDALKTEFKANAEKTSGSNVKIVKVKKE',
                'MSSHKTFTIKRFLAKKQKQNRPIPQWIQMKPGSKIRYNSKRRHWRRTKLGL',
                'MTSWPGGSFGPDPLLALLVVILLARLILWSCLGTYIDYRLAQRRPQKPKQD']
    elif "length_50" in name and "synthetic" in name and "multimodal" in name and "modes_5" in name:
        return ['MVAYWRQAGLSYIRYSQICAKAVRDALKTEFKANAEKTSGSNVKIVKVKKE',
                'MSSHKTFTIKRFLAKKQKQNRPIPQWIQMKPGSKIRYNSKRRHWRRTKLGL',
                'MTSWPGGSFGPDPLLALLVVILLARLILWSCLGTYIDYRLAQRRPQKPKQD',
                'MVQECCSQSLYYEELHSYHIVPYASENAIYEMGYTSSHLEQNSQLLIYKMN',
                'MSGPLSPVCSCPQLPFMLSPCHMHHHPGHVALSQTVSPASLLTQGLGLPQH']
    else:
        raise AssertionError("Dataset name not found")



