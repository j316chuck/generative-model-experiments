import pandas as pd
import numpy as np
import time 
import argparse
import sys 

from pomegranate import State, DiscreteDistribution, HiddenMarkovModel
from utils import load_gfp_data, count_substring_mismatch, get_wild_type_amino_acid_sequence, get_all_amino_acids
from hmm import GenerativeHMM, hmm_amino_acid_args

def get_data(X_train, length, n = 100, random = True): 
    """
    gets n random sequences of size length from the dataset X_train
    """
    if not random: 
        data = X_train[0:length]
    else: 
        indexes = np.random.choice(len(X_train), n)
        data = X_train[indexes]
    return np.array([list(x[0:length]) for x in data])

def sample_and_score(hmm, base_str, n = 100, length = 100, logger = None):
    """
    use the hmm model to sample n sequences of size = length. 
    then use the wild_type to count how far off the average sample is from the wild_type
    prints all results in the logger file
    """
    assert(len(base_str) == length)
    samples = hmm.sample(n, length)        
    average_diff = np.mean([count_substring_mismatch(seq, base_str) for seq in samples])
    print("Average difference: {0:.2f}, or {1:.2f} mismatches per letter".format(average_diff, 
                                                                 average_diff / length), file = logger)
    print("Example sequence {0}".format(samples[np.random.randint(0, n)]), file = logger)
    return average_diff

def train_and_save_hmm(X, args):
    start_time = time.time()
    hmm = GenerativeHMM(args)
    logger = open("./logs/{0}.txt".format(hmm.name), "w")
    sys.stdout = logger
    hmm.fit(X)
    print("Finished training in {:.2f} seconds".format(time.time() - start_time), file = logger)
    print("HMM Parameters:", file = logger)
    print(hmm.get_args(), file = logger)
    sample_and_score(hmm, wild_type_amino_acid[0:args["length"]], 100, args["length"], logger = logger)
    wild_type_prob = np.e ** hmm.predict([list(wild_type_amino_acid[0:args["length"]])])
    mutation_prob = np.e ** hmm.predict([list(wild_type_amino_acid[0:args["length"] - 3] + "ACG")])
    print("Wild type prob: {0}. Mutation prob: {1}".format(wild_type_prob, mutation_prob), file = logger)
    model_path = "./models/{0}.json".format(hmm.name)
    hmm.save_model(model_path)
    cached_hmm = GenerativeHMM(args)
    cached_hmm.load_model(model_path)
    try: 
        for i in get_all_amino_acids():
            for j in get_all_amino_acids(): 
                np.testing.assert_almost_equal(hmm.predict([list(i + j)]), cached_hmm.predict([list(i + j)]))
        print("Successfully finished training and saving {0} model!".format(hmm.name), file = logger)
        if logger: logger.close()
    except:
        print("Error in loading {0} hmm".format(hmm.name), file = logger)
        if logger: logger.close()

def get_args(parser_args):
    args = hmm_amino_acid_args()
    args["n_jobs"] = parser_args.n_jobs
    args["hidden_size"] = parser_args.hidden_size
    args["max_iterations"] = parser_args.max_iterations
    args["name"] = parser_args.name
    args["length"] = parser_args.length
    return args

parser = argparse.ArgumentParser(description='Process the arguments for the HMM Model')
parser.add_argument("-nj", "--n_jobs", default=5, required=False, help="number of jobs/cpus to train the hmm", type=int)
parser.add_argument("-hi", "--hidden_size", default=50, required=False, help="hidden_size of hmm", type=int)
parser.add_argument("-ma", "--max_iterations", default=100, required=False, help="max iterations to run the hmm model", type=int)
parser.add_argument("-nu", "--num_sequences", default=100, required=False, help="number of sequences used to train the hmm", type=int)
parser.add_argument("-le", "--length", default=238, required=False, help="length of sequences", type=int)
parser.add_argument("-na", "--name", default="base", required=False, help="name of hmm model", type=str)
parser_args = parser.parse_args()

X_train, X_test, y_train, y_test = load_gfp_data("./data/gfp_amino_acid_")
wild_type_amino_acid = get_wild_type_amino_acid_sequence()
wild_type_length = len(wild_type_amino_acid)
X = get_data(X_train, parser_args.length, parser_args.num_sequences)
assert(X_train[0] == wild_type_amino_acid)
assert(count_substring_mismatch(wild_type_amino_acid, X_train[1000]) == 8)

args = get_args(parser_args)
train_and_save_hmm(X, args)
