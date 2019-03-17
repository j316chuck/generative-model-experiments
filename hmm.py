
# coding: utf-8

# ## Imports

# In[1]:


import pandas as pd
import numpy as np
import time 
import json

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

from pomegranate import State, DiscreteDistribution, HiddenMarkovModel
from sklearn.model_selection import train_test_split
from utils import *
from Bio.Alphabet import IUPAC


# ## Models

# In[8]:


class GenerativeHMM(): 
    
    def __init__(self, args, x_train=None, weights=None, verbose=True): 
        """
        Initializes the HMM to perform generative tasks
        Parameters
        ----------
        args : dictionary
            defines the hyper-parameters of the HMM
        args.name : string 
            defines the name of the HMM
        args.hidden_size : int 
            defines the hidden size
        args.max_iterations: int
            sets the max iterations
        args.n_jobs: int
            sets the number of cores to use
        args.batch_size : int
            sets the batch size
        args.epochs : int 
            sets the epoch size 
        args.char_to_int : dict
            a map from characters to index (integer) in the sequences
        args.build_from_samples : boolean
            build model from samples
        """
        self.name = args["name"]
        self.hidden_size = args["hidden_size"]
        self.max_iterations = args["max_iterations"]
        self.n_jobs = args["n_jobs"]
        self.batch_size = args["batch_size"]
        self.epoch = args["epoch"]
        self.char_to_int = args["char_to_int"]
        self.vocabulary = [pair[0] for pair in sorted(self.char_to_int.items(), key = lambda x : x[1])]
        self.indexes = [pair[1] for pair in sorted(self.char_to_int.items(), key = lambda x : x[1])]
        self.emission_size = len(self.indexes)
        if args["build_from_samples"] and x_train is not None: 
            self.model = HiddenMarkovModel.from_samples(DiscreteDistribution, 
                                                    n_components = self.hidden_size, 
                                                    X = x_train, 
                                                    algorithm = 'baum-welch', 
                                                    return_history = True,
                                                    verbose = verbose,
                                                    max_iterations = self.max_iterations,
                                                    n_jobs = self.n_jobs, 
                                                    weights = weights,
                                                    #batch_size = self.batch_size,
                                                    #batches_per_epoch = self.epochs
                                               )[0]
            
        else: 
            self.build_model()
        self.model.bake()

    
    def build_model(self): 
        distributions = []
        for _ in range(self.hidden_size): 
            emission_probs = np.random.random(self.emission_size)
            emission_probs = emission_probs / emission_probs.sum()
            distributions.append(DiscreteDistribution(dict(zip(self.vocabulary, emission_probs))))
        trans_mat = np.random.random((self.hidden_size, self.hidden_size))
        trans_mat = trans_mat / trans_mat.sum(axis = 1, keepdims = 1)
        starts = np.random.random((self.hidden_size))
        starts = starts / starts.sum()
        # testing initializations
        np.testing.assert_almost_equal(starts.sum(), 1)
        np.testing.assert_array_almost_equal(np.ones(self.hidden_size), trans_mat.sum(axis = 1))
        self.model = HiddenMarkovModel.from_matrix(trans_mat, distributions, starts)

    def fit(self, x_train, weights=None, verbose=True):
        """
        Fits the model on an HMM with self.hidden_size
        """    
        return self.model.fit(x_train, 
                        algorithm = 'baum-welch', 
                        return_history = True, 
                        verbose = verbose,
                        max_iterations = self.max_iterations,
                        n_jobs = self.n_jobs, 
                        weights = weights,
                        #batch_size = self.batch_size,
                        #batches_per_epoch = self.epochs
                   )
    
    def sample(self, n, length):
        """
        Input:
        n is number of samples
        length is how long you want each sample to be
        """
        return np.array(["".join(seq) for seq in self.model.sample(n = n, length = length)])
            
        
    def predict(self, x_test): 
        """
        predict the log probability of obtaining the sequences in x_test
        log(P(X1, X2, ..., X_test)) = sum(log(P(Xi)))
        Input: x_test a list of sequences. should be 2 or 3 dimensional
        """
        assert(len(np.array(x_test).shape) == 2 or len(np.array(x_test).shape) == 3)
        return sum([self.model.log_probability(seq) for seq in np.array(x_test)])
                
    def show_model(self): 
        self.model.plot()
        
    def save_model(self, path): 
        with open(path, 'w') as f:
            json.dump(self.model.to_json(), f)
    
    def load_model(self, path): 
        with open(path, 'r') as f:
            json_model = json.load(f)
        self.model = HiddenMarkovModel.from_json(json_model)


# ## Model Functions

# In[7]:


def hmm_base_args(): 
    return {
        "name" : "base HMM",
        "hidden_size" : 5,
        "max_iterations" : 10,
        "n_jobs" : 1,
        "batch_size" : 5,
        "epoch" : 2,
        "char_to_int" : {"A" : 0, "C" : 1, "T" : 2, "G" : 3},
        "build_from_samples" : False
    }

def hmm_build_from_samples_args(): 
    args = hmm_base_args()
    args["build_from_samples"] = True
    return args

def hmm_amino_acid_args(): 
    args = hmm_base_args()
    amino_acids = get_all_amino_acids()
    indexes = list(range(len(amino_acids)))
    assert(len(amino_acids) == 21)
    assert(amino_acids == "*" + IUPAC.protein.letters) #*ACDEFGHIKLMNPQRSTVWY
    args["char_to_int"] = dict(zip(amino_acids, indexes))
    return args
    


# ## Tests

# In[17]:


def test_fit_dna_hmm(X_train_sequences): 
    args = hmm_base_args()
    hmm = GenerativeHMM(args)
    hmm.fit(X_train_sequences)
    args["max_iterations"] = 20
    args["n_jobs"] = 5
    hmm = GenerativeHMM(args)
    hmm.fit(X_train_sequences)
    assert(hmm.char_to_int == {"A" : 0, "C" : 1, "T" : 2, "G" : 3})
    assert(hmm.indexes == sorted(list(range(4))))
    assert(hmm.vocabulary == list("ACTG"))
    
def test_fit_amino_acid_hmm(X_train_sequences):
    args = hmm_amino_acid_args()
    hmm = GenerativeHMM(args)
    hmm.fit(X_train_sequences)
    args["max_iterations"] = 20
    args["n_jobs"] = 5
    hmm = GenerativeHMM(args)
    hmm.fit(X_train_sequences)
    amino_acid_alphabet = get_all_amino_acids()
    assert(hmm.char_to_int == dict(zip(amino_acid_alphabet, list(range(len(amino_acid_alphabet))))))
    assert(hmm.indexes == sorted(list(range(len(amino_acid_alphabet)))))
    assert(hmm.vocabulary == list(amino_acid_alphabet))
    
def test_sample_and_predict_dna_hmm(x_train_sequences): 
    args = hmm_base_args()
    hmm = GenerativeHMM(args)
    hmm.fit(x_train_sequences, verbose=False)
    seq1, seq2 = tuple(hmm.sample(2, 714))
    np.testing.assert_almost_equal(hmm.model.probability(seq1), np.e ** hmm.predict([list(seq1)]))
    total = 0
    for i in "ACTG": 
        for j in "ACTG": 
            for k in "ACTG":
                codon = i + j + k
                np.testing.assert_almost_equal(hmm.model.probability(codon), np.e ** hmm.predict([list(codon)]))
                total += np.e ** hmm.predict([list(codon)])
    np.testing.assert_almost_equal(1, total)
    
def test_sample_and_predict_amino_acid_hmm(x_train_sequences): 
    args = hmm_amino_acid_args()
    hmm = GenerativeHMM(args)
    hmm.fit(x_train_sequences, verbose=False)
    wild_type_amino_acid = get_wild_type_amino_acid_sequence()
    seq1, seq2 = tuple(hmm.sample(2, len(wild_type_amino_acid)))
    print("{0} amino acids away from wild type!".format(count_substring_mismatch(seq1, 
                                                            wild_type_amino_acid)))
    np.testing.assert_almost_equal(hmm.model.probability(seq1), np.e ** hmm.predict([list(seq1)]))
    total = 0
    amino_acid_alphabet = get_all_amino_acids()
    for i in amino_acid_alphabet: 
        for j in amino_acid_alphabet: 
            amino_acid = i + j
            np.testing.assert_almost_equal(hmm.model.probability(amino_acid), np.e ** hmm.predict([list(amino_acid)]))
            total += np.e ** hmm.predict([list(amino_acid)])
    np.testing.assert_almost_equal(total, 1)   
    
def test_fit_dna_hmm_from_samples(X_train_sequences): 
    args = hmm_build_from_samples_args()
    hmm = GenerativeHMM(args, X_train_sequences)
    #test max iterations and n_jobs 
    args["max_iterations"] = 20
    args["n_jobs"] = 5
    hmm = GenerativeHMM(args, X_train_sequences)
    assert(hmm.char_to_int == {"A" : 0, "C" : 1, "T" : 2, "G" : 3})
    assert(hmm.indexes == sorted(list(range(4))))
    assert(hmm.vocabulary == list("ACTG"))

def test_fit_dna_hmm_weights(X_train_sequences):
    args = hmm_base_args()
    weights = np.identity(4)
    weights = np.vstack([weights, [0.25, 0.25, 0.25, 0.25]])
    for weight in weights:
        counts = {"A" : 0, "C" : 0, "T" : 0, "G" : 0}
        hmm = GenerativeHMM(args)
        hmm.fit(X_train_sequences, weight, verbose=False)
        json_model = json.loads(hmm.model.to_json())
        for state in json_model["states"]: 
            if state is not None and state["distribution"] is not None:
                mp = state["distribution"]["parameters"][0]
                for k, v in mp.items(): 
                    counts[k] = counts[k] + v
        print("Weights:", weight, "\nCounts:", counts)    
        

def test_fit_dna_hmm_from_samples_weights(X_train_sequences):
    args = hmm_build_from_samples_args()
    weights = np.identity(4)
    weights = np.vstack([weights, [0.25, 0.25, 0.25, 0.25]])
    for weight in weights: 
        counts = {"A" : 0, "C" : 0, "T" : 0, "G" : 0}
        hmm = GenerativeHMM(args, X_train_sequences, weight, verbose=False)
        json_model = json.loads(hmm.model.to_json())
        for state in json_model["states"]: 
            if state is not None and state["distribution"] is not None:
                mp = state["distribution"]["parameters"][0]
                for k, v in mp.items(): 
                    counts[k] = counts[k] + v
        print("Weights:", weight, "\nCounts:", counts) 
        
def test_save_and_load_hmm(X_train_sequences, amino_acid = True): 
    if amino_acid:
        args = hmm_amino_acid_args()
    else:
        args = hmm_base_args()
    hmm = GenerativeHMM(args)
    hmm.fit(X_train_sequences, verbose=False)
    hmm.save_model("./models/test.json")
    cached_hmm = GenerativeHMM(args)
    cached_hmm.load_model("./models/test.json")
    for i in "ACTG": 
        for j in "ACTG": 
            for k in "ACTG":
                codon = i + j + k
                np.testing.assert_almost_equal(hmm.predict([list(codon)]), cached_hmm.predict([list(codon)]))


# In[18]:


test_fit_amino_acid_hmm(test_amino_acid_sequences)
test_sample_and_predict_amino_acid_hmm(test_amino_acid_sequences)
test_save_and_load_hmm(test_amino_acid_sequences)


# In[19]:


test_fit_dna_hmm(test_dna_sequences)
test_sample_and_predict_dna_hmm(test_dna_sequences)
test_fit_dna_hmm_from_samples(test_dna_sequences)
test_save_and_load_hmm(test_dna_sequences)

synthetic_data = np.array([["A", "A", "A", "T"], ["C", "C", "C", "G"], ["T", "T", "T", "A"], ["G", "G", "G", "C"]])
test_fit_dna_hmm_weights(synthetic_data)
test_fit_dna_hmm_from_samples_weights(synthetic_data)


# ## Code

# In[10]:


print("Loading data...")
start_time = time.time()
X_train, X_test, y_train, y_test = load_gfp_data("./data/gfp_dna_") #./data/gfp_dna_
mutated_df = load_saved_mutated_gfp_data()
assert(X_train[0] == get_wild_type_dna_sequence())
assert(count_substring_mismatch(X_train[999], get_wild_type_dna_sequence()) == 3)
print("Finished loading data in {0:.2f} seconds".format(time.time() - start_time))

base_args = hmm_base_args()
build_from_samples_args = hmm_build_from_samples_args()
test_dna_sequences = np.array([list(seq) for seq in X_train[0:100]])


# In[11]:


print("Loading data...")
start_time = time.time()
X_train, X_test, y_train, y_test = load_gfp_data("./data/gfp_amino_acid_") #./data/gfp_amino_acid_
assert(X_train[0] == get_wild_type_amino_acid_sequence())
assert(count_substring_mismatch(X_train[999], get_wild_type_amino_acid_sequence()) == 3)
print("Finished loading data in {0:.2f} seconds".format(time.time() - start_time))

base_amino_acid_args = hmm_amino_acid_args()
test_amino_acid_sequences = np.array([list(seq) for seq in X_train[0:100]])

