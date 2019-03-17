import pandas as pd
import numpy as np
import time 
import json
from pomegranate import State, DiscreteDistribution, HiddenMarkovModel
from sklearn.model_selection import train_test_split
from utils import *
from Bio.Alphabet import IUPAC

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
