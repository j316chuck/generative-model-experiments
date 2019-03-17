import pandas as pd
import numpy as np
import time 
import pixiedust

from pomegranate import State, DiscreteDistribution, HiddenMarkovModel
from sklearn.model_selection import train_test_split
from utils import *
from hmm import GenerativeHMM, hmm_amino_acid_args

print("Loading data...")
start_time = time.time()
X_train, X_test, y_train, y_test = load_gfp_data("./data/gfp_amino_acid_")
mutated_df = load_saved_mutated_gfp_data()
print("Finished loading data in {0:.2f} seconds".format(time.time() - start_time))


# In[3]:


wild_type_amino_acid = get_wild_type_amino_acid_sequence()
assert(X_train[0] == wild_type_amino_acid)
assert(count_substring_mismatch(wild_type_amino_acid, X_train[1000]) == 8)
assert(count_substring_mismatch(wild_type_amino_acid, mutated_df["mutated_amino_acid_sequence"].values[0]) == 1)


# In[5]:


def get_data(X_train, length, n = 100, random=True): 
    if not random: 
        data = X_train[0:length]
    else: 
        indexes = np.random.choice(len(X_train), n)
        data = X_train[indexes]
    return np.array([list(x)[0:length] for x in data])

def sample_and_score(hmm, wild_type, n = 100, length = 100, file = None):
    """
    use the hmm model to sample n sequences of size = length. 
    then use the wild_type to count how far off the average sample is from the wild_type
    """
    assert(len(wild_type) == length)
    samples = hmm.sample(n, length)        
    average_diff = np.average([count_substring_mismatch(seq, wild_type) for seq in samples])
    print("Average difference: {0:.2f}, or {1:.2f} mismatches per letter".format(average_diff, 
                                                                 average_diff / length), file = file)
    print("Example sequence {0}".format(samples[np.random.randint(0, n)]), file = file)
    return average_diff

small_length, medium_length, large_length = 15, len(wild_type_amino_acid) // 4, len(wild_type_amino_acid)
small_X = get_data(X_train, small_length, 100)
medium_X = get_data(X_train, medium_length, 100)
large_X = get_data(X_train, large_length, 100)


# In[6]:


diffs = [count_substring_mismatch(i, wild_type_amino_acid[0:small_length]) for i in small_X]
print("Small diffs:", diffs)
diffs = [count_substring_mismatch(i, wild_type_amino_acid[0:medium_length]) for i in medium_X]
print("Medium diffs:", diffs)
diffs = [count_substring_mismatch(i, wild_type_amino_acid[0:large_length]) for i in large_X]
print("Large diffs:", diffs)


# In[19]:


def train_and_save_hmm(args):
    start_time = time.time()
    hmm = GenerativeHMM(args)
    hmm.fit(small_X)
    log_path = "./logs/{0}.txt".format(hmm.name)
    logger = open(log_path, "w")
    print("Finished training in {:.2f} seconds".format(time.time() - start_time), file = logger)
    print("HMM Parameters:", file = logger)
    print(hmm.get_args(), file = logger)
    sample_and_score(hmm, wild_type_amino_acid[0:small_length], 100, small_length, file = logger)
    model_path = "./models/{0}.json".format(hmm.name)
    hmm.save_model(model_path)
    cached_hmm = GenerativeHMM(args)
    cached_hmm.load_model(model_path)
    try: 
        for i in get_all_amino_acids():
            for j in get_all_amino_acids(): 
                np.testing.assert_almost_equal(hmm.predict([list(i + j)]), cached_hmm.predict([list(i + j)]))
        print("Successfully finished training and saving {0} hmm!".format(hmm.name), file = logger)
        logger.close()
    except:
        print("Error in loading {0} hmm".format(hmm.name))
        logger.close()

def get_small_args():
    base_args = hmm_amino_acid_args()
    base_args["name"] = "amino_acid_small"
    base_args["max_iterations"] = 100
    base_args["hidden_size"] = 50
    base_args["n_jobs"] = 5
    return base_args


# In[20]:


train_and_save_hmm(get_small_args())


# In[ ]:


10, 100 sequences, 100 iterations, 100 sequences. 
10, 200, 1e8, 100 sequences, 
10, 200, 1000, 100 sequences. 
10, 500, 1000, 100 sequences. 
10, 200, 1000, 10000 seqeunces. 


## Fit 3 types.
## Fit small data -> large data. 
## Fit with different hidden sizes 10, 50, 200. 
## fit until it 1e8, 1e2 iterations
## all with more cores 5
## record times of all these. 


