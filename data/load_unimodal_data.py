import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from utils import count_substring_mismatch, load_base_sequence

start_index, end_index = 1, 1
for data_path in glob.glob("./synthetic_unimodal_data_*.csv"):
    # load dataset
    name = data_path[:-4]
    print(name)
    df = pd.read_csv(data_path)
    x_train = np.load(name + "_x_train.npy")
    x_test = np.load(name + "_x_test.npy")
    base_sequence = load_base_sequence(name)
    # check if number of mismatches is equivalent to the mutation count
    for i, row in df.iterrows():
        assert(count_substring_mismatch(row["base_sequence"], row["mutated_string"]) == row["mutation_count"])
        assert(count_substring_mismatch(row["base_sequence"][:start_index], row["mutated_string"][:start_index]) == 0)
        assert(count_substring_mismatch(row["base_sequence"][-end_index:], row["mutated_string"][-end_index:]) == 0)
        assert(row["base_sequence"] == base_sequence)
        assert("*" not in row["mutated_string"])

    # check if loaded x_train and x_test are right
    assert((set(x_train) | set(x_test)) == set(df["mutated_string"].values))
    # plot distribution to see if mismatches distribution is right
    plt.title(name)
    plt.hist(df["mutation_count"].values, bins=10)
    plt.xlabel("mutations")
    plt.ylabel("counts")
    plt.show()
