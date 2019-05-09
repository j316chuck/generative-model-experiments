import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
from utils import count_substring_mismatch, load_base_sequence

start_index, end_index = 1, 1
base_sequences_lst = load_base_sequence("synthetic_multimodal_data_modes_5_length_50_uniform")
dataset_size = 20000
for data_path in glob.glob("./synthetic_multimodal_data_*.csv"):
    # load dataset
    name = data_path[:-4]
    print(name)
    mode = int(name[int(name.find("modes")) + 6])  # only for single digit modes
    df = pd.read_csv(data_path)
    x_train = np.load(name + "_x_train.npy")
    x_test = np.load(name + "_x_test.npy")
    y_train = np.load(name + "_y_train.npy")
    y_test = np.load(name + "_y_test.npy")
    base_sequence = load_base_sequence(name)
    # check if number of mismatches is equivalent to the mutation count
    for i, row in df.iterrows():
        assert(count_substring_mismatch(row["base_sequence"], row["mutated_string"]) == row["mutation_count"])
        assert(count_substring_mismatch(row["base_sequence"][:start_index], row["mutated_string"][:start_index]) == 0)
        assert(count_substring_mismatch(row["base_sequence"][-end_index:], row["mutated_string"][-end_index:]) == 0)
        assert(row["base_sequence"] == base_sequences_lst[int(row["base_sequence_index"])])
        assert("*" not in row["base_sequence"])
    # check shape of mutated dataframe is correct
    print(df.shape[0], mode)
    assert(df.shape[0] == dataset_size * mode)
    for i in range(mode):
        assert(df["base_sequence"].values.tolist().count(base_sequences_lst[i]) == dataset_size)

    # check if loaded x_train, x_test, y_train, y_test are right
    assert((set(x_train) | set(x_test)) == set(df["mutated_string"].values))
    assert((set(y_train) | set(y_test)) == set(df["base_sequence"].values))
    # plot distribution to see if mismatches distribution is right
    plt.title(name)
    plt.hist(df["mutation_count"].values, bins=10)
    plt.xlabel("mutations")
    plt.ylabel("counts")
    plt.show()
