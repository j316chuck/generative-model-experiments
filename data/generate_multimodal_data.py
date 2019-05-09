import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from utils import get_all_amino_acids, generate_mutations_df, generate_discrete_uniform_distribution
from utils import load_base_sequence

# get data distributions
base_sequences_lst = load_base_sequence("synthetic_multimodal_data_modes_5_length_50_uniform")
modes_lst = [2, 3, 5]
num_samples = 10000
uniform_lst = list(generate_discrete_uniform_distribution(num_samples, low=1, high=10)[1].values())

# generate mutation datasets
mutations_lst = list(range(1, 11))
dataset_size = 20000
mutations_count_lst = [int(dataset_size//10)] * 10
alphabet = get_all_amino_acids(gap=False)
assert(len(alphabet) == 20)
start_mutation_index = 1
end_mutation_index = 1
for mode in modes_lst:
    dataset_name = "synthetic_multimodal_data_modes_{0}_length_50_uniform".format(mode)
    mutated_df = pd.DataFrame(columns=['mutated_string', 'mutation_count', 'base_sequence', 'base_sequence_index'])
    for i in range(mode):
        new_mutated_df = generate_mutations_df(base_sequences_lst[i], mutations_lst, mutations_count_lst, alphabet,
                                               start_mutation_index, end_mutation_index, verbose=True)
        new_mutated_df['base_sequence_index'] = np.ones(dataset_size) * i
        mutated_df = mutated_df.append(new_mutated_df, ignore_index=True)
    mutated_df.sample(frac=1).reset_index(drop=True)
    mutated_df.to_csv(dataset_name + ".csv", index=False)
    x_train, x_test, y_train, y_test = train_test_split(mutated_df["mutated_string"].values,
                                                        mutated_df["base_sequence"], test_size=0.2)
    np.save(dataset_name + "_x_train.npy", x_train)
    np.save(dataset_name + "_x_test.npy", x_test)
    np.save(dataset_name + "_y_train.npy", y_train)
    np.save(dataset_name + "_y_test.npy", y_test)

