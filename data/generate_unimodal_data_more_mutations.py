import numpy as np

from sklearn.model_selection import train_test_split
from utils import get_all_amino_acids, generate_mutations_df, generate_discrete_uniform_distribution

# get data distributions
base_sequences_lst = ["MDILLDLGWHFSNCDEDTFYSPVQNTEGDLLFFDHNLKTDRGHVERSVMD",
                      "MQKPCKENEGKPKCSVPKREEKRPYGEFERQQTEGNFRQRLLQSLEEFKEDIDYRHFKDEEMTREGDEMERCLEEIRGLRKKFRALHSNHRHSRDRPYPI"]
num_samples = 50000
maximum_mutations = 21
uniform_lst = list(generate_discrete_uniform_distribution(num_samples, low=1, high=maximum_mutations)[1].values())
mutations_lst = list(range(1, maximum_mutations + 1))
dataset_map = {
    "uniform": uniform_lst,
}

# generate mutation datasets
alphabet = get_all_amino_acids(gap=False)
assert(len(alphabet) == 20)
start_mutation_index = 1
end_mutation_index = 1
for base_sequence in base_sequences_lst:
    dataset_name = "synthetic_mutations_{0}_unimodal_data_length_{1}".format(maximum_mutations, len(base_sequence))
    for distribution_name, mutations_count_lst in dataset_map.items():
        name = dataset_name + "_" + distribution_name
        mutated_df = generate_mutations_df(base_sequence, mutations_lst, mutations_count_lst, alphabet,
                                           start_mutation_index, end_mutation_index, verbose=True)
        mutated_df = mutated_df.sample(frac=1).reset_index(drop=True)  # shuffle
        mutated_df.to_csv(name + ".csv", index=False)
        x_train, x_test = train_test_split(mutated_df["mutated_string"].values, test_size=0.2)
        np.save(name + "_x_train.npy", x_train)
        np.save(name + "_x_test.npy", x_test)
