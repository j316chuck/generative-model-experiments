import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from utils import get_all_amino_acids, generate_mutations_df, generate_discrete_uniform_distribution
from utils import generate_discrete_gaussian_distribution, generate_discrete_skewed_gaussian_distribution

# get data distributions
base_sequences_lst = ["NLYIQWLKDGGPSSGRPPPS",
                      "MDILLDLGWHFSNCDEDTFYSPVQNTEGDLLFFDHNLKTDRGHVERSVMD",
                      "MQKPCKENEGKPKCSVPKREEKRPYGEFERQQTEGNFRQRLLQSLEEFKEDIDYRHFKDEEMTREGDEMERCLEEIRGLRKKFRALHSNHRHSRDRPYPI"]
num_samples = 10000
uniform_lst = list(generate_discrete_uniform_distribution(num_samples, low=1, high=10)[1].values())
skewed_gaussian_lst = list(generate_discrete_skewed_gaussian_distribution(num_samples, a=4, mean=3.5, std=2, low=1, high=10)[1].values())
gaussian_lst = list(generate_discrete_gaussian_distribution(num_samples, mean=5.5, std=2, low=1, high=10)[1].values())
mutations_lst = list(range(1, 11))
dataset_map = {
    "uniform": uniform_lst,
    "skewed_gaussian": skewed_gaussian_lst,
    "gaussian": gaussian_lst
}

# plot data distributions
plt.title("data distributions")
plt.plot(uniform_lst, label="uniform")
plt.plot(skewed_gaussian_lst,label="skewed_gaussian")
plt.plot(gaussian_lst, label="gaussian")
plt.legend()
plt.xlabel("mutations")
plt.xticks(mutations_lst)
plt.ylabel("counts")
plt.savefig("./data_distributions.png")
plt.close()

# generate mutation datasets
alphabet = get_all_amino_acids()
start_mutation_index = 1
end_mutation_index = 1
for base_sequence in base_sequences_lst:
    dataset_name = "synthetic_unimodal_data_length_{0}".format(len(base_sequence))
    for distribution_name, mutations_count_lst in dataset_map.items():
        name = dataset_name + "_" + distribution_name
        mutated_df = generate_mutations_df(base_sequence, mutations_lst, mutations_count_lst, alphabet,
                                           start_mutation_index, end_mutation_index, verbose=True)
        mutated_df = mutated_df.sample(frac=1).reset_index(drop=True)  # shuffle
        mutated_df.to_csv(name + ".csv", index=False)
        x_train, x_test = train_test_split(mutated_df["mutated_string"].values, test_size=0.5)
        np.save(name + "_x_train.npy", x_train)
        np.save(name + "_x_test.npy", x_test)
