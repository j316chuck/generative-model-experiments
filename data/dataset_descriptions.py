import glob
import pandas as pd
import numpy as np
from utils import get_wild_type_amino_acid_sequence, load_base_sequences

columns = ["dataset", "sequence_length", "num_data", "distribution", "base_sequence", "notes"]
df = pd.DataFrame(columns=columns)

# unimodal descriptions
for file in glob.glob("./synthetic_unimodal_*.csv"):
    values = []
    values.append(file[2:-4])  # name
    start_index = file.find("length") + 7
    end_index = file[start_index:].find("_")
    values.append(int(file[start_index: start_index + end_index]))  # sequence length
    values.append(pd.read_csv(file).shape[0])  # num_data
    sub_index = file[::-1].find("_")
    if "skewed" in file:
        values.append("skewed_" + file[len(file) - sub_index:-4])  # distribution
    else:
        values.append(file[len(file) - sub_index:-4])  # distribution
    values.append(pd.read_csv(file).iloc[0, 2])  # base_sequence
    values.append("no * character, vocabulary-20 amino acids")
    df.loc[df.shape[0]] = values

# gfp descriptions
values = ["gfp", 238, 58417, "skewed_gaussian", get_wild_type_amino_acid_sequence(), "* character, vocabulary-21 amino acids"]
df.loc[df.shape[0]] = values

# multimodal descriptions
for file in glob.glob("./synthetic_multimodal_*.csv"):
    values = []
    values.append(file[2:-4])  # name
    start_index = file.find("length") + 7
    end_index = file[start_index:].find("_")
    values.append(int(file[start_index: start_index + end_index]))  # sequence length
    values.append(pd.read_csv(file).shape[0])  # num_data
    if "skewed" in file:
        values.append("skewed_" + file[len(file) - sub_index:-4])  # distribution
    else:
        values.append(file[len(file) - sub_index:-4])  # distribution
    values.append(load_base_sequences(file))  # base_sequences
    sub_index = file[::-1].find("_")
    values.append("no * character, vocabulary-20 amino acids")
    df.loc[df.shape[0]] = values

# output to csv
print(df)
df.to_csv("./dataset_descriptions.csv", index=False)
