import unidecode
import string
import random
import re
import time
import torch
import torch.nn as nn
import numpy as np 
import os 
import matplotlib.pyplot as plt
import sys 
import argparse

from torch.autograd import Variable
from torchviz import make_dot
from torch.utils.data import TensorDataset, DataLoader
from utils import load_gfp_data, get_all_amino_acids, get_wild_type_amino_acid_sequence 
from utils import count_substring_mismatch, string_to_tensor, string_to_numpy
from rnn import GenerativeRNN, get_test_args

def get_dataloader(X_train, length, character_to_int, n = 100, batch_size = 1, shuffle = True, random = True):
    """
	returns a dataloader that loads n sequences of length size broken into batch_size chunks. 
	seqences may be randomly drawn from initial dataset and shuffled every epoch
    """
    if not random: 
        data = X_train[0:n]
    else: 
        indexes = np.random.choice(len(X_train), n)
        data = X_train[indexes]        
    dataset = np.array([string_to_numpy(x[0:length], character_to_int) for x in data])
    input = torch.from_numpy(dataset[:, :-1]).long()
    output = torch.from_numpy(dataset[:, 1:]).long()
    tensor_dataset = TensorDataset(input, output)
    return DataLoader(tensor_dataset, batch_size = batch_size, shuffle = shuffle)


""" 
Parsing Arguments
"""
parser = argparse.ArgumentParser(description='Process the arguments for the RNN Model')
parser.add_argument("-m", "--model", default="rnn", required=False, help="model to use", type=str)
parser.add_argument("-de", "--default", default=0, required=False, help="use default args", type=int)
parser.add_argument("-da", "--dataset", default="gfp_amino_acid_", required=False, help="which dataset to use", type=str)
parser.add_argument("-n", "--name", default="rnn_test_sample", required=False, help="name of model", type=str)
parser.add_argument("-l", "--layers", default=2, required=False, help="layer size", type=int)
parser.add_argument("-hi", "--hidden_size", default=200, required=False, help="hidden size of rnn", type=int)
parser.add_argument("-lr", "--learning_rate", default=0.005, required=False, help="learning_rate", type=float)
parser.add_argument("-e", "--epochs", default=10, required=False, help="number of epochs to train", type=int)
parser.add_argument("-d", "--num_data", default=100, required=False, help="number of data points to train on", type=int)
parser.add_argument("-b", "--batch_size", default=10, required=False, help="batch_size", type=int)
parser_args = parser.parse_args()
print(parser_args)

if parser_args.model != "rnn":
	sys.exit("not rnn model") 
if parser_args.default == 1:
	args = get_test_args()
	args["dataset"] = "gfp_amino_acid_"
else: 
	args = vars(parser_args)
	args["description"] = "name: {0}, layers {1}, hidden size {2}, lr {3}, epochs {4}".format(args["name"], 
    					args["layers"], args["hidden_size"], args["learning_rate"], args["epochs"])
	if args["dataset"] == "gfp_amino_acid_":
		args["vocabulary"] = get_all_amino_acids()
	else:
		sys.exit("not gfp dataset")
    

"""
Loading Data
"""
wild_type = get_wild_type_amino_acid_sequence()
seq_length = len(wild_type)
X_train, X_test, y_train, y_test = load_gfp_data(os.path.join('./data', args["dataset"]))
char_to_int = dict(zip(args["vocabulary"], range(len(args["vocabulary"]))))
dataloader = get_dataloader(X_train, length = seq_length, character_to_int = char_to_int, n = args["num_data"], batch_size = args["batch_size"], shuffle = True, random = True)


"""
Training and Evaluating RNN
"""
logger = open("./logs/{0}.txt".format(args["name"]), "w")
rnn = GenerativeRNN(args)
print("Training RNN\nModel Description: {0}".format(rnn.description), file=logger)
rnn.fit(dataloader=dataloader, logger=logger)
rnn.show_model()
rnn.plot_history("./logs/{0}_training_history".format(args["name"]))
rnn.plot_model("./logs/{0}_model_architecture".format(args["name"]))
temperature_lst = [0.2, 0.8, 1.0, 1.2, 1.8]
for temperature in temperature_lst: 
    generated_sequence = rnn.sample(predict_len = len(wild_type) - 1, prime_str = "S", temperature = temperature)
    mismatches = count_substring_mismatch(wild_type, generated_sequence)
    print("temperature: {0}. generated sequence: {1} with {2} mismatches from the wild type.".format(temperature, generated_sequence, mismatches), file = logger) 

""" 
Extra Checks
"""
def enumerate_all_sequences(model, string, base = "S", depth = 3): 
    if depth == 0: 
        return np.e ** model.predict_log_prob(string, base)
    total = 0
    for c in get_all_amino_acids(): 
        total += enumerate_all_sequences(model, string + c, base, depth - 1)
    return total
for depth in range(1, 3):
    for base in "QRST":
        np.testing.assert_almost_equal(1, enumerate_all_sequences(rnn, "", base, depth))


epoch = 8
load_rnn = GenerativeRNN(args)
load_rnn.load_model("./checkpoint_{0}.pt".format(epoch))
total_loss = []
for i, (input, target) in enumerate(dataloader):
    batch_size, seq_length = input.shape[0], input.shape[1]
    hidden = load_rnn.model.init_hidden(batch_size)
    loss = 0
    for c in range(seq_length):
        output, hidden = load_rnn.model(input[:, c], hidden)
        loss += load_rnn.criterion(output.view(batch_size, -1), target[:, c])
    total_loss.append(loss.item() / seq_length)
print("Loading Model checkpoint {0}".format(epoch), file=logger) 
print("Loaded Model loss", total_loss, file=logger)
print("Checkpoint loss", torch.load("./models/rnn_test_sample/checkpoint_{0}.pt".format(epoch))["loss"], file=logger)
logger.close()

