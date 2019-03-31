import matplotlib
matplotlib.use('Agg')
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

from torch.autograd import Variable
from torchviz import make_dot
from torch.utils.data import TensorDataset, DataLoader
from utils import load_gfp_data, get_all_amino_acids, get_wild_type_amino_acid_sequence 
from utils import count_substring_mismatch, string_to_tensor, string_to_numpy

# https://github.com/spro/char-rnn.pytorch/blob/master/model.py
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model="lstm", n_layers=1):
        super(RNN, self).__init__()
        self.model = model.lower()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.model == "gru":
            self.rnn = nn.GRU(hidden_size, hidden_size, n_layers)
        elif self.model == "lstm":
            self.rnn = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        # input is of shape (batch_size, 1) where each input[x, 0] is the word index
        # char RNN so we generate one character at a time. 
        batch_size = input.size(0)
        encoded = self.encoder(input)
        output, hidden = self.rnn(encoded.view(1, batch_size, -1), hidden)
        output = self.decoder(output.view(batch_size, -1))
        return output, hidden
    
    def init_hidden(self, batch_size):
        if self.model == "lstm":
            return (Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)),
                    Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size)))
        return Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
    
class GenerativeRNN(): 
    
    def __init__(self, args):     
        """
        Initializes the RNN to be a generative char RNN
        Parameters
        ----------
        args : dictionary
            defines the hyper-parameters of the neural network
        args.name : string 
            defines the name of the neural network
        args.description: string
            describes the architecture of the neural network
        args.layers : int
            specifies the number of stacked layers we want in the LSTM
        args.hidden_size : int
            the size of the hidden layer
        args.learning_rate : float
            sets the learning rate
        args.epochs : int 
            sets the epoch size 
        args.vocabulary : string
            all the characters in the context of the problem
        """
        self.name = args["name"]
        self.description = args["description"]
        self.layers = args["layers"]
        self.hidden_size = args["hidden_size"]
        self.learning_rate = args["learning_rate"]
        self.epochs = args["epochs"]
        self.all_characters = args["vocabulary"]
        self.num_characters = len(self.all_characters)
        self.character_to_int = dict(zip(self.all_characters, range(self.num_characters)))
        self.int_to_character = dict(zip(range(self.num_characters), self.all_characters))
        self.model = RNN(self.num_characters, self.hidden_size, self.num_characters, "lstm", self.layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.loss_history = []
        

    def fit(self, dataloader, verbose=True, logger=None, save_model=True):
        # amino acid dataset specific checks
        wild_type = get_wild_type_amino_acid_sequence()
        three_mutation = "".join([self.int_to_character[np.random.randint(0, self.num_characters)] if i % 3 == 1 else wild_type[i] for i in range(10)])
        ten_mutation = "".join([self.int_to_character[np.random.randint(0, self.num_characters)] for i in range(10)])
        print("Sampled sequences", wild_type[0:10], three_mutation, ten_mutation, file=logger)
        
        if not os.path.isdir("./models/{0}".format(self.name)):
            os.mkdir("./models/{0}".format(self.name))
        
        start_time = time.time()
        self.loss_history = []
        for epoch in range(1, self.epochs + 1):
            total_loss = []
            for i, (input, target) in enumerate(dataloader):
                batch_size, seq_length = input.shape[0], input.shape[1]
                hidden = self.model.init_hidden(batch_size)
                self.model.zero_grad()
                
                loss = 0
                for c in range(seq_length):
                    output, hidden = self.model(input[:, c], hidden)
                    loss += self.criterion(output.view(batch_size, -1), target[:, c])
                
                loss.backward()
                self.optimizer.step()
                total_loss.append(loss.item() / seq_length)
            
            self.loss_history.append(np.mean(total_loss))
            generated_sequence = self.sample(predict_len = len(wild_type) - 1, prime_str = "S")
            mismatches = count_substring_mismatch(wild_type, generated_sequence)
            wild_prob, mutation_three_prob, mutation_ten_prob = self.predict_log_prob(wild_type[1:10]), self.predict_log_prob(three_mutation[1:10]), self.predict_log_prob(ten_mutation[1:10])
            
            if verbose: 
                print("epoch {0}. loss: {1:.2f}. time: {2:.2f} seconds.".format(epoch, self.loss_history[-1], time.time() - start_time), file = logger)
                print("generated sequence: {0}\n{1} mismatches from the wild type".format(generated_sequence, mismatches), file = logger) 
                print("wild type log prob: {0}. 3 mutations log prob: {1}. 10 mutations log prob: {2}.\n" \
                      .format(wild_prob, mutation_three_prob, mutation_ten_prob), file = logger)
            if save_model:
                self.save_model(epoch, total_loss)        

    def predict_log_prob(self, sequence, prime_str = "S"):
        hidden = self.model.init_hidden(1) 
        prime_input = string_to_tensor(prime_str, self.character_to_int)
        for p in range(len(prime_str) - 1):
            _, hidden = self.model(prime_input[p], hidden)
        input = prime_input[-1]

        log_prob = 0
        for char in sequence:
            output, hidden = self.model(input.view(1, -1), hidden)
            softmax = nn.Softmax(dim = 1)
            probs = softmax(output).view(-1)
            i = self.character_to_int[char]
            log_prob += np.log(probs[i].item())
        return log_prob

    def sample(self, predict_len, prime_str = 'S', temperature = 1):
        hidden = self.model.init_hidden(1)
        prime_input = string_to_tensor(prime_str, self.character_to_int)
        predicted = prime_str

        # Use priming string to "build up" hidden state
        for p in range(len(prime_str) - 1):
            output, hidden = self.model(prime_input[p], hidden)
        input = prime_input[-1]

        for p in range(predict_len):
            output, hidden = self.model(input.view(1, -1), hidden)

            # Sample from the network as a multinomial distribution
            output_dist = output.data.view(-1).div(temperature).exp()
            top_i = torch.multinomial(output_dist, 1)[0].item()

            # Add predicted character to string and use as next input
            predicted_char = self.int_to_character[top_i]
            predicted += predicted_char
            input = string_to_tensor(predicted_char, self.character_to_int)

        return predicted

            
    def load_model(self, model_path):
        checkpoint = torch.load("./models/{0}/{1}".format(self.name, model_path))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    def save_model(self, epoch=None, loss=None): 
        torch.save({
                    'epoch': epoch,
                    'loss': loss,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()
                }, "./models/{0}/checkpoint_{1}.pt".format(self.name, epoch))

    
    def show_model(self): 
        print(self.model)
    
    def plot_model(self, save_dir, verbose=True): 
        hidden = self.model.init_hidden(1)
        out, _ = self.model(string_to_tensor("S", self.character_to_int), hidden)
        graph = make_dot(out)
        if save_dir is not None:
            graph.format = "png"
            graph.render(save_dir) 
        if verbose:
            graph.view()
            
    def plot_history(self, save_fig_dir): 
        plt.figure()
        plt.title("Training Loss Curve")
        plt.plot(self.loss_history)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.xticks(range(self.epochs))
        if save_fig_dir:
            plt.savefig(save_fig_dir)
        plt.show()
    
def get_test_args():
    args = {
        "name" : "rnn_test_sample",
        "layers" : 2, 
        "hidden_size" : 200,
        "learning_rate" : 0.005,
        "epochs" : 10,
        "vocabulary" : get_all_amino_acids(),
        "num_data" : 100, 
        "batch_size" : 10
    }
    args["description"] = "name: {0}, layers {1}, hidden size {2}, lr {3}, epochs {4}".format(args["name"], 
                            args["layers"], args["hidden_size"], args["learning_rate"], args["epochs"])

    return args
