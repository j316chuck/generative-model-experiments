import time
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torchviz import make_dot
from torch.nn import functional as F
from utils import string_to_tensor
from models import Model


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


class GenerativeRNN(Model):

    def __init__(self, args):
        Model.__init__(self, args)
        self.name = args["name"]
        self.model_type = args["model_type"]
        self.input = args["input"]
        self.layers = args["layers"]
        self.hidden_size = args["hidden_size"]
        self.learning_rate = args["learning_rate"]
        self.epochs = args["epochs"]
        self.all_characters = args["vocabulary"]
        self.batch_size = args["batch_size"]
        self.pseudo_count = args["pseudo_count"]
        self.device = args["device"]
        self.seq_length = args["seq_length"]
        self.num_characters = len(self.all_characters)
        self.indexes = list(range(self.num_characters))
        self.character_to_int = dict(zip(self.all_characters, self.indexes))
        self.int_to_character = dict(zip(self.indexes, self.all_characters))
        self.initial_probs = dict(zip(self.indexes, np.zeros(self.num_characters)))
        self.initial_probs_tensor = []
        self.model = RNN(self.num_characters, self.hidden_size, self.num_characters, "lstm", self.layers)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loss_history, self.valid_loss_history = [], []
        assert(self.seq_length * self.num_characters == self.input)

    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model=True, weights=None,
            **kwargs):
        start_time = time.time()
        self.model.train()
        self.train_loss_history, self.valid_loss_history = [], []
        # fit initial distribution of starting characters with la place smoothing
        self.initial_probs = dict(zip(self.indexes, np.ones(self.num_characters) * self.pseudo_count))
        for inp, _ in train_dataloader:
            for char_index in inp[:, 0]:
                self.initial_probs[char_index.item()] += 1
        dataset_length = len(train_dataloader.dataset)
        smoothing_count = self.num_characters * self.pseudo_count
        for char_index in self.initial_probs.keys():
            self.initial_probs[char_index] = self.initial_probs[char_index] / (dataset_length + smoothing_count)
        # initial distribution to sample from
        self.initial_probs_tensor = torch.Tensor(list(self.initial_probs.values()))

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_loss = 0
            for i, (inp, target) in enumerate(train_dataloader):
                batch_size, seq_length = inp.shape[0], inp.shape[1]
                hidden = self.model.init_hidden(batch_size)
                self.model.zero_grad()
                loss = 0
                for c in range(seq_length):
                    output, hidden = self.model(inp[:, c], hidden)
                    loss += self.criterion(output.view(batch_size, -1), target[:, c])  # mean cross entropy loss
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * batch_size
            train_loss /= len(train_dataloader.dataset)
            self.train_loss_history.append(train_loss)
            self.model.eval()
            if valid_dataloader:
                valid_loss = self.evaluate(valid_dataloader, verbose=False, logger=logger)
                self.valid_loss_history.append(valid_loss)
            if verbose:
                print("epoch {0}, train neg log prob: {1:.4f}, test neg log probability {2:.4f}, time: {3:.2f} sec".format(
                        epoch, train_loss, valid_loss, time.time() - start_time), file=logger)
            if epoch % self.save_epochs == 0 and save_model:
                path = os.path.join(self.base_log, self.name, "{0}_checkpoint_{1}.pt".format(self.model_type, epoch))
                print(path)
                self.save_model(path, epoch=epoch, loss=loss, initial_probs=True)

    def evaluate(self, dataloader, verbose=False, logger=None, weights=None, **kwargs):
        total_loss = 0
        for inp, target in dataloader:
            batch_size, seq_length = inp.shape[0], inp.shape[1]
            for starting_char_index in inp[:, 0]:
                total_loss += -np.log(self.initial_probs[starting_char_index.item()]) #neg log probability of starting character
            hidden = self.model.init_hidden(batch_size)
            for c in range(seq_length):
                output, hidden = self.model(inp[:, c], hidden)
                total_loss += (self.criterion(output.view(batch_size, -1), target[:, c]) * batch_size)
        total_loss = total_loss.item() / len(dataloader.dataset)
        if verbose:
            print('total loss: {0:.4f}'.format(total_loss), file=logger)
        return total_loss

    def sample(self, num_samples, length, to_string=True, **kwargs):
        hidden = self.model.init_hidden(num_samples)
        input = torch.multinomial(input=self.initial_probs_tensor, num_samples=num_samples, replacement=True)
        predicted_strings = input.reshape(1, num_samples).long()
        sampled_probabilities = torch.stack([self.initial_probs_tensor for _ in range(num_samples)])
        sampled_probabilities = sampled_probabilities.reshape(1, num_samples, self.num_characters)
        for _ in range(1, length):
            output, hidden = self.model(input.view(num_samples, 1), hidden)
            output = F.softmax(output, dim=-1)
            input = torch.Tensor([torch.multinomial(input=prob, num_samples=1, replacement=True)[0] for prob in output]).long()
            sampled_probabilities = torch.cat([sampled_probabilities, output.reshape(1, num_samples, self.num_characters)])
            predicted_strings = torch.cat([predicted_strings, input.reshape(1, num_samples)])
        sampled_probabilities = sampled_probabilities.permute(1, 0, 2)
        predicted_strings = predicted_strings.permute(1, 0).detach().numpy()
        if to_string:
            sampled_strings = []
            for string in predicted_strings:
                sampled_strings.append("".join([self.int_to_character[index] for index in string]))
            return sampled_strings
        else:
            return sampled_probabilities.detach().numpy()

    def show_model(self, logger=None, **kwargs):
        print(self.model, file=logger)

    def plot_model(self, save_fig_dir, show=False, **kwargs):
        hidden = self.model.init_hidden(1)
        out, _ = self.model(string_to_tensor("S", self.character_to_int), hidden)
        graph = make_dot(out)
        if save_fig_dir is not None:
            graph.format = "png"
            graph.render(save_fig_dir)
        if show:
            graph.view()

    def save_model(self, path, **kwargs):
        d = dict()
        d['model_state_dict'] = self.model.state_dict()
        d['optimizer_state_dict'] = self.optimizer.state_dict()
        if 'initial_probs' in kwargs:
            d['initial_probs'] = self.initial_probs
        if 'epoch' in kwargs:
            d['epoch'] = kwargs['epoch']
        if 'loss' in kwargs:
            d['loss'] = kwargs
        torch.save(d, path)

    def load_model(self, path, **kwargs):
        saved_dict = torch.load(path)
        self.model.load_state_dict(saved_dict["model_state_dict"])
        self.optimizer.load_state_dict(saved_dict["optimizer_state_dict"])
        if 'initial_probs' in kwargs:
            self.initial_probs = saved_dict['initial_probs']
            self.initial_probs_tensor = torch.Tensor(list(self.initial_probs.values()))

    def plot_history(self, save_fig_dir, **kwargs):
        plt.figure()
        plt.title("{0} training history".format(self.name))
        for name, history_lst in self.__dict__.items():
            if "history" in name:
                plt.plot(history_lst, label=name)
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        if save_fig_dir:
            plt.savefig(save_fig_dir)
        plt.close()

