import torch
import torch.utils.data
import numpy as np
import time
import matplotlib.pyplot as plt
import os

from torch import nn, optim
from torch.nn import functional as F
from torchviz import make_dot
from utils import to_tensor, sample_tensor_to_string
from models import Model


class VAE(nn.Module):

    def __init__(self, input_size, hidden_size, latent_dim, num_characters, seq_length):
        super(VAE, self).__init__()
        self.num_characters = num_characters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_dim)
        self.fc22 = nn.Linear(hidden_size, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        # input should be one hot encoded. shape - (batch_size, alphabet x sequence_length)
        h1 = F.elu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        # mu and logvar should be of shape - (batch_size, hidden_size)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, softmax=False):
        # z should be of shape - (batch_size, hidden_size)
        batch_size = z.shape[0]
        h3 = F.elu(self.fc3(z))
        if softmax:
            return F.softmax(self.fc4(h3).view(batch_size, -1, self.num_characters), dim=2)
        else:
            return self.fc4(h3).view(batch_size, -1, self.num_characters)

    def forward(self, x):
        """
        :param x: one hot encoded vector with dimension (batch_size x (seq_length * num_characters))
        :return: recon_x: one hot encoded vector with dimension (batch_size x seq_length x num_characters)
                mu: hidden state mean with dimension (batch_size x hidden_size)
                logvar: hidden state log variance with dimension (batch_size x hidden_size)
        """
        mu, logvar = self.encode(x.view(-1, self.input_size))
        # sampled hidden state
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, softmax=False)
        return recon_x, mu, logvar


class GenerativeVAE(Model):

    def __init__(self, args):
        Model.__init__(self, args)
        self.model_type = args["model_type"]
        self.name = args["name"]
        self.input = args["input"]
        self.hidden_size = args["hidden_size"]
        self.latent_dim = args["latent_dim"]
        self.device = args["device"]
        self.learning_rate = args["learning_rate"]
        self.epochs = args["epochs"]
        self.all_characters = args["vocabulary"]
        self.seq_length = args["seq_length"]
        self.batch_size = args["batch_size"]
        self.learning_rate = args["learning_rate"]
        self.num_characters = len(self.all_characters)
        self.character_to_int = dict(zip(self.all_characters, range(self.num_characters)))
        self.int_to_character = dict(zip(range(self.num_characters), self.all_characters))
        self.indexes = list(range(self.num_characters))
        self.model = VAE(self.input, self.hidden_size, self.latent_dim, self.num_characters, self.seq_length)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_loss_history, self.train_recon_loss_history, self.train_kld_loss_history = [], [], []
        self.valid_loss_history, self.valid_recon_loss_history, self.valid_kld_loss_history = [], [], []
        assert(self.seq_length * self.num_characters == self.input)

    def elbo_loss(self, recon_x, x, mu, logvar):
        """
        :param recon_x: one hot encoded vector that is the output of the vae (batch_size x seq_length x num_characters)
        :param x: one hot encoded vector that is the expected x value (batch_size x seq_length x num_characters)
        :param mu: hidden state mean with dimension (batch_size x hidden_size)
        :param logvar: hidden state log variance with dimension (batch_size x hidden_size)
        :return: kld_loss + negative log likelihood loss
        """
        return self.cross_entropy_loss(recon_x, x) + self.kld_loss(mu, logvar)

    def cross_entropy_loss(self, recon_x, x):
        loss = nn.CrossEntropyLoss(reduction='sum')
        inp = recon_x.permute(0, 2, 1)  # reshape to format in CrossEntropy Form (batch_size x num_characters x seq_length)
        _, target = x.view(x.shape[0], -1, self.num_characters).max(dim=2)  # get in CrossEntropy Form (batch_size x seq_length)
        target = target.long()
        return loss(inp, target)

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model_dir=None, weights=None, **kwargs):
        start_time = time.time()
        self.train_loss_history, self.train_recon_loss_history, self.train_kld_loss_history = [], [], []
        self.valid_loss_history, self.valid_recon_loss_history, self.valid_kld_loss_history = [], [], []
        for epoch in range(1, self.epochs + 1):
            # train model
            self.model.train()
            total_train_loss, total_recon_loss, total_kld_loss = 0, 0, 0
            for batch_idx, (x, _) in enumerate(train_dataloader):
                x = x.to(self.device)
                self.optimizer.zero_grad()
                recon_x, mu, logvar = self.model(x)
                rloss, kloss = self.cross_entropy_loss(recon_x, x), self.kld_loss(mu, logvar)
                loss = (rloss + kloss) / (x.shape[0])
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item() * x.shape[0]
                total_recon_loss += rloss.item()
                total_kld_loss += kloss.item()
            self.train_loss_history.append(total_train_loss / len(train_dataloader.dataset)) # len(train_dataloader.dataset is size of data)
            self.train_recon_loss_history.append(total_recon_loss / len(train_dataloader.dataset))
            self.train_kld_loss_history.append(total_kld_loss / len(train_dataloader.dataset))
            # evaluate model
            self.model.eval()
            if valid_dataloader:
                valid_loss, valid_recon_loss, valid_kld_loss = self.evaluate(valid_dataloader, verbose=False, logger=logger)
                self.valid_loss_history.append(valid_loss)
                self.valid_recon_loss_history.append(valid_recon_loss)
                self.valid_kld_loss_history.append(valid_kld_loss)
            if verbose:
                print("-" * 50, file=logger)
                print('epoch: {0}. train loss: {1:.4f}. train cross entropy loss: {2:.4f}. train kld loss: {3:.4f}'.format(
                        epoch, self.train_loss_history[-1], self.train_recon_loss_history[-1], self.train_kld_loss_history[-1]), file=logger)
                print('time: {0:.2f} sec. valid loss: {1:.4f}. valid cross entropy loss: {2:.4f}, valid kld loss {3:.4f}'.format(
                        time.time() - start_time, self.valid_loss_history[-1], self.valid_recon_loss_history[-1], self.valid_kld_loss_history[-1]), file=logger)
                print("-" * 50, file=logger)
            if epoch % self.save_epochs == 0 and save_model_dir:
                self.save_model(os.path.join(save_model_dir, "checkpoint_{0}.pt".format(epoch)), epoch=epoch, loss=loss)

    def evaluate(self, dataloader, verbose=True, logger=None, weights=None, **kwargs):
        self.model.eval()
        total_loss, total_recon_loss, total_kld_loss = 0, 0, 0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(self.device)
            recon_x, mu, logvar = self.model(x)
            rloss, kloss = self.cross_entropy_loss(recon_x, x), self.kld_loss(mu, logvar)
            total_recon_loss += rloss
            total_kld_loss += kloss
            total_loss += rloss + kloss
        total_loss /= len(dataloader.dataset)
        total_recon_loss /= len(dataloader.dataset)
        total_kld_loss /= len(dataloader.dataset)
        if verbose:
            print('total loss: {0:.4f} cross entropy loss: {1:.4f}. kld loss: {2:.4f}'.format(
                total_loss, total_recon_loss, total_kld_loss), file=logger)
        return total_loss.item(), total_recon_loss.item(), total_kld_loss.item()

    def decoder(self, z, softmax=False):
        assert (z.shape[1] == self.latent_dim)
        if type(z) != torch.Tensor:
            z = to_tensor(z, self.device)
        return self.model.decode(z, softmax=softmax)

    def encoder(self, x, reparameterize=False):
        assert (x.shape[1] == self.input)
        if type(x) != torch.Tensor:
            x = to_tensor(x, self.device)
        mu, log_var = self.model.encode(x)
        if reparameterize:
            return self.model.reparameterize(mu, log_var), mu, log_var
        else:
            return mu, log_var

    def sample(self, num_samples, length, to_string=True, **kwargs):
        assert(length <= self.input / self.num_characters)
        if "z" in kwargs:
            z = kwargs["z"]
        else:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
        sampled_probabilities = self.decoder(z, softmax=True)
        sampled_probabilities = sampled_probabilities[:, :length, :]
        if to_string:
            return [sample_tensor_to_string(prob, self.int_to_character) for prob in sampled_probabilities]
        else:
            return sampled_probabilities.detach().numpy()

    def show_model(self, logger=None):
        print(self.model, file=logger)

    def plot_model(self, save_fig_dir, show=False):
        x = np.random.randn(self.batch_size, self.seq_length, self.num_characters)
        x = to_tensor(x, self.device)
        out, _, _ = self.model(x)
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
        if 'epoch' in kwargs:
            d['epoch'] = kwargs['epoch']
        if 'loss' in kwargs:
            d['loss'] = kwargs
        torch.save(d, path)

    def load_model(self, path, **kwargs):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

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

