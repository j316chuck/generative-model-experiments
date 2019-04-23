import torch
import torch.utils.data
import numpy as np
import time
import matplotlib.pyplot as plt

from torch import nn, optim
from torch.nn import functional as F
from torchviz import make_dot
from utils import one_hot_encode, to_tensor
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
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, softmax=False):
        batch_size = z.shape[0]
        h3 = F.elu(self.fc3(z))
        if softmax:
            return F.softmax(self.fc4(h3).view(batch_size, -1, self.num_characters), dim=2)
        else:
            return self.fc4(h3).view(batch_size, -1, self.num_characters)

    def forward(self, x):
        """
        Input: x is the one hot encoded batch_size x (seq_length * num_characters)
        Output: recon_x: is the one hot encoded batch_size x seq_length x num_characters vector
                mu: is the hidden state mean with dimension batch_size x hidden_size
                logvar: is the hidden state log variance with dimension batch_size x hidden_size
        """
        mu, logvar = self.encode(x.view(-1, self.input_size))
        z = self.reparameterize(mu, logvar)
        return self.decode(z, softmax=False), mu, logvar

    def elbo_loss(self, recon_x, x, mu, logvar):
        """
        Input: x is the one hot encoded batch_size x (seq_length * num_characters)
               recon_x is the unormalized outputs of the decoder in the same shape as x
               mu and logvar are the hidden states of size self.hidden_size
        Output: elbo_loss
        """
        outputs = F.log_softmax(recon_x, dim=2)
        CE = (-1 * outputs * x.view(x.shape[0], -1, self.num_characters)).sum()
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return CE + KLD

    def cross_entropy_loss(self, recon_x, x):
        loss = nn.CrossEntropyLoss(reduction='sum')
        input = recon_x.permute(0, 2, 1)
        _, target = x.view(x.shape[0], -1, self.num_characters).max(dim=2)
        target = target.long()
        return loss(input, target)

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


class GenerativeVAE(Model):

    def __init__(self, args):
        """
        Initializes the VAE to be a generative VAE
        Parameters
        ----------
        args : dictionary
            defines the hyper-parameters of the neural network
        args.model_type : string
            defines the type of the neural network
        args.name : string
            defines the name of the neural network
        args.input : int
            the size of the input
        args.hidden_size : int
            the size of the hidden layer
        args.latent_dim: int
            the size of the latent dimension
        args.device : device
            the device used: cpu or gpu
        args.learning_rate : float
            sets the learning rate
        args.epochs : int
            sets the epoch size
        args.vocabulary : string
            all the characters in the context of the problem
        args.seq_length : int
            maximum seq length of the DNA sequence
        args.batch_size : int
            batch size of the model
        args.learning_rate : float
            initial learning rate
        """
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
        self.model = VAE(self.input, self.hidden_size, self.latent_dim, self.num_characters, self.seq_length)
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.train_loss_history = []
        self.valid_loss_history = []
        assert(self.seq_length * self.num_characters == self.input)

    def elbo_loss(self, recon_x, x, mu, logvar):
        """
        Input: x is the one hot encoded batch_size x (seq_length * num_characters)
               recon_x is the unormalized outputs of the decoder in the same shape as x
               mu and logvar are the hidden states of size self.hidden_size
        Output: elbo_loss
        """
        return self.cross_entropy_loss(recon_x, x) + self.kld_loss(mu, logvar)

    def cross_entropy_loss(self, recon_x, x):
        loss = nn.CrossEntropyLoss(reduction='sum')
        input = recon_x.permute(0, 2, 1)
        _, target = x.view(x.shape[0], -1, self.num_characters).max(dim=2)
        target = target.long()
        return loss(input, target)

    def kld_loss(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model=True, weights=None, **kwargs):
        start_time = time.time()
        self.train_loss_history, self.valid_loss_history = [], []
        self.valid_recon_loss_history, self.valid_kld_loss_history = [], []
        self.train_recon_loss_history, self.train_kld_loss_history = [], []
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
            self.train_loss_history.append(total_train_loss / len(train_dataloader.dataset))
            self.train_recon_loss_history.append(total_recon_loss / len(train_dataloader.dataset))
            self.train_kld_loss_history.append(total_kld_loss / len(train_dataloader.dataset))

            # evaluate model
            self.model.eval()
            if valid_dataloader:
                valid_loss, valid_recon_loss, valid_kld_loss = self.evaluate(valid_dataloader, verbose=False,
                                                                             logger=logger)
                self.valid_loss_history.append(valid_loss)
                self.valid_recon_loss_history.append(valid_recon_loss)
                self.valid_kld_loss_history.append(valid_kld_loss)
            if verbose:
                print("-" * 50, file=logger)
                print(
                    'epoch: {0}. train loss: {1:.4f}. train cross entropy loss: {2:.4f}. train kld loss: {3:.4f}'.format(
                        epoch, self.train_loss_history[-1], self.train_recon_loss_history[-1],
                        self.train_kld_loss_history[-1]), file=logger)
                print(
                    'time: {0:.2f}. valid loss: {1:.4f}. valid cross entropy loss: {2:.4f}, valid kld loss {3:.4f}'.format(
                        time.time() - start_time, self.valid_loss_history[-1], self.valid_recon_loss_history[-1],
                        self.valid_kld_loss_history[-1]), file=logger)
                print("-" * 50, file=logger)
            if epoch % 50 == 0:
                self.save_model(epoch, self.train_loss_history[-1])

    def predict_elbo_prob(self, sequences, string=True):
        """
        Input: list of sequences in string or one_hot_encoded form
        Output: list of the elbo probability for each sequence
        Example: predict_elbo_prob(["ACT", "ACG"]) = [0.2, 0.75]
        predict_elbo_prob([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]]) = [0.2, 0.75]
        note: alphabet in this example is ACTG and the wild type is probably ACG***
        """
        if string:
            sequences = one_hot_encode(sequences, self.all_characters)
        if type(sequences) != torch.Tensor:
            x = to_tensor(sequences, device=self.device)
        recon_x, mu, logvar = self.model(x)
        return self.elbo_loss(recon_x, x, mu, logvar)

    def evaluate(self, dataloader, verbose=True, logger=None):
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
        """ note that the outputs are unnormalized """
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

    def sample(self, num_samples=1, z=None, softmax=True):
        if z is None:
            z = torch.randn(num_samples, self.latent_dim).to(self.device)
        return self.decoder(z, softmax=softmax), z

    def load_model(self, model_path):
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def save_model(self, epoch=None, loss=None):
        torch.save({
            'epoch': epoch,
            'loss': loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, "./models/{0}/checkpoint_{1}.pt".format(self.name, epoch))

    def show_model(self, logger=None):
        print(self.model, file=logger)

    def plot_model(self, save_dir, verbose=False):
        x = np.random.randn(self.batch_size, self.seq_length, self.num_characters)
        x = to_tensor(x, self.device)
        out, _, _ = self.model(x)
        graph = make_dot(out)
        if save_dir is not None:
            graph.format = "png"
            graph.render(save_dir)
        if verbose:
            graph.view()

    def plot_history(self, save_fig_dir):
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
