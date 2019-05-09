import matplotlib.pyplot as plt

class Model(object):

    def __init__(self, args):
        """
        initializes the model with all its hyper-parameters
        :param args: dictionary, defines the hyper-parameters of the model
        :param args.model_type: string, defines the type of the model
        :param args.name: string, defines the name of the model
        :param args.input: int, the size of the input
        :param args.hidden_size: int, the size of the hidden layer
        :param args.latent_dim: int, the size of the latent dimension
        :param args.seq_length: int, the size of the sequence length
        :param args.pseudo_count: int, the pseudo count to be added to each output distribution
        :param args.n_jobs: int, the number of jobs to parallelize the computation
        :param args.device: device, the device used: cpu or gpu
        :param args.learning_rate: float, sets the learning rate
        :param args.epochs: int, sets the epoch size
        :param args.batch_size: int, batch size of the model
        :param args.layers: int, the number of layers in the model
        :param args.dataset: string, the dataset used
        :param args.num_data: int, the number of data points in the train, test, and valid datasets
        :param args.vocabulary: string, all the characters in the context of the problem
        """
        self.args = args
        self.save_epochs = 50
        self.model_type = args["model_type"]
        self.base_log = args["base_log"]
        self.name = args["name"]
        self.input = args["input"]
        self.hidden_size = args["hidden_size"]
        self.latent_dim = args["latent_dim"]
        self.seq_length = args["seq_length"]
        self.pseudo_count = args["pseudo_count"]
        self.n_jobs = args["n_jobs"]
        self.device = args["device"]
        self.learning_rate = args["learning_rate"]
        self.epochs = args["epochs"]
        self.batch_size = args["batch_size"]
        self.layers = args["layers"]
        self.dataset = args["dataset"]
        self.num_data = args["num_data"]
        self.all_characters = args["vocabulary"]
        self.num_characters = len(self.all_characters)
        self.character_to_int = dict(zip(self.all_characters, range(self.num_characters)))
        self.int_to_character = dict(zip(range(self.num_characters), self.all_characters))
        self.indexes = list(range(self.num_characters))
        self.train_loss_history = []
        self.valid_loss_history = []
        assert(self.seq_length * self.num_characters == self.input)

    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model=True, weights=None):
        """
        fits the model on a training set, using the validation set for hyper-parameter tuning
        :param train_dataloader: a dataloader that iterates through the training dataset in batches
        :param valid_dataloader: a dataloader that iterates through the validation dataset in batches
        :param verbose: whether or not to print output
        :param logger: the file to which the output will be printed to
        :param save_model: whether or not to save the model every save_epochs iterations.
        All models will be saved as checkpoint_{epoch_number}
        :param weights: weights of the training data
        """
        raise NotImplementedError
        
    def evaluate(self, dataloader, verbose=False, logger=None, save_model_dir=None, weights=None):
        """
        evaluates the model on the dataloader dataset
        :param dataloader: a dataloader in which the mean loss per data point will be evaluated from
        :param verbose: whether or not to print output
        :param logger: the file to which the output will be printed to
        :param save_model_dir: the directory to save the model to
        :param weights: weights of the data
        :return: mean loss per data point
        """
        raise NotImplementedError

    def sample(self, num_samples, length, to_string=True):
        """
        sample a number of strings from the generative model
        :param num_samples: number of samples to be taken from generative model
        :param length: the length of the string to be returned
        :param to_string: return the strings in one_hot format or string format
        :return: list of sampled strings from the generative model
        """
        raise NotImplementedError

    def show_model(self):
        """
        prints the model architecture
        :return: None
        """
        raise NotImplementedError

    def save_model(self, path, **kwargs):
        """
        saves the model weights and optimizer in a specific path
        :param path: path to save the model in
        :param kwargs: additional arguments
        :return: None
        """
        raise NotImplementedError

    def load_model(self, path):
        """
        loads the optimizer and weights for the model from a specific path
        :param path: path to load the model from
        :return: None
        """
        raise NotImplementedError

    def plot_model(self, save_fig_dir):
        """
        plot the model in a path
        :param save_fig_dir: directory to save the architecture of the model
        :return:
        """
        raise NotImplementedError

    def plot_history(self, save_fig_dir, **kwargs):
        """
        plot the training and validation history of the model. should be called after fit.
        :param save_fig_dir: directory to save the training and validation history in
        :return: None
        """
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
