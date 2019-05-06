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
        :param args.device: device, the device used: cpu or gpu
        :param args.learning_rate: float, sets the learning rate
        :param args.epochs: int, sets the epoch size
        :param args.vocabulary: string, all the characters in the context of the problem
        :param args.seq_length: int, maximum seq length of the DNA sequence
        :param args.batch_size: int, batch size of the model
        :param args.learning_rate: float, initial learning rate
        :param args.pseudo_count: int, the pseudocount to add to each character
        """
        self.args = args
        self.save_epochs = 50

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

    def plot_history(self, save_fig_dir):
        """
        plot the training and validation history of the model. should be called after fit.
        :param save_fig_dir: directory to save the training and validation history in
        :return: None
        """
        raise NotImplementedError
