class Model(object):

    def __init__(self, args):
        self.args = args
        self.save_epochs = 50

    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model=True, weights=None):
        raise NotImplementedError
        
    def evaluate(self, dataloader, verbose=False, logger=None, save_model_dir=None, weights=None):
        raise NotImplementedError

    def sample(self, num_samples, length, to_string=True):
        raise NotImplementedError

    def show_model(self):
        raise NotImplementedError

    def save_model(self, path, **kwargs):
        raise NotImplementedError

    def load_model(self, path):
        raise NotImplementedError

    def plot_model(self, save_fig_dir):
        raise NotImplementedError

    def plot_history(self, save_fig_dir):
        raise NotImplementedError
