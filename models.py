class Model:
    """ TODO add documentation here """
    def __init__(self, args):
        self.args = args
        self.save_epochs = 50

    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model=True, weights=None, **kwargs):
        pass

    def evaluate(self, dataloader, verbose=False, logger=None, weights=None, **kwargs):
        pass

    def sample(self, num_samples, length, **kwargs):
        pass

    def show_model(self, logger=None, **kwargs):
        pass

    def plot_model(self, save_fig_dir, show=False, **kwargs):
        pass

    def save_model(self, path, **kwargs):
        pass

    def load_model(self, path, **kwargs):
        pass

    def plot_history(self, save_fig_dir, **kwargs):
        pass

    def print_vars(self):
        print(self.__dict__)
