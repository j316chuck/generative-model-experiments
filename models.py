class Model:

    def __init__(self, args):
        self.args = args

    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model=True, weights=None):
        pass

    def predict(self, dataloader, verbose=False, logger=None, save_model=True, weights=None):
        pass

    def evaluate(self, dataloader, verbose=False, logger=None, save_model=True, weights=None):
        pass

    def sample(self, num_samples=1, length=10):
        pass

    def show_model(self):
        pass

    def save_model(self, path, **kwargs):
        pass

    def load_model(self, path):
        pass

    def plot_model(self, save_fig_dir):
        pass

    def plot_history(self, save_fig_dir):
        pass
