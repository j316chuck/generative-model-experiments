import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
       from https://github.com/Bjarten/early-stopping-pytorch
    """

    def __init__(self, patience=7, verbose=False):
        """
        :param patience: (int) How long to wait after last time validation loss improved.
                            Default: 7
        :param verbose: (bool) If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.checkpoint = True
        elif score < self.best_score:
            self.counter += 1
            self.checkpoint = False
            if self.verbose:
                print(f'early stopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if self.verbose:
                print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.best_score = score
            self.checkpoint = True
            self.counter = 0

