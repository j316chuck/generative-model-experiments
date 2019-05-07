import json
from pomegranate import DiscreteDistribution, HiddenMarkovModel
from utils import *
from models import Model
import os


class GenerativeHMM(Model):

    def __init__(self, args):
        Model.__init__(self, args)
        self.model_type = args["model_type"]
        self.name = args["name"]
        self.hidden_size = args["hidden_size"]
        self.epochs = args["epochs"]
        self.n_jobs = args["n_jobs"]
        self.batch_size = args["batch_size"]
        self.pseudo_count = args["pseudo_count"]
        self.all_characters = args["vocabulary"]
        self.num_characters = len(self.all_characters)
        self.character_to_int = dict(zip(self.all_characters, range(self.num_characters)))
        self.int_to_character = dict(zip(range(self.num_characters), self.all_characters))
        self.indexes = list(range(self.num_characters))
        self.model = None
        self.build_model()
        self.train_neg_log_prob = []
        self.valid_neg_log_prob = []
    
    def build_model(self): 
        distributions = []
        for _ in range(self.hidden_size):
            emission_probs = np.random.random(self.num_characters)
            emission_probs = emission_probs / emission_probs.sum()
            distributions.append(DiscreteDistribution(dict(zip(self.all_characters, emission_probs))))
        trans_mat = np.random.random((self.hidden_size, self.hidden_size))
        trans_mat = trans_mat / trans_mat.sum(axis=1, keepdims=1)
        starts = np.random.random(self.hidden_size)
        starts = starts / starts.sum()
        # testing initializations
        np.testing.assert_almost_equal(starts.sum(), 1)
        np.testing.assert_array_almost_equal(np.ones(self.hidden_size), trans_mat.sum(axis=1))
        self.model = HiddenMarkovModel.from_matrix(trans_mat, distributions, starts)
        self.model.bake()

    def fit(self, train_dataloader, valid_dataloader=None, verbose=True, logger=None, save_model=True, weights=None, **kwargs):
        start_time = time.time()
        for epoch in range(1, self.epochs + 1):
            _, hist = self.model.fit(train_dataloader, max_iterations=1, pseudocount=self.pseudo_count,
                                     n_jobs=self.n_jobs, return_history=True)
            train_neg_log_prob = self.evaluate(train_dataloader) / len(train_dataloader)
            self.train_neg_log_prob.append(train_neg_log_prob)
            if valid_dataloader:
                test_neg_log_prob = self.evaluate(valid_dataloader) / len(valid_dataloader)
                self.valid_neg_log_prob.append(test_neg_log_prob)
            if verbose:
                print("epoch {0}, train neg log prob: {1:.4f}, test neg log probability {2:.4f}, time: {3:.2f} sec".format(
                    epoch, train_neg_log_prob, test_neg_log_prob, time.time() - start_time), file=logger)
            if epoch % self.save_epochs == 0 and save_model:
                path = os.path.join(self.base_log, self.name, "{0}_checkpoint_{1}.json".format(self.model_type, epoch))
                print(path)
                self.save_model(path)

    def evaluate(self, dataloader, verbose=False, logger=None, weights=None, **kwargs):
        assert(len(np.array(dataloader).shape) == 2 or len(np.array(dataloader).shape) == 3)
        neg_log_prob = -sum([self.model.log_probability(seq) for seq in np.array(dataloader)])
        if verbose:
            print("Average neg log prob: {0:.4f}".format(neg_log_prob / len(dataloader)), file=logger)
        if "pos_log_prob" in kwargs:
            return -neg_log_prob
        else:
            return neg_log_prob

    def sample(self, num_samples, length, to_string=True, **kwargs):
        return ["".join(x) for x in self.model.sample(n=num_samples, length=length)]

    def show_model(self, logger=None, **kwargs):
        print(self.model, logger)

    def plot_model(self, save_fig_dir, show=False, **kwargs):
        # self.model.plot() does not plot legible graphs for hidden size > 10
        pass
        
    def save_model(self, path, **kwargs):
        with open(path, 'w') as f:
            json.dump(self.model.to_json(), f)
    
    def load_model(self, path, **kwargs):
        with open(path, 'r') as f:
            json_model = json.load(f)
        self.model = HiddenMarkovModel.from_json(json_model)

    def plot_history(self, save_fig_dir, **kwargs):
        plt.figure()
        plt.title("{0} training history".format(self.name))
        for name, history_lst in self.__dict__.items():
            if "prob" in name:
                plt.plot(history_lst, label=name)
        plt.legend()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        if save_fig_dir:
            plt.savefig(save_fig_dir)
        plt.close()



