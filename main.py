import torch
import os
import argparse

from hmm import GenerativeHMM
from vae import GenerativeVAE
from rnn import GenerativeRNN
from torch.utils.data import TensorDataset, DataLoader
from utils import load_gfp_data, get_all_amino_acids, get_wild_type_amino_acid_sequence
from utils import one_hot_encode, plot_mismatches_histogram, string_to_tensor


def get_dataloader(args):
    """
    returns a dataloader that loads n sequences of length size broken into batch_size chunks.
    sequences may be randomly drawn from initial dataset and shuffled every epoch
    wild type added to args
    """
    x_train, x_test, y_train, y_test = None, None, None, None
    train_loader, valid_loader, test_loader = None, None, None
    if args["dataset"] == "gfp_amino_acid":
        x_train, x_test, y_train, y_test = load_gfp_data("./data/gfp_amino_acid_shuffle_")
        args["vocabulary"] = get_all_amino_acids()
        args["wild_type"] = get_wild_type_amino_acid_sequence()

    if args["model_type"] == "vae":
        one_hot_x_valid = one_hot_encode(x_train[args["num_data"]:2 * args["num_data"]], args["vocabulary"])
        one_hot_x_train = one_hot_encode(x_train[:args["num_data"]], args["vocabulary"])
        one_hot_x_test = one_hot_encode(x_test[:args["num_data"]], args["vocabulary"])
        y_valid = y_train[args["num_data"]:2 * args["num_data"]]
        y_train = y_train[:args["num_data"]]
        y_test = y_test[:args["num_data"]]
        train_dataset = TensorDataset(torch.from_numpy(one_hot_x_train).float(),
                                      torch.from_numpy(y_train.reshape(-1, 1)).float())
        valid_dataset = TensorDataset(torch.from_numpy(one_hot_x_valid).float(),
                                      torch.from_numpy(y_valid.reshape(-1, 1)).float())
        test_dataset = TensorDataset(torch.from_numpy(one_hot_x_test).float(),
                                     torch.from_numpy(y_test.reshape(-1, 1)).float())
        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)
    elif args["model_type"] == "hmm":
        train_loader = [list(x) for x in x_train[:args["num_data"]]]
        valid_loader = [list(x) for x in x_train[args["num_data"]:2 * args["num_data"]]]
        test_loader = [list(x) for x in x_test[:args["num_data"]]]
    elif args["model_type"] == "rnn":
        character_to_int = dict(zip(args["vocabulary"], list(range(len(args["vocabulary"])))))
        train_input = torch.stack([string_to_tensor(string[:-1], character_to_int) for string in x_train[:args["num_data"]]]).long()
        valid_input = torch.stack([string_to_tensor(string[:-1], character_to_int) for string in x_train[args["num_data"]:2*args["num_data"]]]).long()
        test_input = torch.stack([string_to_tensor(string[:-1], character_to_int) for string in x_test[:args["num_data"]]]).long()
        train_output = torch.stack([string_to_tensor(string[1:], character_to_int) for string in x_train[:args["num_data"]]]).long()
        valid_output = torch.stack([string_to_tensor(string[1:], character_to_int) for string in x_train[args["num_data"]:2*args["num_data"]]]).long()
        test_output = torch.stack([string_to_tensor(string[1:], character_to_int) for string in x_test[:args["num_data"]]]).long()
        train_dataset = TensorDataset(train_input, train_output)
        valid_dataset = TensorDataset(valid_input, valid_output)
        test_dataset = TensorDataset(test_input, test_output)
        train_loader = DataLoader(train_dataset, batch_size=args["batch_size"], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=args["batch_size"], shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args["batch_size"], shuffle=True)

    return train_loader, valid_loader, test_loader


def get_model(args):
    model = None
    if args["model_type"] == 'vae':
        model = GenerativeVAE(args)
    elif args["model_type"] == 'hmm':
        model = GenerativeHMM(args)
    elif args["model_type"] == 'rnn':
        model = GenerativeRNN(args)
    return model


def main(args):
    # loading data and model
    train_loader, valid_loader, test_loader = get_dataloader(args)
    model = get_model(args)
    assert(model is not None and train_loader is not None)
    logger = open("./logs/{0}/{1}.txt".format(args["model_type"], args["name"]), "w")
    # logger = None
    print("Training {0} \nArgs:".format(args["name"]), file=logger)
    for arg, value in model.__dict__.items():
        print(arg, "--", value, file=logger)

    # training and evaluating model
    # torch.manual_seed(1)  # set seed to get reproducible results
    print("*" * 50 + "\ntraining model on train and validation datasets...", file=logger)
    model.fit(train_dataloader=train_loader, valid_dataloader=valid_loader, verbose=True, logger=logger, save_model=True)
    model.plot_model("./logs/{0}/{1}_model_architecture".format(model.model_type, model.name))
    model.plot_history("./logs/{0}/{1}_training_history.png".format(model.model_type, model.name))
    print("*" * 50 + "\nevaluating model on test dataset:", file=logger)
    model.evaluate(dataloader=test_loader, verbose=True, logger=logger)
    if logger:
        logger.close()

    # sample from model and see if generated sequences are reasonable
    sampled_sequences = model.sample(num_samples=1000, length=len(args["wild_type"]))
    save_fig_dir = "./logs/{0}/{1}_mismatches.png".format(model.model_type, model.name)
    plot_mismatches_histogram(sampled_sequences, args["wild_type"], save_fig_dir=save_fig_dir, show=False)


if __name__ == '__main__':
    # parsing arguments
    parser = argparse.ArgumentParser(description='Process the arguments for the Model')
    parser.add_argument("-mt", "--model_type", default="vae", required=False, help="type of model to use", type=str)
    parser.add_argument("-n", "--name", default="vae_test_sample", required=False, help="name of model", type=str)
    parser.add_argument("-i", "--input", default=-1, required=False, help="size of input", type=int)
    parser.add_argument("-hi", "--hidden_size", default=-1, required=False, help="hidden size of model", type=int)
    parser.add_argument("-la", "--latent_dim", default=-1, required=False, help="latent dim of model", type=int)
    parser.add_argument("-se", "--seq_length", default=-1, required=False, help="seq length of the vocabulary", type=int)
    parser.add_argument("-ps", "--pseudo_count", default=1, required=False, help="pseudocounts to be added to outputs", type=int)
    parser.add_argument("-nj", "--n_jobs", default=1, required=False, help="num of jobs for parallelizing hmm", type=int)
    parser.add_argument("-de", "--device", default="cpu", required=False, help="device to use to train model", type=str)
    parser.add_argument("-lr", "--learning_rate", default=0.001, required=False, help="learning_rate", type=float)
    parser.add_argument("-e", "--epochs", default=10, required=False, help="number of epochs to train", type=int)
    parser.add_argument("-b", "--batch_size", default=10, required=False, help="batch_size", type=int)
    parser.add_argument("-l", "--layers", default=2, required=False, help="layer size", type=int)
    parser.add_argument("-da", "--dataset", default="gfp_amino_acid", required=False, help="which dataset to use", type=str)
    parser.add_argument("-d", "--num_data", default=100, required=False, help="number of data points to train on", type=int)
    args = vars(parser.parse_args())
    if args["device"] == "cpu":
        args["device"] = torch.device("cpu")
    else:
        args["device"] = torch.device("gpu")

    # creating paths for the models to be logged and saved
    if not os.path.exists("./logs/{0}".format(args["model_type"])):
        os.makedirs("./logs/{0}".format(args["model_type"]))
    if not os.path.exists("./models/{0}".format(args["name"])):
        os.makedirs('./models/{0}'.format(args["name"]))

    # main training and testing function
    main(args)
