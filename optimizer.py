import pandas as pd
import os
import numpy as np
from run_model import run_experiment


def optimize():
    """
    run many different sets of hyperparameters and datasets and models and report the test scores and sample loss scores
    :return: dataframe with the model name, loss, and sample loss scores
    """
    # manually tune these parameters every time.
    args = {
        "model_type": "vae",
        "base_log": "./logs/vae_small_optimization/",
        "name": "vae",
        "input": 4998,
        "hidden_size": -1,
        "latent_dim": 20,
        "seq_length": 238,
        "pseudo_count": 1,
        "n_jobs": 5,
        "learning_rate": -1,
        "epochs": 50,
        "batch_size": -1,
        "layers": 2,
        "dataset": "gfp_amino_acid",
        "num_data": 1000
    }
    base_name = args["name"]
    device = "cpu"
    # creating the paths
    if not os.path.exists(args["base_log"]):
        os.makedirs(args["base_log"])
    parameters = pd.read_csv("./models/parameters/vae_small_optimization.csv")
    types = {"batch_size": np.int32, "hidden_size": np.int32, "learning_rate": float}
    parameters = parameters.astype(types)

    model_results = []
    hyperparameter_names = parameters.columns
    for index in range(parameters.shape[0]):
        name = base_name
        for hyperparameter in hyperparameter_names:
            args[hyperparameter] = parameters.loc[index, hyperparameter]
            name += "_{0}_{1}".format(hyperparameter, args[hyperparameter])
        args["device"] = device
        args["name"] = name
        mismatches, test_score, model = run_experiment(args)
        if test_score is not float:
            test_score = test_score[0]
        model_results.append([name, test_score, mismatches])
    optimize_results_df = pd.DataFrame(np.array(model_results), columns=["name", "test_score", "mismatches"])
    optimize_results_df.sort_values(ascending=False, inplace=True, by="test_score")
    print(optimize_results_df)
    optimize_results_df.to_csv(os.path.join(args["base_log"], "model_results.csv"), index=False)
    return optimize_results_df


if __name__ == '__main__':
    optimize()
