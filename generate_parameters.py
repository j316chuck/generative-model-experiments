import itertools
import pandas as pd

learning_rate_lst = [0.003, 0.001]
hidden_size_lst = [50, 200]
batch_size_lst = [10, 20]
columns = ["learning_rate", "hidden_size", "batch_size"]
combinations = pd.DataFrame(columns=columns)
for parameter in itertools.product(learning_rate_lst, hidden_size_lst, batch_size_lst):
    combinations = combinations.append(pd.Series(parameter, index=combinations.columns), ignore_index=True)
parameter_path = "./models/parameters/vae_small_optimization.csv"
combinations.to_csv(parameter_path, index=False)
parameters = pd.read_csv(parameter_path)
print(parameters)

