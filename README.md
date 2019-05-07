## generative model experiments 

**generate_parameters.py**: produces the set of hyperparameters and models we want to grid search over. <br/> 
**run_experiments.py**: is the main loop which loads the datasets and models -> fits and trains the model according to the train and validation sets -> evaluates the model according to the test set -> generates 1000 samples and counts the mismatches from the wild type <br/>
**optimizer.py**: searches over the set of hyperparameters generated from generate_parameters.py and calls run_experiments.py to fit and valuate each model using a train, valid, and test set. (this is also the part that can be parallelized). It then sorts each model by their test set score. <br/> 
**models.py**: all models inherit from this class and contain an init, fit, evaluate, sample, show_model, plot_model, save, and load function. all models contain an internal object called self.model which is the actual underlying model. <br/> 
**utils.py**: helper functions <br/>
**vae.py**: the vae implementation <br/>
**rnn.py**: the rnn implementation <br/>
**hmm.py**: the hmm implementation <br/> 
