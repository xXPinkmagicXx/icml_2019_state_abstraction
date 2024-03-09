# icml_2019_state_abstraction
Code for the 2019 ICML submission, "Learning State Abstractions for Transfer in Continuous Control".

To run experiments:

	> python run_learning_experiments_(domain).py

Where "(domain)" can be each of {puddle, lunar, cartpole}. To run the transfer experiments, open the file and set params['multitask'] to True.

To reproduce figure 3b, uncomment the call to _num_training_data_experiment()_ in _run_learning_experiments_puddle.py_ and run it.

## To run the environment

1. First install tensorflow version 1.15.0 with conda
```
conda install tensorflow=1.15.0
```
2. Then install the `requirements.txt`
```
pip install -r requirements.txt
```
