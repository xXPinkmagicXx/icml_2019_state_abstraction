# Bachelor Project 
This repo is a fork of https://github.com/anonicml2019/icml_2019_state_abstraction

The work done in this repo will be used as a submodule in this repo https://github.com/borchand/Bachelor-Project ( currently private )

## Train a agent
From the bachelor-project folder

```
python run_icml.py -a algo -e gym_name 
```

## train an agent from stable baselines

```
python baselines.py -a ppo -e CartPole-v1 -t 100000
```
this trains a agent with ppo algorithm in the Cartpole-v1 environment

The `baselines.py` takes the following parameters
- -a --algo: str - The algorithm to train
- -e --env: str - The gym environment to run
- -t --time-steps: int (deafult=100000) - The number of steps to train the algorithm
- -tr --train: bool (default=True) A bool that determines whether to train or not
- -r  --render: bool (default=Fale), if true renders one episode of the algorithm in the environment
- -s --save: bool (default=True), if true saves the trained model
- -sh --show: bool (default=False), if true renders one episode of the algorithm in the enviroment, and does not train

## To run the environment
What is expected to run the environment
- python=3.10
- PyTorch
- TensorFlow=2.9.1
- Stable-Baselines3
- rl_zoo3

```
1. Then install the `requirements.txt`
```
pip install -r requirements.txt
```
## Train policy and run learning experiment (run.py)
```
python run.py "MountainCar-v0"
```
Where "MountainCar-v0" can be switched out with other implemented gym_envs
Currently implemented gym environments {"MountainCar-v0", "CartPole-v0"}

## Directory Structure
```
|-- experiments/
|   |-- abstraction/
|   |   |-- abstraction_network.py
|   |   |-- NNStateAbstrClass.py
|   |-- Lunar_dqn/
|   |-- policies/
|   |-- results/
|   |   | -- gym_env_name/
|   |-- simple_rl/
|   |-- utils/
|   |-- visuals/
|   |-- run_learning_experiment.py
|-- lunar_variants/
|-- mac/
|   |-- learned_policy/
|   |-- actor_network.py
|   |-- critic_network.py
|   |-- mac.py
|   |-- run.py
|   |-- utils.py
|-- run.py
|-- README.md
|-- requirements.txt
```
- experiments/run_learning_experiment.py: main file for running experiments
- experiments/abstraction : Contains the abstraction implementation
- experiments/policies : The implemented task specific policy classes  
- experiments/results : Results saved as `gym_env_name/` e.g gym-CartPole-v0/ containing csv, txt, and plot
- experiments/simple_rl : this is the simple_rl repo https://github.com/david-abel/simple_rl
- mac/ : This contains the mean actor critic implementation
- requirements.txt : the requirements for running the code
- run.py : this file trains a policy and run learning experiment 


# README for icml_2019_state_abstraction
Code for the 2019 ICML submission, "Learning State Abstractions for Transfer in Continuous Control".

To run experiments:

	> python run_learning_experiments_(domain).py

Where "(domain)" can be each of {puddle, lunar, cartpole}. To run the transfer experiments, open the file and set params['multitask'] to True.

To reproduce figure 3b, uncomment the call to _num_training_data_experiment()_ in _run_learning_experiments_puddle.py_ and run it.
