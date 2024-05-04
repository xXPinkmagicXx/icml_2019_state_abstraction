import keras
import sys
from keras.models import model_from_json
import numpy as np
from simple_rl.tasks import GymMDP
from policies.PolicySB import PolicySB
from stable_baselines3 import DQN 

class CartPolePolicySB(PolicySB):
	
	def __init__(self, gym_env: GymMDP, algo: str, policy_train_episodes: int, experiment_episodes: int, seed: int):
		
		super().__init__(
			gym_env=gym_env,
			algo=algo,
			policy_train_episodes=policy_train_episodes,
			experiment_episodes=experiment_episodes,
			seed=seed)
	
	def get_params(self):
		
		params={}
		params['env_name']="CartPole-v1"
		params['obs_size'] = 4
		params['num_iterations_for_abstraction_learning'] = 300
		params['learning_rate_for_abstraction_learning'] = 0.001
		params['abstraction_network_hidden_layers'] = 2
		params['abstraction_network_hidden_nodes'] = 40
		params['num_samples_from_demonstrator'] = 5000
		params['rl_learning_rate'] = 0.001
    
		return params