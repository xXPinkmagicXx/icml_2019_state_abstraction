import keras
import sys
from keras.models import model_from_json
import numpy as np
from simple_rl.tasks import GymMDP
from policies.PolicySB import PolicySB
from stable_baselines3 import DQN
import os
class LunarLanderPolicySB(PolicySB):
	
	"""
	This class loads the pre-trained model for LunarLander-v2 environment.
	"""

	def __init__(self, gym_env: GymMDP, algo: str = "dqn"):
		super().__init__(gym_env, algo)
		
	def get_params(self):
		params={}
		params['multitask'] = False
		params['env_name'] = "LunarLander-v2"
		params['obs_size'] = 8
		params['num_iterations_for_abstraction_learning'] = 500
		params['learning_rate_for_abstraction_learning'] = 0.005
		params['abstraction_network_hidden_layers'] = 2
		params['abstraction_network_hidden_nodes'] = 200
		params['num_samples_from_demonstrator'] = 10000
		params['episodes'] = 200
		params['steps']=1000
		params['num_instances']=5
		params['rl_learning_rate']=0.005

		return params
	