import keras
import sys
from keras.models import model_from_json
import numpy as np
from simple_rl.tasks import GymMDP
from policies.PolicySB import PolicySB
import os

class PendulumPolicySB(PolicySB):
	
	def __init__(self, gym_env: GymMDP, algo: str = "ppo", policy_train_steps=100_000):
		
		super().__init__(gym_env, algo, policy_train_steps)
		
	def get_params(self):
		params={}
		params['multitask'] = False
		params['env_name'] = "Pendulum-v0"
		params['obs_size'] = 3
		params['num_iterations_for_abstraction_learning'] = 500
		params['learning_rate_for_abstraction_learning'] = 0.005
		params['abstraction_network_hidden_layers'] = 2
		params['abstraction_network_hidden_nodes'] = 200
		params['num_samples_from_demonstrator'] = 10000
		params['episodes'] = 200
		params['steps']=200 # is truncated at 200 in gym environment
		params['num_instances']=5
		params['rl_learning_rate']=0.005

		return params