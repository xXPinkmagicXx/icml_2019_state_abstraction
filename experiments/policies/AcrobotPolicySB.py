import keras
import sys
from keras.models import model_from_json
import numpy as np
from simple_rl.tasks import GymMDP
from policies.PolicySB import PolicySB

class AcrobotPolicySB(PolicySB):
	
	def __init__(self, gym_env: GymMDP, algo: str, policy_train_episodes: int, experiment_episodes: int, seed: int):
		
		super().__init__(
			gym_env,
			algo,
			policy_train_episodes,
			experiment_episodes,
			seed=seed
			)
	
	def get_params(self):
		
		params={}
		## steps in acrobot above 500 is truncated
		steps = 500
		learning_rate = 0.05
		## Env
		params['env_name']= "Acrobot-v1"
		params['obs_size'] = 6
		## Abstraction
		params['num_iterations_for_abstraction_learning']= 500
		params['learning_rate_for_abstraction_learning']=learning_rate
		params['abstraction_network_hidden_layers'] = 2
		params['abstraction_network_hidden_nodes'] = 200
		params['num_samples_from_demonstrator'] = 20000
		
		params['steps']=steps
		params['rl_learning_rate']=learning_rate

		return params
	