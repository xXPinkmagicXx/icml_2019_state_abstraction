import keras
import sys
from keras.models import model_from_json
import numpy as np
from simple_rl.tasks import GymMDP
from policies.PolicySB import PolicySB
from stable_baselines3 import DQN 

class CartPolePolicySB(PolicySB):
	
	def __init__(self, gym_env: GymMDP, algo: str = "dqn"):
		# self.gym_env = gym_env
		# self.model = DQN.load(path_to_learned_policy ,env=gym_env.env)
		# self.params = self.get_params()
		# self.demo_policy = self.expert_policy
		super().__init__(gym_env, algo)
	
	def get_params(self):
		
		params={}
		params['env_name']="CartPole-v0"
		params['multitask']=True
		params['obs_size']=4
		params['num_iterations_for_abstraction_learning']=500
		params['learning_rate_for_abstraction_learning']=0.001
		params['abstraction_network_hidden_layers']=2
		params['abstraction_network_hidden_nodes']=40
		params['num_samples_from_demonstrator']=5000
		params['episodes'] = 50
		params['steps']=200
		params['num_instances']=100
		params['rl_learning_rate']=0.001
    
		return params
	
	# def update_params(self, params):
	# 	self.params = params

	# def expert_policy(self, state):
		
	# 	s_size=len(state)
	# 	s_array=np.array(state).reshape(1,s_size)
	# 	temp = self.model.predict(s_array)
		
	# 	return np.argmax(temp[0])	

	# def sample_unif_random(self, num_samples = 5000):
	# 	'''
	# 	Args:
	# 		mdp (simple_rl.MDP)
	# 		num_samples (int)
	# 		epsilon (float)

	# 	Returns:
	# 		(list): A collection of (s, a, mdp_id) tuples.
	# 	'''

	# 	samples = []

	# 	for _ in range(num_samples):
	# 		cur_state = self.gym_env.env.observation_space.sample()
	# 		self.gym_env.env.state = cur_state
	# 		print("cur_state: ", cur_state)
	# 		# Get demo action.
	# 		best_action = self.demo_policy(cur_state)
	# 		samples.append((cur_state, best_action, 0))

	# 	return samples