import keras
import sys
from keras.models import model_from_json
import numpy as np
from simple_rl.tasks import GymMDP
import policies.Policy as Policy

class CartPolePolicy(Policy.Policy):
	
	def __init__(self, gym_env: GymMDP, policy_train_episodes: int, experiment_episodes: int):
		
		super().__init__(gym_env, policy_train_episodes, experiment_episodes)

	
	def get_params(self):
		
		params={}
		params['env_name']="CartPole-v1"
		params['obs_size'] = 4
		params['num_iterations_for_abstraction_learning'] = 100
		params['learning_rate_for_abstraction_learning'] = 0.001
		params['abstraction_network_hidden_layers'] = 2
		params['abstraction_network_hidden_nodes'] = 40
		params['num_samples_from_demonstrator']=5000
		params['num_instances'] = 100
		params['rl_learning_rate'] = 0.001
    
		return params
	
	def update_params(self, params):
		self.params = params

	def expert_policy(self, state):
		
		s_size=len(state)
		s_array=np.array(state).reshape(1,s_size)
		temp=self.loaded_model.predict(s_array)
		
		return np.argmax(temp[0])

	def sample_unif_random(self, num_samples = 5000):
		'''
		Args:
			mdp (simple_rl.MDP)
			num_samples (int)
			epsilon (float)

		Returns:
			(list): A collection of (s, a, mdp_id) tuples.
		'''

		samples = []

		for _ in range(num_samples):
			cur_state = self.gym_env.env.reset()
			cur_state = self.gym_env.env.observation_space.sample()
			self.gym_env.env.state = cur_state

			# Get demo action.
			best_action = self.demo_policy(cur_state)
			#action_index = mdp.get_actions().index(best_action)
			samples.append((cur_state, best_action, 0))

		return samples