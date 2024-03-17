import keras
import sys
from keras.models import model_from_json
import numpy as np
from simple_rl.tasks import GymMDP
import policies.Policy as Policy 

class AcrobotPolicy(Policy.Policy):
	
	def __init__(self, gym_env: GymMDP):
		
		super().__init__(gym_env)
	
	def get_params(self):
		params={}
		## steps in acrobot above 500 is truncated
		steps = 500
		learning_rate = 0.005
		## Env
		params['env_name']= "Acrobot-v1"
		params['multitask']=False
		params['obs_size']=6
		## Abstraction
		params['num_iterations_for_abstraction_learning']= steps
		params['learning_rate_for_abstraction_learning']=learning_rate
		params['abstraction_network_hidden_layers']=2
		params['abstraction_network_hidden_nodes']=200
		params['num_samples_from_demonstrator']=15000
		
		params['episodes'] = 250
		params['steps']=steps
		params['num_instances']=1
		params['rl_learning_rate']=learning_rate

		return params
	
	def update_params(self, params):
		self.params = params

	def expert_policy(self, state):
		
		s_size=len(state)
		s_array=np.array(state).reshape(1,s_size)
		temp=self.loaded_model.predict(s_array)
		
		return np.argmax(temp[0])

	def sample_unif_random(self, num_samples = 15000):
		'''
		Args:
			self (simple_rl.MDP)
			num_samples (int) default=5000
		
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
			samples.append((cur_state, best_action, 0))

		return samples 

	
	def _load_model(self):
		json_file = open('../mac/learned_policy/Acrobot-v1.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		return loaded_model