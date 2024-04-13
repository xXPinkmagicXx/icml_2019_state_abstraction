import keras
import sys
from simple_rl.tasks import GymMDP
from keras.models import model_from_json
import abc
import os
import numpy as np
from stable_baselines3 import DQN
# load json and create model
# load weights into new model

class PolicySB:
    __metaclass__ = abc.ABCMeta
	
    def __init__(self, gym_env: GymMDP, algo: str):
		
        self.gym_env = gym_env
        self.params = self.get_params()
        self.env_name = gym_env.env_name
		
        ## Get current working directory
        cwd = os.getcwd().split('\\')[-1]
		
        path_to_trained_agents = './rl-trained-agents/'
        
        if cwd == "icml_2019_state_abstraction":
            path_to_trained_agents = '../rl-trained-agents/'
        elif cwd == "Bachelor-Project":
            path_to_trained_agents = './rl-trained-agents/'	
        ## . if called as submodule or .. if called from experiments/
        
        path_to_agent = path_to_trained_agents + algo + '_' + self.env_name

        self.model = self._load_agent(path_to_agent)
        self.demo_policy = self.expert_policy
        self.num_mdps = 1
    
    def get_num_actions(self):
        return len(list(self.gym_env.get_actions()))
       
    def _load_agent(self, path_to_agent):

        return DQN.load(path_to_agent, env=self.gym_env.env)

    @abc.abstractmethod
    def get_params(self):
        """
        Args:
            self (Policy: class)
        Returns:
            (dict): A dictionary of parameters for the policy.
        
        This should return a dictionary with the following
        """
        params={}
        return params
    
    def expert_policy(self, state): 
        s_size=len(state)
        s_array=np.array(state).reshape(1,s_size)
        temp = self.model.predict(s_array)
		
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
            cur_state = self.gym_env.env.observation_space.sample()
            self.gym_env.env.state = cur_state

            # Get demo action.
            best_action = self.demo_policy(cur_state)
            samples.append((cur_state, best_action, 0))

        return samples