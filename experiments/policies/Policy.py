import keras
import sys
from simple_rl.tasks import GymMDP
import abc
import os
# load json and create model
# load weights into new model

class Policy:
    __metaclass__ = abc.ABCMeta
	
    def __init__(self, gym_env: GymMDP):
		
        self.gym_env = gym_env
        self.params = self.get_params()
        self.env_name = self.params['env_name']
		## Get current working directory
        cwd = os.getcwd().split('\\')[-1]
        print("this is the current working dir",cwd)
		## . if called as submodule or .. if called from experiments/
        path_to_learned_policy = './mac/learned_policy/' if "icml_2019_state_abstraction" == cwd else '../mac/learned_policy/'
        
        self.loaded_model = self._load_model(path_to_learned_policy)
        self.demo_policy = self.expert_policy
        self.num_mdps = 1
    
    def get_num_actions(self):
        return len(list(self.gym_env.get_actions()))

    @abc.abstractmethod
    def _load_model(self, path_to_learned_policy):
        return
	
    @abc.abstractmethod
    def get_params(self):
        """
        This should return a dictionary with the following
        """
        params={}
        return params
    
    @abc.abstractmethod
    def expert_policy(self, state): 
        return 

    @abc.abstractmethod
    def sample_unif_random(self, num_samples):
        '''
        Args:
            self (Policy: class)
            num_samples (int)

        Returns:
            (list): A collection of (s, a, mdp_id) tuples.
        '''
        samples = []
        return samples