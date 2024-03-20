import keras
import sys
from simple_rl.tasks import GymMDP
from keras.models import model_from_json
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
		
        ## . if called as submodule or .. if called from experiments/
        path_to_learned_policy = './mac/learned_policy/' if "icml_2019_state_abstraction" == cwd else '../mac/learned_policy/'
        
        self.loaded_model = self._load_model(path_to_learned_policy)
        self.demo_policy = self.expert_policy
        self.num_mdps = 1
    
    def get_num_actions(self):
        return len(list(self.gym_env.get_actions()))

    def _load_model(self, path_to_learned_policy):
        """
        Args:
            path_to_learned_policy (str): path to the learned policy
        Returns:
            () a loaded model
        
        """
        # json file
        json_file_name = path_to_learned_policy + self.env_name + '.json'
        json_file = open(json_file_name, 'r')
        # Load Model
        loaded_model_json = json_file.read()
        json_file.close()

        # Load weights into new model
        loaded_model = model_from_json(loaded_model_json)
        weights_file_name = path_to_learned_policy + self.env_name + '.h5'
        loaded_model.load_weights(weights_file_name)
        
        # return the loaded model
        return loaded_model
	
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