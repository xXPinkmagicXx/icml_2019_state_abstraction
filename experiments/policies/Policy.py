import keras
import sys
from simple_rl.tasks import GymMDP
# from keras.models import model_from_json
import tensorflow as tf
import numpy as np
import abc
import os
# load json and create model
# load weights into new model

class Policy:
    __metaclass__ = abc.ABCMeta
	
    def __init__(self, gym_env: GymMDP, policy_train_episodes: int, experiment_episodes: int, seed: int, k_bins: int = 1):
		
        self.gym_env = gym_env
        self.params = self.get_params()
        print("this is the env name: ", self.params['env_name'])
        self.env_name = self.params['env_name']
        self.policy_train_episodes = policy_train_episodes
        self.algo = "mac"
        self.k_bins = k_bins
        self.seed = seed

        self.params['size_a'] = self.get_num_actions()
        self.params['algo'] = 'mac'
        self.params['policy_train_episodes'] = policy_train_episodes
        self.params['episodes'] = experiment_episodes
        self.params['k_bins'] = k_bins
        abstract_agent_save_path = "trained-abstract-agents/" + str(policy_train_episodes) + '/'
        if os.path.exists(abstract_agent_save_path) == False:
            os.makedirs(abstract_agent_save_path)
        print("this is k_bins: ", k_bins)
        if self.k_bins > 1:
            model =  str(self.k_bins) + "_" + str(self.algo)
        else:
            model =  str(self.algo)
        # model =  str(self.k_bins) + "_" + str(self.algo) if self.k_bins > 1 else str(self.algo)  
        self.params['results_save_name'] = "Q-learning_phi_" + model
        
        # if action space is continuous in the environment it is discretized into k_bins
        if k_bins > 1:
            abstract_agent_save_path = abstract_agent_save_path + str(k_bins) + "_"
        # Save path cannot end in /, as it is used to save the model with keras
        self.params['save_path'] = abstract_agent_save_path + "mac_" + self.env_name 
        if not os.path.exists(self.params['save_path']):
            os.makedirs(self.params['save_path'])
            print("Created directory: ", self.params['save_path'])
        
        self.path_to_learned_policy = self._get_path_to_learned_policy()
        self.loaded_model = self._load_model(self.path_to_learned_policy) 
        self.demo_policy = self.expert_policy
        self.num_mdps = 1
    
    def get_num_actions(self):
        return len(list(self.gym_env.get_actions()))

    def get_policy_train_time(self):
        
        path_to_learned_policy = self._get_path_to_learned_policy()
        
        with open(path_to_learned_policy + "_time.txt", "r") as file:
            policy_train_time = file.readline().strip()
        
        return float(policy_train_time)
    
    def _load_trained_policy(self):
        
        self.path_to_learned_policy = self._get_path_to_learned_policy()
        loaded_model = self._load_model(self.path_to_learned_policy)
        return loaded_model
    
    def _get_path_to_learned_policy(self):
        cwd = os.getcwd()
        print("this is the cwd: ", cwd)
        
        path_to_learned_policy = './mac/learned_policy/'
		# If called as submodule or .. if called from experiments/
        if "icml_2019_state_abstraction" in cwd:
            path_to_learned_policy = './mac/learned_policy/'
        elif "Bachelor-Project" in cwd:
            path_to_learned_policy = './icml_2019_state_abstraction/mac/learned_policy/'	
        ## . if called as submodule or .. if called from experiments/
        path_to_learned_policy += str(self.policy_train_episodes) + "/"
        if self.k_bins > 1:
            path_to_learned_policy += str(self.k_bins) + "_"
        path_to_learned_policy = path_to_learned_policy + self.env_name + "_" + str(self.seed)
        
        return path_to_learned_policy
    def _load_model(self, path_to_learned_policy):
        """
        Args:
            path_to_learned_policy (str): path to the learned policy
        Returns:
            () a loaded model
        
        """
        # Load weights into new model
        loaded_model = keras.models.load_model(path_to_learned_policy + '.h5', compile=False)
        
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

    def _sample_y_data(self, num_samples=10_000, x_train=None):
        
        if x_train is None:
            x_train, y_train = self.sample_training_data(num_samples)
        else:
            y_train = []
            for state in x_train:
                y_train.append(self.demo_policy(state))
            y_train = np.array(y_train)
        
        return x_train, y_train
    def sample_training_data(self, num_samples=10_000, verbose=True):
        '''
        Args:
            self (Policy: class)
            num_samples (int)

        Returns:
            (list): A collection of (s, a, mdp_id) tuples.
        '''
        x = []
        y = []
        for n in range(num_samples):
            cur_state = self.gym_env.env.observation_space.sample()
            self.gym_env.env.state = cur_state
            best_action = self.demo_policy(cur_state)
            x.append(cur_state)
            y.append(best_action)
            if verbose and n % 1000 == 0:
                print("Sampled ", n, " samples. out of ", num_samples, " samples.")
        # normalize and conver the data
        x_train = keras.utils.normalize(x, axis=0)
        y_train = np.array(y)

        return x_train, y_train

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