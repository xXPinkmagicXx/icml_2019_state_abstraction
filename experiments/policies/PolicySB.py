from simple_rl.tasks import GymMDP
from keras.models import model_from_json
import abc
import os
from gymnasium.vector.utils import batch_space
import numpy as np
import tensorflow as tf
import keras
from stable_baselines3 import DQN, PPO, SAC, TD3, DDPG

class PolicySB:
    __metaclass__ = abc.ABCMeta
	
    def __init__(self, gym_env: GymMDP, algo: str, policy_train_episodes: int, experiment_episodes: int, seed: int, k_bins: int = 1):
		
        # environment 
        self.gym_env = gym_env
        self.env_name = gym_env.env_name
        self.algo = algo
        self.policy_train_episodes = policy_train_episodes
        self.k_bins = k_bins
        self.seed = seed
        
        # Get the model class based on the algorithm
        self._model_class = self._get_model_class(algo)

        # Get the parameters for the policy
        self.params = self.get_params()
        self.params['env_name'] = self.env_name
        self.params['algo'] = algo
        self.params['episodes'] = experiment_episodes
        self.params['k_bins'] = k_bins

        self.params['num_actions'] = self.get_num_actions()
        self.params['size_a'] = self.params['num_actions']
        self.params['policy_train_episodes'] = policy_train_episodes
        self.params['plot_path'] = 'results/' + 'gym-'+ self.params['env_name'] + '/' + str(policy_train_episodes)
        self.params['results_folder_path'] = 'results/' + 'gym-'+ self.params['env_name'] + '/' + str(policy_train_episodes) + '/'

        model =  str(self.k_bins) + "_" + str(self.algo) if self.k_bins > 1 else str(self.algo)  
        self.params['results_save_name'] = "Q-learning_phi_" + model
        
        abstract_agent_save_path = "trained-abstract-agents/" + str(policy_train_episodes) + '/'
        if os.path.exists(abstract_agent_save_path) == False:
            os.makedirs(abstract_agent_save_path)
        
        # if action space is continuous in the environment it is discretized into k_bins
        if k_bins > 1:
            abstract_agent_save_path = abstract_agent_save_path + str(k_bins) + "_"
        
        self.params['save_path'] = abstract_agent_save_path + self.params['algo'] + '_' + self.params['env_name'] + "_" + str(seed)
        if not os.path.exists(self.params['save_path']):
            os.makedirs(self.params['save_path'])
            print("Created directory: ", self.params['save_path'])

       
        path_to_agent = self._get_path_to_agent()
        self.model = self._load_agent(path_to_agent)
        self.demo_policy = self.expert_policy
        self.num_mdps = 1
    
    def get_policy_train_time(self):
        
        path_to_agent = self._get_path_to_agent()
        
        with open(path_to_agent + "_time.txt", "r") as f:
            policy_trian_time = f.readline().strip()
        
        return float(policy_trian_time)

    def _get_path_to_agent(self):
         ## Get current working directory
        cwd = os.getcwd()
		
        path_to_trained_agents = './rl-trained-agents/'
        
        if cwd == "icml_2019_state_abstraction":
            path_to_trained_agents = '../rl-trained-agents/'
        elif cwd == "Bachelor-Project":
            path_to_trained_agents = './rl-trained-agents/'	
        elif cwd == "experiments":
            path_to_trained_agents = '../../rl-trained-agents/'
        ## . if called as submodule or .. if called from experiments/
        path_to_trained_agents += str(self.policy_train_episodes) + '/'
        if self.k_bins > 1:
            path_to_trained_agents += str(self.k_bins) + "_"
        
        # print("this is the path to trained agents:", path_to_trained_agents)
        path_to_agent = path_to_trained_agents + self.algo + '_' + self.env_name + "_" + str(self.seed)
        return path_to_agent

    def get_num_actions(self):
        return len(list(self.gym_env.get_actions()))
       
    def _load_agent(self, path_to_agent):
        
        return self._model_class.load(path_to_agent, env=self.gym_env.env)

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
        temp, _ = self.model.predict(s_array)
        best_action = np.argmax(temp)	
        # print("this is the state:", state)
        # print("this is the s_array:", s_array)
        # print("this is the temp:", temp, "with the best action", best_action)
        return temp[0]


    def export_policy_batch(self, state_batch):
        """
        Args:
            state_batch (list): List of states
        Returns:
            (list): List of actions
        """
        print("this is the shape of the state_batch", np.shape(state_batch))
        print("this is state_batch", state_batch)
        temp, _ = self.model.predict(state_batch)
        
        print("this is the shape of the temp", np.shape(temp))

        return temp
    
    def sample_unif_random(self, num_samples = 5000, verbose = False):
        '''
		Args:
			mdp (simple_rl.MDP)
			num_samples (int)
			epsilon (float)

		Returns:
			(list): A collection of (s, a, mdp_id) tuples.
		'''

        samples = []

        for i in range(num_samples):
            cur_state = self.gym_env.env.observation_space.sample()
            self.gym_env.env.state = cur_state
            self.model.env.state = cur_state
            # Get demo action.
            best_action = self.demo_policy(cur_state)
            # print("this is the best action:", best_action)
            if verbose and i % 1000 == 0:
                print("this is the number of samples:", i, "out of", num_samples, "samples.")
            samples.append((cur_state, best_action, 0))

        return samples
    
    def sample_training_data(self, num_samples = 10000, verbose = True):
        """
        Args:
            num_samples (int): Number of samples to collect
        Returns:
            x_train, y_train: (np.array, np.array): Tuple of states and actions
        """
        x = []
        y = []
        for n in range(num_samples):
            cur_state = self.gym_env.env.observation_space.sample()
            self.gym_env.env.state = cur_state
            self.model.env.state = cur_state
            best_action = self.demo_policy(cur_state)
            x.append(cur_state)
            y.append(best_action)
            if n % 1000 == 0 and verbose:
                print("Sampled ", n, " samples. out of ", num_samples, " samples.")

        # normalize and conver the data
        return np.array(x), np.array(y)
    
    def _get_model_class(self, algo_name: str):
        if algo_name == 'ppo':
            return PPO
        elif algo_name == 'dqn':
            return DQN
        elif algo_name == 'sac':
            return SAC
        elif algo_name == 'td3':
            return TD3
        elif algo_name == 'ddpg':
            return DDPG
        else:
            raise ValueError('Invalid algorithm name')