import keras
import sys
from simple_rl.tasks import GymMDP
from keras.models import model_from_json
import abc
import os
from gymnasium.vector.utils import batch_space
import numpy as np
import tensorflow as tf
from stable_baselines3 import DQN, PPO, A2C, SAC, TD3, DDPG
# load json and create model
# load weights into new model

class PolicySB:
    __metaclass__ = abc.ABCMeta
	
    def __init__(self, gym_env: GymMDP, algo: str):
		
        # environment 
        self.gym_env = gym_env
        self.env_name = gym_env.env_name
        self.algo = algo
        # Get the model class based on the algorithm
        self._model_class = self._get_model_class(algo)

        # Get the parameters for the policy
        self.params = self.get_params()
        self.params['env_name'] = self.env_name
        self.params['algo'] = algo
        self.params['save_path'] = "trained-abstract-agents/" + self.params['algo'] + '_' + self.params['env_name'] 
        ## Get current working directory
        cwd = os.getcwd().split('\\')[-1]
		
        path_to_trained_agents = './rl-trained-agents/'
        
        if cwd == "icml_2019_state_abstraction":
            path_to_trained_agents = '../rl-trained-agents/'
        elif cwd == "Bachelor-Project":
            path_to_trained_agents = './rl-trained-agents/'	
        elif cwd == "experiments":
            path_to_trained_agents = '../../rl-trained-agents/'
        ## . if called as submodule or .. if called from experiments/
        print("this is the path to trained agents:", path_to_trained_agents)
        path_to_agent = path_to_trained_agents + algo + '_' + self.env_name

        self.model = self._load_agent(path_to_agent)
        self.demo_policy = self.expert_policy
        self.num_mdps = 1
    
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

        for i in range(num_samples):
            cur_state = self.gym_env.env.observation_space.sample()
            self.gym_env.env.state = cur_state
            self.model.env.state = cur_state
            # Get demo action.
            best_action = self.demo_policy(cur_state)
            # print("this is the best action:", best_action)
            if i % 100 == 0:
                print("this is the best action:", best_action)
                print("Appending this to the sample", (cur_state, best_action, 0))
            samples.append((cur_state, best_action, 0))

        return samples
    
    def sample_training_data(self, num_samples = 10000):
        """
        Args:
            num_samples (int): Number of samples to collect
        Returns:
            x_train, y_train: (np.array, np.array): Tuple of states and actions
        """
        x = []
        y = []
        for _ in range(num_samples):
            cur_state = self.gym_env.env.observation_space.sample()
            self.gym_env.env.state = cur_state
            self.model.env.state = cur_state
            best_action = self.demo_policy(cur_state)
            x.append(cur_state)
            y.append(best_action)

        # normalize and conver the data
        x_train = tf.keras.utils.normalize(np.array(x), axis=0)
        y_train = np.array(y)

        return x_train, y_train
    
    def _get_model_class(self, algo_name: str):
        if algo_name == 'ppo':
            return PPO
        elif algo_name == 'a2c':
            return A2C
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