import keras
import sys
import numpy as np
from simple_rl.tasks import GymMDP
import policies.Policy as Policy
import os
from stable_baselines3 import DQN
from policies.PolicySB import PolicySB


class MountainCarContunuousPolicySB(PolicySB):
    
    def __init__(self, gym_env: GymMDP, algo: str, policy_train_episodes: int, experiment_episodes: int, k_bins: int):    
        
        super().__init__(gym_env, algo, policy_train_episodes, experiment_episodes, k_bins)
        
        
    def get_params(self):
		
        params={}
        params['env_name']="MountainCarContinuous-v0"
        params['obs_size']=self.gym_env.env.observation_space.shape[0]
        
        params['num_iterations_for_abstraction_learning'] = 500
        params['learning_rate_for_abstraction_learning'] = 0.001
        params['abstraction_network_hidden_layers'] = 2
        params['abstraction_network_hidden_nodes'] = 64
        params['num_samples_from_demonstrator']= 10_000
        params['num_instances'] = 100
        params['rl_learning_rate'] = 0.001
    
        return params