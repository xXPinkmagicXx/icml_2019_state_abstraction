import keras
import sys
import numpy as np
from simple_rl.tasks import GymMDP
import policies.Policy as Policy
import os
from stable_baselines3 import DQN



class MountainCarPolicySB():
    
    def __init__(self, checkpoint, gym_env: GymMDP):    
        self.checkpoint = checkpoint
        self.gym_env = gym_env
        self.model = DQN.load(checkpoint, env=gym_env.env)
        self.params = self.get_params()
        self.demo_policy = self.expert_policy
    
    def get_params(self):
		
        params={}
        params['env_name']="MountainCar-v0"
        params['multitask']=False
        params['obs_size']=self.gym_env.env.observation_space.shape[0]
        params['num_iterations_for_abstraction_learning']=500
        params['learning_rate_for_abstraction_learning']=0.001
        params['abstraction_network_hidden_layers']=2
        params['abstraction_network_hidden_nodes']=64
        params['num_samples_from_demonstrator']=100
        params['episodes'] = 50
        params['steps']=200
        params['num_instances']=100
        params['rl_learning_rate']=0.001
    
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

        for _ in range(100):
            cur_state = self.gym_env.env.observation_space.sample()
            # print("this is the current state", cur_state)
            self.gym_env.env.state = cur_state

            # Get demo action.
            best_action = self.demo_policy(cur_state)
            # print("this is the best_action", best_action)
            #action_index = mdp.get_actions().index(best_action)
            samples.append((cur_state, best_action, 0))

        return samples