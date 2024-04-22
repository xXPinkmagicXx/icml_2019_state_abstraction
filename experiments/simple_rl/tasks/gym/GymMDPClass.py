'''
GymMDPClass.py: Contains implementation for MDPs of the Gym Environments.
'''

# Python imports.
import random
import sys
import os
import numpy as np
import random
from collections import defaultdict

# Other imports.
import gymnasium as gym
from simple_rl.mdp.MDPClass import MDP
from simple_rl.tasks.gym.GymStateClass import GymState
from gymnasium import spaces
class GymMDP(MDP):
    ''' Class for Gym MDPs '''


    def __init__(self, gym_env, render=False, render_every_n_episodes=0, k=20):
        '''
        Args:
            env_name (str)
            render (bool): If True, renders the screen every time step.
            render_every_n_epsiodes (int): @render must be True, then renders the screen every n episodes.
            k (int): Number of bins to discretize the action space into. Only used if the action space is continuous.
        '''
        # self.render_every_n_steps = render_every_n_steps
        self.render_every_n_episodes = render_every_n_episodes
        self.episode = 0
        self.env_name = gym_env.spec.id
        env = gym_env
        
        self.env = env
        self.render = render
        obs, info = self.env.reset()
        MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=GymState(obs))

    def get_parameters(self):
        '''
        Returns:
            (dict) key=param_name (str) --> val=param_val (object).
        '''
        param_dict = defaultdict(int)
        param_dict["env_name"] = self.env_name
   
        return param_dict

    def _reward_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self.render and (self.render_every_n_episodes == 0 or self.episode % self.render_every_n_episodes == 0):
            self.env.render()

        self.next_state = GymState(obs, is_terminal=terminated)

        return reward

    def _transition_func(self, state, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (State)
        '''
        return self.next_state

    def reset(self):
        self.env.reset()
        self.episode += 1

    def __str__(self):
        return "gym-" + str(self.env_name)
