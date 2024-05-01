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
        self.max_steps = gym_env.spec.max_episode_steps
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

    def _reward_func(self, state: GymState, action):
        '''
        Args:
            state (AtariState)
            action (str)

        Returns
            (float)
        '''
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        reward = self._reward_shaping(state, reward)
        
        # Reward shaping end of episode
        if terminated or truncated:
            reward = self._reward_shaping_end(state, terminated, truncated)
        
        if self.render and (self.render_every_n_episodes == 0 or self.episode % self.render_every_n_episodes == 0):
            self.env.render()

        self.next_state = GymState(obs, is_terminal=(terminated or truncated))

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

    def _reward_shaping(self, state: GymState, reward):
        '''
        Args:
            state (AtariState)
            reward (float)

        Returns
            (float)
        Summary:
            Does reward shaping if necessary. Otherwise, returns the reward as is.
        '''
        if self.env_name == "MountainCar-v0" or self.env_name == "MountainCarContinuous-v0":
            return reward + self._reward_MountainCar(state.data[1])
        
        if self.env_name == "Pendulum-v1":
            return reward + self._reward_Pendulum(state)
            

        return reward
    
    def _reward_MountainCar(self, currentVelocity):
        return 100 * abs(currentVelocity)
    
    def _reward_Pendulum(self, state: GymState):
        x, y, velocity = state.data
        if abs(y) < 0.2 and x > 0 and abs(velocity) < 0.2:
            return 10
        return 0
    
    def _reward_shaping_end(self, state: GymState, terminated, truncated):
        '''
        Args:
            state (AtariState)

        Returns
            (float)
        Summary:
            Reward shaping for when the agent reaches the goal.
            If no reward shaping is necessary, return 0 (default).
        '''
        if self.env_name == "CartPole-v1":
            # if terminated and pole angle is more than 12 degrees or out of bounds
            if terminated:
                return -1000
        if self.env_name == "MountainCar-v1":
            return 1000
        if self.env_name == "Acrobot-v1" and terminated:
            return 1000
        if self.env_name == "Pendulum-v1":
            # if upright and low velocity
            x, y, velocity = state.data

            if abs(y) < 0.1 and x > 0 and abs(velocity) < 0.1:
                return 1000
        
        return 0

    def reset(self):
        self.env.reset()
        self.episode += 1

    def __str__(self):
        return "gym-" + str(self.env_name)
