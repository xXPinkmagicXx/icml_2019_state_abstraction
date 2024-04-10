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
from gym import spaces

class GymMDP(MDP):
    ''' Class for Gym MDPs '''


    def __init__(self, env_name='CartPole-v0', render=False, render_every_n_episodes=0, k=20):
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
        self.env_name = env_name
        env = gym.make(env_name)
        if isinstance(env.action_space, gym.spaces.Box):
            env = self.discretizing_wrapper(env, k)
        self.env = env
        self.render = render
        print("this is the reset", self.env.reset())
        # if env_name == "MountainCar-v0":
        #     MDP.__init__(self, range(self.env.action_space.n), self._transition_func, self._reward_func, init_state=GymState(self.env.reset()))
        # else: 
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

    def discretizing_wrapper(self, env, K):
        """
        # discretize each action dimension to K bins
        """
        unwrapped_env = env.unwrapped
        unwrapped_env.orig_step_ = unwrapped_env.step
        unwrapped_env.orig_reset_ = unwrapped_env.reset
        
        action_low, action_high = env.action_space.low, env.action_space.high
        naction = action_low.size
        action_table = np.reshape([np.linspace(action_low[i], action_high[i], K) for i in range(naction)], [naction, K])
        assert action_table.shape == (naction, K)

        def discretizing_reset():
            obs = unwrapped_env.orig_reset_()
            return obs

        def discretizing_step(action):
            # action is a sequence of discrete indices
            action_cont = action_table[np.arange(naction), action]
            obs, rew, terminated, truncated, info, = unwrapped_env.orig_step_(action_cont)
            
            return (obs, rew, terminated, info)

        # change observation space
        # In the case where the action space is a single value
        if naction == 1:
            env.action_space = spaces.Discrete(K)
        else:
            env.action_space = spaces.MultiDiscrete([[0, K-1] for _ in range(naction)])

        unwrapped_env.step = discretizing_step
        unwrapped_env.reset = discretizing_reset

        return env