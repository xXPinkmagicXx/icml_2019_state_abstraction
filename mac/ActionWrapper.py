import numpy as np
from gymnasium import spaces
import gymnasium as gym
# Three line comment
# This code is copied from 
# https://github.com/robintyh1/onpolicybaselines/blob/master/onpolicyalgos/discrete_actions_space/ppo_discrete/wrapper.py
"""
wrapper for discretizing continuous action space
"""
def discretizing_wrapper(env, K):
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

    def discretizing_reset(seed=None):
        obs, info = unwrapped_env.orig_reset_()
        return obs, info

    def discretizing_step(action):
        # action is a sequence of discrete indices
        action_cont = action_table[np.arange(naction), action]
        obs, rew, terminated, truncated, info, = unwrapped_env.orig_step_(action_cont)
        
        return (obs, rew, terminated, truncated, info)

    # change observation space
    # In the case where the action space is a single value
    if naction == 1:
        env.action_space = spaces.Discrete(K)
    else:
        action_space = [K for _ in range(naction)]
        # print("This is the action space", action_space)
        a = []
        for _ in range(naction):
            a.append(K)
        action_space = spaces.MultiDiscrete(a) 
        action_space.sample()
        print("This is a sample", action_space.sample())

        env.action_space = spaces.MultiDiscrete(a)

    unwrapped_env.step = discretizing_step
    unwrapped_env.reset = discretizing_reset

    return env