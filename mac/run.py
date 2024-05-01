from .mac import mac 
import numpy
import sys,random
import tensorflow as tf
# Disable extensive logging
# tf.keras.utils.disable_interactive_logging()
import gymnasium as gym
from keras import backend as K
import matplotlib.pyplot as plt
import time
import os
# Import action wrapper
from .ActionWrapper import discretizing_wrapper
from .HyperParameters import AlgorithmParameters, MetaParameters, make_parameters
from Code.icml import icml_config
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# def get_config(env_name) -> dict:
	
# 	# # Get the config for the environment
# 	if env_name == "MountainCar-v0":
# 		return icml_config.MOUNTAIN_CAR
# 	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
# 		return icml_config.CARTPOLE
# 	elif env_name == "Acrobot-v1":
# 		return icml_config.ACROBOT
# 	elif env_name == "LunarLander-v2":
# 		return icml_config.LUNAR_LANDER
# 	# Continuous action space
# 	elif env_name == "Pendulum-v1":
# 		return icml_config.PENDULUM
# 	elif env_name == "MountainCarContinuous-v0":
# 		return icml_config.MOUNTAIN_CAR_CONTINUOUS
# 	elif env_name == "Swimmer-v4":
# 		return icml_config.SWIMMER
# 	else:
# 		raise ValueError("Invalid environment name")

# def get_params(env, env_name, env_render, episodes, n_actions, k_bins, seed, verbose) -> dict:

# 	## Get the config for the environment
# 	config = get_config(env_name)

# 	config['env'] = env
# 	config['env_render'] = env_render
# 	config['n_actions'] = n_actions
# 	config['A'] = env.action_space.n
# 	config['k'] = k_bins
# 	config['state_dimension']=env.observation_space.shape[0]
# 	config['episodes'] = episodes
# 	config['seed'] = seed
# 	config['verbose'] = verbose
# 	config['k_bins'] = k_bins
# 	config['env_name'] = env_name
# 	config['gamma'] = 0.99
# 	config['plot'] = False
	
# 	return config

def main_from_config(config: dict, seed=None, verbose=False, time_limit_sec=None):

	main(
		config['env_name'],
		episodes=config['policy_episodes'],
		k_bins=config['k_bins'],
		train=config['train'],
		seed=seed,
		verbose=verbose,
		config=config,
		time_limit_sec=time_limit_sec)

def main(
		env_name: str,
		episodes: int,
		k_bins: int,
		seed: int=None,
		train=True,
		verbose=False,
		render=True,
		time_limit_sec=None,
		config=None
		):
	
	## The neural nets are created in version 1 of tensorflow
	## This is to ensure compatibility and the code runs faster  
	tf.compat.v1.disable_v2_behavior()
	tf.compat.v1.disable_eager_execution()

	# Params for all environments.
	env = gym.make(env_name)
	env_render = gym.make(env_name, render_mode="human")
	# How to discretize the action space for the environment
	## Discretize the action space for Pendulum and MountainCarContinuous
	n_actions_continuous = None
	if isinstance(env.action_space, gym.spaces.Box):
		n_actions_continuous = env.action_space.shape[0]
		env = discretizing_wrapper(env, k_bins)
		env_render = discretizing_wrapper(env_render, k_bins)
	else:
		# in case k_bins was set and the action space is not continuous
		k_bins = 1

	n_actions = env.action_space.n if n_actions_continuous is None else n_actions_continuous
	# Get the config for the environment
	# Params for specific environments.
	params = config
	params['env'] = env
	params['env_render'] = env_render
	params['n_actions'] = n_actions
	params['A'] = env.action_space.n
	params['k'] = k_bins
	params['state_dimension']=env.observation_space.shape[0]
	params['episodes'] = episodes
	params['seed'] = seed
	params['verbose'] = verbose
	params['k_bins'] = k_bins
	params['env_name'] = env_name
	params['gamma'] = 0.99
	params['plot'] = False

	# set seeds to ensure results are reproducible
	numpy.random.seed(seed)
	random.seed(seed)
	tf.compat.v1.set_random_seed(seed)

	if verbose:
		print("this is the action space", env.action_space)
	
	# create a MAC agent and run
	agent = mac(params)
	
	# train the agent and time it
	if train:
		start_time = time.time()
		returns, rewards = agent.train(time_limit_sec)
		end_time = time.time()
		
		with open(agent.learned_policy_path + "_time.txt", 'w') as f:
			f.write(str(end_time - start_time))
		
		if verbose:
			print("this is the returns: ", returns)
			print("this is the rewards: ", rewards)
	
	# render one episode after training
	if render:
		# to make the policy deterministic
		agent.params['episilon'] = 0.0
		agent.interactOneEpisode(render=True)
	
	

if __name__ == "__main__":
	
	gym_env = sys.argv[1]
	
	main(gym_env)
