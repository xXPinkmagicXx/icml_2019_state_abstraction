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

def get_config(env_name) -> dict:
	
	# # Get the config for the environment
	if env_name == "MountainCar-v0":
		return icml_config.MOUNTAIN_CAR
	if env_name == "CartPole-v0" or env_name == "CartPole-v1":
		return icml_config.CARTPOLE
	elif env_name == "Acrobot-v1":
		return icml_config.ACROBOT
	elif env_name == "Pendulum-v1":
		return icml_config.PENDULUM
	elif env_name == "MountainCarContinuous-v0":
		return icml_config.MOUNTAIN_CAR_CONTINUOUS
	elif env_name == "LunarLander-v2":
		return icml_config.LUNAR_LANDER
	elif env_name == "Swimmer-v4":
		return icml_config.SWIMMER
	else:
		raise ValueError("Invalid environment name")

def get_params(env, env_name, env_render, episodes, k_bins, seed, verbose) -> dict:

	## Get the config for the environment
	config = get_config(env_name)

	config['env'] = env
	config['env_render'] = env_render
	config['A'] = env.action_space.n
	config['n_actions'] = 1
	config['k'] = k_bins
	config['state_dimension']=env.observation_space.shape[0]
	config['episodes'] = episodes
	config['seed'] = seed
	config['verbose'] = verbose
	config['k_bins'] = k_bins
	config['env_name'] = env_name
	config['gamma'] = 0.99
	config['plot'] = False
	
	return config

def main(env_name: str, episodes=200, k_bins=1, seed=42, verbose=False, render=True):
	
	## The neural nets are created in version 1 of tensorflow
	## This is to ensure compatibility and the code runs faster  
	tf.compat.v1.disable_v2_behavior()
	tf.compat.v1.disable_eager_execution()

	# Params for all environments.
	env = gym.make(env_name)
	env_render = gym.make(env_name, render_mode="human")
	# How to discretize the action space for the environment
	## Discretize the action space for Pendulum-v0 and MountainCarContinuous
	if isinstance(env.action_space, gym.spaces.Box):
		env = discretizing_wrapper(env, k_bins)
		env_render = discretizing_wrapper(env_render, k_bins)
	
	# Get the config for the environment
	params = get_params(env, env_name, env_render, episodes, k_bins, seed, verbose)
	
	# Params for specific environments.
	## ensure results are reproducible
	
	# set seeds
	numpy.random.seed(seed)
	random.seed(seed)
	tf.compat.v1.set_random_seed(seed)

	# set session
	# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
	# tf.compat.v1.keras.set_session(sess)

	if verbose:
		print("this is the action space", env.action_space)
	#create a MAC agent and run
		
	agent = mac(params)
	
	# train the agent and time it
	start_time = time.time()
	returns, rewards = agent.train()
	end_time = time.time()
	
	# render one episode
	agent.interactOneEpisode(render=True)
	
	with open(agent.learned_policy_path + "_time.txt", 'w') as f:
		f.write(str(end_time - start_time))
	
	if verbose:
		print("this is the returns: ", returns)
		print("this is the rewards: ", rewards)
	## make plot of returns pr time step
	
	#create a MAC agent and run

if __name__ == "__main__":
	
	gym_env = sys.argv[1]
	
	main(gym_env)
