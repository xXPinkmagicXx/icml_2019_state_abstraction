from .mac import mac 
import numpy
import sys,random
import tensorflow as tf
# Disable extensive logging
tf.keras.utils.disable_interactive_logging()
import gymnasium as gym
from keras import backend as K
import matplotlib.pyplot as plt
import time

# Import action wrapper
from .ActionWrapper import discretizing_wrapper
from .HyperParameters import AlgorithmParameters, MetaParameters, make_parameters

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(env_name: str, episodes=200, k_bins=1, seed=42, verbose=False):
	
	## The neural nets are created in version 1 of tensorflow
	## This is to ensure compatibility and the code runs faster  
	tf.compat.v1.disable_v2_behavior()
	tf.compat.v1.disable_eager_execution()

	#get and set hyper-parameters
	print("default environment is Lunar Lander ...")

	# Params for all environments.
	env = gym.make(env_name)
	
	# How to discretize the action space for the environment
	## Discretize the action space for Pendulum-v0 and MountainCarContinuous
	if isinstance(env.action_space, gym.spaces.Box):
		env = discretizing_wrapper(env, k_bins)

	# Params for specific environments.
	if env_name =='CartPole-v0' or env_name == 'CartPole-v1':
		
		meta_params = MetaParameters(
			env=env,
			env_name=env_name,
			episodes=episodes,
			gamma=0.99,
			seed=seed)
		
		print("this is the observation space", env.observation_space.shape[0])
		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=env.observation_space.shape[0],
			action_space=env.action_space.n,
			k=1,
			epsilon=0.3,
			actor_num_h=1,
			actor_h=64,
			actor_lr=0.001,
			critic_num_h=1,
			critic_h=32,
			critic_lr=0.01,
			critic_batch_size=32,
			critic_num_epochs=40,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_TD',
			verbose=verbose)

	if  env_name =='LunarLander-v2':

		meta_params = MetaParameters(
			env=env,
			env_name="LunarLander-v2",
			episodes=episodes,
			gamma=0.99,
			seed=seed)
		
		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=env.observation_space.shape[0],
			action_space=env.action_space.n,
			k=k,
			epsilon=0.3,
			actor_num_h=2,
			actor_h=128,
			actor_lr=0.00025,
			critic_num_h=2,
			critic_h=128,
			critic_lr=0.005,
			critic_batch_size=32,
			critic_num_epochs=10,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_TD',
			verbose=verbose)

	if env_name == 'Acrobot-v1':
		
		meta_params = MetaParameters(
			env=env,
			env_name="Acrobot-v1",
			max_learning_episodes=3000,
			gamma=0.99,
			seed=seed)
		
		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=env.observation_space.shape[0],
			action_space=env.action_space.n,
			k=k,
			epsilon=0.3,
			actor_num_h=2,
			actor_h=128,
			actor_lr=0.00025,
			critic_num_h=2,
			critic_h=128,
			critic_lr=0.005,
			critic_batch_size=32,
			critic_num_epochs=10,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_TD',
			verbose=verbose)

	if env_name =='MountainCar-v0':
		
		meta_params = MetaParameters(
			env=env,
			env_name="MountainCar-v0",
			episodes=episodes,
			gamma=0.99,
			seed=seed)

		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=len(env.reset()[0]),
			action_space=env.action_space.n,
			epsilon=0.3,
			actor_num_h=2,
			actor_h=64,
			actor_lr=0.001,
			critic_num_h=2,
			critic_h=64,
			critic_lr=0.001,
			critic_batch_size=32,
			critic_num_epochs=10,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_TD'
		)

	if env_name == "Pendulum-v1":
		# The episode truncates at 200 time steps.
		meta_params = MetaParameters(
			env=env,
			env_name="Pendulum-v1",
			episodes=episodes,
			gamma=0.99,
			seed=seed)

		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=3,
			action_space=k,
			epsilon=0.3,
			actor_num_h=2,
			actor_h=64,
			actor_lr=0.0001,
			critic_num_h=2,
			critic_h=64,
			critic_lr=0.001,
			critic_batch_size=32,
			critic_num_epochs=10,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_TD')

	if env_name == "MountainCarContinuous-v0":
		meta_params = MetaParameters(
			env=env,
			env_name="MountainCarContinuous-v0",
			episodes=episodes,
			gamma=0.8,
			seed=seed)

		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=len(env.observation_space.shape[0]),
			action_space= k,
			epsilon=0.6,
			actor_num_h=2,
			actor_h=40,
			actor_lr=0.01,
			critic_num_h=2,
			critic_h=40,
			critic_lr=0.01,
			critic_batch_size=64,
			critic_num_epochs=10,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_monte_carlo')

	if env_name == "Swimmer-v4":
		meta_params = MetaParameters(
			env=env,
			env_name="Swimmer-v4",
			episodes=episodes,
			gamma=0.99,
			seed=seed)

		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=8,
			action_space= 2,
			k=k,
			epsilon=0.3,
			actor_num_h=2,
			actor_h=128,
			actor_lr=0.0001,
			critic_num_h=2,
			critic_h=128,
			critic_lr=0.001,
			critic_batch_size=32,
			critic_num_epochs=10,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_TD',
			verbose=verbose)
		
	## ensure results are reproducible
	
	# set seeds
	numpy.random.seed(seed)
	random.seed(seed)
	if not isinstance(meta_params, MetaParameters):
		meta_params['env'].seed(seed)
	tf.compat.v1.set_random_seed(seed)

	# set session
	session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
	# tf.compat.v1.keras.set_session(sess)

	if verbose:
		print("this is the action space", env.action_space)
	#create a MAC agent and run
	action_space = env.action_space.n
	state_dimension = len(env.reset())
	actor_lr = [0.01]
	critic_lr = [0.01]
	critic_batch_size = [32]
	critic_train_type = ['model_free_critic_TD', 'model_free_critic_monte_carlo']
	epsilon = [0.3, 0.1, 0.5]
	max_buffer_size = [10000]

	actor_h = [128]
	critic_h = [128]

	algo_parameter_list = make_parameters(
							action_space,
							state_dimension,
							k_bins,
							actor_h,
							critic_h,
							actor_lr,
							critic_lr,
							critic_batch_size,
							critic_train_type,
							epsilon,
							max_buffer_size)

	DO_PARAMETER_SEARCH = False

	# alg_params.verbose = True

	if DO_PARAMETER_SEARCH:
		print("Starting parameter search...")
		for alg_param in algo_parameter_list:
			
			print(str(alg_param))
			params = {**alg_param.to_Dictionary(), **meta_params.to_Dictionary()}
			agent = mac(alg_param.to_Dictionary())
			agent.train()
	else:
		
		print("Starting normal training...")
		params = {**alg_params.to_Dictionary(), **meta_params.to_Dictionary()}
		agent = mac(params)

		# train the agent and time it
		start_time = time.time()
		returns, rewards = agent.train()
		end_time = time.time()

		with open(agent.learned_policy_path + "_time.txt", 'w') as f:
			f.write(str(end_time - start_time))

		print("this is the returns: ", returns)
		print("this is the rewards: ", rewards)
		## make plot of returns pr time step
	
	#create a MAC agent and run

if __name__ == "__main__":
	
	gym_env = sys.argv[1]
	
	main(gym_env)
