from .mac import mac 
import numpy
import gym,sys,random
import tensorflow as tf
from keras import backend as K

# Import action wrapper
from .ActionWrapper import discretizing_wrapper
from .HyperParameters import AlgorithmParameters, MetaParameters, make_parameters

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(env_name, seed=42, verbose=False):
	
	#get and set hyper-parameters
	meta_params,alg_params={},{}
	print("default environment is Lunar Lander ...")

	# Params for all environments.
	env = gym.make(env_name)
	
	# How to discretize the action space for the environment
	## Discretize the action space for Pendulum-v0
	k = 1
	if isinstance(env.action_space, gym.spaces.Box):
		k = 100

		env = discretizing_wrapper(env, k)

	# Params for specific environments.
	if env_name =='CartPole-v0':

		meta_params = MetaParameters(
			env=env,
			env_name="CartPole-v0",
			max_learning_episodes=400,
			gamma=0.99,
			seed=seed)
		
		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=len(env.reset()),
			action_space=env.action_space.n,
			k=k,
			epsilon=0.3,
			actor_num_h=1,
			actor_h=64,
			actor_lr=0.001,
			critic_num_h=1,
			critic_h=64,
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
			max_learning_episodes=3000,
			gamma=0.99,
			seed=seed)
		
		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=len(env.reset()),
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
			state_dimension=len(env.reset()),
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
			max_learning_episodes=2000,
			gamma=0.99,
			seed=seed)

		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=len(env.reset()),
			action_space=env.action_space.n,
			epsilon=0.3,
			actor_num_h=2,
			actor_h=40,
			actor_lr=0.01,
			critic_num_h=2,
			critic_h=40,
			critic_lr=0.01,
			critic_batch_size=64,
			critic_num_epochs=10,
			critic_target_net_freq=1,
			critic_train_type='model_free_critic_monte_carlo'
		)

	if env_name == "Pendulum-v1":
		# The episode truncates at 200 time steps.
		meta_params = MetaParameters(
			env=env,
			env_name="Pendulum-v1",
			max_learning_episodes=3000,
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
			max_learning_episodes=600,
			gamma=0.8,
			seed=seed)

		alg_params = AlgorithmParameters(
			max_buffer_size=10000,
			state_dimension=len(env.reset()),
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
			max_learning_episodes=2000,
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
	tf.set_random_seed(seed)

	# set session
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)

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
							k,
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
			
			agent = mac(alg_param.to_Dictionary())
			agent.train(meta_params.to_Dictionary())
	else:
		
		print("Starting normal training...")
		params = {**alg_params.to_Dictionary(), **meta_params.to_Dictionary()}
		params["verbose"] = True
		agent = mac(params)
		agent.train(meta_params.to_Dictionary())
	
	#create a MAC agent and run

if __name__ == "__main__":
	
	gym_env = sys.argv[1]
	
	main(gym_env)
