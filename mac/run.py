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

def main(gym_env, seed=42):
	#get and set hyper-parameters
	meta_params,alg_params={},{}
	print("default environment is Lunar Lander ...")
	meta_params['env_name']=gym_env

	# Params for all environments.
	env = gym.make(meta_params['env_name'])
	
	# How to discretize the action space for the environment
	k = 20
	## Discretize the action space for Pendulum-v0
	if meta_params['env_name']=='Pendulum-v1':

		env = discretizing_wrapper(env, k)

	meta_params['env'] = env
	meta_params['gamma']=0.9999
	meta_params['plot']=False
	alg_params={}
	alg_params["epsilon"] = 0.3
	alg_params['state_|dimension|']=len(meta_params['env'].reset())

	# Params for specific environments.
	if meta_params['env_name']=='CartPole-v0':
		meta_params['max_learning_episodes']=400
		alg_params['|A|']=meta_params['env'].action_space.n
		alg_params['critic_num_h']=1
		alg_params['critic_|h|']=64
		alg_params['critic_lr']=0.01
		alg_params['actor_num_h']=1
		alg_params['actor_|h|']=64
		alg_params['actor_lr']=0.001
		alg_params['critic_batch_size']=32
		alg_params['critic_num_epochs']=40
		alg_params['critic_target_net_freq']=1
		alg_params['max_buffer_size']=10000
		alg_params['critic_train_type']='model_free_critic_TD'#or model_free_critic_monte_carlo

	if meta_params['env_name']=='LunarLander-v2':
		meta_params['max_learning_episodes']=3000
		alg_params['state_|dimension|']=len(meta_params['env'].reset())
		alg_params['|A|']=meta_params['env'].action_space.n
		alg_params['critic_num_h']=2
		alg_params['critic_|h|']=128
		alg_params['critic_lr']=0.005
		alg_params['actor_num_h']=2
		alg_params['actor_|h|']=128
		alg_params['actor_lr']=0.00025
		alg_params['critic_batch_size']=32
		alg_params['critic_num_epochs']=10
		alg_params['critic_target_net_freq']=1
		alg_params['max_buffer_size']=10000
		alg_params['critic_train_type']='model_free_critic_TD'#or model_free_critic_monte_carlo

	if meta_params['env_name'] == 'Acrobot-v1':
		
		meta_params['max_learning_episodes']=3000
		alg_params['max_buffer_size']=5000
		alg_params['state_|dimension|']=len(meta_params['env'].reset())

		# Actor
		alg_params['actor_num_h']=2
		alg_params['actor_|h|']=128
		alg_params['actor_lr']=0.00025
		
		## critic
		alg_params['|A|']= meta_params['env'].action_space.n
		alg_params['critic_num_h']=2
		alg_params['critic_|h|']=128
		alg_params['critic_lr']=0.005
		alg_params['critic_batch_size']=32
		alg_params['critic_num_epochs']=10
		alg_params['critic_target_net_freq']=1
		alg_params['critic_train_type']='model_free_critic_TD'

	if meta_params['env_name']=='MountainCar-v0':
		
		meta_params['max_learning_episodes']=3000
		alg_params['state_|dimension|']=len(meta_params['env'].reset())
		alg_params['|A|']=meta_params['env'].action_space.n
		print("this is the action space", alg_params['|A|'])
		print("this is the state space", alg_params['state_|dimension|'])
		alg_params['max_buffer_size']=10000
		
		## Actor
		alg_params['actor_num_h']=2
		alg_params['actor_|h|']=40
		alg_params['actor_lr']=0.01
		alg_params['epsilon'] = 0.6
		## Critic
		alg_params['critic_num_h']=2
		alg_params['critic_|h|']=40
		alg_params['critic_lr']=0.01
		alg_params['critic_batch_size']=64
		alg_params['critic_num_epochs']=10
		alg_params['critic_target_net_freq']=1
		alg_params['critic_train_type']='model_free_critic_monte_carlo' #'model_free_critic_TD'

	if meta_params['env_name'] == "Pendulum-v1":
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


	#create a MAC agent and run
	action_space = env.action_space.n
	state_dimension = len(env.reset())
	actor_lr = [0.0001, 0.001, 0.01]
	critic_lr = [0.0001, 0.001, 0.01]
	critic_batch_size = [32, 64, 128]
	critic_train_type = ['model_free_critic_TD', 'model_free_critic_monte_carlo']
	epsilon = [0.1, 0.3, 0.5]
	max_buffer_size = [1000, 5000, 10000]

	algo_parameter_list = make_parameters(action_space,state_dimension, actor_lr, critic_lr, critic_batch_size, critic_train_type, epsilon, max_buffer_size)

	if isinstance(alg_params, AlgorithmParameters):
		agent = mac(alg_params.to_Dictionary())
		returns = agent.train(meta_params.to_Dictionary())
		print(returns)
	else:
		agent = mac(alg_params)
		agent.train(meta_params)
	
	#create a MAC agent and run

if __name__ == "__main__":
	
	gym_env = sys.argv[1]
	
	main(gym_env)
