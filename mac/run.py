from .mac import mac 
import numpy
import gym,sys,random
import tensorflow as tf

# Import action wrapper
from .ActionWrapper import discretizing_wrapper

tf.disable_v2_behavior()
tf.compat.v1.disable_eager_execution()

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(gym_env):
	#get and set hyper-parameters
	meta_params,alg_params={},{}
	try:
		meta_params['env_name']=gym_env
		meta_params['seed_number']= 1
		print("the env is set to", meta_params['env_name'])
	except:
		print("default environment is Lunar Lander ...")
		meta_params['env_name']='LunarLander-v2'
		meta_params['seed_number']=0
	meta_params['seed_number']= 1

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
		meta_params['max_learning_episodes']=3000
		alg_params['state_|dimension|']=len(meta_params['env'].reset())
		alg_params['|A|']= k
		alg_params['max_buffer_size']=10000
		alg_params['actor_num_h']=2
		alg_params['actor_|h|']=64
		alg_params['actor_lr']=0.0001
		alg_params['critic_num_h']=2
		alg_params['critic_|h|']=64
		alg_params['critic_lr']=0.001
		alg_params['critic_batch_size']=32
		alg_params['critic_num_epochs']=10
		alg_params['critic_target_net_freq']=1
		alg_params['critic_train_type']='model_free_critic_TD'


	#ensure results are reproducible
	numpy.random.seed(meta_params['seed_number'])
	random.seed(meta_params['seed_number'])
	session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
	from keras import backend as K
	tf.set_random_seed(meta_params['seed_number'])
	sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
	K.set_session(sess)

	meta_params['env'].seed(meta_params['seed_number'])
	#ensure results are reproducible

	#create a MAC agent and run
	agent=mac(alg_params)
	agent.train(meta_params)
	#create a MAC agent and run

if __name__ == "__main__":
	
	gym_env = sys.argv[1]
	
	main(gym_env)
