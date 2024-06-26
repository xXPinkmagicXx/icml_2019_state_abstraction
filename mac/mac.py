import numpy
from .critic_network import critic
from .actor_network import actor
import gymnasium as gym
import sys
import os
from collections import deque
import time
import numpy as np
from tqdm import tqdm
import keras


class mac:
	'''
	a class representing the Mean Actor-Critic algorithm.
	It contains and actor and a critic + a train function.
	'''
	def __init__(self, params):
		'''
		This initializes MAC agent by creating an actor
		a critic.
		'''
		self.params=params
		self.time_limit_sec = self.params["time_limit_sec"]
		self.memory = deque(maxlen=self.params['max_buffer_size'])
		self.actor=actor(self.params)
		self.critic=critic(self.params)
		self.epsilon = params['epsilon']
		# get current working directory
		cwd = os.getcwd()
		learned_policy_path = "./learned_policy/"
		# If called as submodule or .. if called from experiments/
		if "icml_2019_state_abstraction" in cwd:
			learned_policy_path = './mac/learned_policy/'
		elif "Bachelor-Project" in cwd:
			learned_policy_path = './icml_2019_state_abstraction/mac/learned_policy/'	
		
		if not os.path.exists(learned_policy_path + str(self.params['episodes'])+'/'):
			os.makedirs(learned_policy_path + str(self.params['episodes'])+'/')
		
		self.learned_policy_path = learned_policy_path  + str(self.params['episodes']) + '/' 
		if self.params["k_bins"] > 1:
			self.learned_policy_path += str(self.params["k_bins"]) + "_"
		self.learned_policy_path += self.params["env_name"] + "_" + str(self.params["seed"])

	def add_2_memory(self,states,actions,rewards):
		T=len(states)
		for index,(s,a,r) in enumerate(zip(states,actions,rewards)):
			if index<T-1:
				self.memory.append((s,a,r,states[index+1],T-index-1))

	def train(self):
		'''
		This function trains a MAC agent for max_learning_episodes episodes.
		It proceeds with interaction for one episode, followed by critic
		and actor updates.

		'''
		print("training has begun...")

		# init arrays for logging
		li_returns=[]
		li_actions=[]
		li_rewards=[]
		li_acc_rewards=[]
		li_time = []
		accumulated_rewards= 0
		start_time = time.time()
		print("Is verbose", self.params["verbose"])
		for episode in range(1,self.params['episodes']):
			
			if self.time_limit_sec is not None:
				# break when time limit is reached
				if time.time() - start_time > self.time_limit_sec:
					break
			# Do one episode of interaction
			states, actions, returns, rewards = self.interactOneEpisode()

			# if self.params["verbose"]:
			# 	print("This is the actions taken", np.unique(actions, return_counts=True))
			# Update epsilon for epsilon greedy policy
			# print("This is the epsilon in actor", self.actor.params['epsilon'], "this is the epsilon in mac", self.epsilon)
			self.actor.params['epsilon'] = max(1 - episode/(self.params["episodes"]*self.epsilon), 0.01)

			#add to memory
			self.add_2_memory(states,actions,rewards)

			# Accumulated rewards
			accumulated_rewards += numpy.sum(rewards)
			if episode % 200 == 0:
				self.save_model()
				if self.params["verbose"]:
					print("Saved latest policy network to disk")

			if episode % 250 == 0:
				self.params['env'].render()
			
			
			# if episode % 10 == 0:
			# 	if self.params["verbose"]:
			# 		print(episode,"return in last 10 episodes",numpy.mean(li_returns[-10:]), "with accumulated rewards", accumulated_rewards)

			li_returns.append(returns[0])
			li_rewards.append(numpy.sum(rewards))
			li_acc_rewards.append(accumulated_rewards)
			sys.stdout.flush()

			#log performance
			if self.params["verbose"]:
				print("episode: ",episode, "out of", self.params["episodes"] , "with returns:", numpy.mean(li_returns[-1]), " and accumulated rewards", accumulated_rewards, "steps:", len(rewards))
			
			#train the Q network
			self.train_critic(states, actions, returns, episode)
			self.actor.train(states, self.critic)
				
		# save the model when training is over
		self.save_model()
		print("Saved latest policy network to disk")

		print("training is finished successfully!")
		# Return the rewards 
  		
		# self.makePlotofReturns(rewards)
		# self.makePlotofReturns(li_acc_rewards, title="Accumulated rewards")
		
		return li_returns, li_rewards

	def train_critic(self, states, actions, returns, episode) -> None:
		'''
		Train the critic network
		'''
		if self.params['critic_train_type']=='model_free_critic_monte_carlo':
			self.critic.train_model_free_monte_carlo(states,actions,returns)

		elif self.params['critic_train_type']=='model_free_critic_TD':
			self.critic.train_model_free_TD(self.memory,self.actor,self.params,self.params,episode)

	
	def interactOneEpisode(self, render=False):
		'''
			given the mac agent and an environment, executes their
			interaction for an episode, and then returns important information
			used for training.
		'''
		env_string = "env_render" if render else "env"
		s0, _ = self.params[env_string].reset()
		rewards = []
		states = []
		actions = []
		t = 0
		s=s0
		while True:
			a = self.actor.select_action(s)
			s_p , r , terminated, truncated , info = self.params[env_string].step(a)
			
			states.append(s)
			actions.append(a)

			# reward shaping, applies shaping if defined or returns the reward as is
			shaped_reward = self.reward_shaping(s, r)
			rewards.append(shaped_reward)
			# update for next iteration
			
			s, t = (s_p,t+1)

			# when done
			if truncated or terminated :
				reward_goal = 0
				
				# we actually store the terminal state!
				states.append(s_p)
				a = self.actor.select_action(s_p)
				actions.append(a),
				
				reward_end = self.reward_shaping_end(s_p, terminated, truncated)
				
				rewards.append(reward_goal)
				break
			if render:
				self.params[env_string].render()
		returns = self.rewardToReturn(rewards)

		return states, actions, returns, rewards
	
	def rewardToReturn(self, rewards) -> list:
		T=len(rewards)
		returns=T*[0]
		returns[T-1]=rewards[T-1]
		for t in range(T-2,-1,-1):
			returns[t]=rewards[t]+self.params['gamma']*returns[t+1]
		return returns
	
	# def makePlotofReturns(self, returns, title="Returns", show=False)->None:
	# 	plt.plot([i for i in range(len(returns))], returns, 'o')
	# 	plt.title(title)
	# 	if show:
	# 		plt.show()

	def save_model(self) -> None:
		
		## Update json file
		
		# serialize weights to HDF5
		keras.models.save_model(self.actor.network, self.learned_policy_path + ".h5", save_format='h5')
	
	def load_model(self) -> None:
		# load json and create model
		if self.params["verbose"]:
			print("Loaded model from disk")
		self.actor.network = keras.models.load_model(self.learned_policy_path + ".h5")
	def reward_shaping(self, current_state, current_reward):
		'''
		Here we can define a reward shaping, to be used for every step.
		'''
		if self.params['env_name'] == "MountainCar-v0":
			return current_reward + self.rewardVelocityMountainCar(current_state[1])
		
		if self.params['env_name'] == "MountainCarContinuous-v0":
			return current_reward + self.rewardVelocityMountainCar(current_state[1])
		
		if self.params['env_name'] == "Pendulum-v1":
			return current_reward + self.reward_Upright_Pendulum(current_state)

		# if no shaping return current reward
		return current_reward
	
	def reward_shaping_end(self, current_state, terminated, truncated):
		'''
		Here we can define a reward shaping function.
		'''
		if self.params['env_name'] == "MountainCar-v0" and terminated:
			return 1000
		if self.params["env_name"] == "Acrobot-v1" and terminated:
			return 1000
		
		if self.params["env_name"] == "Pendulum-v1":
			# if upright and low velocity
			x, y, velocity = current_state
			if abs(y) < 0.1 and x > 0 and abs(velocity) < 0.1:
				return 1000
			
		if self.params["env_name"] == "CartPole-v1":
			# if terminated pole angle is more than 12 degrees or out of bounds
			if terminated:
				return -1000
			else: 
				return 1
		
		# if no shaping return 0, which is default
		return 0
	def reward_Upright_Pendulum(self, current_state):
		# Reward for being upright
		x, y, velocity = current_state
		if abs(y) < 0.2 and x > 0 and abs(velocity) < 0.2:
			return 10
		return 0

	def rewardVelocityMountainCar(self, current_velocity):
		# Reward for velocity
		return 100 * abs(current_velocity)


