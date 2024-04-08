import numpy
from .critic_network import critic
from .actor_network import actor
import gym
import sys
import os
from collections import deque
import time

class mac:
	'''
	a class representing the Mean Actor-Critic algorithm.
	It contains and actor and a critic + a train function.
	'''
	def __init__(self,params):
		'''
		This initializes MAC agent by creating an actor
		a critic.
		'''
		self.params=params
		self.memory = deque(maxlen=self.params['max_buffer_size'])
		self.actor=actor(self.params)
		self.critic=critic(self.params)
		self.epsilon = params['epsilon']
		# get current working directory
		cwd = os.getcwd().split('\\')[-1]
		learned_policy_path = "./learned_policy/"
		# If called as submodule or .. if called from experiments/
		if cwd == "icml_2019_state_abstraction":
			learned_policy_path = './mac/learned_policy/'
		elif cwd == "Bachelor-Project":
			learned_policy_path = './icml_2019_state_abstraction/mac/learned_policy/'
		self.learned_policy_path = learned_policy_path

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
		li_time = []
		accumulated_rewards= 0

		for episode in range(1,self.params['max_learning_episodes']):
			# Do one episode of interaction
			start_time = time.time()
			states, actions, returns, rewards = self.interactOneEpisode(episode)
			end_time = time.time()
			
			# Update epsilon for epsilon greedy policy
			# print("This is the epsilon in actor", self.actor.params['epsilon'], "this is the epsilon in mac", self.epsilon)
			self.actor.params['epsilon'] = max(1 - episode/(self.params["max_learning_episodes"]*self.epsilon), 0.01)

			# Accumulated rewards
			accumulated_rewards += numpy.sum(rewards)

			# Time
			episode_time = end_time - start_time
			li_time.append(episode_time)

			#add to memory
			self.add_2_memory(states,actions,rewards)

			#log performance
			if episode % 10 == 0:
				print(episode,"return in last 10 episodes",numpy.mean(li_returns[-10:]), "with accumulated rewards", accumulated_rewards, "this was the last 10 epiode time", numpy.sum(li_time[-10:]))

			li_returns.append(returns[0])
			li_rewards.append(numpy.sum(rewards))
			sys.stdout.flush()

			#train the Q network
			if self.params['critic_train_type']=='model_free_critic_monte_carlo':
				self.critic.train_model_free_monte_carlo(states,actions,returns)

			elif self.params['critic_train_type']=='model_free_critic_TD':
				self.critic.train_model_free_TD(self.memory,self.actor,self.params,self.params,episode)

			self.actor.train(states,self.critic)

		print("training is finished successfully!")
		# Return the rewards 
		return li_returns, li_rewards

	def interactOneEpisode(self,episode):
		'''
			given the mac agent and an environment, executes their
			interaction for an episode, and then returns important information
			used for training.
		'''

		s0 = self.params['env'].reset()
		rewards = []
		states = []
		actions = []
		t = 0
		s=s0
		s_max = -0.5
		s_min = -0.5
		while True:
			a = self.actor.select_action(s)
			s_p , r , done , info = self.params['env'].step(a)

			if episode%250==0:
				self.params['env'].render()

			ra = self.rewardToMounainCar(s, s_p, r, a, episode)
			
			states.append(s)
			actions.append(a)
			rewards.append(ra)
			# update for next iteration
			s, t = (s_p,t+1)

			# when done
			if done==True:
				if not "TimeLimit.truncated" in info:
					rewards[-1]=100
					print("Reached the goal!")
				states.append(s_p)#we actually store the terminal state!
				a=self.actor.select_action(s_p)
				actions.append(a),
				rewards.append(0)
				break

		returns = self.rewardToReturn(rewards)

		## Save to disk every 200 episodes
		if episode%200==0:
			model_json = self.actor.network.to_json()
			with open(self.learned_policy_path+self.params['env_name']+".json", "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			self.actor.network.save_weights(self.learned_policy_path+self.params['env_name']+".h5")
			print("Saved latest policy network to disk")

		return states, actions, returns, rewards
	def rewardToReturn(self, rewards):
		T=len(rewards)
		returns=T*[0]
		returns[T-1]=rewards[T-1]
		for t in range(T-2,-1,-1):
			returns[t]=rewards[t]+self.params['gamma']*returns[t+1]
		return returns

	def rewardToMounainCar(self, current_state, next_state, reward, action, episode):
		
		valley = -0.5
		velocity_max = 0.07
		velocity_min = -0.07
		goal_position = 0.5
		action_right = action == 2
		no_action = action == 1
		
		## Simple reward reaching left hight 

		## Exp reward for position

		## Exp reward for velocity when crossing valley

		## Potential energi reward

		if self.params["env_name"] == "MountainCar-v0":
			## moving right
			# pos_change = (next_state[0] - current_state[0])
			# moving_right = pos_change > 0
			
			# reward for position
			# pos_reward = self.rewardPositionMountainCar(next_state[0])
			# vel_valley_reward = self.rewardVelocityWhenCrossingValley(current_state, next_state)
			# vel_reward = self.rewardVelocityMountainCar(current_state[1])
			# reward += pos_reward + vel_valley_reward + vel_reward

			# potential_energy = self.findPotentialEnergyMountainCar(current_state[0])
			# kinetic_energy = self.findKineticEnergyMountainCar(current_state[1])
			vel_reward = self.rewardVelocityMountainCar(current_state[1])
			reward += vel_reward   

			# Reward for reaching the goal
			if next_state[0] >= 0.5:
				reward = 1000

			# r += position_reward + velocity_reward
			if self.params["verbose"] and episode%100==0 or episode==1:
				# print("posistion: ",corrected,"velocity",s_p[1],"action: ",a,"reward =",position_reward, "+", velocity_reward)
				print("posistion: ",next_state, "velocity",next_state[1],"action: ",action,"reward: ",reward ,"velocity_reward:", vel_reward)
			# position= max(0, 0.5 - abs(s_p[0] - 0.5))  # Reward for being close to the goal position
			# velocity= max(0, 0.07 - abs(s_p[1]))
		
		return reward
	def findPotentialEnergyMountainCar(self, pos):
		m = 1000
		g = 0.0025
		# The funciton of the mountain road
		h = numpy.sin(numpy.pi * pos) + 1

		# Correct pos to be 0 in valley
		potential_energy = m * g * (h *100)
		# Reward for potential energy
		
		return potential_energy
	def findKineticEnergyMountainCar(self, velocity):
		m = 1000
		velocity_ratio = abs(velocity) / 0.07
		velocity_percentage = velocity_ratio * 100
		kinetic_energy = 0.5 * m * abs(velocity_ratio)**2
		# Reward for kinetik energy
		return kinetic_energy

	def rewardVelocityMountainCar(self, current_velocity):
		valley = -0.5
		velocity_max = 0.07
		velocity_min = -0.07
		velocity_reward = 0
		
		velocity_ratio = abs(current_velocity) / velocity_max
		velocity_reward = (2+ velocity_ratio)**2

		# Reward for velocity
		return velocity_reward


	def rewardVelocityWhenCrossingValley(self, current_state, next_state):
		valley = -0.5
		velocity_max = 0.07
		velocity_min = -0.07
		current_pos = current_state[0]
		next_pos = next_state[0]
		current_velocity = current_state[1]
		next_velocity = next_state[1]
		velocity_reward = 0
		velocity_ratio = 0
		# Reward for velocity when crossing valley
		if current_pos < valley and next_pos >= valley:
			
			velocity_ratio = next_velocity / velocity_max
			velocity_reward = 100 * velocity_ratio
		
		elif current_pos > valley and next_pos <= valley:
			
			velocity_ratio = next_velocity / velocity_min
			velocity_reward = 100 * velocity_ratio

		return velocity_reward
	def rewardPositionMountainCar(self, pos):
		valley = -0.5
		# Correct pos to be 0 in valley
		corrected_pos = pos - valley
		
		# Reward for being close to the goal position
		pos_reward = (1 / (1-abs(corrected_pos)**2))
		
		return pos_reward
		
