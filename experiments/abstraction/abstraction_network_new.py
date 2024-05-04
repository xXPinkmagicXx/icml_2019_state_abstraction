import tensorflow as tf
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
import sys
import numpy as np
import keras

class abstraction_network_new():
	
	def __init__(self, params, num_abstract_states):
		
		self.save_path = params['save_path']
		self.policy_train_episodes = params['policy_train_episodes'] # the number of steps the pre-trained policy is trained for
		self.obs_size = params['obs_size']
		self.action_size = params['size_a'] 
		self.num_nodes = params['abstraction_network_hidden_nodes']
		self.learning_rate = params['learning_rate_for_abstraction_learning']

		self.activation_output = 'softmax' if self.action_size > 2 else 'softmax'
		self.output_nodes = num_abstract_states if num_abstract_states > 2 else 2
		
		self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
		
		if self.action_size >= 2:
			self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
		else:
			self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=False)
		
		self.net = keras.models.Sequential()
		# self.net.add(tf.keras.layers.Flatten(input_shape=(self.obs_size,)))
		
		# # hidden layers
		self.net.add(keras.layers.Dense(self.num_nodes, activation='relu', input_dim=self.obs_size))
		for _ in range(params['abstraction_network_hidden_layers']-1):
		
			self.net.add(keras.layers.Dense(self.num_nodes, activation='relu'))
		
		# output layer
		self.net.add(keras.layers.Dense(self.output_nodes, activation=self.activation_output))
		# compile the model
		self.net.compile(
			loss=self.loss_fn,
			optimizer=self.optimizer,
			metrics=['accuracy']
		)
		# output summary in console
		self.net.summary()
		print("Created network with loss function", self.loss_fn, "and optimizer", self.optimizer, "and activation function", self.activation_output, "and output nodes", self.output_nodes)
	def predict(self, x):
		
		x = np.array(x).reshape(1, self.obs_size)
		li = self.net.predict(x)
		
		return li
	
	def save_model(self, abstraction_net_training_time):
		
		print("Saving abstraction network to disk...")
		print("This is the save path", self.save_path)
		keras.models.save_model(self.net, self.save_path + ".keras")
		# self.net.save(self.save_path)	
		print("Saved abstraction network to disk")
		with open(self.save_path + "/abstraction_training_time.txt", "w") as f:
			f.write(str(abstraction_net_training_time))
		print("Saved abstraction network training time to disk")
	
	def load_model(self, path):
		
		self.net = keras.models.load_model(path)

