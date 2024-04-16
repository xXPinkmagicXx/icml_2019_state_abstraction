import tensorflow as tf
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import sys
import numpy as np

class abstraction_network_new():
	
	def __init__(self, params, num_abstract_states):
		
		self.obs_size = params['obs_size']
		self.action_size = params['size_a'] 
		self.num_nodes = params['abstraction_network_hidden_nodes']

		self.activation_output = 'softmax' if self.action_size > 2 else 'sigmoid'
		self.output_nodes = num_abstract_states if num_abstract_states > 2 else 1
		
		self.optimizer = 'RMSprop'

		if self.action_size > 2:
			self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		else:
			self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
		
		self.net = tf.keras.models.Sequential()
		# self.net.add(tf.keras.layers.Flatten(input_shape=(self.obs_size,)))
		
		# # hidden layers
		self.net.add(tf.keras.layers.Dense(self.num_nodes, activation='relu', input_dim=self.obs_size))
		self.net.add(tf.keras.layers.Dense(self.num_nodes, activation='relu'))
		
		# output layer
		self.net.add(tf.keras.layers.Dense(self.output_nodes, activation=None))
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

