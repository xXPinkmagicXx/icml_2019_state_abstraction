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

		# self.activation_output = 'softmax' if self.action_size > 2 else 'sigmoid'
		# self.output_nodes = num_abstract_states if num_abstract_states > 2 else 1
		self.activation_output = 'softmax'
		self.output_nodes = num_abstract_states
		self.optimizer = 'sgd'

		# if self.action_size > 2:
		self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
		# else:
		# 	self.loss_fn = tf.keras.losses.BinaryCrossentropy()
		
		self.net = tf.keras.models.Sequential()
		self.net.add(tf.keras.layers.Flatten(input_shape=(self.obs_size,)))
		
		# # hidden layers
		self.net.add(tf.keras.layers.Dense(self.num_nodes, activation='relu'))
		self.net.add(tf.keras.layers.Dense(self.num_nodes, activation='relu'))
		
		# output layer
		self.net.add(tf.keras.layers.Dense(self.output_nodes, activation=self.activation_output))

		self.net.compile(
			loss=self.loss_fn,
			optimizer=self.optimizer,
			metrics=['accuracy']
		)
		# output summary in console
		self.net.summary()
	def predict(self, x):
		x = np.array(x).reshape(1, self.obs_size)
		li = self.net.predict(x)
		
		return li

