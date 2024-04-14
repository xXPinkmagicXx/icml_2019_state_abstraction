import tensorflow as tf
import sys, numpy

class abstraction_network(tf.keras.Model):
	def __init__(self,params,num_abstract_states):
		
		# init sequential model
		self.model = tf.keras.Sequential()
		
		self.model.add(tf.keras.layers.Dense(params['abstraction_network_hidden_nodes'], activation='relu', input_shape=(params['obs_size'],)))
		
		for _ in range(params['abstraction_network_hidden_layers']-1):
			self.model.add(tf.keras.layers.Dense(params['abstraction_network_hidden_nodes'], activation='relu'))

		self.model.add(tf.keras.layers.Dense(num_abstract_states), activation='softmax')

		self.loss_fn = tf.keras.losses.Adam(learning_rate=params['learning_rate_for_abstraction_learning'], from_logits=True)		

		self.compile()

	def compile(self):
		self.model.compile(
			optimizer='adam',
			loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
			metrics=['accuracy'])

	def call(self, x):
		return self.model(x)

	@tf.function
	def train_step(self, training_data):
		self.model.fit(
			training_data,


		)
		

	def predict(self, samples):
		predictions = self.model.predict(samples)	
		return predictions
		## TODO: Params in
		# self.obs=tf.compat.v1.placeholder(tf.float32, [None, self.obs_size], name = 'obs')
		# self.Pr_a_given_z=tf.compat.v1.placeholder(tf.float32, [None,self.num_abstract_states], name = 'prob_of_all_a_given_z')
		# h=self.obs
		# for _ in range(params['abstraction_network_hidden_layers']):
		# 	# h=tf.layers.dense(
		# 	#     inputs = h,
		# 	#     units = params['abstraction_network_hidden_nodes'],
		# 	#     activation = tf.nn.relu)
			
			# h=tf.keras.layers.Dense(
			# 	units = params['abstraction_network_hidden_nodes'],
			# 	activation = tf.nn.relu)(h)
			

		# self.logits=tf.keras.layers.Dense(
		# 	# inputs = h,
		# 	units = self.num_abstract_states)(h)

		self.Pr_z_given_s = tf.nn.softmax(self.logits)
		self.Pr_z_given_s=tf.clip_by_value(
				self.Pr_z_given_s,
				clip_value_min=0.0001,
				clip_value_max=0.9999,
			)

		# self.loss=-tf.reduce_mean(tf.multiply(self.Pr_a_given_z,tf.compat.v1.log(self.Pr_z_given_s)))							  
		# self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
		

		#self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
	def predict(self,samples):
		# logits
		predictions = self.model(samples)
		tf.nn.softmax(predictions)
		return li

	def train(self,samples, a_in_z):
		s_li=[]
		prob_a_in_z_for_mdp_li=[]
		# print("this is the samples: ", samples)
		print("This is a_in_z: ", a_in_z)
		for sample in samples:

			s_li.append(numpy.array(sample[0]))
			action=sample[1]
			mdp_number=sample[2]
			
			a_in_z_for_mdp=a_in_z[:,mdp_number]
			
			prob_a_in_z_for_mdp=[float(action==x) for x in a_in_z_for_mdp]
			# print("prob a in z for mdp", prob_a_in_z_for_mdp)
			prob_a_in_z_for_mdp_li.append(numpy.array(prob_a_in_z_for_mdp))
		
		# print("this is the list:", prob_a_in_z_for_mdp_li)
		_ ,l = self.sess.run([self.optimizer,self.loss],
					  		feed_dict={self.obs:s_li,
							self.Pr_a_given_z:prob_a_in_z_for_mdp_li})
		print("this is l: ", l)
		return l

