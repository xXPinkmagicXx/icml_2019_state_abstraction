import keras
import sys
from keras.models import model_from_json
import numpy


# load json and create model
# json_file = open('../mac/learned_policy/LunarLander-v2.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# load weights into new model

# def load_weights():
# 	loaded_model.load_weights('../mac/learned_policy/LunarLander-v2.h5')


def expert_mountaincar_policy(state):
	
	s_size=len(state.data)


	#print(temp)
	return numpy.random.choice(4)