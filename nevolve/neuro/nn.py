import copy
import numpy as np
from nevolve.neuro.dnn import DNN


class NeuralNetwork:

	def __init__(self, model):
		assert isinstance(model, DNN)
		self.model = model

	def copy(self):
		return NeuralNetwork(copy.deepcopy(self.model))

	def mutate(self, rate):
		weights, biases = self.model.get_weights()

		for i in range(len(weights)):
			self._mutate_np_array(weights[i], rate)
			self._mutate_np_array(biases[i], rate)

		self.model.set_weights(weights, biases)

	def dispose(self):
		pass

	def predict(self, inputs):
		return self.model.predict(inputs)

	def _mutate_np_array(self, arr, rate):

		if len(arr.shape) == 1:
			for x in range(arr.shape[0]):
				if np.random.random() < rate:
					arr[x] += np.random.normal()

		if len(arr.shape) == 2:
			for x in range(arr.shape[0]):
				for y in range(arr.shape[1]):
					if np.random.random() < rate:
						arr[x][y] += np.random.normal()
