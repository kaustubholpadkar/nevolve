import copy
import numpy as np
from nevolve.neuro.dnn import DNN


class NeuralNetwork:
	"""
	Class for Neural Network
	"""

	def __init__(self, model):
		"""
		Constructor of NeuralNetwork
		:param model: instance of DNN
		"""
		assert isinstance(model, DNN)
		self.model = model

	def copy(self):
		"""
		Create a copy of NeuralNetwork
		:return: instance of NeuralNetwork
		"""
		return NeuralNetwork(copy.deepcopy(self.model))

	def mutate(self, rate):
		"""
		Mutate the Neural Network
		:param rate: mutation rate
		"""
		weights, biases = self.model.get_weights()

		for i in range(len(weights)):
			self._mutate_np_array(weights[i], rate)
			self._mutate_np_array(biases[i], rate)

		self.model.set_weights(weights, biases)

	def dispose(self):
		"""
		Dispose the Neural Network
		"""
		pass

	def predict(self, inputs):
		"""
		Predict output for given input
		:param inputs: numpy array
		:return: numpy array
		"""
		return self.model.predict(inputs)

	def _mutate_np_array(self, arr, rate):
		"""
		Mutate the given numpy array
		:param arr: numpy array
		:param rate: mutation rate
		"""
		if len(arr.shape) == 1:
			for x in range(arr.shape[0]):
				if np.random.random() < rate:
					arr[x] += np.random.normal()

		if len(arr.shape) == 2:
			for x in range(arr.shape[0]):
				for y in range(arr.shape[1]):
					if np.random.random() < rate:
						arr[x][y] += np.random.normal()
