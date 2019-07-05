"""
Module for Neural Network implementation with Evolutionary Functionality like Mutation and Crossover.
"""


import copy
import numpy as np
from nevolve.neuro.dnn import DNN


class NeuralNetwork:
	"""
	Class for Neural Network

	Args:
		model: instance of DNN

	Attributes:
		model: instance of DNN
	"""

	def __init__(self, model):
		assert isinstance(model, DNN)
		self.model = model

	def copy(self):
		"""
		Create a copy of NeuralNetwork

		Returns:
			instance of NeuralNetwork
		"""
		return NeuralNetwork(copy.deepcopy(self.model))

	def mutate(self, rate):
		"""
		Mutate the Neural Network

		Args:
			rate: mutation rate
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

		Args:
			inputs: numpy array

		Returns:
			numpy array
		"""
		return self.model.predict(inputs)

	def _mutate_np_array(self, arr, rate):
		"""
		Mutate the given numpy array

		Args:
			arr: numpy array
			rate: mutation rate
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
