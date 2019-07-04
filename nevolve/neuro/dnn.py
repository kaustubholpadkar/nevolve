"""
Module for Deep Neural Network implementation.
"""


import numpy as np
from nevolve.neuro import activations


class DNN:
	"""
	Class for Deep Neural Network


	Args:
		config: dict - Neural Network Configuration

	Attributes:
		weights: list of weights for all each layer
		biases: list of biases for all each layer
		activations: list of activation functions for all each layer
	"""

	def __init__(self, config):

		self.weights = []
		self.biases = []
		self.activations = []

		for cfg in config:
			input_dim = cfg["input_dim"]
			output_dim = cfg["output_dim"]
			activation = cfg["activation"]

			self.weights.append(np.random.normal(size=(input_dim, output_dim)))
			self.biases.append(np.random.normal(size=(1, output_dim)))
			self.activations.append(activation)

	def predict(self, x):
		"""
		Predict the output for given input - Forward Pass of Neural Network

		Args:
			x: numpy array - input data

		Returns:
			prediction - numpy array
		"""

		A = x
		for weight, bias, activation in zip(self.weights, self.biases, self.activations):
			Z = np.matmul(A, weight) + bias
			A = activations.activate(Z, activation)
		return A

	def get_weights(self):
		"""
		Get the Weights

		Returns:
			tuple - (weights, biases)
		"""
		return self.weights, self.biases

	def set_weights(self, weights, biases):
		"""
		Set the Weights

		Args:
			weights: numpy array
			biases: numpy array
		"""
		self.weights, self.biases = weights, biases

	def serialize(self):
		"""
		Serialize the Neural Network

		Returns:
			tuple
		"""
		return self.weights, self.biases, self.activations

	def deserialize(self, data):
		"""
		Deserialize the Neural Network

		Args:
			data: tuple - (weights, biases, activations)
		"""
		weights, biases, activations = data
		self.weights = [weight for weight in weights]
		self.biases = [bias for bias in biases]
		self.activations = [activation for activation in activations]
