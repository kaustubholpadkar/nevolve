import numpy as np
from nevolve.neuro import activations


class DNN:
	"""
	Class for Deep Neural Network
	"""

	def __init__(self, config):
		"""
		Constructor of DNN
		:param config: Neural Network Configuration
		"""

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
		Predict the output for given input
		:param x: input data
		:return: prediction
		"""

		A = x
		for weight, bias, activation in zip(self.weights, self.biases, self.activations):
			Z = np.matmul(A, weight) + bias
			A = activations.activate(Z, activation)
		return A

	def get_weights(self):
		return self.weights, self.biases

	def set_weights(self, weights, biases):
		self.weights, self.biases = weights, biases
