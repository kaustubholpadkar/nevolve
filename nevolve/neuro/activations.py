"""
Module for various activation functions for Neural Networks.
"""


import numpy as np


def sigmoid(x, derivative=False):
	"""
	Sigmoid activation function.
	Args:
		x: numpy array
		derivative: bool - True if derivative of sigmoid function is needed

	Returns:
		numpy array - sigmoid or derivative of sigmoid for given input
	"""
	sig = 1. / (1. + np.exp(-x))
	if derivative:
		return sig * (1. - sig)
	return sig


def softmax(x):
	"""
	Softmax activation function.

	Args:
		x: numpy array

	Returns:
		numpy array - softmax for given input

	"""
	x = x[0]
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def tanh(x, derivative=False):
	"""
	Tanh activation function.
	Args:
		x: numpy array
		derivative: bool - True if derivative of sigmoid function is needed

	Returns:
		numpy array - tanh or derivative of tanh for given input

	"""
	sig = (2. / (1. + np.exp(-2 * x))) - 1.
	if derivative:
		return 1. - sig ** 2
	return sig


def relu(x):
	"""
	Relu activation function.
	Args:
		x: numpy array

	Returns:
		numpy array - Relu for given input

	"""
	return np.maximum(x, 0)


activation_map = {
	"sigmoid": sigmoid,
	"tanh": tanh,
	"softmax": softmax,
	"linear": lambda x: x,
	"relu": relu
}
"""A dictionary mapping string to activation function"""


def activate(z, activation):
	"""
	Function to apply activation function on numpy array z.

	Args:
		z: numpy array
		activation: str - activation function

	Returns:
		numpy array

	"""
	return activation_map[activation](z)
