import numpy as np


def sigmoid(x, derivative=False):
	"""
	Sigmoid activation function
	:param x: input
	:param derivative: bool - True if derivative of sigmoid function is needed
	:return: sigmoid or derivative of sigmoid for given input
	"""
	sig = 1. / (1. + np.exp(-x))
	if derivative:
		return sig * (1. - sig)
	return sig


def softmax(x):
	"""
	Softmax activation function
	:param x: input
	:return: softmax for given input
	"""
	x = x[0]
	e_x = np.exp(x - np.max(x))
	return e_x / e_x.sum()


def tanh(x, derivative=False):
	"""
	Tanh activation function
	:param x: input
	:param derivative: bool - True if derivative of sigmoid function is needed
	:return: sigmoid or derivative of sigmoid for given input
	"""
	sig = (2. / (1. + np.exp(-2 * x))) - 1.
	if derivative:
		return 1. - sig ** 2
	return sig


# dictionary mapping string to activation function
activation_map = {
	"sigmoid": sigmoid,
	"tanh": tanh,
	"softmax": softmax,
	"linear": lambda x: x,
	"relu": lambda x: np.maximum(x, 0)
}


def activate(z, activation):
	"""
	Function to apply activation function on numpy array z
	:param z: numpy array
	:param activation: activation function (str)
	:return: numpy array
	"""
	return activation_map[activation](z)
