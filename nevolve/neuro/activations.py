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
	"""Compute softmax values for each sets of scores in x."""
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


activation_map = {
	"sigmoid": sigmoid,
	"tanh": tanh,
	"softmax": softmax,
	"linear": lambda x: x
}


def activate(Z, activation):
	return activation_map[activation](Z)