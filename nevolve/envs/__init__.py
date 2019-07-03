from abc import ABCMeta

from nevolve.neuro import nn, dnn


class Environment(metaclass=ABCMeta):
	"""
	Generic Environment class
	"""

	def __init__(self):
		"""
		Constructor of Environment
		"""
		self.brain = None
		self.env = None
		self.configuration = None
		self.cls = None
		self.dead = False
		self.action = None
		self.fitness = 0
		self.score = 0

	def get_brain(self):
		"""
		Get Neural Network Architecture
		:return: tuple - weights, biases, activations
		"""
		return self.brain.model.serialize()

	def set_brain(self, data):
		"""
		Set Neural Network Architecture
		:param data: tuple - weights, biases, activations
		"""
		self.brain.model.deserialize(data)

	def create_brain(self):
		"""
		Create NeuralNetwork instance
		:return: instance of NeuralNetwork
		"""
		model = dnn.DNN(config=self.configuration)
		return nn.NeuralNetwork(model)

	def think(self):
		"""
		Think!
		"""
		raise NotImplementedError()

	def act(self):
		"""
		Act!
		"""
		raise NotImplementedError()

	def mutate(self, mutation_rate):
		"""
		Mutate the Brain!
		:param mutation_rate: mutation rate
		"""
		self.brain.mutate(mutation_rate)

	def show(self):
		"""
		Show!
		"""
		raise NotImplementedError()

	def clone(self):
		"""
		Clone the instance of Environment
		:return: clone of current Environment
		"""
		clone = self.__class__()
		clone.brain = self.brain.copy()
		return clone

	def close(self):
		"""
		Close!
		"""
		raise NotImplementedError()

	def calculate_fitness(self):
		"""
		Calculate Fitness!
		"""
		raise NotImplementedError()

	def get_config(self):
		"""
		Get Neural Network Configuration
		"""
		raise NotImplementedError()


class GymEnvironment(Environment, metaclass=ABCMeta):
	"""
	Environment class for OpenAI Gym Environments
	"""

	def __init__(self):
		"""
		Constructor of GymEnvironment
		"""
		super().__init__()

	def show(self):
		"""
		Render the Environment
		"""
		self.env.render()

	def close(self):
		"""
		Close the Environment
		"""
		self.env.close()
