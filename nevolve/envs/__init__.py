from abc import ABCMeta

from nevolve.neuro import nn, dnn


class Environment(metaclass=ABCMeta):

	def __init__(self):
		self.brain = None
		self.env = None
		self.configuration = None
		self.cls = None
		self.dead = False
		self.action = None
		self.fitness = 0
		self.score = 0

	def get_brain(self):
		return self.brain.model.serialize()

	def set_brain(self, data):
		self.brain.model.deserialize(data)

	def create_brain(self):
		model = dnn.DNN(config=self.configuration)
		return nn.NeuralNetwork(model)

	def think(self):
		raise NotImplementedError()

	def act(self):
		raise NotImplementedError()

	def mutate(self, mutation_rate):
		self.brain.mutate(mutation_rate)

	def show(self):
		raise NotImplementedError()

	def clone(self):
		clone = self.__class__()
		clone.brain = self.brain.copy()
		return clone

	def close(self):
		raise NotImplementedError()

	def calculate_fitness(self):
		raise NotImplementedError()

	def get_config(self):
		raise NotImplementedError()


class GymEnvironment(Environment, metaclass=ABCMeta):

	def __init__(self):
		super().__init__()

	def show(self):
		self.env.render()

	def close(self):
		self.env.close()
