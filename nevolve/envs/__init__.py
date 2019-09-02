"""
Package including various Environments for Neuro-Evolution.
"""


from abc import ABCMeta

from nevolve.neuro import nn, dnn


class Environment(metaclass=ABCMeta):
	"""
	Generic Environment class.

	Attributes:
		brain: NeuralNetwork instance
		env: Environment instance
		configuration: Neural Network Configuration
		cls: Environment Class Reference
		dead: bool - if agent is dead
		action: Action to be taken at next step
		fitness: Fitness of agent
		score: Score of agent
		observation: Observation of Environment
	"""

	def __init__(self):
		self.brain = None
		self.env = None
		self.configuration = None
		self.cls = None
		self.dead = False
		self.action = None
		self.fitness = 0
		self.score = 0
		self.observation = None

	def get_brain(self):
		"""
		Get Neural Network Architecture

		Returns:
			tuple - weights, biases, activations
		"""
		return self.brain.model.serialize()

	def set_brain(self, data):
		"""
		Set Neural Network Architecture

		Args:
			data: tuple - weights, biases, activations
		"""
		self.brain.model.deserialize(data)

	def create_brain(self):
		"""
		Create NeuralNetwork instance

		Returns:
			instance of NeuralNetwork
		"""
		model = dnn.DNN(config=self.configuration)
		return nn.NeuralNetwork(model)

	def think(self):
		"""
		Think!

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()

	def act(self):
		"""
		Act!

		Raises:
			NotImplementedError
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

		Raises:
			NotImplementedError
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

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()

	def reset(self):
		"""
		Reset the Environment
		"""
		raise NotImplementedError()

	def calculate_fitness(self):
		"""
		Calculate Fitness!

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()

	def get_config(self):
		"""
		Get Neural Network Configuration

		Raises:
			NotImplementedError
		"""
		raise NotImplementedError()


class GymEnvironment(Environment, metaclass=ABCMeta):
	"""
	Environment class for OpenAI Gym Environments.

	Attributes:
		brain: NeuralNetwork instance
		env: Environment instance
		configuration: Neural Network Configuration
		cls: Environment Class Reference
		dead: bool - if agent is dead
		action: Action to be taken at next step
		fitness: Fitness of agent
		score: Score of agent
		observation: Observation of Environment
	"""

	def __init__(self):
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

	def reset(self):
		"""
		Reset the Environment
		"""
		self.observation = self.env.reset()
		self.dead = False
