from nevolve.neuro import nn, dnn


class Environment:

	def __init__(self):
		self.brain = None
		self.env = None
		self.configuration = None

	def get_brain(self):
		return self.brain.model.serialize()

	def set_brain(self, data):
		self.brain.model.deserialize(data)

	def create_brain(self):
		model = dnn.DNN(config=self.configuration)
		return model

	def think(self):
		pass

	def act(self):
		pass

	def mutate(self, mutation_rate):
		self.brain.mutate(mutation_rate)

	def show(self):
		self.env.render()

	def clone(self):
		pass

	def close(self):
		self.env.close()

	def calculate_fitness(self):
		pass
