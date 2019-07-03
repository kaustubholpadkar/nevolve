import gym
import numpy as np
from nevolve.envs import GymEnvironment


gym.logger.set_level(40)


class MountainCarV0(GymEnvironment):
	"""
	Wrapper for MountainCar-v0 environment of OpenAI Gym
	"""

	def __init__(self):
		"""
		Constructor of MountainCarV0
		"""
		super().__init__()

		self.env = gym.make("MountainCar-v0")
		self.observation = self.env.reset()

		self.configuration = self.get_config()
		self.brain = self.create_brain()

	def think(self):
		"""
		Function for inference based on current observation
		"""
		inputs = self.observation
		inputs[0] += 1.2
		inputs[0] /= 1.8
		inputs[1] += 0.07
		inputs[1] /= 0.14
		inputs = np.reshape(inputs, newshape=(1, -1))
		outputs = self.brain.predict(inputs)
		self.action = np.argmax(outputs)

	def act(self):
		"""
		Function to apply the action and get observation and rewards
		"""
		self.observation, reward, self.dead, info = self.env.step(self.action)
		self.fitness = max(self.observation[0], self.fitness)
		self.score = self.fitness

	def calculate_fitness(self):
		"""
		Function to set fitness while applying natural selection
		"""
		self.fitness = self.score

	def get_config(self):
		"""
		Get Neural Network Configuration
		:return: list
		"""
		input_size = self.env.observation_space.shape[0]
		hidden_size = 10
		output_size = self.env.action_space.n
		return [
			{"input_dim": input_size, "output_dim": hidden_size, "activation": "sigmoid"},
			{"input_dim": hidden_size, "output_dim": output_size, "activation": "softmax"}
		]
